import os
import csv
import argparse
import codecs
import re
import logging

import numpy as np
import scipy.io as sio
from natsort import natsorted
from scipy.signal import butter, sosfiltfilt, welch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def recommend_bandpass_on_combineds(
    combined_list,
    fs,
    nperseg=1024,
    low_pct=0.05,
    high_pct=0.95,
    top_k_per_session=5,
    use_energy_metric=True,
    verbose=True
):
    """
    - combined_list: list of combined arrays (channels, samples). 不会触发 I/O。
    - fs: sampling rate (Hz)
    - nperseg: nperseg for welch (auto clipped to signal length)
    - low_pct/high_pct: cumulative energy percentiles for selecting low/high (e.g. 0.05/0.95)
    - top_k_per_session: per session, select top K channels by energy (if None -> use all channels)
    - use_energy_metric: whether to use time-domain energy (sum(x^2)) to rank channels (fast).
    返回: (low_freq, high_freq, metrics)
    """
    if fs <= 0:
        raise ValueError("fs must be > 0")
    nyq = fs / 2.0
    total_psd = None
    f_ref = None # reference frequency bins from the first successful welch
    spec_count = 0 # total spectra accumulated across sessions/channels
    sessions_used = 0

    for combined in combined_list:
        ch, total = combined.shape
        if total <= 0:
            continue

        # choose channels: rank by time-domain energy or variance (fast)
        if top_k_per_session is None or top_k_per_session >= ch:
            ch_indices = list(range(ch))
        else:
            if use_energy_metric:
                # energy = sum(x^2)
                energies = np.sum(np.asarray(combined, dtype=np.float64)**2, axis=1)
            else:
                # variance
                energies = np.var(np.asarray(combined, dtype=np.float64), axis=1)
            # take top-k indices
            k = min(top_k_per_session, ch)
            ch_indices = np.argsort(energies)[-k:].tolist()

        # for each selected channel compute welch and accumulate
        for c in ch_indices:
            sig = combined[c, :]
            if sig.size <= 0:
                continue
            seg_nperseg = min(nperseg, sig.size)
            try:
                f, Pxx = welch(sig, fs=fs, nperseg=seg_nperseg)
            except Exception as e:
                logging.debug("recommend_bandpass_on_combineds: welch failed for session channel %d: %s", c, e)
                continue

            if total_psd is None:
                total_psd = np.zeros_like(Pxx)
                f_ref = f
            else:
                if f.shape != f_ref.shape or not np.allclose(f, f_ref):
                    logging.debug("recommend_bandpass_on_combineds: mismatched freq bins, skipping")
                    continue

            total_psd += Pxx
            spec_count += 1

        sessions_used += 1

    if spec_count == 0 or total_psd is None:
        logging.warning("recommend_bandpass_on_combineds: no spectra collected; returning defaults (0, nyquist).")
        return 0.0, nyq, {"spec_count": 0, "sessions_used": sessions_used}

    # compute mean PSD across accumulated spectra
    mean_psd = total_psd / float(spec_count)
    total_energy = mean_psd.sum()
    if total_energy <= 0:
        logging.warning("recommend_bandpass_on_combineds: total energy <= 0; returning defaults")
        return 0.0, nyq, {"spec_count": spec_count, "sessions_used": sessions_used}

    cum = np.cumsum(mean_psd) / total_energy

    # find indices for percentiles
    low_idx = np.where(cum >= low_pct)[0]
    high_idx = np.where(cum >= high_pct)[0]

    if low_idx.size == 0:
        low_freq = float(f_ref[0])
    else:
        low_freq = float(f_ref[low_idx[0]])

    if high_idx.size == 0:
        high_freq = float(f_ref[-1])
    else:
        high_freq = float(f_ref[high_idx[0]])

    # boundary checks
    if high_freq >= nyq:
        logging.warning("recommend_bandpass_on_combineds: high_freq (%.3f) >= Nyquist (%.3f); clipping", high_freq, nyq)
        high_freq = max(nyq * 0.999, 0.0)

    low_freq = float(np.round(low_freq, 2))
    high_freq = float(np.round(high_freq, 2))

    metrics = {
        "spec_count": spec_count,
        "sessions_used": sessions_used,
        "f_max": float(f_ref[-1]),
    }
    if verbose:
        logging.info("recommend_bandpass_on_combineds -> spec_count=%d, sessions_used=%d, low=%.2f, high=%.2f",
                     spec_count, sessions_used, low_freq, high_freq)

    return low_freq, high_freq, metrics

# -------------------------------------------------
# 带宽滤波
# -------------------------------------------------
def bandpass(signal, fs, low=5.0, high=300.0, order=4):
    """
    Apply a Butterworth bandpass (zero-phase via sosfiltfilt) to a 1D signal.
    """
    sos = butter(order, [low, high], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, signal)

# -------------------------------------------------
# 解析 label txt
# -------------------------------------------------
def parse_label_txt(label_path):

    info = {
        "class": None,
        "channel": None,
        "intensity": None,
        "env": ""
    }

    try:
        with codecs.open(label_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:

                line = line.strip()

                if not line:
                    continue

                # 前三行都有 =
                if "=" in line:

                    key, val = line.split("=", 1)

                    key = key.lower()

                    if "data_path" in key:
                        info["class"] = val.strip()

                    elif "channel" in key:
                        info["channel"] = int(val)

                    elif "intensity" in key:
                        info["intensity"] = int(val)

                # 第四行（环境描述）
                else:
                    parts = line.split()
                    if parts:
                        info["env"] = parts[-1]

    except Exception as e:
        logging.exception("读取label失败: %s | %s", label_path, e)

    return info

# -------------------------------------------------
# 找 label
# -------------------------------------------------
def find_label_for_event_and_parse(label_root, class_name, session_folder):

    candidate_dir = os.path.join(label_root, class_name)

    if not os.path.isdir(candidate_dir):
        return None, None

    s1 = re.sub(r'[^0-9]', '', session_folder)

    for fn in os.listdir(candidate_dir):
        name, ext = os.path.splitext(fn)
        s2 = re.sub(r'[^0-9]', '', name)

        if s1 in s2 or s2 in s1:

            full = os.path.join(candidate_dir, fn)

            if ext.lower() == ".txt":

                parsed = parse_label_txt(full)

                return full, parsed

    logging.debug("No label found in %s for session %s", candidate_dir, session_folder)
    return None, None

# -------------------------------------------------
# 读取 mat
# -------------------------------------------------
def load_and_combine_mat_files(session_path):
    """
    Load all .mat files in session_path, extract variable 's' from each, and concatenate along time axis.
    Expectation: each mat['s'] is an array of shape (ch, samples).
    """

    mat_files = [f for f in os.listdir(session_path) if f.endswith(".mat")]

    mat_files = natsorted(mat_files)

    arrays = []

    for f in mat_files:

        file_path = os.path.join(session_path, f)

        try:

            mat = sio.loadmat(file_path)

            if "s" not in mat:
                logging.debug("MAT file %s does not contain 's', skipping", file_path)
                continue

            arr = mat["s"]
            if arr.ndim == 1:
                arr = arr[np.newaxis, :] # Ensure 2D
            arrays.append(arr)

        except Exception as e:
            logging.exception("读取失败: %s | %s", file_path, e)

    if not arrays:
        return None

    combined = np.concatenate(arrays, axis=1)

    return combined

# -------------------------------------------------
# 保存窗口
# -------------------------------------------------
def save_windows_and_metadata(
        combined_arr,
        out_dir,
        class_name,
        session_folder,
        window_len,
        hop,
        label_parsed,
        meta_writer
):

    ch, total = combined_arr.shape

    os.makedirs(out_dir, exist_ok=True)

    idx = 0

    for start in range(0, total - window_len + 1, hop):

        win = combined_arr[:, start:start + window_len]

        save_arr = win.T.astype(np.float32)

        fname = f"{class_name}__{session_folder}__{idx:06d}.npy"

        path = os.path.join(out_dir, fname)

        np.save(path, save_arr)

        cls = label_parsed.get("class") if label_parsed else ""

        chn = label_parsed.get("channel") if label_parsed else ""

        inten = label_parsed.get("intensity") if label_parsed else ""

        env = label_parsed.get("env") if label_parsed else ""

        meta_writer.writerow([
            fname,
            class_name,
            session_folder,
            start,
            start + window_len,
            cls,
            chn,
            inten,
            env
        ])

        idx += 1

    return idx
    # -------------------------------------------------
# 主函数
# -------------------------------------------------
def main(args):

    data_root = args.data_root
    label_root = args.label_root
    out_root = args.out_root

    window_len = args.window_len
    hop = args.hop

    fs = args.fs
    low = args.low
    high = args.high
    order = args.order
    apply_bp = args.apply_bandpass

    sample_combineds = []  # will store combined arrays for sampling
    deferred_sessions = []
    auto_bandpass_done = False
    auto_max_sessions = getattr(args, "auto_max_sessions", 50)
    auto_channels_per_session = getattr(args, "auto_channels_per_session", None)
    auto_nperseg = getattr(args, "auto_nperseg", 1024)
    auto_low_pct = getattr(args, "auto_low_pct", 0.05)
    auto_high_pct = getattr(args, "auto_high_pct", 0.95)
    auto_top_k_per_session = getattr(args, "auto_top_k_per_session", 5)
    auto_bandpass_enabled = getattr(args, "auto_bandpass", False)

    os.makedirs(out_root, exist_ok=True)

    metadata_path = os.path.join(out_root, "metadata.csv")

    with open(metadata_path, "w", newline='', encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow([
            "filename",
            "class",
            "session_folder",
            "start",
            "end",
            "label_class",
            "label_channel",
            "label_intensity",
            "label_env"
        ])

        for class_name in os.listdir(data_root):

            class_dir = os.path.join(data_root, class_name)

            if not os.path.isdir(class_dir):
                continue

            logging.info("处理类别: %s", class_name)

            out_class_dir = os.path.join(out_root, class_name)
            session_list = os.listdir(class_dir)

            for i, session_folder in enumerate(session_list):

                session_path = os.path.join(class_dir, session_folder)
                combined = load_and_combine_mat_files(session_path)

                if combined is None:
                    logging.warning("  无有效数据 for session %s", session_folder)
                    continue

                label_path, label_parsed = find_label_for_event_and_parse(
                    label_root,
                    class_name,
                    session_folder
                )
                sample_combineds.append(combined.copy())
                deferred_sessions.append((out_class_dir, class_name, session_folder, label_parsed))

                if auto_bandpass_enabled and not auto_bandpass_done:
                    logging.info("Auto bandpass enabled but not chosen yet: deferring session %s", session_folder)
                    is_last_session = (i == len(session_list) - 1)

                    if len(sample_combineds) >= auto_max_sessions or is_last_session:
                        low, high, rec_meta = recommend_bandpass_on_combineds(
                            sample_combineds,
                            fs=args.fs,
                            nperseg=auto_nperseg,
                            low_pct=auto_low_pct,
                            high_pct=auto_high_pct,
                            top_k_per_session=auto_top_k_per_session,
                            verbose=True
                        )
                        auto_bandpass_done = True
                        logging.info("Auto bandpass chosen (early): low=%.2f, high=%.2f, meta=%s", low, high, rec_meta)

            if deferred_sessions:
                logging.info("Processing %d deferred sessions with chosen bandpass", len(deferred_sessions))
                for (d_out_class_dir, d_class_name, d_session_folder, d_label_parsed) in deferred_sessions:
                    d_combined = sample_combineds.pop(0)  # get the corresponding combined array
                    try:
                        if apply_bp:
                            logging.info("  Applying bandpass to deferred session %s: %s - %s Hz", d_session_folder, low, high)
                            for c in range(d_combined.shape[0]):
                                d_combined[c, :] = bandpass(d_combined[c, :].astype(np.float64), fs=fs, low=low, high=high, order=order)
                    except Exception as e:
                        logging.exception("  Bandpass failed for deferred session %s: %s", d_session_folder, e)
                        continue

                    save_windows_and_metadata(
                        d_combined,
                        d_out_class_dir,
                        d_class_name,
                        d_session_folder,
                        window_len,
                        hop,
                        d_label_parsed,
                        writer
                    )
                deferred_sessions = []
                sample_combineds = []
                auto_bandpass_done = False

    logging.info("预处理完成")
    logging.info("metadata: %s", metadata_path)
# -------------------------------------------------
# CLI
# -------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", required=True)
    parser.add_argument("--label_root", required=True)
    parser.add_argument("--out_root", required=True)
    
    parser.add_argument("--window_len", type=int, default=512, help="window length in samples (default 512)")
    parser.add_argument("--hop", type=int, default=64, help="hop/stride in samples (default 64)")

    # --- bandpass params added for MAT files
    parser.add_argument("--fs", type=float, default=100.0, help="sampling rate in Hz for bandpass (must match your MAT data)")
    parser.add_argument("--low", type=float, default=5.0, help="bandpass low cutoff (Hz)")
    parser.add_argument("--high", type=float, default=49.0, help="bandpass high cutoff (Hz)")
    parser.add_argument("--order", type=int, default=4, help="Butterworth filter order")
    parser.add_argument("--apply_bandpass", action="store_true", help="apply bandpass to mat channels before windowing")

    parser.add_argument("--auto_bandpass", action="store_true", help="scan dataset and auto-select low/high via PSD")
    parser.add_argument("--auto_nperseg", type=int, default=1024, help="nperseg for welch when auto scanning")
    parser.add_argument("--auto_low_pct", type=float, default=0.05, help="lower cumulative energy percentile for auto band (e.g. 0.05)")
    parser.add_argument("--auto_high_pct", type=float, default=0.95, help="upper cumulative energy percentile for auto band (e.g. 0.95)")
    parser.add_argument("--auto_max_sessions", type=int, default=50, help="max sessions to scan when auto selecting band")
    parser.add_argument("--auto_channels_per_session", type=int, default=5, help="how many channels to sample per session when scanning (None = all)")

    args = parser.parse_args()

    main(args)