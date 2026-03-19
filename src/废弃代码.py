import os
import csv
import codecs
import re
import json
import logging
import numpy as np
import scipy.io as sio
from natsort import natsorted
from scipy.signal import butter, sosfiltfilt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# =========================
# 改动点 1：三分类映射
# 0 = 无事件, 1 = 爬网, 2 = 翻网
# =========================
CLASS2ID = {
    "无事件": 0,
    "no_event": 0,
    "无": 0,
    "爬网": 1,
    "crawl": 1,
    "翻网": 2,
    "flip": 2,
}

def bandpass(signal, fs, low=5.0, high=300.0, order=4):
    sos = butter(order, [low, high], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, signal)

def parse_label_txt(label_path):
    """
    仍然沿用你原来的 label 解析方式。
    返回:
      class: 类别名
      channel: 事件通道
      intensity: 强度
      env: 环境信息
    """
    info = {"class": None, "channel": None, "intensity": None, "env": ""}
    try:
        with codecs.open(label_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
    except Exception as e:
        logging.exception("读取label失败: %s | %s", label_path, e)
        return info

    for line in text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            k = k.strip().lower()
            v = v.strip()
            if "data_path" in k or "class" in k:
                info["class"] = v
            elif "channel" in k:
                try:
                    info["channel"] = int(re.search(r"\d+", v).group())
                except:
                    pass
            elif "intensity" in k:
                try:
                    info["intensity"] = int(re.search(r"\d+", v).group())
                except:
                    pass
        else:
            parts = line.split()
            if parts:
                info["env"] = parts[-1]
    return info

def load_and_combine_mat_files(session_path):
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
                arr = arr[np.newaxis, :]
            arrays.append(arr)
        except Exception as e:
            logging.exception("读取失败: %s | %s", file_path, e)

    if not arrays:
        return None

    combined = np.concatenate(arrays, axis=1)
    return combined

def detect_event_intervals(combined, fs, channel_hint=None, win_len=256, thr_mul=2.5, min_samples=5):
    """
    用短时 RMS 在单个通道上粗略检测事件区间。
    返回: [(t0, t1), ...]，样本索引坐标
    """
    ch, total = combined.shape
    if channel_hint is None:
        return []

    c = int(channel_hint)
    # 防止越界；如果你的通道是 1-based，建议在 parse_label_txt 里先减 1
    c = max(0, min(c, ch - 1))

    sig = combined[c, :].astype(np.float64)
    sq = sig ** 2
    kernel = np.ones(win_len, dtype=np.float64)
    rms = np.sqrt(np.convolve(sq, kernel, mode='same') / float(win_len))

    mu = rms.mean()
    sigma = rms.std()
    thr = mu + thr_mul * sigma

    mask = rms > thr
    if not mask.any():
        return []

    idx = np.where(mask)[0]
    runs = []
    s = idx[0]
    p = idx[0]
    for i in idx[1:]:
        if i == p + 1:
            p = i
        else:
            runs.append((s, p))
            s = i
            p = i
    runs.append((s, p))

    runs = [r for r in runs if (r[1] - r[0] + 1) >= min_samples]
    if not runs:
        return []

    return [(int(a), int(b)) for a, b in runs]

def _intervals_overlap(t0, t1, a, b):
    return not (t1 < a or t0 > b)

def _clip_interval_to_window(t0, t1, win_start, win_end):
    """
    把 session 级区间裁剪到 window 内。
    返回 window 相对坐标区间 (a, b)，如果没有重叠则返回 None
    """
    a = max(t0, win_start)
    b = min(t1, win_end)
    if a > b:
        return None
    return int(a - win_start), int(b - win_start)

def choose_window_class(window_events):
    """
    一个窗口里如果有多个事件，这里选“主类”作为窗口级三分类标签。
    规则：按事件持续时间累计，谁最长就选谁。
    """
    if not window_events:
        return 0

    score = {}
    for ev in window_events:
        cid = int(ev["class_id"])
        dur = max(1, int(ev["t1"]) - int(ev["t0"]) + 1)
        score[cid] = score.get(cid, 0) + dur

    return max(score.items(), key=lambda x: x[1])[0]

def save_windows_and_metadata(combined_arr, out_dir, class_name, session_folder,
                              window_len, hop, label_parsed, meta_writer,
                              class_id,
                              fs=1000.0,
                              detector_win_len=256,
                              detector_thr_mul=2.5,
                              detector_min_samples=5):
    """
    保存滑窗并写 metadata。

    metadata 列：
      filename, class_name, label_class_id, session_folder, start, end, events_json

    events_json:
      一个 window 内所有事件的列表。
      每个事件格式：
        {"class_id": 1/2, "t0": 相对window起点, "t1": 相对window起点, "channel": 通道号}
    """
    ch, total = combined_arr.shape
    os.makedirs(out_dir, exist_ok=True)

    # =========================
    # 改动点 2：先得到 session 级事件列表
    # 这里仍然沿用“单通道 + RMS 检测”
    # =========================
    session_events = []
    if class_id > 0 and label_parsed and label_parsed.get("channel") is not None:
        detected = detect_event_intervals(
            combined_arr,
            fs=fs,
            channel_hint=label_parsed.get("channel"),
            win_len=detector_win_len,
            thr_mul=detector_thr_mul,
            min_samples=detector_min_samples
        )

        # 只要检测到了多个时间区间，就都作为事件
        for a, b in detected:
            session_events.append({
                "class_id": int(class_id),
                "t0": int(a),
                "t1": int(b),
                "channel": int(label_parsed["channel"]),
            })

        logging.info(
            "Detected %d event interval(s) for session %s: %s",
            len(session_events), session_folder, detected
        )

    idx = 0
    for start in range(0, total - window_len + 1, hop):
        win = combined_arr[:, start:start + window_len]
        save_arr = win.T.astype(np.float32)  # (T, C)

        fname = f"{class_name}__{session_folder}__{idx:06d}.npy"
        np.save(os.path.join(out_dir, fname), save_arr)

        win_end = start + window_len - 1

        # =========================
        # 改动点 3：窗口内事件裁剪
        # 支持一个窗口多个事件
        # =========================
        window_events = []
        for ev in session_events:
            clipped = _clip_interval_to_window(
                ev["t0"], ev["t1"], start, win_end
            )
            if clipped is None:
                continue

            a, b = clipped
            window_events.append({
                "class_id": int(ev["class_id"]),
                "t0": int(a),
                "t1": int(b),
                "channel": int(ev["channel"]),
            })

        # =========================
        # 改动点 4：窗口级三分类标签
        # 没有事件 -> 0
        # 有事件 -> 取主类
        # =========================
        window_class_id = choose_window_class(window_events)

        meta_writer.writerow([
            fname,
            class_name,
            window_class_id,
            session_folder,
            start,
            win_end,
            json.dumps(window_events, ensure_ascii=False),
        ])

        idx += 1

    return idx

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

    os.makedirs(out_root, exist_ok=True)
    metadata_path = os.path.join(out_root, "metadata.csv")

    with open(metadata_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename",
            "class_name",
            "label_class_id",
            "session_folder",
            "start",
            "end",
            "events_json",
        ])

        for class_name in os.listdir(data_root):
            class_dir = os.path.join(data_root, class_name)
            if not os.path.isdir(class_dir):
                continue

            class_id = CLASS2ID.get(class_name, -1)
            if class_id < 0:
                logging.warning("未知类别文件夹: %s，跳过", class_name)
                continue

            logging.info("处理类别: %s -> %d", class_name, class_id)

            session_list = natsorted(os.listdir(class_dir))
            for session_folder in session_list:
                session_path = os.path.join(class_dir, session_folder)
                combined = load_and_combine_mat_files(session_path)
                if combined is None:
                    logging.warning("  无有效数据 for session %s", session_folder)
                    continue

                label_parsed = {}

                # 找 label txt
                candidate_dir = os.path.join(label_root, class_name)
                if os.path.isdir(candidate_dir):
                    s1 = re.sub(r'[^0-9]', '', session_folder)
                    for fn in os.listdir(candidate_dir):
                        name, ext = os.path.splitext(fn)
                        s2 = re.sub(r'[^0-9]', '', name)
                        if s1 in s2 or s2 in s1:
                            full = os.path.join(candidate_dir, fn)
                            if ext.lower() == ".txt":
                                label_parsed = parse_label_txt(full)
                                break

                # 可选 bandpass
                if apply_bp:
                    try:
                        for c in range(combined.shape[0]):
                            combined[c, :] = bandpass(
                                combined[c, :].astype(np.float64),
                                fs=fs, low=low, high=high, order=order
                            )
                    except Exception as e:
                        logging.exception("Bandpass failed: %s", e)

                save_windows_and_metadata(
                    combined,
                    out_root,
                    class_name,
                    session_folder,
                    window_len,
                    hop,
                    label_parsed,
                    writer,
                    class_id=class_id,
                    fs=fs,
                    detector_win_len=256,
                    detector_thr_mul=2.5,
                    detector_min_samples=5,
                )

    logging.info("预处理完成，metadata saved to %s", metadata_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--label_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--window_len", type=int, default=1024)
    parser.add_argument("--hop", type=int, default=128)
    parser.add_argument("--fs", type=float, default=1000.0)
    parser.add_argument("--low", type=float, default=5.0)
    parser.add_argument("--high", type=float, default=300.0)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--apply_bandpass", action="store_true")
    args = parser.parse_args()
    main(args)