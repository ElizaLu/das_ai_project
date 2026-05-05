import os
import csv
import json
import logging
import argparse
import numpy as np

from utils.io_utils import (
    load_and_combine_mat_files,
    ensure_dir,
    save_metadata_header,
    crop_channels,
)
from utils.signal_utils import bandpass_2d
from utils.picking_utils import (
    detect_event_time_intervals,
    pick_event_channels,
    window_event_ratio,
)


CLASS2ID = {
    "nature": 0,
    "human": 1,
    "mechanical": 2,
}

IGNORED_CLASSES = {"both"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def save_windows_and_metadata(
    combined_arr,
    out_dir,
    class_name,
    class_id,
    session_folder,
    window_len,
    hop,
    meta_writer,
    event_intervals,
    event_channels,
    event_channel_scores,
    min_event_ratio=0.3,
):
    """
    CHANGED:
    1) class_id == 0 (nature): 仍然保留全部窗口，标签固定为 0。
    2) class_id == 1/2: 只有当窗口满足 ratio >= min_event_ratio 时才保留；
       否则直接跳过，不再写成 0。
    """
    _, total = combined_arr.shape
    ensure_dir(out_dir)

    idx = 0
    saved_count = 0  # CHANGED: 统计真正保存的窗口数

    for start in range(0, total - window_len + 1, hop):
        win = combined_arr[:, start:start + window_len]
        save_arr = win.T.astype(np.float32)  # (time, channel)

        window_t0 = start
        window_t1 = start + window_len - 1

        # 计算该窗口与事件区间的重叠比例
        ratio = window_event_ratio(window_t0, window_t1, event_intervals)

        # CHANGED: 只对 nature 保留全部窗口；
        #         对 human/mechanical，如果不满足阈值，直接丢弃
        if class_id in (1, 2) and ratio < min_event_ratio:
            idx += 1
            continue

        # 只有保留下来的窗口才真正保存
        fname = f"{class_name}__{session_folder}__{idx:06d}.npy"
        np.save(os.path.join(out_dir, fname), save_arr)

        # CHANGED: nature 仍然标 0；class 1/2 则保留原始 class_id
        window_label = 0 if class_id == 0 else class_id

        channels_str = "" if event_channels is None else ",".join(
            map(str, np.asarray(event_channels, dtype=int).tolist())
        )
        scores_str = "" if event_channel_scores is None else json.dumps(
            np.asarray(event_channel_scores, dtype=float).tolist()
        )

        meta_writer.writerow([
            fname,
            class_name,
            class_id,
            session_folder,
            window_t0,
            window_t1,
            channels_str,
            scores_str,
            window_label,
            f"{ratio:.6f}",
        ])

        saved_count += 1
        idx += 1

    return saved_count


def build_session_event_info(combined, fs, args):
    event_intervals = detect_event_time_intervals(
        combined,
        fs=fs,
        win_len=args.detector_win_len,
        thr_mul=args.detector_thr_mul,
        min_samples=args.detector_min_samples,
        aggregate=args.detector_aggregate,
        smooth_len=args.detector_smooth_len,
    )

    event_channels = None
    event_channel_scores = None
    channel_thr = None

    if event_intervals:
        t0 = min(a for a, b in event_intervals)
        t1 = max(b for a, b in event_intervals)

        event_channels, event_channel_scores, channel_thr = pick_event_channels(
            combined,
            t0=t0,
            t1=t1,
            method=args.channel_pick_method,
            ratio=args.channel_pick_ratio,
            min_channels=args.channel_pick_min_channels,
        )

    return event_intervals, event_channels, event_channel_scores, channel_thr


def main(args):
    data_root = args.data_root
    out_root = args.out_root

    ensure_dir(out_root)
    metadata_path = os.path.join(out_root, "metadata.csv")

    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        save_metadata_header(writer)

        for class_name in os.listdir(data_root):
            class_dir = os.path.join(data_root, class_name)
            if not os.path.isdir(class_dir):
                continue

            class_name_lower = class_name.lower()

            if class_name_lower == "nature":
                class_id = 0
                logging.info("Processing background class: %s -> %d", class_name, class_id)
            elif class_name_lower in IGNORED_CLASSES:
                logging.info("Skip ignored class folder: %s", class_name)
                continue
            elif class_name_lower not in CLASS2ID:
                logging.info("Skip unknown class folder: %s", class_name)
                continue
            else:
                class_id = CLASS2ID[class_name_lower]
                logging.info("Processing event class: %s -> %d", class_name, class_id)

            session_list = sorted(os.listdir(class_dir))
            for session_folder in session_list:
                session_path = os.path.join(class_dir, session_folder)
                if not os.path.isdir(session_path):
                    continue

                combined = load_and_combine_mat_files(session_path)
                combined = crop_channels(combined, 20, 101)
                if combined is None:
                    logging.warning("No valid data for session %s", session_folder)
                    continue

                if args.apply_bandpass:
                    try:
                        combined = bandpass_2d(
                            combined,
                            fs=args.fs,
                            low=args.low,
                            high=args.high,
                            order=args.order,
                        )
                    except Exception as e:
                        logging.exception("Bandpass failed for session %s: %s", session_folder, e)

                if class_name_lower == "nature":
                    event_intervals = []
                    event_channels = None
                    event_channel_scores = None
                    channel_thr = None
                else:
                    event_intervals, event_channels, event_channel_scores, channel_thr = build_session_event_info(
                        combined, fs=args.fs, args=args
                    )

                logging.info(
                    "Session %s | intervals=%s | channels=%s",
                    session_folder,
                    event_intervals,
                    None if event_channels is None else event_channels.tolist(),
                )

                session_json = {
                    "class_name": class_name,
                    "class_id": class_id,
                    "session_folder": session_folder,
                    "event_intervals": event_intervals,
                    "event_channels": None if event_channels is None else event_channels.tolist(),
                    "channel_threshold": channel_thr,
                    "note": (
                        "CHANGED: for class_id 1/2, windows with ratio < min_event_ratio are dropped "
                        "instead of being labeled as 0."
                    ),
                }
                with open(
                    os.path.join(out_root, f"{class_name}__{session_folder}__pick.json"),
                    "w",
                    encoding="utf-8",
                ) as jf:
                    json.dump(session_json, jf, ensure_ascii=False, indent=2)

                saved_count = save_windows_and_metadata(
                    combined_arr=combined,
                    out_dir=out_root,
                    class_name=class_name,
                    class_id=class_id,
                    session_folder=session_folder,
                    window_len=args.window_len,
                    hop=args.hop,
                    meta_writer=writer,
                    event_intervals=event_intervals,
                    event_channels=event_channels,
                    event_channel_scores=event_channel_scores,
                    min_event_ratio=args.min_event_ratio,
                )

                logging.info(
                    "Session %s finished | saved_windows=%d",
                    session_folder,
                    saved_count,
                )

    logging.info("Preprocessing finished. metadata saved to %s", metadata_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_root", required=True)

    parser.add_argument("--window_len", type=int, default=1024)
    parser.add_argument("--hop", type=int, default=128)

    parser.add_argument("--fs", type=float, default=1000.0)
    parser.add_argument("--low", type=float, default=5.0)
    parser.add_argument("--high", type=float, default=300.0)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--apply_bandpass", action="store_true")

    parser.add_argument("--detector_win_len", type=int, default=256)
    parser.add_argument("--detector_thr_mul", type=float, default=-0.5)
    parser.add_argument("--detector_min_samples", type=int, default=5)
    parser.add_argument("--detector_aggregate", type=str, default="p90", choices=["p90", "max", "mean", "median"])
    parser.add_argument("--detector_smooth_len", type=int, default=11)

    parser.add_argument("--channel_pick_method", type=str, default="peak_rms", choices=["rms", "peak_rms"])
    parser.add_argument("--channel_pick_ratio", type=float, default=0.3)
    parser.add_argument("--channel_pick_min_channels", type=int, default=1)

    parser.add_argument("--min_event_ratio", type=float, default=0.3)

    args = parser.parse_args()
    main(args)