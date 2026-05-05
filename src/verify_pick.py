import os
import json
import argparse
import logging
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from utils.io_utils import load_and_combine_mat_files, crop_channels
from utils.signal_utils import bandpass_2d


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def contiguous_groups(idxs):
    idxs = np.asarray(sorted(set(map(int, idxs))), dtype=int)
    if idxs.size == 0:
        return []

    groups = []
    cur = [int(idxs[0])]

    for c in idxs[1:]:
        c = int(c)
        if c == cur[-1] + 1:
            cur.append(c)
        else:
            groups.append(cur)
            cur = [c]
    groups.append(cur)
    return groups


def block_reduce_time(arr, max_width=4000, reduce="mean"):
    """
    Downsample only along the time axis for visualization.
    arr: (channels, samples)
    """
    ch, total = arr.shape
    if total <= max_width:
        return arr, 1

    stride = int(math.ceil(total / float(max_width)))
    pad = (-total) % stride
    if pad > 0:
        arr = np.pad(arr, ((0, 0), (0, pad)), mode="edge")

    new_total = arr.shape[1] // stride
    x = arr.reshape(ch, new_total, stride)

    if reduce == "max":
        out = x.max(axis=2)
    elif reduce == "median":
        out = np.median(x, axis=2)
    else:
        out = x.mean(axis=2)

    return out, stride


def build_mask(combined_shape, event_intervals, event_channels):
    """
    Build a binary mask in original resolution: (channels, samples)
    """
    ch, total = combined_shape
    mask = np.zeros((ch, total), dtype=np.uint8)

    if event_channels is None:
        event_channels = []
    event_channels = np.asarray(event_channels, dtype=int)

    for a, b in event_intervals:
        a = max(0, int(a))
        b = min(total - 1, int(b))
        if b < a:
            continue
        if event_channels.size > 0:
            mask[event_channels, a:b + 1] = 1

    return mask


def save_session_figure(
    combined,
    pick_info,
    out_png,
    fs,
    max_plot_width=4000,
    figsize=(20, 7),
    dpi=180,
):
    """
    Save one figure for one session.
    The display data is downsampled in time, but rectangles use original indices.
    """
    ch, total = combined.shape

    event_intervals = pick_info.get("event_intervals", [])
    event_channels = pick_info.get("event_channels", [])

    # Build original-resolution mask
    mask = build_mask((ch, total), event_intervals, event_channels)

    # Downsample for plotting only
    combined_plot, stride = block_reduce_time(combined, max_width=max_plot_width, reduce="mean")
    mask_plot, _ = block_reduce_time(mask.astype(np.float32), max_width=max_plot_width, reduce="max")

    total_plot = combined_plot.shape[1]

    t_full = np.arange(total_plot, dtype=np.float64) * stride / float(fs)
    z = np.arange(ch, dtype=np.float64)

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        combined_plot,
        aspect="auto",
        origin="lower",
        extent=(0, (total - 1) / float(fs), 0, ch - 1),
        cmap="cubehelix",
    )

    ax.imshow(
        mask_plot,
        aspect="auto",
        origin="lower",
        extent=(0, (total - 1) / float(fs), 0, ch - 1),
        cmap="Reds",
        alpha=0.35,
        vmin=0,
        vmax=1,
    )

    # Draw rectangles using original indices
    if event_channels is None:
        event_channels = []
    event_channels = np.asarray(event_channels, dtype=int)

    groups = contiguous_groups(event_channels)

    for a, b in event_intervals:
        a = int(a)
        b = int(b)
        if b < a:
            continue

        t0 = a / float(fs)
        t1 = b / float(fs)

        for g in groups:
            if len(g) == 0:
                continue
            c0 = min(g)
            c1 = max(g)

            rect = Rectangle(
                (t0, c0),
                max(1e-9, t1 - t0),
                c1 - c0 + 1,
                linewidth=1.5,
                edgecolor="red",
                facecolor="none",
            )
            ax.add_patch(rect)

    class_name = pick_info.get("class_name", "")
    session_folder = pick_info.get("session_folder", "")
    class_id = pick_info.get("class_id", "")

    ax.set_title(f"{class_name} / {session_folder} / class_id={class_id}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Channel")

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Amplitude")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logging.info("Saved figure: %s", out_png)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", required=True, help="Root folder containing class/session .mat data")
    parser.add_argument("--pick_root", required=True, help="Root folder containing preprocess output json files")
    parser.add_argument("--out_dir", required=True, help="Where to save figures")

    parser.add_argument("--fs", type=float, default=1000.0)

    parser.add_argument("--apply_bandpass", action="store_true")
    parser.add_argument("--low", type=float, default=5.0)
    parser.add_argument("--high", type=float, default=300.0)
    parser.add_argument("--order", type=int, default=4)

    parser.add_argument("--max_plot_width", type=int, default=4000)
    parser.add_argument("--dpi", type=int, default=180)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for class_name in sorted(os.listdir(args.data_root)):
        if class_name == "nature" or class_name == "both":
            continue
        class_dir = os.path.join(args.data_root, class_name)
        if not os.path.isdir(class_dir):
            continue

        for session_folder in sorted(os.listdir(class_dir)):
            session_path = os.path.join(class_dir, session_folder)
            if not os.path.isdir(session_path):
                continue

            pick_json = os.path.join(args.pick_root, f"{class_name}__{session_folder}__pick.json")
            if not os.path.isfile(pick_json):
                logging.warning("Missing pick json: %s", pick_json)
                continue

            combined = load_and_combine_mat_files(session_path)
            combined = crop_channels(combined, 20, 101)
            if combined is None:
                logging.warning("No valid data: %s", session_path)
                continue

            if args.apply_bandpass:
                combined = bandpass_2d(
                    combined,
                    fs=args.fs,
                    low=args.low,
                    high=args.high,
                    order=args.order,
                )

            with open(pick_json, "r", encoding="utf-8") as f:
                pick_info = json.load(f)

            out_png = os.path.join(
                args.out_dir,
                f"{class_name}__{session_folder}__verify.png"
            )

            save_session_figure(
                combined=combined,
                pick_info=pick_info,
                out_png=out_png,
                fs=args.fs,
                max_plot_width=args.max_plot_width,
                dpi=args.dpi,
            )


if __name__ == "__main__":
    main()