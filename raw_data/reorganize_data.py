#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def safe_copy(src: Path, dst: Path, mode: str) -> None:
    """
    Copy or move src to dst.
    Raise if dst already exists to avoid silent overwrites.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise FileExistsError(f"Target already exists: {dst}")

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "move":
        shutil.move(str(src), str(dst))
    else:
        raise ValueError(f"Unknown mode: {mode}")


def process_data_event_class(src_event_dir: Path, out_root: Path, mode: str) -> int:
    """
    src_event_dir:
        .../date/data/event_class
    out_root:
        out_root/
    """
    copied = 0

    for session_dir in sorted(p for p in src_event_dir.iterdir() if p.is_dir()):
        dst_session_dir = out_root / "data" / src_event_dir.name / session_dir.name
        dst_session_dir.mkdir(parents=True, exist_ok=True)

        mat_files = sorted(
            p for p in session_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".mat"
        )

        for mat in mat_files:
            safe_copy(mat, dst_session_dir / mat.name, mode)

        copied += 1
    print(f"{src_event_dir} data的session数：{copied}")
    return copied


def process_label_event_class(src_event_dir: Path, out_root: Path, mode: str) -> int:
    """
    Compatible with both label layouts:

    Case A:
        .../date/label/event_class/session.txt
        -> out_root/label/event_class/session.txt

    Case B:
        .../date/label/event_class/session/session.txt
        -> out_root/label/event_class/session.txt
    """
    copied = 0
    dst_event_dir = out_root / "label" / src_event_dir.name
    dst_event_dir.mkdir(parents=True, exist_ok=True)

    for item in sorted(src_event_dir.iterdir()):
        if item.is_file() and item.suffix.lower() == ".txt":
            # Already a txt file directly under event_class
            safe_copy(item, dst_event_dir / item.name, mode)
            copied += 1
        
    print(f"{src_event_dir} label的session数：{copied}")
    return copied


def reorganize_dataset(source_root: Path, target_root: Path, mode: str) -> None:
    if not source_root.exists():
        raise FileNotFoundError(f"Source root not found: {source_root}")

    target_root.mkdir(parents=True, exist_ok=True)

    total_data = 0
    total_label = 0

    # Top level: date folders
    date_dirs = sorted(p for p in source_root.iterdir() if p.is_dir())

    for date_dir in date_dirs:
        data_dir = date_dir / "data"
        label_dir = date_dir / "label"

        if data_dir.exists():
            for event_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
                total_data += process_data_event_class(event_dir, target_root, mode)

        if label_dir.exists():
            for event_dir in sorted(p for p in label_dir.iterdir() if p.is_dir()):
                total_label += process_label_event_class(event_dir, target_root, mode)

    print(f"[DONE] data files processed : {total_data}")
    print(f"[DONE] label files processed: {total_label}")
    print(f"[DONE] output root: {target_root}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reorganize DAS dataset hierarchy by removing the date layer."
    )
    parser.add_argument(
        "--src",
        required=True,
        help="Source root, e.g. /home/sente/DAS_data",
    )
    parser.add_argument(
        "--dst",
        required=True,
        help="Target root, e.g. /home/sente/DAS_data_reorg",
    )
    parser.add_argument(
        "--mode",
        choices=["copy", "move"],
        default="copy",
        help="copy keeps original files, move transfers files.",
    )

    args = parser.parse_args()
    source_root = Path(args.src).expanduser().resolve()
    target_root = Path(args.dst).expanduser().resolve()

    reorganize_dataset(source_root, target_root, args.mode)


if __name__ == "__main__":
    main()