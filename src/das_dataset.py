from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


def load_metadata(csv_path: str | Path) -> List[dict]:
    csv_path = Path(csv_path)
    rows: List[dict] = []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "fname": row["filename"],
                    "class_id": int(row["class_id"]),
                    "session_folder": row.get("session_folder", ""),
                }
            )
    return rows


def split_by_session(
    rows: Sequence[dict],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)

    session_to_rows = defaultdict(list)
    for row in rows:
        session_to_rows[row["session_folder"]].append(row)

    sessions = list(session_to_rows.keys())
    rng.shuffle(sessions)

    n_val = max(1, int(len(sessions) * val_ratio))
    val_sessions = set(sessions[:n_val])

    train_rows, val_rows = [], []
    for s, rs in session_to_rows.items():
        if s in val_sessions:
            val_rows.extend(rs)
        else:
            train_rows.extend(rs)

    return train_rows, val_rows


def balance_rows_to_target(
    rows: Sequence[dict],
    target_class_id: int = 1,
    seed: int = 42,
) -> List[dict]:
    rng = random.Random(seed)

    class_to_rows: Dict[int, List[dict]] = defaultdict(list)
    for row in rows:
        class_to_rows[row["class_id"]].append(row)

    if target_class_id not in class_to_rows or len(class_to_rows[target_class_id]) == 0:
        counts = [len(v) for v in class_to_rows.values() if len(v) > 0]
        if not counts:
            return list(rows)
        target = int(np.median(counts))
    else:
        target = len(class_to_rows[target_class_id])

    balanced: List[dict] = []
    for cid, cls_rows in class_to_rows.items():
        if len(cls_rows) >= target:
            chosen = rng.sample(cls_rows, target)
        else:
            chosen = cls_rows[:] + rng.choices(cls_rows, k=target - len(cls_rows))
        balanced.extend(chosen)

    rng.shuffle(balanced)
    return balanced


class DASWindowDataset(Dataset):
    """
    Reads only:
      - fname
      - class_id

    Assumes each npy is (T, C).
    If it is (C, T), it will be transposed.
    """
    def __init__(
        self,
        rows: Sequence[dict],
        data_root: str | Path,
        expected_channels: int = 81,
    ):
        self.rows = list(rows)
        self.data_root = Path(data_root)
        self.expected_channels = expected_channels

    def __len__(self) -> int:
        return len(self.rows)

    def _ensure_tc(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape={x.shape}")

        if x.shape[1] == self.expected_channels:
            return x
        if x.shape[0] == self.expected_channels:
            return x.T

        raise ValueError(
            f"Cannot infer (T,C) for shape={x.shape}, expected_channels={self.expected_channels}"
        )

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        fp = self.data_root / row["fname"]

        x = np.load(fp)
        x = self._ensure_tc(x).astype(np.float32)
        y = np.int64(row["class_id"])

        return {
            "data": torch.from_numpy(x),
            "label": torch.tensor(y, dtype=torch.long),
            "fname": row["fname"],
        }
