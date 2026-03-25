# train.py
import os
import csv
import time
import random
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import DASMILTFClassifier, DASMILLoss, LossWeights


# =========================================================
# Utils
# =========================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_item(x: torch.Tensor) -> float:
    return float(x.detach().cpu().item())


# =========================================================
# Dataset: one session = one npy file, shape (T, C)
# =========================================================

class SessionNpyDataset(Dataset):
    """
    CSV manifest format:
        path,presence,type

    where:
        path     -> relative or absolute path to .npy file
        presence -> 0/1
        type     -> 0/1 (positive samples meaningful; negative samples can be -1 or any placeholder)

    Each .npy file must contain one session with shape:
        (T, C)
    """

    def __init__(self, manifest_path: str | Path, root_dir: Optional[str | Path] = None):
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.root_dir = Path(root_dir) if root_dir is not None else None

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        self.rows: List[Dict[str, Any]] = []
        with open(self.manifest_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = {"path", "presence", "type"}
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError(
                    f"CSV must contain columns {required}, got {reader.fieldnames}"
                )
            for row in reader:
                self.rows.append(row)

        if len(self.rows) == 0:
            raise ValueError(f"Empty manifest: {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_path(self, p: str) -> Path:
        pp = Path(p)
        if pp.is_absolute():
            return pp
        if self.root_dir is not None:
            return self.root_dir / pp
        return (self.manifest_path.parent / pp).resolve()

    @staticmethod
    def _load_npy(path: Path) -> torch.Tensor:
        arr = np.load(path)
        x = torch.from_numpy(arr).float()

        if x.dim() == 2:
            # (T, C) -> (1, T, C)
            x = x.unsqueeze(0)
        elif x.dim() == 3:
            # allow already saved as (1, T, C)
            pass
        else:
            raise ValueError(f"Expected npy shape (T, C) or (1, T, C), got {tuple(x.shape)}")

        return x

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.rows[idx]
        path = self._resolve_path(row["path"])
        if not path.exists():
            raise FileNotFoundError(f"Sample file not found: {path}")

        x = self._load_npy(path)  # (1, T, C)
        presence = torch.tensor([float(row["presence"])], dtype=torch.float32)
        typ = torch.tensor(int(row["type"]), dtype=torch.long)
        return x, presence, typ


def collate_sessions(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """
    Each item:
        x: (1, T, C)
        presence: (1,)
        type: scalar

    Output:
        x: (B, 1, T_max, C)
        presence: (B, 1)
        type: (B,)
    """
    xs, pres, typs = zip(*batch)
    max_t = max(x.shape[1] for x in xs)
    c = xs[0].shape[2]

    padded = []
    for x in xs:
        if x.shape[2] != c:
            raise ValueError(f"Channel mismatch: expected {c}, got {x.shape[2]}")
        pad_t = max_t - x.shape[1]
        if pad_t > 0:
            # pad time dimension: (left, right, top, bottom) for last two dims
            x = F.pad(x, (0, 0, 0, pad_t))
        padded.append(x)

    x = torch.stack(padded, dim=0)
    presence = torch.stack(pres, dim=0)
    typ = torch.stack(typs, dim=0)
    return x, presence, typ


# =========================================================
# Metrics
# =========================================================

@dataclass
class ConfusionBinary:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred = pred.view(-1).long().cpu()
        target = target.view(-1).long().cpu()
        self.tp += int(((pred == 1) & (target == 1)).sum())
        self.fp += int(((pred == 1) & (target == 0)).sum())
        self.tn += int(((pred == 0) & (target == 0)).sum())
        self.fn += int(((pred == 0) & (target == 1)).sum())

    def compute(self) -> Dict[str, float]:
        acc = (self.tp + self.tn) / max(self.tp + self.tn + self.fp + self.fn, 1)
        precision = self.tp / max(self.tp + self.fp, 1)
        recall = self.tp / max(self.tp + self.fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        return {"acc": acc, "precision": precision, "recall": recall, "f1": f1}


@dataclass
class ConfusionMulticlass:
    num_classes: int
    cm: torch.Tensor

    @classmethod
    def create(cls, num_classes: int):
        return cls(num_classes=num_classes, cm=torch.zeros(num_classes, num_classes, dtype=torch.long))

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred = pred.view(-1).long().cpu()
        target = target.view(-1).long().cpu()
        for t, p in zip(target.tolist(), pred.tolist()):
            if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                self.cm[t, p] += 1

    def compute(self) -> Dict[str, float]:
        cm = self.cm.float()
        total = cm.sum().clamp_min(1.0)
        acc = torch.diag(cm).sum() / total

        f1s = []
        for k in range(self.num_classes):
            tp = cm[k, k]
            fp = cm[:, k].sum() - tp
            fn = cm[k, :].sum() - tp
            precision = tp / (tp + fp).clamp_min(1.0)
            recall = tp / (tp + fn).clamp_min(1.0)
            f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
            f1s.append(f1.item())

        return {"acc": acc.item(), "macro_f1": sum(f1s) / len(f1s)}


# =========================================================
# Netron export
# =========================================================

class ExportWrapper(nn.Module):
    def __init__(self, model: DASMILTFClassifier):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return (
            out["presence_logit"],
            out["type_logits"],
            out["presence_prob"],
            out["type_prob"],
            out["window_attn"],
            out["channel_attn"],
        )


def export_model_for_netron(
    model: DASMILTFClassifier,
    example_x: torch.Tensor,
    export_dir: str | Path,
    name: str = "dasmil_model",
) -> Path:
    export_dir = ensure_dir(export_dir)
    wrapper = ExportWrapper(model).eval()

    onnx_path = export_dir / f"{name}.onnx"
    ts_path = export_dir / f"{name}.pt"

    with torch.no_grad():
        try:
            torch.onnx.export(
                wrapper,
                example_x,
                onnx_path.as_posix(),
                export_params=True,
                opset_version=18,
                do_constant_folding=True,
                input_names=["x"],
                output_names=[
                    "presence_logit",
                    "type_logits",
                    "presence_prob",
                    "type_prob",
                    "window_attn",
                    "channel_attn",
                ],
                dynamic_axes={
                    "x": {0: "batch", 2: "time"},
                    "presence_logit": {0: "batch"},
                    "type_logits": {0: "batch"},
                    "presence_prob": {0: "batch"},
                    "type_prob": {0: "batch"},
                    "window_attn": {0: "batch", 2: "windows"},
                    "channel_attn": {0: "batch", 1: "windows"},
                },
            )
            return onnx_path
        except Exception as e:
            print(f"[Netron] ONNX export failed, fallback to TorchScript. Reason: {e}")
            traced = torch.jit.trace(wrapper, example_x)
            traced.save(ts_path.as_posix())
            return ts_path


# =========================================================
# Batch size probing for T4
# =========================================================

def representative_sample(dataset: Dataset, max_check: int = 8):
    n = min(len(dataset), max_check)
    best = None
    best_t = -1
    for i in range(n):
        x, presence, typ = dataset[i]
        if x.shape[1] > best_t:
            best = (x, presence, typ)
            best_t = x.shape[1]
    if best is None:
        raise RuntimeError("Unable to fetch representative sample")
    return best


def make_probe_batch(sample: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_size: int):
    x, presence, typ = sample
    batch = [(x.clone(), presence.clone(), typ.clone()) for _ in range(batch_size)]
    return collate_sessions(batch)


def probe_batch_size(
    model: nn.Module,
    loss_fn: nn.Module,
    dataset: Dataset,
    device: torch.device,
    start_bs: int = 1,
    max_bs: int = 64,
    verbose: bool = True,
) -> int:
    if device.type != "cuda":
        return start_bs

    sample = representative_sample(dataset, max_check=min(len(dataset), 8))
    best_bs = start_bs

    def can_run(bs: int) -> bool:
        try:
            x, presence, typ = make_probe_batch(sample, bs)
            x = x.to(device, non_blocking=True)
            presence = presence.to(device, non_blocking=True)
            typ = typ.to(device, non_blocking=True)

            model.train()
            model.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=True):
                out = model(x)
                loss = loss_fn(out, presence, typ)["total"]
            loss.backward()
            torch.cuda.synchronize()
            return True
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                return False
            raise
        finally:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

    bs = start_bs
    while bs <= max_bs:
        ok = can_run(bs)
        if verbose:
            print(f"[BatchProbe] bs={bs} -> {'OK' if ok else 'OOM'}")
        if ok:
            best_bs = bs
            bs *= 2
        else:
            break

    lo = best_bs
    hi = min(max_bs, max(best_bs * 2 - 1, best_bs))
    while lo <= hi:
        mid = (lo + hi) // 2
        ok = can_run(mid)
        if verbose:
            print(f"[BatchProbe] binary bs={mid} -> {'OK' if ok else 'OOM'}")
        if ok:
            best_bs = mid
            lo = mid + 1
        else:
            hi = mid - 1

    return max(best_bs, 1)


# =========================================================
# Train / Eval
# =========================================================

@dataclass
class EpochResult:
    loss_total: float
    loss_presence: float
    loss_type: float
    loss_channel_entropy: float
    loss_window_entropy: float
    presence_acc: float
    presence_precision: float
    presence_recall: float
    presence_f1: float
    type_acc: float
    type_macro_f1: float
    num_samples: int
    num_pos_samples: int


def compute_epoch_results(
    loss_sums: Dict[str, float],
    n_samples: int,
    presence_cm: ConfusionBinary,
    type_cm: ConfusionMulticlass,
    num_pos: int,
) -> EpochResult:
    p = presence_cm.compute()
    t = type_cm.compute() if num_pos > 0 else {"acc": 0.0, "macro_f1": 0.0}
    denom = max(n_samples, 1)
    return EpochResult(
        loss_total=loss_sums["total"] / denom,
        loss_presence=loss_sums["presence"] / denom,
        loss_type=loss_sums["type"] / denom,
        loss_channel_entropy=loss_sums["channel_entropy"] / denom,
        loss_window_entropy=loss_sums["window_entropy"] / denom,
        presence_acc=p["acc"],
        presence_precision=p["precision"],
        presence_recall=p["recall"],
        presence_f1=p["f1"],
        type_acc=t["acc"],
        type_macro_f1=t["macro_f1"],
        num_samples=n_samples,
        num_pos_samples=num_pos,
    )


def train_one_epoch(
    model: DASMILTFClassifier,
    loss_fn: DASMILLoss,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    grad_clip: float = 1.0,
) -> EpochResult:
    model.train()

    loss_sums = {"total": 0.0, "presence": 0.0, "type": 0.0, "channel_entropy": 0.0, "window_entropy": 0.0}
    presence_cm = ConfusionBinary()
    type_cm = ConfusionMulticlass.create(num_classes=model.num_type_heads)
    n_samples = 0
    num_pos = 0

    optimizer.zero_grad(set_to_none=True)

    for x, presence_target, type_target in loader:
        x = x.to(device, non_blocking=True)
        presence_target = presence_target.to(device, non_blocking=True)
        type_target = type_target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            outputs = model(x)
            loss_dict = loss_fn(outputs, presence_target, type_target)
            loss = loss_dict["total"]

        scaler.scale(loss).backward()

        if grad_clip is not None and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        bs = x.size(0)
        n_samples += bs
        num_pos += int((presence_target.squeeze(-1) > 0.5).sum().item())

        for k in loss_sums:
            loss_sums[k] += safe_item(loss_dict[k]) * bs

        presence_pred = (outputs["presence_prob"] >= 0.5).long()
        presence_cm.update(presence_pred, presence_target)

        pos_mask = presence_target.squeeze(-1) > 0.5
        if pos_mask.any():
            type_pred = outputs["type_logits"][pos_mask].argmax(dim=-1)
            type_cm.update(type_pred, type_target[pos_mask])

    return compute_epoch_results(loss_sums, n_samples, presence_cm, type_cm, num_pos)


@torch.no_grad()
def evaluate(
    model: DASMILTFClassifier,
    loss_fn: DASMILLoss,
    loader: DataLoader,
    device: torch.device,
) -> EpochResult:
    model.eval()

    loss_sums = {"total": 0.0, "presence": 0.0, "type": 0.0, "channel_entropy": 0.0, "window_entropy": 0.0}
    presence_cm = ConfusionBinary()
    type_cm = ConfusionMulticlass.create(num_classes=model.num_type_heads)
    n_samples = 0
    num_pos = 0

    for x, presence_target, type_target in loader:
        x = x.to(device, non_blocking=True)
        presence_target = presence_target.to(device, non_blocking=True)
        type_target = type_target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            outputs = model(x)
            loss_dict = loss_fn(outputs, presence_target, type_target)

        bs = x.size(0)
        n_samples += bs
        num_pos += int((presence_target.squeeze(-1) > 0.5).sum().item())

        for k in loss_sums:
            loss_sums[k] += safe_item(loss_dict[k]) * bs

        presence_pred = (outputs["presence_prob"] >= 0.5).long()
        presence_cm.update(presence_pred, presence_target)

        pos_mask = presence_target.squeeze(-1) > 0.5
        if pos_mask.any():
            type_pred = outputs["type_logits"][pos_mask].argmax(dim=-1)
            type_cm.update(type_pred, type_target[pos_mask])

    return compute_epoch_results(loss_sums, n_samples, presence_cm, type_cm, num_pos)


def log_epoch(writer: SummaryWriter, split: str, epoch: int, result: EpochResult, lr: float):
    p = f"{split}/"
    writer.add_scalar(p + "loss_total", result.loss_total, epoch)
    writer.add_scalar(p + "loss_presence", result.loss_presence, epoch)
    writer.add_scalar(p + "loss_type", result.loss_type, epoch)
    writer.add_scalar(p + "loss_channel_entropy", result.loss_channel_entropy, epoch)
    writer.add_scalar(p + "loss_window_entropy", result.loss_window_entropy, epoch)

    writer.add_scalar(p + "presence_acc", result.presence_acc, epoch)
    writer.add_scalar(p + "presence_precision", result.presence_precision, epoch)
    writer.add_scalar(p + "presence_recall", result.presence_recall, epoch)
    writer.add_scalar(p + "presence_f1", result.presence_f1, epoch)

    writer.add_scalar(p + "type_acc", result.type_acc, epoch)
    writer.add_scalar(p + "type_macro_f1", result.type_macro_f1, epoch)

    writer.add_scalar(p + "num_samples", result.num_samples, epoch)
    writer.add_scalar(p + "num_pos_samples", result.num_pos_samples, epoch)
    writer.add_scalar(p + "lr", lr, epoch)


# =========================================================
# Checkpoint
# =========================================================

def save_checkpoint(
    ckpt_path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
    best_metric: float,
    args: argparse.Namespace,
):
    ckpt = {
        "epoch": epoch,
        "best_metric": best_metric,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "args": vars(args),
    }
    torch.save(ckpt, Path(ckpt_path).as_posix())


def verify_checkpoint(
    ckpt_path: str | Path,
    model_ctor,
    sample_x: torch.Tensor,
    device: torch.device,
) -> bool:
    payload = torch.load(Path(ckpt_path), map_location="cpu")
    model = model_ctor().to(device)
    missing, unexpected = model.load_state_dict(payload["model_state"], strict=False)

    if len(missing) > 0 or len(unexpected) > 0:
        raise RuntimeError(
            f"Checkpoint key mismatch:\nmissing={missing}\nunexpected={unexpected}"
        )

    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
        out = model(sample_x.to(device))

    required = ["presence_logit", "type_logits", "presence_prob", "type_prob", "window_attn", "channel_attn"]
    for k in required:
        if k not in out:
            raise RuntimeError(f"Missing output key: {k}")

    return True


# =========================================================
# Main
# =========================================================

def build_datasets(args):
    train_ds = SessionNpyDataset(args.train_manifest, root_dir=args.data_root)
    val_ds = SessionNpyDataset(args.val_manifest, root_dir=args.data_root)
    return train_ds, val_ds


def build_model(args) -> DASMILTFClassifier:
    return DASMILTFClassifier(
        in_ch=args.in_ch,
        base_filters=args.base_filters,
        num_type_heads=args.num_types,
        embed_dim=args.embed_dim,
        temporal_hidden_dim=args.temporal_hidden_dim,
        temporal_module=args.temporal_module,
        window_size=args.window_size,
        stride=args.stride,
        attn_temperature=args.attn_temperature,
        max_windows=args.max_windows,
    )


def main():
    parser = argparse.ArgumentParser("DASMIL training for per-session npy files")
    # data
    parser.add_argument("--data-root", type=str, default=".")
    parser.add_argument("--train-manifest", type=str, required=True)
    parser.add_argument("--val-manifest", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=4)

    # model
    parser.add_argument("--in-ch", type=int, default=1)
    parser.add_argument("--base-filters", type=int, default=32)
    parser.add_argument("--num-types", type=int, default=2)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--temporal-hidden-dim", type=int, default=128)
    parser.add_argument("--temporal-module", type=str, default="gru", choices=["gru", "transformer", "conv"])
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--attn-temperature", type=float, default=1.0)
    parser.add_argument("--max-windows", type=int, default=512)

    # training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="runs/dasmil_exp")
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--auto-batch-size", action="store_true")
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--start-batch-size", type=int, default=1)

    # loss weights
    parser.add_argument("--presence-loss-w", type=float, default=1.0)
    parser.add_argument("--type-loss-w", type=float, default=1.0)
    parser.add_argument("--channel-entropy-w", type=float, default=0.01)
    parser.add_argument("--window-entropy-w", type=float, default=0.01)

    # scheduler
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--plateau-patience", type=int, default=3)
    parser.add_argument("--plateau-min-lr", type=float, default=1e-6)

    # netron
    parser.add_argument("--export-netron", action="store_true")
    parser.add_argument("--netron-name", type=str, default="dasmil_model")

    args = parser.parse_args()

    set_seed(args.seed)

    save_dir = ensure_dir(args.save_dir)
    log_dir = ensure_dir(args.log_dir if args.log_dir is not None else (save_dir / "tb"))
    ckpt_dir = ensure_dir(save_dir / "checkpoints")
    best_dir = ensure_dir(save_dir / "best")

    writer = SummaryWriter(log_dir=log_dir.as_posix())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] device = {device}")

    train_ds, val_ds = build_datasets(args)
    print(f"[Info] train size = {len(train_ds)}, val size = {len(val_ds)}")

    model = build_model(args).to(device)
    loss_fn = DASMILLoss(
        LossWeights(
            presence=args.presence_loss_w,
            type=args.type_loss_w,
            channel_entropy=args.channel_entropy_w,
            window_entropy=args.window_entropy_w,
        )
    ).to(device)

    if args.auto_batch_size:
        print("[Info] probing batch size ...")
        bs = probe_batch_size(
            model=model,
            loss_fn=loss_fn,
            dataset=train_ds,
            device=device,
            start_bs=args.start_batch_size,
            max_bs=args.max_batch_size,
            verbose=True,
        )
        print(f"[Info] selected batch size = {bs}")
    else:
        bs = args.batch_size
        print(f"[Info] fixed batch size = {bs}")

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_sessions,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, bs),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_sessions,
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.plateau_factor,
        patience=args.plateau_patience,
        min_lr=args.plateau_min_lr,
        verbose=True,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    start_epoch = 1
    best_val_loss = float("inf")
    best_ckpt_path = best_dir / "best.pth"
    last_ckpt_path = ckpt_dir / "last.pth"

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            payload = torch.load(resume_path, map_location="cpu")
            model.load_state_dict(payload["model_state"], strict=True)
            optimizer.load_state_dict(payload["optimizer_state"])
            if payload.get("scheduler_state") is not None:
                scheduler.load_state_dict(payload["scheduler_state"])
            if payload.get("scaler_state") is not None:
                scaler.load_state_dict(payload["scaler_state"])
            start_epoch = int(payload["epoch"]) + 1
            best_val_loss = float(payload.get("best_metric", best_val_loss))
            print(f"[Info] resumed from {resume_path}, start_epoch={start_epoch}")
        else:
            print(f"[Warn] resume checkpoint not found: {resume_path}")

    # Netron export
    if args.export_netron:
        sample_x, _, _ = representative_sample(train_ds, max_check=min(8, len(train_ds)))
        sample_x = sample_x.unsqueeze(0).to(device)  # (1,1,T,C)
        model.eval()
        export_path = export_model_for_netron(
            model=model,
            example_x=sample_x,
            export_dir=save_dir / "netron",
            name=args.netron_name,
        )
        print(f"[Info] Netron export: {export_path}")

    def model_ctor():
        return build_model(args).to(device)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_res = train_one_epoch(
            model=model,
            loss_fn=loss_fn,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            grad_clip=args.grad_clip,
        )

        val_res = evaluate(
            model=model,
            loss_fn=loss_fn,
            loader=val_loader,
            device=device,
        )

        scheduler.step(val_res.loss_total)
        current_lr = optimizer.param_groups[0]["lr"]

        log_epoch(writer, "train", epoch, train_res, current_lr)
        log_epoch(writer, "val", epoch, val_res, current_lr)

        save_checkpoint(
            ckpt_path=last_ckpt_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_metric=best_val_loss,
            args=args,
        )

        # verify last checkpoint
        sample_x, _, _ = representative_sample(val_ds, max_check=min(8, len(val_ds)))
        sample_x = sample_x.unsqueeze(0).to(device)
        verify_checkpoint(last_ckpt_path, model_ctor, sample_x, device)

        # save best
        if val_res.loss_total < best_val_loss:
            best_val_loss = val_res.loss_total
            save_checkpoint(
                ckpt_path=best_ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_metric=best_val_loss,
                args=args,
            )
            verify_checkpoint(best_ckpt_path, model_ctor, sample_x, device)
            print(f"[Info] new best checkpoint saved: {best_ckpt_path}")

        dt = time.time() - t0
        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_res.loss_total:.4f} "
            f"val_loss={val_res.loss_total:.4f} "
            f"val_presence_f1={val_res.presence_f1:.4f} "
            f"val_type_f1={val_res.type_macro_f1:.4f} "
            f"lr={current_lr:.3e} "
            f"time={dt:.1f}s"
        )

    writer.close()
    print(f"[Done] best checkpoint: {best_ckpt_path}")


if __name__ == "__main__":
    main()