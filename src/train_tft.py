from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from das_dataset import DASWindowDataset, balance_rows_to_target, load_metadata, split_by_session
from tft_model import _autocast, _make_scaler, build_model


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(log_path: str | Path) -> logging.Logger:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("train_tft")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    all_true: List[int] = []
    all_pred: List[int] = []

    with torch.no_grad():
        for batch in loader:
            x = batch["data"].to(device, dtype=torch.float32)
            y = batch["label"].to(device)

            with _autocast(device):
                _, logits = model(x)
                loss = criterion(logits, y)

            pred = logits.argmax(dim=1)
            total_loss += float(loss.detach())
            total_correct += (pred == y).sum().item()
            total_count += y.numel()

            all_true.extend(y.detach().cpu().numpy().tolist())
            all_pred.extend(pred.detach().cpu().numpy().tolist())

    avg_loss = total_loss / max(len(loader), 1)
    acc = total_correct / max(total_count, 1)
    return avg_loss, acc, np.asarray(all_true, dtype=np.int64), np.asarray(all_pred, dtype=np.int64)


def compute_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> Dict[str, np.ndarray]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(n_classes)),
        average=None,
        zero_division=0,
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    epoch: int,
    best_val_acc: float,
    args: argparse.Namespace,
) -> None:
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_acc": best_val_acc,
        "args": vars(args),
    }
    if scaler is not None:
        try:
            ckpt["scaler_state_dict"] = scaler.state_dict()
        except Exception:
            ckpt["scaler_state_dict"] = None
    torch.save(ckpt, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    logger: logging.Logger,
) -> Tuple[int, float]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        try:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        except Exception as e:
            logger.warning("Failed to load scaler state, continue without it: %s", e)

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_val_acc = float(ckpt.get("best_val_acc", -1.0))
    logger.info("Resumed from %s (next epoch = %d, best_val_acc = %.6f)", path, start_epoch, best_val_acc)
    return start_epoch, best_val_acc


def run_train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir / "log.txt")
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    logger.info("Arguments: %s", vars(args))
    logger.info("Using device: %s", device)

    rows = load_metadata(args.metadata_csv)
    logger.info("Loaded metadata rows: %d", len(rows))

    train_rows, val_rows = split_by_session(rows, val_ratio=args.val_ratio, seed=args.seed)
    logger.info("Split by session -> train=%d, val=%d", len(train_rows), len(val_rows))

    if args.balance_train:
        train_rows = balance_rows_to_target(
            train_rows,
            target_class_id=args.balance_target_class,
            seed=args.seed,
        )
        logger.info("Balanced train rows: %d", len(train_rows))

    train_ds = DASWindowDataset(train_rows, data_root=args.data_root, expected_channels=args.in_channels)
    val_ds = DASWindowDataset(val_rows, data_root=args.data_root, expected_channels=args.in_channels)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = build_model(
        in_channels=args.in_channels,
        n_classes=args.n_classes,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_tokens=args.max_tokens,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = _make_scaler(device)

    start_epoch = 1
    best_val_acc = -1.0
    last_ckpt = output_dir / "last.pt"
    best_ckpt = output_dir / "best.pt"

    if args.resume:
        if last_ckpt.exists():
            start_epoch, best_val_acc = load_checkpoint(
                last_ckpt, model, optimizer, scaler, device, logger
            )
        else:
            logger.warning("resume=True but %s not found. Training from scratch.", last_ckpt)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            x = batch["data"].to(device, dtype=torch.float32)
            y = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with _autocast(device):
                _, logits = model(x)
                loss = criterion(logits, y)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss += float(loss.detach())
            train_correct += (logits.argmax(dim=1) == y).sum().item()
            train_total += y.numel()
            print(f"\rEpoch {epoch}/{args.epochs} - Batch {train_total}/{len(train_loader.dataset)} - Loss: {train_loss / max(train_total // args.batch_size, 1):.4f}", end="")

        avg_train_loss = train_loss / max(len(train_loader), 1)
        train_acc = train_correct / max(train_total, 1)

        val_loss, val_acc, y_true, y_pred = evaluate_loss(model, val_loader, criterion, device)
        metrics = compute_class_metrics(y_true, y_pred, args.n_classes)

        logger.info(
            "[Epoch %03d] train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f",
            epoch,
            avg_train_loss,
            train_acc,
            val_loss,
            val_acc,
        )

        for cid in range(args.n_classes):
            logger.info(
                "  class %d | precision=%.4f recall=%.4f f1=%.4f support=%d",
                cid,
                float(metrics["precision"][cid]),
                float(metrics["recall"][cid]),
                float(metrics["f1"][cid]),
                int(metrics["support"][cid]),
            )

        macro_p = float(np.mean(metrics["precision"]))
        macro_r = float(np.mean(metrics["recall"]))
        macro_f1 = float(np.mean(metrics["f1"]))
        logger.info("  macro | precision=%.4f recall=%.4f f1=%.4f", macro_p, macro_r, macro_f1)

        writer.add_scalar("loss/train", avg_train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("acc/train", train_acc, epoch)
        writer.add_scalar("acc/val", val_acc, epoch)
        writer.add_scalar("macro/precision", macro_p, epoch)
        writer.add_scalar("macro/recall", macro_r, epoch)
        writer.add_scalar("macro/f1", macro_f1, epoch)

        for cid in range(args.n_classes):
            writer.add_scalar(f"class_{cid}/precision", float(metrics["precision"][cid]), epoch)
            writer.add_scalar(f"class_{cid}/recall", float(metrics["recall"][cid]), epoch)
            writer.add_scalar(f"class_{cid}/f1", float(metrics["f1"][cid]), epoch)
            writer.add_scalar(f"class_{cid}/support", int(metrics["support"][cid]), epoch)

        save_checkpoint(last_ckpt, model, optimizer, scaler, epoch, best_val_acc, args)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(best_ckpt, model, optimizer, scaler, epoch, best_val_acc, args)
            logger.info("New best checkpoint saved to %s (best_val_acc=%.6f)", best_ckpt, best_val_acc)

    writer.close()
    logger.info("Training finished. Best val acc = %.6f", best_val_acc)
    logger.info("Last checkpoint: %s", last_ckpt)
    logger.info("Best checkpoint: %s", best_ckpt)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--metadata_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="runs/tft_exp")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--balance_train", action="store_true")
    parser.add_argument("--balance_target_class", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="resume from output_dir/last.pt")

    parser.add_argument("--in_channels", type=int, default=81)
    parser.add_argument("--n_classes", type=int, default=3)

    parser.add_argument("--d_model", type=int, default=96)
    parser.add_argument("--n_heads", type=int, default=3)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--d_ff", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_tokens", type=int, default=1024)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)

    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run_train(args)
