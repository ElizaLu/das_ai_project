from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Union, Dict, Any, Tuple

import numpy as np
import torch

from tft_model import build_model

from scipy.signal import butter, sosfiltfilt

def bandpass_2d(arr, fs, low=5.0, high=300.0, order=4):
    """
    arr: [T, C]
    沿时间轴（axis=0）一次性滤波所有通道
    """
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={arr.shape}")

    sos = butter(order, [low, high], btype="band", fs=fs, output="sos")

    out = sosfiltfilt(sos, arr, axis=0)

    return out.astype(np.float32)


CLASS_NAMES = {0: "nature", 1: "human", 2: "mechanical"}


def load_checkpoint(ckpt_path: str | Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt.get("args", {})
    model = build_model(
        in_channels=int(args.get("in_channels", 81)),
        n_classes=int(args.get("n_classes", 3)),
        d_model=int(args.get("d_model", 96)),
        n_heads=int(args.get("n_heads", 3)),
        num_layers=int(args.get("num_layers", 2)),
        d_ff=int(args.get("d_ff", 192)),
        dropout=float(args.get("dropout", 0.1)),
        max_tokens=int(args.get("max_tokens", 1024)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, args


def load_input(input_data: Union[str, Path, np.ndarray]) -> np.ndarray:
    """
    统一输入：
    - 如果是 npy 路径 -> np.load
    - 如果是矩阵 -> np.asarray
    """
    if isinstance(input_data, (str, Path)):
        return np.load(input_data)
    return input_data


def ensure_layout(x: np.ndarray, input_layout: str = "tc") -> np.ndarray:
    """
    input_layout:
        - 'tc' : x.shape == [T, C]，默认就是这个
        - 'ct' : x.shape == [C, T]，会自动转置成 [T, C]
    """
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={x.shape}")

    if input_layout == "tc":
        return x
    if input_layout == "ct":
        return x.T

    raise ValueError(f"Unsupported input_layout={input_layout}, use 'tc' or 'ct'")


def make_channel_windows(x_tc: np.ndarray, window_channels: int = 81) -> Tuple[np.ndarray, int, int]:
    """
    x_tc: [T, C]
    return:
        windows: [B, T, window_channels]
        usable_channels: 实际参与推理的通道数
        n_windows: 窗口数
    """
    if x_tc.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={x_tc.shape}")

    t, c = x_tc.shape
    n_windows = c // window_channels
    if n_windows <= 0:
        raise ValueError(f"Channel count {c} is smaller than window_channels={window_channels}")

    usable_channels = n_windows * window_channels

    # 只保留能整除的部分，丢弃尾巴
    x_used = np.ascontiguousarray(x_tc[:, :usable_channels], dtype=np.float32)

    # [T, usable_channels] -> [T, n_windows, 81] -> [n_windows, T, 81]
    windows = x_used.reshape(t, n_windows, window_channels).transpose(1, 0, 2)
    return windows, usable_channels, n_windows


@torch.inference_mode()
def predict_input_batch(
    model,
    input_data: Union[str, Path, np.ndarray],
    device: torch.device,
    input_layout: str = "tc",
    window_channels: int = 81,
    batch_size: int = 256,
    fs: int = 1000,
    low_cut: float = 5.0,
    high_cut: float = 300.0,
    order: int = 4,
) -> Dict[str, Any]:
    x = load_input(input_data)
    x = ensure_layout(x, input_layout=input_layout)

    x = bandpass_2d(x, fs=fs, low=low_cut, high=high_cut, order=order)

    # x_windows: [B, T, 81]
    x_windows, usable_channels, n_windows = make_channel_windows(
        x, window_channels=window_channels
    )

    xt = torch.from_numpy(x_windows)

    if device.type == "cuda":
        xt = xt.pin_memory()

    all_probs = []
    all_pred_ids = []

    for start in range(0, n_windows, batch_size):
        end = min(start + batch_size, n_windows)
        batch = xt[start:end].to(device, non_blocking=True)

        _, logits = model(batch)             # [B, n_classes]
        prob = torch.softmax(logits, dim=1)   # [B, n_classes]
        prob_np = prob.float().cpu().numpy()

        all_probs.append(prob_np)
        all_pred_ids.append(prob_np.argmax(axis=1))

    probs = np.concatenate(all_probs, axis=0)        # [B, n_classes]
    pred_ids = np.concatenate(all_pred_ids, axis=0)  # [B]

    window_results = []
    for i in range(n_windows):
        pid = int(pred_ids[i])

        # [修改4] 只保留 1 / 2 类窗口
        if pid not in (1, 2):
            continue

        pname = CLASS_NAMES.get(pid, str(pid))
        if i < 6:
            window_results.append(
                {
                    "window_idx": i,
                    "channel_start": i * window_channels,
                    "channel_end": (i + 1) * window_channels - 1,
                    "pred_id": pid,
                    "pred_name": pname,
                    # [修改5] numpy array 改成 list，方便 JSON 序列化
                    "prob": probs[i].tolist(),
                }
            )

    # [修改6] 补齐返回字段，网页和 main() 都能直接读
    return {
        "has_positive_windows": len(window_results) > 0,
        "window_results": window_results,
        "usable_channels": usable_channels,
        "dropped_channels": x.shape[1] - usable_channels,
        "n_windows": n_windows,
        "window_channels": window_channels,
        "fs": fs,
        "low_cut": low_cut,
        "high_cut": high_cut,
    }


class TFTInferencer:
    """
    实时推断器：模型只初始化一次，后续直接对矩阵/文件推断。
    """

    def __init__(self, ckpt_path: str | Path, device: str = "cuda"):
        self.device = torch.device(
            device if (torch.cuda.is_available() or device == "cpu") else "cpu"
        )
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        self.model, self.ckpt_args = load_checkpoint(ckpt_path, self.device)
        self.window_channels = int(self.ckpt_args.get("in_channels", 81))

    def predict(
        self,
        input_data: Union[str, Path, np.ndarray],
        input_layout: str = "tc",
        batch_size: int = 256,
        fs: int = 1000,
        low_cut: float = 5.0,
        high_cut: float = 300.0,
        order: int = 4,
    ):
        return predict_input_batch(
            self.model,
            input_data,
            self.device,
            input_layout=input_layout,
            window_channels=self.window_channels,
            batch_size=batch_size,
            fs=fs,
            low_cut=low_cut,
            high_cut=high_cut,
            order=order,
        )


def main(args):
    inferencer = TFTInferencer(args.checkpoint, args.device)

    paths: List[Path] = []
    p = Path(args.input_path)

    if p.is_dir():
        paths = sorted(p.glob("*.npy"))
    elif p.suffix == ".npy":
        paths = [p]
    else:
        raise ValueError("input_path must be a .npy file or a directory containing .npy files")

    if not paths:
        raise FileNotFoundError(f"No npy files found under {p}")

    for idx, fp in enumerate(paths, 1):
        result = inferencer.predict(
            fp,
            input_layout=args.input_layout,
            batch_size=args.batch_size,
        )

        print(f"\n=== 输入 {idx} ===")
        print(
            f"windows={result['n_windows']} | "
            f"used_channels={result['usable_channels']} | "
            f"dropped_channels={result['dropped_channels']}"
        )

        if not result["has_positive_windows"]:
            print(f"pred=0(nature)")
            continue

        for w in result["window_results"]:
            wp = w["prob"]
            wprob_str = ", ".join(
                [f"{CLASS_NAMES[i]}={wp[i]:.4f}" for i in range(len(wp))]
            )
            print(
                f"  window[{w['window_idx']:02d}] "
                f"channels[{w['channel_start']:04d}:{w['channel_end']:04d}] "
                f"-> pred={w['pred_id']}({w['pred_name']}) | {wprob_str}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input_path", type=str, required=True, help="single .npy file or a directory")
    parser.add_argument("--device", type=str, default="cuda")

    # tc: [T, C]，默认推荐
    # ct: [C, T]
    parser.add_argument("--input_layout", type=str, default="tc", choices=["tc", "ct"])
    parser.add_argument("--batch_size", type=int, default=256)

    args = parser.parse_args()
    main(args)