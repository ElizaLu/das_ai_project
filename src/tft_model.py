from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from torch.amp import autocast as _autocast_new, GradScaler as _GradScalerNew

    def _autocast(device: torch.device):
        return _autocast_new("cuda") if device.type == "cuda" else nullcontext()

    def _make_scaler(device: torch.device):
        return _GradScalerNew("cuda") if device.type == "cuda" else None

except Exception:
    from torch.cuda.amp import autocast as _autocast_old, GradScaler as _GradScalerOld

    def _autocast(device: torch.device):
        return _autocast_old(enabled=(device.type == "cuda"))

    def _make_scaler(device: torch.device):
        return _GradScalerOld(enabled=(device.type == "cuda"))


def _to_btc(x: torch.Tensor, in_channels: int) -> torch.Tensor:
    if x.dim() == 4:
        x = x.squeeze(1)
    if x.dim() != 3:
        raise ValueError(f"Unexpected input shape: {tuple(x.shape)}")
    if x.shape[-1] == in_channels:
        return x
    if x.shape[1] == in_channels:
        return x.transpose(1, 2)
    raise ValueError(f"Cannot infer (T,C) from shape {tuple(x.shape)} with in_channels={in_channels}")


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 20000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1), :]


class VariableSelection(nn.Module):
    def __init__(self, in_channels: int, d_model: int):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.var_projs = nn.ModuleList([nn.Linear(1, d_model) for _ in range(in_channels)])
        self.weight_net = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        if C != self.in_channels:
            raise ValueError(f"VariableSelection expected C={self.in_channels}, got {C}")
        w = self.weight_net(x)
        fused = 0.0
        for c in range(C):
            xc = x[..., c:c + 1]
            ec = self.var_projs[c](xc)
            wc = w[..., c:c + 1]
            fused = fused + ec * wc
        return fused, w


class AttentionPooling(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model))
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = torch.tanh(self.key(x))
        v = self.value(x)
        scores = torch.einsum("btd,d->bt", k, self.query)
        attn = torch.softmax(scores, dim=1)
        return torch.einsum("bt,btd->bd", attn, v)


class TemporalFusionTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        d_model: int = 96,
        n_heads: int = 3,
        num_layers: int = 2,
        d_ff: int = 192,
        dropout: float = 0.1,
        max_len: int = 20000,
        max_tokens: int = 1024,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.in_channels = in_channels
        self.d_model = d_model
        self.max_tokens = max_tokens
        self.var_sel = VariableSelection(in_channels, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pool = AttentionPooling(d_model)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = _to_btc(x, self.in_channels)
        _, T, _ = x.shape
        if T > self.max_tokens:
            k = math.ceil(T / self.max_tokens)
            x = F.avg_pool1d(x.transpose(1, 2), kernel_size=k, stride=k, ceil_mode=True).transpose(1, 2)
        z, _weights = self.var_sel(x)
        z = self.pos_enc(z)
        z = self.encoder(z)
        feats = self.pool(z)
        logits = self.classifier(feats)
        return feats, logits


def build_model(
    in_channels: int = 81,
    n_classes: int = 3,
    d_model: int = 96,
    n_heads: int = 3,
    num_layers: int = 2,
    d_ff: int = 192,
    dropout: float = 0.1,
    max_tokens: int = 1024,
) -> TemporalFusionTransformer:
    return TemporalFusionTransformer(
        in_channels=in_channels,
        n_classes=n_classes,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout,
        max_tokens=max_tokens,
    )
