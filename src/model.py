import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


# -----------------------------
# Utilities
# -----------------------------

def _safe_log(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.log(x.clamp_min(eps))


def attention_entropy(attn: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """Entropy regularizer for softmax attention.

    Lower entropy => sharper / sparser attention.
    Expected attn shape: (..., N)
    """
    return -(attn * _safe_log(attn, eps)).sum(dim=dim).mean()


def extract_sliding_windows(
    x: torch.Tensor,
    window_size: int,
    stride: int,
) -> Tuple[torch.Tensor, int]:
    """Extract overlapping windows from x.

    Args:
        x: (B, 1, T, C)
        window_size: window length along time axis
        stride: sliding step along time axis

    Returns:
        windows: (B * N, 1, window_size, C)
        N: number of windows per session
    """
    if x.dim() != 4:
        raise ValueError(f"Expected x with shape (B,1,T,C), got {tuple(x.shape)}")

    b, ch, t, c = x.shape
    if t < window_size:
        pad_t = window_size - t
        x = F.pad(x, (0, 0, 0, pad_t))
        t = window_size

    windows = x.unfold(dimension=2, size=window_size, step=stride)
    # (B, 1, N, window_size, C)
    windows = windows.permute(0, 2, 1, 3, 4).contiguous()
    b, n, ch, w, c = windows.shape
    windows = windows.view(b * n, ch, w, c)
    return windows, n


# -----------------------------
# Backbone
# -----------------------------

class DASBackbone(nn.Module):
    """A simple 2D conv backbone that preserves the channel axis.

    Input:  (B, 1, T, C)
    Output: (B, D, T', C)
    """

    def __init__(self, in_ch: int = 1, base_filters: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, base_filters, kernel_size=(7, 1), padding=(3, 0))
        self.bn1 = nn.BatchNorm2d(base_filters)

        self.conv2 = nn.Conv2d(
            base_filters,
            base_filters * 2,
            kernel_size=(5, 1),
            stride=(2, 1),
            padding=(2, 0),
        )
        self.bn2 = nn.BatchNorm2d(base_filters * 2)

        self.layer1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(base_filters * 2, base_filters * 2, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(),
            nn.Conv2d(base_filters * 2, base_filters * 2, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(base_filters * 2),
        )

        self.layer2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(),
            nn.Conv2d(base_filters * 4, base_filters * 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(base_filters * 4),
        )

        self.proj = nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        r1 = self.layer1(x) + x
        r2 = self.layer2(r1) + self.proj(r1)
        return r2


class DualBranchWindowEncoder(nn.Module):
    """Time-domain + frequency-domain window encoder.

    Time branch:        window -> backbone -> avg over time -> (B, D, C)
    Frequency branch:   window -> rFFT magnitude -> backbone -> avg over freq -> (B, D, C)
    Fuse: concatenate both branches -> class-specific channel attention.
    """

    def __init__(
        self,
        in_ch: int = 1,
        base_filters: int = 32,
        num_type_heads: int = 2,
        attn_temperature: float = 1.0,
        embed_dim: int = 128,
    ):
        super().__init__()
        self.num_type_heads = num_type_heads
        self.attn_temperature = attn_temperature

        self.time_backbone = DASBackbone(in_ch=in_ch, base_filters=base_filters)
        self.freq_backbone = DASBackbone(in_ch=in_ch, base_filters=base_filters)

        d = base_filters * 4
        fused_d = d * 2

        # One channel-attention head per event type.
        self.channel_attn = nn.Conv1d(fused_d, num_type_heads, kernel_size=1)

        self.shared_proj = nn.Sequential(
            nn.Linear(fused_d, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim),
        )

        self.class_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fused_d, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
            )
            for _ in range(num_type_heads)
        ])

        self.embed_dim = embed_dim
        self.fused_d = fused_d

    def forward(self, window_x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """window_x: (B, 1, Tw, C)"""
        # Time branch
        t_feat = self.time_backbone(window_x)   # (B, D, T', C)
        t_pool = t_feat.mean(dim=2)            # (B, D, C)

        # Frequency branch
        x0 = window_x.squeeze(1)               # (B, Tw, C)
        fft_mag = torch.fft.rfft(x0, dim=1).abs()
        fft_mag = torch.log1p(fft_mag).unsqueeze(1)  # (B, 1, F, C)
        f_feat = self.freq_backbone(fft_mag)   # (B, D, F', C)
        f_pool = f_feat.mean(dim=2)            # (B, D, C)

        fused = torch.cat([t_pool, f_pool], dim=1)   # (B, 2D, C)
        shared_window = self.shared_proj(fused.mean(dim=-1))  # (B, E)

        att_logits = self.channel_attn(fused)        # (B, K, C)
        channel_attn = torch.softmax(att_logits / self.attn_temperature, dim=-1)
        class_pooled = torch.einsum("bkc,bdc->bkd", channel_attn, fused)  # (B, K, 2D)

        class_window_emb = torch.stack(
            [self.class_proj[k](class_pooled[:, k, :]) for k in range(self.num_type_heads)],
            dim=1,
        )  # (B, K, E)

        return {
            "shared_window_emb": shared_window,
            "class_window_emb": class_window_emb,
            "channel_attn": channel_attn,
        }


# -----------------------------
# Temporal aggregators
# -----------------------------

class TemporalGRUAggregator(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, num_types: int, bidirectional: bool = True):
        super().__init__()
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.window_attn = nn.Linear(out_dim, num_types)
        self.presence_head = nn.Linear(out_dim, 1)
        self.type_classifier = nn.Sequential(
            nn.Linear(out_dim * num_types, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, num_types),
        )
        self.out_dim = out_dim
        self.num_types = num_types

    def forward(self, window_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """window_emb: (B, N, E)"""
        h, _ = self.gru(window_emb)                           # (B, N, H)
        att_logits = self.window_attn(h)                      # (B, N, K)
        window_attn = torch.softmax(att_logits.transpose(1, 2), dim=-1)  # (B, K, N)
        session_repr = torch.einsum("bkn,bnh->bkh", window_attn, h)      # (B, K, H)

        type_logits = self.type_classifier(session_repr.flatten(1))       # (B, K)
        presence_logit = self.presence_head(h.mean(dim=1))                # (B, 1)

        return {
            "type_logits": type_logits,
            "presence_logit": presence_logit,
            "window_attn": window_attn,
            "context_seq": h,
            "session_repr": session_repr,
        }


class TemporalConvAggregator(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, num_types: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.window_attn = nn.Linear(hidden_dim, num_types)
        self.presence_head = nn.Linear(hidden_dim, 1)
        self.type_classifier = nn.Sequential(
            nn.Linear(hidden_dim * num_types, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_types),
        )

    def forward(self, window_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.net(window_emb.transpose(1, 2)).transpose(1, 2)           # (B, N, H)
        att_logits = self.window_attn(h)                                  # (B, N, K)
        window_attn = torch.softmax(att_logits.transpose(1, 2), dim=-1)   # (B, K, N)
        session_repr = torch.einsum("bkn,bnh->bkh", window_attn, h)      # (B, K, H)

        type_logits = self.type_classifier(session_repr.flatten(1))       # (B, K)
        presence_logit = self.presence_head(h.mean(dim=1))                # (B, 1)

        return {
            "type_logits": type_logits,
            "presence_logit": presence_logit,
            "window_attn": window_attn,
            "context_seq": h,
            "session_repr": session_repr,
        }


class TemporalTransformerAggregator(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, num_types: int, max_windows: int = 512, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(embed_dim, hidden_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_windows, hidden_dim))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, # dimension of the input features
            nhead=nhead,
            dim_feedforward=hidden_dim * 4, # FFN,引入非线性 
            batch_first=True,
            dropout=0.1,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.window_attn = nn.Linear(hidden_dim, num_types)
        self.presence_head = nn.Linear(hidden_dim, 1)
        self.type_classifier = nn.Sequential(
            nn.Linear(hidden_dim * num_types, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_types),
        )

    def forward(self, window_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, n, _ = window_emb.shape
        h = self.input_proj(window_emb)
        h = h + self.pos_emb[:, :n, :]
        h = self.encoder(h)

        att_logits = self.window_attn(h)
        window_attn = torch.softmax(att_logits.transpose(1, 2), dim=-1)
        session_repr = torch.einsum("bkn,bnh->bkh", window_attn, h)

        type_logits = self.type_classifier(session_repr.flatten(1))
        presence_logit = self.presence_head(h.mean(dim=1))

        return {
            "type_logits": type_logits,
            "presence_logit": presence_logit,
            "window_attn": window_attn,
            "context_seq": h,
            "session_repr": session_repr,
        }


# -----------------------------
# Main model
# -----------------------------

class DASMILTFClassifier(nn.Module):
    """Session-level MIL classifier for DAS data.

    Hierarchy:
      Stage 1: presence vs no-event
      Stage 2: mechanical vs human (only meaningful for positive sessions)

    Recommended labels:
      presence_target: (B, 1), binary
      type_target: (B,), integer class index for positive sessions only
          0 = mechanical
          1 = human

    The model supports:
      - overlapping window extraction
      - dual-branch time/frequency window encoder
      - class-specific softmax channel attention
      - temporal context modeling across windows
      - attention-based MIL pooling
    """

    def __init__(
        self,
        in_ch: int = 1,
        base_filters: int = 32,
        num_type_heads: int = 2, # 类别数
        embed_dim: int = 128,
        temporal_hidden_dim: int = 128,
        temporal_module: str = "gru",   # "gru" | "transformer" | "conv"
        window_size: int = 256,
        stride: int = 128,
        attn_temperature: float = 1.0,
        max_windows: int = 512,
    ):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.num_type_heads = num_type_heads

        self.window_encoder = DualBranchWindowEncoder(
            in_ch=in_ch,
            base_filters=base_filters,
            num_type_heads=num_type_heads,
            attn_temperature=attn_temperature,
            embed_dim=embed_dim,
        )

        temporal_module = temporal_module.lower().strip()
        if temporal_module == "gru":
            self.temporal = TemporalGRUAggregator(embed_dim, temporal_hidden_dim, num_type_heads)
        elif temporal_module in {"transformer", "trans"}:
            self.temporal = TemporalTransformerAggregator(
                embed_dim,
                temporal_hidden_dim,
                num_type_heads,
                max_windows=max_windows,
            )
        elif temporal_module in {"conv", "conv1d"}:
            self.temporal = TemporalConvAggregator(embed_dim, temporal_hidden_dim, num_type_heads)
        else:
            raise ValueError(f"Unknown temporal_module={temporal_module}")

        self.temporal_module = temporal_module

    def encode_windows(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        windows, n = extract_sliding_windows(x, self.window_size, self.stride)
        enc = self.window_encoder(windows)
        shared = enc["shared_window_emb"].view(x.size(0), n, -1)
        class_window_emb = enc["class_window_emb"].view(x.size(0), n, self.num_type_heads, -1)
        channel_attn = enc["channel_attn"].view(x.size(0), n, self.num_type_heads, -1)
        return {
            "shared_window_emb": shared,
            "class_window_emb": class_window_emb,
            "channel_attn": channel_attn,
            "num_windows": n,
        }

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """x: (B, 1, T, C)"""
        enc = self.encode_windows(x)
        shared_window_emb = enc["shared_window_emb"]  # (B, N, E)
        temporal_out = self.temporal(shared_window_emb)

        presence_logit = temporal_out["presence_logit"]  # (B, 1)
        type_logits = temporal_out["type_logits"]        # (B, 2)
        window_attn = temporal_out["window_attn"]        # (B, 2, N)

        presence_prob = torch.sigmoid(presence_logit)
        type_prob = torch.softmax(type_logits, dim=-1)
        gated_type_prob = presence_prob * type_prob

        return {
            "presence_logit": presence_logit,
            "type_logits": type_logits,
            "presence_prob": presence_prob,
            "type_prob": type_prob,
            "gated_type_prob": gated_type_prob,
            "window_attn": window_attn,
            "channel_attn": enc["channel_attn"],
            "shared_window_emb": shared_window_emb,
            "class_window_emb": enc["class_window_emb"],
            "context_seq": temporal_out["context_seq"],
            "session_repr": temporal_out["session_repr"],
        }


# -----------------------------
# Training loss helper
# -----------------------------

@dataclass
class LossWeights:
    presence: float = 1.0
    type: float = 1.0
    channel_entropy: float = 0.01
    window_entropy: float = 0.01


class DASMILLoss(nn.Module):
    """Loss for the 2-stage hierarchical setup.

    Targets:
      presence_target: (B, 1) binary
      type_target: (B,) integer labels for positive sessions only
         0 = mechanical
         1 = human

    Stage-2 loss is masked to positive samples only.
    """

    def __init__(self, weights: Optional[LossWeights] = None):
        super().__init__()
        self.weights = weights or LossWeights()
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        presence_target: torch.Tensor,
        type_target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        presence_logit = outputs["presence_logit"]
        type_logits = outputs["type_logits"]
        window_attn = outputs["window_attn"]
        channel_attn = outputs["channel_attn"]

        loss_presence = self.bce(presence_logit, presence_target.float())

        if type_target is None:
            loss_type = torch.zeros((), device=presence_logit.device)
        else:
            pos_mask = presence_target.squeeze(-1) > 0.5
            if pos_mask.any():
                loss_type = self.ce(type_logits[pos_mask], type_target[pos_mask].long()).mean()
            else:
                loss_type = torch.zeros((), device=presence_logit.device)

        loss_channel_entropy = attention_entropy(channel_attn, dim=-1)
        loss_window_entropy = attention_entropy(window_attn, dim=-1)

        total = (
            self.weights.presence * loss_presence
            + self.weights.type * loss_type
            + self.weights.channel_entropy * loss_channel_entropy
            + self.weights.window_entropy * loss_window_entropy
        )

        return {
            "total": total,
            "presence": loss_presence,
            "type": loss_type,
            "channel_entropy": loss_channel_entropy,
            "window_entropy": loss_window_entropy,
        }