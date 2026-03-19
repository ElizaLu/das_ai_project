# src/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DASBackbone(nn.Module):
    """
    Simple separable conv backbone that preserves spatial resolution (no pooling on width).
    Input: (B, 1, T, C)
    Output: feature map (B, D, T', C)
    """
    def __init__(self, in_ch=1, base_filters=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, base_filters, kernel_size=(7,1), padding=(3,0))
        self.bn1 = nn.BatchNorm2d(base_filters)
        self.conv2 = nn.Conv2d(base_filters, base_filters*2, kernel_size=(5,1), stride=(2,1), padding=(2,0))
        self.bn2 = nn.BatchNorm2d(base_filters*2)
        self.proj = nn.Conv2d(base_filters*2, base_filters*4, kernel_size=(1,1), padding=(0,0))
        # a few residual blocks with time-stride=1, keep width same
        self.layer1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(base_filters*2, base_filters*2, kernel_size=(3,1), padding=(1,0)),
            nn.BatchNorm2d(base_filters*2),
            nn.ReLU(),
            nn.Conv2d(base_filters*2, base_filters*2, kernel_size=(3,1), padding=(1,0)),
            nn.BatchNorm2d(base_filters*2),
        )
        self.layer2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(base_filters*2, base_filters*4, kernel_size=(3,1), padding=(1,0)),
            nn.BatchNorm2d(base_filters*4),
            nn.ReLU(),
            nn.Conv2d(base_filters*4, base_filters*4, kernel_size=(3,1), padding=(1,0)),
            nn.BatchNorm2d(base_filters*4),
        )

    def forward(self, x):
        # x: (B,1,T,C)
        x = F.relu(self.bn1(self.conv1(x)))   # (B, f, T, C)
        x = F.relu(self.bn2(self.conv2(x)))   # (B, f2, T/2, C)
        r1 = self.layer1(x) + x               # residual
        r2 = self.layer2(r1) + self.proj(r1)
        return r2  # (B, D, T', C)

class DASAttentionClassifier(nn.Module):
    """
    Backbone -> spatial attention -> classification
    Outputs:
      - cls_logit: (B,1)  (binary classification score)
      - att_map: (B, C)   (spatial attention map in [0,1])
    """
    def __init__(self, in_ch=1, base_filters=32):
        super().__init__()
        self.backbone = DASBackbone(in_ch=in_ch, base_filters=base_filters)
        D = base_filters*4
        # attention module: compress time dimension then conv1d across channels
        # after backbone, features: (B, D, T', C)
        self.att_conv = nn.Conv1d(in_channels=D, out_channels=1, kernel_size=1)  # operates on (B,D,C)
        # classification head: FC from pooled features
        self.cls_fc = nn.Sequential(
            nn.Linear(D, D//2),
            nn.ReLU(),
            nn.Linear(D//2, 1)  # binary logit
        )

    def forward(self, x):
        # x: (B,1,T,C)
        feat = self.backbone(x)              # (B, D, T', C)
        # squeeze time dim by average
        fpool = feat.mean(dim=2)             # (B, D, C)
        # attention logits across channels: for Conv1d we input (B, D, C) -> permute to (B, D, C)
        att_logits = self.att_conv(fpool)
        #NOTE: our att_conv declared as Conv1d(D->1) works because input shape (B,D,C)
        # att_logits -> (B,1,C)
        att_logits = att_logits.squeeze(1)   # (B, C)
        att_map = torch.sigmoid(att_logits)  # (B, C) in [0,1]
        # apply attention to pooled features
        weighted = fpool * att_map.unsqueeze(1)  # (B, D, C)
        # aggregate across spatial dimension to get classifier input
        pooled = weighted.mean(dim=2)        # (B, D)
        cls_logit = self.cls_fc(pooled)   # (B, 1)
        return cls_logit, att_map