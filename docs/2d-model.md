抱歉，前版确实加了你数据里没有的东西。下面这版严格按你原来的结构来改，只做你要的三件事：

1. 三分类：`无事件 / 爬网 / 翻网`
2. 2D heatmap：`(class, time, channel)`
3. 多时间支持：一个窗口里可有多个事件区间

默认还是你原来的单通道事件标注方式，不加 `c0/c1` 之类的字段。
如果你的 `label_channel` 是从 1 开始编号，把 `parse_label_txt()` 里读到的通道减 1 就行。

---

## `src/preprocess.py`

```python
import os
import csv
import codecs
import re
import json
import logging
import numpy as np
import scipy.io as sio
from natsort import natsorted
from scipy.signal import butter, sosfiltfilt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# =========================
# 改动点 1：三分类映射
# 0 = 无事件, 1 = 爬网, 2 = 翻网
# =========================
CLASS2ID = {
    "无事件": 0,
    "no_event": 0,
    "无": 0,
    "爬网": 1,
    "crawl": 1,
    "翻网": 2,
    "flip": 2,
}

def bandpass(signal, fs, low=5.0, high=300.0, order=4):
    sos = butter(order, [low, high], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, signal)

def parse_label_txt(label_path):
    """
    仍然沿用你原来的 label 解析方式。
    返回:
      class: 类别名
      channel: 事件通道
      intensity: 强度
      env: 环境信息
    """
    info = {"class": None, "channel": None, "intensity": None, "env": ""}
    try:
        with codecs.open(label_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
    except Exception as e:
        logging.exception("读取label失败: %s | %s", label_path, e)
        return info

    for line in text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            k = k.strip().lower()
            v = v.strip()
            if "data_path" in k or "class" in k:
                info["class"] = v
            elif "channel" in k:
                try:
                    info["channel"] = int(re.search(r"\d+", v).group())
                except:
                    pass
            elif "intensity" in k:
                try:
                    info["intensity"] = int(re.search(r"\d+", v).group())
                except:
                    pass
        else:
            parts = line.split()
            if parts:
                info["env"] = parts[-1]
    return info

def load_and_combine_mat_files(session_path):
    mat_files = [f for f in os.listdir(session_path) if f.endswith(".mat")]
    mat_files = natsorted(mat_files)
    arrays = []

    for f in mat_files:
        file_path = os.path.join(session_path, f)
        try:
            mat = sio.loadmat(file_path)
            if "s" not in mat:
                logging.debug("MAT file %s does not contain 's', skipping", file_path)
                continue
            arr = mat["s"]
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]
            arrays.append(arr)
        except Exception as e:
            logging.exception("读取失败: %s | %s", file_path, e)

    if not arrays:
        return None

    combined = np.concatenate(arrays, axis=1)
    return combined

def detect_event_intervals(combined, fs, channel_hint=None, win_len=256, thr_mul=2.5, min_samples=5):
    """
    用短时 RMS 在单个通道上粗略检测事件区间。
    返回: [(t0, t1), ...]，样本索引坐标
    """
    ch, total = combined.shape
    if channel_hint is None:
        return []

    c = int(channel_hint)
    # 防止越界；如果你的通道是 1-based，建议在 parse_label_txt 里先减 1
    c = max(0, min(c, ch - 1))

    sig = combined[c, :].astype(np.float64)
    sq = sig ** 2
    kernel = np.ones(win_len, dtype=np.float64)
    rms = np.sqrt(np.convolve(sq, kernel, mode='same') / float(win_len))

    mu = rms.mean()
    sigma = rms.std()
    thr = mu + thr_mul * sigma

    mask = rms > thr
    if not mask.any():
        return []

    idx = np.where(mask)[0]
    runs = []
    s = idx[0]
    p = idx[0]
    for i in idx[1:]:
        if i == p + 1:
            p = i
        else:
            runs.append((s, p))
            s = i
            p = i
    runs.append((s, p))

    runs = [r for r in runs if (r[1] - r[0] + 1) >= min_samples]
    if not runs:
        return []

    return [(int(a), int(b)) for a, b in runs]

def _intervals_overlap(t0, t1, a, b):
    return not (t1 < a or t0 > b)

def _clip_interval_to_window(t0, t1, win_start, win_end):
    """
    把 session 级区间裁剪到 window 内。
    返回 window 相对坐标区间 (a, b)，如果没有重叠则返回 None
    """
    a = max(t0, win_start)
    b = min(t1, win_end)
    if a > b:
        return None
    return int(a - win_start), int(b - win_start)

def choose_window_class(window_events):
    """
    一个窗口里如果有多个事件，这里选“主类”作为窗口级三分类标签。
    规则：按事件持续时间累计，谁最长就选谁。
    """
    if not window_events:
        return 0

    score = {}
    for ev in window_events:
        cid = int(ev["class_id"])
        dur = max(1, int(ev["t1"]) - int(ev["t0"]) + 1)
        score[cid] = score.get(cid, 0) + dur

    return max(score.items(), key=lambda x: x[1])[0]

def save_windows_and_metadata(combined_arr, out_dir, class_name, session_folder,
                              window_len, hop, label_parsed, meta_writer,
                              class_id,
                              fs=1000.0,
                              detector_win_len=256,
                              detector_thr_mul=2.5,
                              detector_min_samples=5):
    """
    保存滑窗并写 metadata。

    metadata 列：
      filename, class_name, label_class_id, session_folder, start, end, events_json

    events_json:
      一个 window 内所有事件的列表。
      每个事件格式：
        {"class_id": 1/2, "t0": 相对window起点, "t1": 相对window起点, "channel": 通道号}
    """
    ch, total = combined_arr.shape
    os.makedirs(out_dir, exist_ok=True)

    # =========================
    # 改动点 2：先得到 session 级事件列表
    # 这里仍然沿用“单通道 + RMS 检测”
    # =========================
    session_events = []
    if class_id > 0 and label_parsed and label_parsed.get("channel") is not None:
        detected = detect_event_intervals(
            combined_arr,
            fs=fs,
            channel_hint=label_parsed.get("channel"),
            win_len=detector_win_len,
            thr_mul=detector_thr_mul,
            min_samples=detector_min_samples
        )

        # 只要检测到了多个时间区间，就都作为事件
        for a, b in detected:
            session_events.append({
                "class_id": int(class_id),
                "t0": int(a),
                "t1": int(b),
                "channel": int(label_parsed["channel"]),
            })

        logging.info(
            "Detected %d event interval(s) for session %s: %s",
            len(session_events), session_folder, detected
        )

    idx = 0
    for start in range(0, total - window_len + 1, hop):
        win = combined_arr[:, start:start + window_len]
        save_arr = win.T.astype(np.float32)  # (T, C)

        fname = f"{class_name}__{session_folder}__{idx:06d}.npy"
        np.save(os.path.join(out_dir, fname), save_arr)

        win_end = start + window_len - 1

        # =========================
        # 改动点 3：窗口内事件裁剪
        # 支持一个窗口多个事件
        # =========================
        window_events = []
        for ev in session_events:
            clipped = _clip_interval_to_window(
                ev["t0"], ev["t1"], start, win_end
            )
            if clipped is None:
                continue

            a, b = clipped
            window_events.append({
                "class_id": int(ev["class_id"]),
                "t0": int(a),
                "t1": int(b),
                "channel": int(ev["channel"]),
            })

        # =========================
        # 改动点 4：窗口级三分类标签
        # 没有事件 -> 0
        # 有事件 -> 取主类
        # =========================
        window_class_id = choose_window_class(window_events)

        meta_writer.writerow([
            fname,
            class_name,
            window_class_id,
            session_folder,
            start,
            win_end,
            json.dumps(window_events, ensure_ascii=False),
        ])

        idx += 1

    return idx

def main(args):
    data_root = args.data_root
    label_root = args.label_root
    out_root = args.out_root

    window_len = args.window_len
    hop = args.hop
    fs = args.fs
    low = args.low
    high = args.high
    order = args.order
    apply_bp = args.apply_bandpass

    os.makedirs(out_root, exist_ok=True)
    metadata_path = os.path.join(out_root, "metadata.csv")

    with open(metadata_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename",
            "class_name",
            "label_class_id",
            "session_folder",
            "start",
            "end",
            "events_json",
        ])

        for class_name in os.listdir(data_root):
            class_dir = os.path.join(data_root, class_name)
            if not os.path.isdir(class_dir):
                continue

            class_id = CLASS2ID.get(class_name, -1)
            if class_id < 0:
                logging.warning("未知类别文件夹: %s，跳过", class_name)
                continue

            logging.info("处理类别: %s -> %d", class_name, class_id)

            session_list = natsorted(os.listdir(class_dir))
            for session_folder in session_list:
                session_path = os.path.join(class_dir, session_folder)
                combined = load_and_combine_mat_files(session_path)
                if combined is None:
                    logging.warning("  无有效数据 for session %s", session_folder)
                    continue

                label_parsed = {}

                # 找 label txt
                candidate_dir = os.path.join(label_root, class_name)
                if os.path.isdir(candidate_dir):
                    s1 = re.sub(r'[^0-9]', '', session_folder)
                    for fn in os.listdir(candidate_dir):
                        name, ext = os.path.splitext(fn)
                        s2 = re.sub(r'[^0-9]', '', name)
                        if s1 in s2 or s2 in s1:
                            full = os.path.join(candidate_dir, fn)
                            if ext.lower() == ".txt":
                                label_parsed = parse_label_txt(full)
                                break

                # 可选 bandpass
                if apply_bp:
                    try:
                        for c in range(combined.shape[0]):
                            combined[c, :] = bandpass(
                                combined[c, :].astype(np.float64),
                                fs=fs, low=low, high=high, order=order
                            )
                    except Exception as e:
                        logging.exception("Bandpass failed: %s", e)

                save_windows_and_metadata(
                    combined,
                    out_root,
                    class_name,
                    session_folder,
                    window_len,
                    hop,
                    label_parsed,
                    writer,
                    class_id=class_id,
                    fs=fs,
                    detector_win_len=256,
                    detector_thr_mul=2.5,
                    detector_min_samples=5,
                )

    logging.info("预处理完成，metadata saved to %s", metadata_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--label_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--window_len", type=int, default=1024)
    parser.add_argument("--hop", type=int, default=128)
    parser.add_argument("--fs", type=float, default=1000.0)
    parser.add_argument("--low", type=float, default=5.0)
    parser.add_argument("--high", type=float, default=300.0)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--apply_bandpass", action="store_true")
    args = parser.parse_args()
    main(args)
```

---

## `src/dataset.py`

```python
import os
import csv
import json
import numpy as np
import torch
from torch.utils.data import Dataset

# =========================
# 改动点 1：二维 heatmap
# shape = (3, T, C)
# =========================
def make_gaussian_heatmap_2d(T, C, t_center, c_center, sigma_t=8.0, sigma_c=1.5):
    tt = np.arange(T, dtype=np.float32)[:, None]   # (T, 1)
    cc = np.arange(C, dtype=np.float32)[None, :]   # (1, C)
    hm = np.exp(
        -((tt - t_center) ** 2) / (2 * sigma_t ** 2)
        -((cc - c_center) ** 2) / (2 * sigma_c ** 2)
    )
    hm = hm / (hm.max() + 1e-8)
    return hm.astype(np.float32)

def make_multiclass_heatmap(events, T, C, num_classes=3, sigma_t=8.0, sigma_c=1.5):
    """
    返回 shape = (3, T, C)

    events:
      [
        {"class_id": 1/2, "t0": 相对window起点, "t1": 相对window起点, "channel": 通道号},
        ...
      ]

    同类多个事件：取 max
    """
    target = np.zeros((num_classes, T, C), dtype=np.float32)

    for ev in events:
        cid = int(ev.get("class_id", -1))
        if cid < 0 or cid >= num_classes:
            continue

        t0 = float(ev["t0"])
        t1 = float(ev["t1"])
        ch = float(ev["channel"])

        t_center = 0.5 * (t0 + t1)
        c_center = ch

        hm = make_gaussian_heatmap_2d(
            T, C, t_center, c_center,
            sigma_t=sigma_t, sigma_c=sigma_c
        )
        target[cid] = np.maximum(target[cid], hm)

    return target

class DASWindowDataset(Dataset):
    """
    读取：
      - samples_dir 下的 .npy 文件，shape = (T, C)
      - metadata.csv

    返回：
      x: (1, T, C)
      y: int64, 三分类标签 0/1/2
      heatmap: (3, T, C)
    """
    def __init__(self, samples_dir, metadata_csv, time_normalize=True, hm_sigma_t=8.0, hm_sigma_c=1.5):
        self.samples_dir = samples_dir
        self.entries = []
        self.time_normalize = time_normalize
        self.hm_sigma_t = hm_sigma_t
        self.hm_sigma_c = hm_sigma_c

        with open(metadata_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                self.entries.append(r)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        row = self.entries[idx]
        fname = row['filename']
        path = os.path.join(self.samples_dir, fname)

        x = np.load(path)  # (T, C)

        if self.time_normalize:
            x = (x - x.mean()) / (x.std() + 1e-8)

        x = torch.from_numpy(x.astype(np.float32)).unsqueeze(0)  # (1, T, C)

        # =========================
        # 改动点 2：三分类标签
        # =========================
        y = torch.tensor(int(row["label_class_id"]), dtype=torch.long)

        # =========================
        # 改动点 3：读取窗口内多个事件
        # =========================
        events_json = row.get("events_json", "[]")
        try:
            events = json.loads(events_json)
        except:
            events = []

        T, C = x.shape[1], x.shape[2]
        heatmap = make_multiclass_heatmap(
            events,
            T=T,
            C=C,
            num_classes=3,
            sigma_t=self.hm_sigma_t,
            sigma_c=self.hm_sigma_c
        )
        heatmap = torch.from_numpy(heatmap)  # (3, T, C)

        return x, y, heatmap
```

---

## `src/models.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DASBackbone(nn.Module):
    """
    输入:  (B, 1, T, C)
    输出:  (B, D, T, C)

    这里不做时间下采样，不做通道下采样，
    方便直接预测 2D heatmap。
    """
    def __init__(self, in_ch=1, base_filters=32):
        super().__init__()
        D = base_filters * 4

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_filters, kernel_size=(7, 1), padding=(3, 0)),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=False),

            nn.Conv2d(base_filters, base_filters * 2, kernel_size=(5, 1), padding=(2, 0)),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=False),

            nn.Conv2d(base_filters * 2, D, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(D),
            nn.ReLU(inplace=False),
        )

        self.block = nn.Sequential(
            nn.Conv2d(D, D, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(D),
            nn.ReLU(inplace=False),
            nn.Conv2d(D, D, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(D),
        )

    def forward(self, x):
        x = self.stem(x)
        x = F.relu(self.block(x) + x, inplace=False)
        return x

class DASHeatmapClassifier(nn.Module):
    """
    改动点：输出两头
      - cls_logits:  (B, 3)
      - heat_logits: (B, 3, T, C)
    """
    def __init__(self, in_ch=1, base_filters=32, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = DASBackbone(in_ch=in_ch, base_filters=base_filters)

        D = base_filters * 4

        # 分类头：窗口级三分类
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B, D, 1, 1)
            nn.Flatten(),             # (B, D)
            nn.Linear(D, D // 2),
            nn.ReLU(inplace=False),
            nn.Linear(D // 2, num_classes)
        )

        # 2D heatmap 头：每个类别一张图
        self.heat_head = nn.Conv2d(D, num_classes, kernel_size=1)

    def forward(self, x):
        feat = self.backbone(x)         # (B, D, T, C)
        cls_logits = self.cls_head(feat)  # (B, 3)
        heat_logits = self.heat_head(feat) # (B, 3, T, C)
        return cls_logits, heat_logits
```

---

## `src/train.py`

```python
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import DASWindowDataset
from models import DASHeatmapClassifier
import argparse

def train_one_epoch(model, loader, optim, device, scaler,
                    cls_loss_fn, heat_loss_fn, lambda_heat):
    model.train()
    total_loss = 0.0
    total_samples = 0
    acc_sum = 0.0

    for x, y, heat in loader:
        x = x.to(device)
        y = y.to(device).long()    # (B,)
        heat = heat.to(device)     # (B, 3, T, C)

        optim.zero_grad()

        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            cls_logits, heat_logits = model(x)  # cls:(B,3), heat:(B,3,T,C)
            loss_cls = cls_loss_fn(cls_logits, y)
            loss_heat = heat_loss_fn(heat_logits, heat)
            loss = loss_cls + lambda_heat * loss_heat

        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

        total_loss += float(loss.item()) * x.size(0)
        total_samples += x.size(0)

        pred = cls_logits.argmax(dim=1)
        acc_sum += (pred == y).float().sum().item()

    return total_loss / total_samples, acc_sum / total_samples

def validate(model, loader, device, cls_loss_fn, heat_loss_fn, lambda_heat):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    acc_sum = 0.0

    with torch.no_grad():
        for x, y, heat in loader:
            x = x.to(device)
            y = y.to(device).long()
            heat = heat.to(device)

            cls_logits, heat_logits = model(x)
            loss_cls = cls_loss_fn(cls_logits, y)
            loss_heat = heat_loss_fn(heat_logits, heat)
            loss = loss_cls + lambda_heat * loss_heat

            total_loss += float(loss.item()) * x.size(0)
            total_samples += x.size(0)

            pred = cls_logits.argmax(dim=1)
            acc_sum += (pred == y).float().sum().item()

    return total_loss / total_samples, acc_sum / total_samples

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_ds = DASWindowDataset(
        args.samples_dir,
        args.metadata,
        time_normalize=True,
        hm_sigma_t=args.hm_sigma_t,
        hm_sigma_c=args.hm_sigma_c
    )

    # =========================
    # 改动点：训练集 / 验证集划分
    # 不再拿同一份数据同时训练和验证
    # =========================
    n_total = len(full_ds)
    n_val = max(1, int(n_total * args.val_ratio))
    n_train = n_total - n_val
    g = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=g)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = DASHeatmapClassifier(
        in_ch=1,
        base_filters=args.base_filters,
        num_classes=3
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    # =========================
    # 改动点：三分类损失
    # =========================
    cls_loss_fn = nn.CrossEntropyLoss()

    # =========================
    # 改动点：2D heatmap 损失
    # heatmap 极稀疏，所以给正样本更高权重
    # =========================
    pos_weight = torch.tensor([args.heat_pos_weight] * 3, device=device)
    heat_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val = 0.0
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(args.epochs):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optim, device, scaler,
            cls_loss_fn, heat_loss_fn, args.lambda_heat
        )
        val_loss, val_acc = validate(
            model, val_loader, device,
            cls_loss_fn, heat_loss_fn, args.lambda_heat
        )

        print(
            f"Epoch {epoch} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"time={time.time()-t0:.1f}s"
        )

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "val_acc": val_acc
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, "latest.pth"))
        if val_acc > best_val:
            best_val = val_acc
            torch.save(ckpt, os.path.join(args.checkpoint_dir, "best.pth"))
            print("Saved best:", best_val)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_dir", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--checkpoint_dir", default="../checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--base_filters", type=int, default=32)

    # 改动点：三分类 + heatmap
    parser.add_argument("--lambda_heat", type=float, default=1.0)
    parser.add_argument("--heat_pos_weight", type=float, default=5.0)

    # heatmap 高斯宽度
    parser.add_argument("--hm_sigma_t", type=float, default=8.0)
    parser.add_argument("--hm_sigma_c", type=float, default=1.5)

    # 验证集划分
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
```

---

## `src/infer.py`

```python
import torch
import numpy as np
from models import DASHeatmapClassifier

def load_model(ckpt_path, device='cuda', base_filters=32):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = DASHeatmapClassifier(in_ch=1, base_filters=base_filters, num_classes=3)
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()
    return model

def infer_sample(model, npy_path, device='cuda'):
    x = np.load(npy_path).astype(np.float32)  # (T, C)
    x = (x - x.mean()) / (x.std() + 1e-8)
    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, T, C)

    with torch.no_grad():
        cls_logits, heat_logits = model(x)  # cls:(1,3), heat:(1,3,T,C)
        cls_prob = torch.softmax(cls_logits, dim=1)[0].cpu().numpy()
        heat_prob = torch.sigmoid(heat_logits)[0].cpu().numpy()

    return cls_prob, heat_prob

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--npy", required=True)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(args.ckpt, device=device)

    cls_prob, heat_prob = infer_sample(model, args.npy, device=device)

    id2class = ["无事件", "爬网", "翻网"]
    pred_cls = int(np.argmax(cls_prob))
    print("pred_class:", id2class[pred_cls])
    print("class_probs:", cls_prob)

    # 每个类别的 heatmap 峰值位置
    for k in range(3):
        peak_idx = np.unravel_index(np.argmax(heat_prob[k]), heat_prob[k].shape)
        t_peak, c_peak = peak_idx
        peak_val = heat_prob[k][peak_idx]
        print(f"class={id2class[k]} peak_t={t_peak} peak_c={c_peak} peak_val={peak_val:.4f}")
```

---

这版的逻辑是干净的：

* `preprocess.py`：把一个 session 切成很多窗口，窗口里有多个事件就都写进 `events_json`
* `dataset.py`：把 `events_json` 变成 `(3, T, C)` 的 heatmap
* `models.py`：输出三分类 logits + 三张 heatmap
* `train.py`：三分类用 `CrossEntropyLoss`，heatmap 用 `BCEWithLogitsLoss`
* `infer.py`：输出类别概率和每个类别 heatmap 的峰值位置

如果你下一步要做，我可以继续把这版再整理成你能直接复制进项目的目录结构，包括文件名、命令行运行方式和一个最小可跑的配置。
