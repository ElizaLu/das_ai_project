# src/dataset.py
import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset

def make_gaussian_heatmap(center_idx, C, sigma=0.5):
    x = np.arange(C)
    hm = np.exp(-((x - center_idx)**2) / (2 * sigma**2))
    if hm.max() > 0:
        hm = hm / hm.max()
    return hm.astype(np.float32)

class DASWindowDataset(Dataset):
    """
    Expects:
      - samples_dir containing .npy files (time x channel)
      - metadata.csv with columns: filename,...,label_channel,event_class
    Returns:
      - x: torch.FloatTensor (1, T, C)
      - label: int (0/1)
      - heatmap: torch.FloatTensor (C,)  (if event_class==1 else zeros)
    """
    def __init__(self, samples_dir, metadata_csv, time_normalize=True, hm_sigma=0.5):
        self.samples_dir = samples_dir
        self.entries = []
        self.hm_sigma = hm_sigma
        with open(metadata_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                self.entries.append(r)
        self.time_normalize = time_normalize

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        row = self.entries[idx]
        fname = row['filename']
        path = os.path.join(self.samples_dir, fname)
        x = np.load(path)  # shape (T, C)
        # normalize per-sample (time-channel)
        if self.time_normalize:
            x = (x - x.mean()) / (x.std() + 1e-8)
        # to tensor shape (1, T, C)
        x = torch.from_numpy(x.astype(np.float32)).unsqueeze(0)
        event_class = int(row.get('event_class', '0') or 0)
        label_channel = row.get('label_channel', '')
        C = x.shape[2]
        if event_class and label_channel not in ('', None):
            try:
                ch_idx = int(label_channel)
            except:
                ch_idx = None
        else:
            ch_idx = None
        if event_class and ch_idx is not None:
            heatmap = make_gaussian_heatmap(ch_idx, C, sigma=self.hm_sigma)
        else:
            heatmap = np.zeros(C, dtype=np.float32)
        heatmap = torch.from_numpy(heatmap)
        return x, torch.tensor([event_class], dtype=torch.float32), heatmap