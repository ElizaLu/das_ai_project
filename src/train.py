# src/train.py
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import DASWindowDataset
from models import DASAttentionClassifier
import argparse
import numpy as np

def train_one_epoch(model, loader, optim, device, scaler, bce_loss, lambda_att):
    model.train()
    total_loss = 0.0; total_samples=0; acc_sum=0.0
    for x, y, heat in loader:
        x = x.to(device); y = y.to(device); heat = heat.to(device)
        optim.zero_grad()
        with torch.amp.autocast("cuda"):
            logits, att_map = model(x)
            pred = torch.sigmoid(logits)
            loss_cls = bce_loss(logits, y)
            loss_att = bce_loss(att_map, heat)
            loss = loss_cls + lambda_att * loss_att
        scaler.scale(loss).backward()
        scaler.step(optim); scaler.update()
        total_loss += float(loss.item()) * x.size(0) #loss.item() = 当前 batch 的平均 loss
        total_samples += x.size(0) #batch size
        preds = (pred>0.5).float()
        acc_sum += (preds==y).float().sum().item()
    return total_loss/total_samples, acc_sum/total_samples

def validate(model, loader, device, bce_loss, lambda_att):
    model.eval()
    total_loss=0.0; total_samples=0; acc_sum=0.0
    with torch.no_grad():
        for x,y,heat in loader:
            x=x.to(device); y=y.to(device); heat=heat.to(device)
            logits, att_map = model(x)
            pred = torch.sigmoid(logits)
            loss_cls = bce_loss(logits, y)
            loss_att = bce_loss(att_map, heat)
            loss = loss_cls + lambda_att * loss_att
            total_loss += float(loss.item()) * x.size(0)
            total_samples += x.size(0)
            preds = (pred>0.5).float()
            acc_sum += (preds==y).float().sum().item()
    return total_loss/total_samples, acc_sum/total_samples

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = DASWindowDataset(args.samples_dir, args.metadata, time_normalize=True, hm_sigma=args.hm_sigma)
    val_ds = DASWindowDataset(args.samples_dir, args.metadata, time_normalize=True, hm_sigma=args.hm_sigma)  # ideally separate val
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)#生成batch
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    model = DASAttentionClassifier(in_ch=1, base_filters=args.base_filters).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    bce_loss = nn.BCEWithLogitsLoss()
    best_val = 0.0
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    for epoch in range(args.epochs):
        t0=time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optim, device, scaler, bce_loss, args.lambda_att)
        val_loss, val_acc = validate(model, val_loader, device, bce_loss, args.lambda_att)
        print(f"Epoch {epoch} train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} time={time.time()-t0:.1f}s")
        # save
        ckpt = {"epoch":epoch, "model":model.state_dict(), "optim":optim.state_dict(), "val_acc":val_acc}
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
    parser.add_argument("--lambda_att", type=float, default=1.0)
    parser.add_argument("--hm_sigma", type=float, default=0.5)
    args = parser.parse_args()
    main(args)