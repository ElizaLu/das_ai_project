# src/infer.py
import torch
import numpy as np
from models import DASAttentionClassifier

def load_model(ckpt_path, device='cuda', base_filters=32):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = DASAttentionClassifier(in_ch=1, base_filters=base_filters)
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()
    return model

def infer_sample(model, npy_path, device='cuda'):
    x = np.load(npy_path).astype(np.float32)  # (T,C)
    x = (x - x.mean())/(x.std()+1e-8)
    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,T,C)
    with torch.no_grad():
        logit, att = model(x)
        prob = torch.sigmoid(logit).item()
        att_np = att.cpu().numpy()[0]  # (C,)
    return prob, att_np

if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--npy", required=True)
    args=parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(args.ckpt, device=device)
    p, att = infer_sample(model, args.npy, device=device)
    print("prob_event:", p)
    peak_idx = int(att.argmax())
    print("predicted_channel_idx (0-based):", peak_idx)