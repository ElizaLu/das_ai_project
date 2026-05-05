#!/bin/bash
set -e

cd /home/sente/das_ai_project

source /home/sente/anaconda3/etc/profile.d/conda.sh
conda activate pytorch_env

python src/verify_pick.py \
  --data_root /home/sente/das_ai_project/data/ \
  --pick_root /home/sente/das_ai_project/data/processed_data \
  --out_dir /home/sente/das_ai_project/data/verify_figs \
  --apply_bandpass

echo "✅ Verification finished!"