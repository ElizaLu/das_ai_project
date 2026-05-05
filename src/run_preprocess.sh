#!/bin/bash
set -e

cd /home/sente/das_ai_project

source /home/sente/anaconda3/etc/profile.d/conda.sh
conda activate pytorch_env

python src/preprocess.py \
  --data_root /home/sente/das_ai_project/data \
  --out_root /home/sente/das_ai_project/data/processed_data_128 \
  --apply_bandpass \
  --channel_pick_ratio 0.5 \

echo "✅ Preprocess finished!"