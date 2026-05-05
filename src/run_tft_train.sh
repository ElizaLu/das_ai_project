#!/usr/bin/env bash
set -euo pipefail
set -e

cd /home/sente/das_ai_project

source /home/sente/anaconda3/etc/profile.d/conda.sh
conda activate pytorch_env

DATA_ROOT="/home/sente/das_ai_project/data/processed_data_128"
CSV_PATH="/home/sente/das_ai_project/data/processed_data_128/metadata.csv"
output_dir="/home/sente/das_ai_project/checkpoints/128_hop"

python src/train_tft.py \
  --data_root "${DATA_ROOT}" \
  --metadata_csv "${CSV_PATH}" \
  --output_dir "${output_dir}" \
  --balance_train