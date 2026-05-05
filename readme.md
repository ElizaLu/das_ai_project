# DAS AI Project

A research-oriented pipeline for **Distributed Acoustic Sensing (DAS)** signal preprocessing, **Temporal Fusion Transformer (TFT)** training, and real-time inference.

This repository is designed as a reproducible experimental framework for turning raw DAS time-series data into model-ready datasets, training a Transformer-based classifier, and deploying the model for streaming inference.

---

## Highlights

- **DAS signal preprocessing** with band-pass filtering, channel cropping, and sample-rate alignment
- **Window-based dataset construction** for multi-channel sequence learning
- **Temporal Fusion Transformer (TFT)** classifier for three-way prediction
- **Checkpointed training** with logging, TensorBoard, and resume support
- **Real-time inference** for incoming UDP/MQTT-driven DAS streams
- **Visualization support** including waterfall-style signal inspection

---

## Motivation

DAS systems produce high-frequency, multi-channel time-series signals that contain rich physical structure but are difficult to interpret directly.  
This project explores how deep learning, especially Transformer-style sequence models, can be used to learn discriminative patterns from these signals for tasks such as:

- signal state recognition,
- anomaly detection,
- event classification,
- real-time condition monitoring.

The codebase is structured to support future research extensions, including better label design, larger-scale experiments, and more advanced Transformer-based architectures.

---

## Repository Structure

```text
das_ai_project/
├── das_server.py
├── run_server.sh
├── src/
│   ├── config.yaml
│   ├── das_dataset.py
│   ├── infer_tft.py
│   ├── preprocess.py
│   ├── run_preprocess.sh
│   ├── run_tft_train.sh
│   ├── tft_model.py
│   ├── train_tft.py
│   ├── verify_pick.py
│   └── verify_pick.sh
├── docs/
└── json/
```

---

## Core Workflow

### 1. Data preparation
Raw DAS matrices are filtered and standardized so that they can be used consistently across training and inference.

### 2. Dataset construction
The pipeline converts long signal sequences into windowed samples suitable for deep learning.

### 3. Model training
A Temporal Fusion Transformer is trained on the prepared samples, with validation metrics and checkpoints saved during training.

### 4. Real-time inference
The trained model can be loaded for online prediction on new DAS matrices or streaming data.

---

## Model Overview

The main classifier is a **Temporal Fusion Transformer** built in PyTorch.

It includes:

- **variable selection** for multi-channel input,
- **sinusoidal positional encoding**,
- **Transformer encoder layers**,
- **attention pooling** for sequence summarization,
- a final **classification head**.

This architecture is well suited for DAS signals because it can model both local channel interactions and longer temporal dependencies.

---

## Training Pipeline

The training script supports:

- seed control for reproducibility,
- train/validation splitting by session,
- optional class balancing,
- mixed-precision training on CUDA,
- TensorBoard logging,
- checkpoint saving (`last.pt` and `best.pt`),
- resume training from saved checkpoints.

Example:

```bash
python src/train_tft.py \
  --data_root /path/to/data_root \
  --metadata_csv /path/to/metadata.csv \
  --output_dir runs/tft_exp \
  --device cuda \
  --epochs 30
```

---

## Inference Pipeline

The inference script loads a trained checkpoint and performs prediction on `.npy` inputs.

It supports:

- single file or directory input,
- `tc` / `ct` input layouts,
- band-pass filtering before inference,
- batched prediction,
- class-wise probability output.

Example:

```bash
python src/infer_tft.py \
  --checkpoint runs/tft_exp/best.pt \
  --input_path /path/to/npy_or_directory \
  --device cuda \
  --input_layout tc
```

---

## Real-Time Server

`das_server.py` implements a streaming-oriented receiver for DAS data.

It is designed to:

- receive UDP packets,
- maintain a rolling signal history,
- trigger inference on the latest matrix,
- publish state and results through MQTT,
- generate waterfall visualizations for inspection.

This makes the project suitable not only for offline experiments, but also for deployment-style research prototypes.

---

## Utility Scripts

- `run_preprocess.sh`: preprocess entry point
- `run_tft_train.sh`: training entry point
- `verify_pick.sh`: verification / inspection helper
- `run_server.sh`: server launch script

---

## Expected Data Format

The project is built around multi-channel time-series matrices.  
In practice, the data pipeline expects:

- 2D signal arrays,
- channel-consistent slicing,
- metadata that links samples to labels or sessions,
- saved `.npy` / CSV artifacts for reproducible experiments.

The exact data specification can be extended later as the dataset grows.

---

## Planned Research Extensions

This repository is intended to evolve into a stronger research platform. Possible future directions include:

- larger-scale DAS benchmark construction,
- more refined label taxonomy,
- contrastive or self-supervised pretraining,
- longer-horizon sequence modeling,
- uncertainty estimation,
- domain adaptation across different sensing environments,
- Transformer-based ablation studies,
- publication-ready evaluation protocols.

---

## Reproducibility Notes

For a cleaner research workflow, it is recommended to keep the following items versioned together with the code:

- preprocessing configuration,
- data split strategy,
- model hyperparameters,
- experiment logs,
- checkpoint metadata,
- inference settings.

This makes it easier to reproduce results and to turn the project into a paper-ready experimental framework.

---

## Citation

If this repository is used in academic work, please cite the project and describe the specific preprocessing and modeling settings used in your experiment.

---

## License

Add a license file before public release.
