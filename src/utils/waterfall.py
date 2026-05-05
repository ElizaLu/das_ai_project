import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path


npy_path = "/home/sente/das_ai_project/data/processed_data/mechanical__wajikai__2021-07-22 15_40_53__000273.npy"

data = np.load(npy_path)

if data.ndim != 2:
    raise ValueError(f"Expected 2D array, got shape={data.shape}")

plt.figure(figsize=(12, 5))
plt.imshow(
    data,
    aspect="auto",
    origin="lower",
    interpolation="nearest",
)

plt.xlabel("Channel")
plt.ylabel("Time step")
plt.title("DAS Waterfall")

plt.colorbar(label="Value")
plt.tight_layout()

filename = Path(npy_path).stem
out_path = f"{filename}.png"

plt.savefig(out_path)
plt.close()

print(f"saved to: {out_path}")