import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_npy_data(npy_path):
    data = np.load(npy_path)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {data.shape}")
    return data


def compute_fft_power(data, fs):
    """
    data: shape (T, C), where T is time, C is channel
    fs: sampling frequency in Hz
    return:
        freqs: (F,)
        power: (F, C)
    """
    T, C = data.shape
    fft_vals = np.fft.rfft(data, axis=0)
    power = (np.abs(fft_vals) ** 2) / T
    freqs = np.fft.rfftfreq(T, d=1.0 / fs)
    return freqs, power


def plot_average_spectrum(freqs, power, save_path, title="Average Power Spectrum"):
    avg_power = power.mean(axis=1)

    plt.figure(figsize=(10, 5))
    plt.plot(freqs, avg_power)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_frequency_channel_map(freqs, power, save_path, title="Frequency-Channel Power Map"):
    plt.figure(figsize=(12, 6))
    plt.imshow(
        power.T,
        aspect="auto",
        origin="lower",
        extent=[freqs[0], freqs[-1], 0, power.shape[1] - 1],
        cmap="viridis"
    )
    plt.colorbar(label="Power")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Channel")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    npy_path = "das_ai_project/data/processed_data/mechanical__wajiwa__2021-07-22 13_15_05__000375.npy"
    out_dir = Path("/home/sente/das_ai_project/data")
    out_dir.mkdir(parents=True, exist_ok=True)

    fs = 1000.0

    data = load_npy_data(npy_path)
    print("Loaded data shape:", data.shape)

    freqs, power = compute_fft_power(data, fs=fs)

    avg_spec_path = out_dir / "average_power_spectrum.png"
    freq_ch_map_path = out_dir / "frequency_channel_power_map.png"

    plot_average_spectrum(
        freqs,
        power,
        save_path=avg_spec_path,
        title="Average Power Spectrum Across Channels"
    )

    plot_frequency_channel_map(
        freqs,
        power,
        save_path=freq_ch_map_path,
        title="Frequency-Channel Power Map"
    )


if __name__ == "__main__":
    main()