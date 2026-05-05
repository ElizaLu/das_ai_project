import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.ndimage import uniform_filter1d


def bandpass(x, fs, low=5.0, high=300.0, order=4):
    sos = butter(order, [low, high], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def bandpass_2d(arr, fs, low=5.0, high=300.0, order=4):
    out = np.empty_like(arr, dtype=np.float64)
    for c in range(arr.shape[0]):
        out[c] = bandpass(arr[c].astype(np.float64), fs=fs, low=low, high=high, order=order)
    return out


def moving_rms_1d(x, win_len=256):
    x = np.asarray(x, dtype=np.float64)
    x2 = x * x
    return np.sqrt(uniform_filter1d(x2, size=win_len, mode="nearest"))


def moving_rms_2d(arr, win_len=256):
    arr = np.asarray(arr, dtype=np.float64)
    out = np.empty_like(arr, dtype=np.float64)
    for c in range(arr.shape[0]):
        out[c] = moving_rms_1d(arr[c], win_len=win_len)
    return out


def sta_lta_1d(x, nsta=25, nlta=250, eps=1e-8):
    x = np.asarray(x, dtype=np.float64)
    if nlta <= nsta:
        nlta = nsta + 1
    absx = np.abs(x)
    sta = uniform_filter1d(absx, size=nsta, mode="nearest")
    lta = uniform_filter1d(absx, size=nlta, mode="nearest")
    return sta / (lta + eps)


def normalize_0_1(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float64)
    xmin = np.min(x)
    xmax = np.max(x)
    return (x - xmin) / (xmax - xmin + eps)


def robust_zscore(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float64)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return (x - med) / (1.4826 * mad + eps)