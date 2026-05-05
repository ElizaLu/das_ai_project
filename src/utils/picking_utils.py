import numpy as np
from .signal_utils import moving_rms_2d


def interval_overlap_len(a0, a1, b0, b1):
    left = max(a0, b0)
    right = min(a1, b1)
    return max(0, right - left + 1)


def merge_runs(idxs):
    if len(idxs) == 0:
        return []

    runs = []
    s = idxs[0]
    p = idxs[0]
    for i in idxs[1:]:
        if i == p + 1:
            p = i
        else:
            runs.append((int(s), int(p)))
            s = i
            p = i
    runs.append((int(s), int(p)))
    return runs


def detect_event_time_intervals(
    combined,
    fs,
    win_len=256,
    thr_mul=1.0,
    min_samples=5,
    aggregate="p90",
    smooth_len=11,
):
    if combined is None or combined.size == 0:
        return []

    rms_2d = moving_rms_2d(combined, win_len=win_len)

    if aggregate == "p90":
        score_t = np.percentile(rms_2d, 90, axis=0)
    elif aggregate == "max":
        score_t = np.max(rms_2d, axis=0)
    elif aggregate == "mean":
        score_t = np.mean(rms_2d, axis=0)
    elif aggregate == "median":
        score_t = np.median(rms_2d, axis=0)
    else:
        raise ValueError(f"Unknown aggregate mode: {aggregate}")

    if smooth_len and smooth_len > 1:
        kernel = np.ones(smooth_len, dtype=np.float64) / float(smooth_len)
        score_t = np.convolve(score_t, kernel, mode="same")

    med = np.median(score_t)
    mad = np.median(np.abs(score_t - med)) + 1e-8
    thr = med + thr_mul * 1.4826 * mad

    mask = score_t > thr
    if not np.any(mask):
        return []

    idx = np.where(mask)[0]
    runs = merge_runs(idx)
    runs = [r for r in runs if (r[1] - r[0] + 1) >= min_samples]
    return runs


def window_event_ratio(window_start, window_end, event_intervals):
    if not event_intervals:
        return 0.0

    win_len = window_end - window_start + 1
    if win_len <= 0:
        return 0.0

    overlap = 0
    for a, b in event_intervals:
        overlap += interval_overlap_len(window_start, window_end, a, b)

    return overlap / float(win_len)


def pick_event_channels(
    combined,
    t0,
    t1,
    method="peak_rms",
    ratio=0.01,
    min_channels=1,
):
    """
    返回一组通道，而不是单个通道。

    规则：
    1) 先算每个通道的得分
    2) 取 scores >= ratio * max(scores) 的通道
    3) 可选只保留最长连续通道段
    """
    if combined is None or combined.size == 0:
        return None, None, None

    ch, total = combined.shape
    t0 = max(0, int(t0))
    t1 = min(total - 1, int(t1))
    if t1 <= t0:
        return None, None, None

    scores = np.zeros(ch, dtype=np.float64)

    for c in range(ch):
        tr = combined[c, t0:t1 + 1].astype(np.float64)

        if method == "rms":
            scores[c] = np.sqrt(np.mean(tr ** 2))

        elif method == "peak_rms":
            if len(tr) < 3:
                scores[c] = np.sqrt(np.mean(tr ** 2))
            else:
                from scipy.ndimage import uniform_filter1d
                win = min(64, max(5, len(tr) // 10))
                rms = np.sqrt(uniform_filter1d(tr ** 2, size=win, mode="nearest"))
                scores[c] = np.max(rms)

        else:
            raise ValueError(f"Unknown channel picking method: {method}")

    max_score = np.max(scores)
    if max_score <= 0:
        return np.array([], dtype=int), scores, 0.0

    thr = ratio * max_score
    selected = np.where(scores >= thr)[0]

    if selected.size < min_channels:
        selected = np.argsort(scores)[-min_channels:]
        selected = np.sort(selected)

    return selected.astype(int), scores, float(thr)