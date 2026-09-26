"""Signal-processing primitives shared by the detectors, metrics and figures."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from analysis import config


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def lowpass_signal(x: np.ndarray, fs: float, cutoff_hz: float = 6.0, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass filter for kinematic signals."""
    values = np.asarray(x, dtype=float)
    if len(values) < max(20, order * 6):
        return values.copy()

    series = pd.Series(values).interpolate(limit_direction="both").to_numpy()
    nyquist = 0.5 * fs
    cutoff = min(cutoff_hz, nyquist * 0.95)
    if cutoff <= 0:
        return series
    sos = butter(order, cutoff / nyquist, btype="lowpass", output="sos")
    return sosfiltfilt(sos, series)


def highpass_detrend(x: np.ndarray, fs: float, cutoff_hz: float = 0.05, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth high-pass filter to remove z-axis drift trends.

    A 0.05 Hz cutoff removes slow drift (period > 20 s) while preserving all
    Tai Chi trunk-rotation content (typically 0.2–1 Hz).  The filter is applied
    with sosfiltfilt so it introduces no phase distortion.

    A pass-through when ``config.IGNORE_HIGH_PASS_FILTER`` is set.
    """
    if config.IGNORE_HIGH_PASS_FILTER:
        return x.copy()

    values = np.asarray(x, dtype=float)
    min_samples = max(20, order * 6)
    # Need at least ~3 periods of the cutoff frequency for the filter to work
    min_samples = max(min_samples, int(3.0 / cutoff_hz * fs))
    if len(values) < min_samples:
        # Too short to filter — fall back to simple linear detrend
        return values - np.linspace(values[0], values[-1], len(values))

    series = pd.Series(values).interpolate(limit_direction="both").to_numpy()
    nyquist = 0.5 * fs
    cutoff = min(cutoff_hz, nyquist * 0.45)  # stay well below Nyquist
    if cutoff <= 0:
        return series
    sos = butter(order, cutoff / nyquist, btype="highpass", output="sos")
    return sosfiltfilt(sos, series)


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


def resample_filtered_window(
    time: np.ndarray,
    signal: np.ndarray,
    fs: float,
    start: int,
    end: int,
    target_fs: float,
    cutoff_hz: float | None = None,
) -> np.ndarray:
    """Anti-alias filter at the source rate, then interpolate to target_fs."""
    cutoff = min(6.0, 0.45 * target_fs) if cutoff_hz is None else cutoff_hz
    filtered = lowpass_signal(signal, fs, cutoff_hz=cutoff)
    n_target = max(2, int(round((end - start) / fs * target_fs)))
    target_time = time[start] + np.arange(n_target) / target_fs
    return np.interp(target_time, time, filtered)


def resample_filtered_full(
    time: np.ndarray,
    signal: np.ndarray,
    fs: float,
    target_fs: float,
    cutoff_hz: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Anti-alias filter a full signal and resample it to a uniform target grid."""
    cutoff = min(12.0, 0.45 * target_fs) if cutoff_hz is None else cutoff_hz
    filtered = lowpass_signal(signal, fs, cutoff_hz=cutoff)
    n_target = max(2, int(round((time[-1] - time[0]) * target_fs)) + 1)
    target_time = time[0] + np.arange(n_target) / target_fs
    target_time = target_time[target_time <= time[-1]]
    return target_time, np.interp(target_time, time, filtered)


# ---------------------------------------------------------------------------
# Analysis primitives
# ---------------------------------------------------------------------------


def extract_contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        if start is not None and (not value or i == len(mask) - 1):
            end = i + 1 if value and i == len(mask) - 1 else i
            if end > start:
                runs.append((start, end))
            start = None
    return runs


LAG_SEARCH_S = 1.0
"""How far :func:`pearson_lag` looks either way.  Every pelvis-chest lag on these
recordings lies between -0.13 and -0.45 s; an offset beyond a second is a
different phase of the movement, not the coordination of one."""

LAG_MIN_OVERLAP = 0.5
"""Fraction of the window that must overlap at a lag for it to be scored."""


def pearson_lag(
    a: np.ndarray,
    b: np.ndarray,
    fs: float,
    max_lag_s: float = LAG_SEARCH_S,
    min_overlap: float = LAG_MIN_OVERLAP,
    work_fs: float = 50.0,
) -> tuple[float, float]:
    """Peak correlation between two signals over a range of lags, and that lag.

    At every lag the Pearson correlation is computed on the overlapping samples
    only, with their own means and standard deviations, so the value is a true
    correlation and cannot exceed 1.  (The overlap-normalised estimator this
    replaces z-scored over the whole window and then divided by the overlap
    count: 7 of 12 trunk rows came out above 1, and on a synthetic turn with a
    known 0.30 s lag it returned 0.00-0.17 s depending on where the turn sat in
    the window.  This returns 0.300 s wherever it sits.)

    Only the most positive peak counts -- an anti-phase peak is not the two
    segments moving together.  The lag is refined to sub-sample precision by a
    parabola through the peak and its neighbours.

    Returns ``(r, lag_s)``; ``lag_s < 0`` means ``a`` leads ``b``.  The lag is
    NaN when the peak sits on the edge of the search range, where it is a
    failure marker rather than a measurement, and both are NaN when no lag has
    enough overlap.  The signals are low-passed at 6 Hz and decimated to about
    ``work_fs`` first, which changes nothing below 6 Hz and makes the search cheap.
    """
    a = lowpass_signal(np.asarray(a, dtype=float), fs, cutoff_hz=6.0)
    b = lowpass_signal(np.asarray(b, dtype=float), fs, cutoff_hz=6.0)
    step = max(1, int(round(fs / work_fs)))
    a, b = a[::step], b[::step]
    rate = fs / step
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    max_lag = int(round(max_lag_s * rate))

    lags, values = [], []
    for k in range(-max_lag, max_lag + 1):
        x, y = (a[k:], b[:n - k]) if k >= 0 else (a[:n + k], b[-k:])
        if len(x) < max(3, min_overlap * n):
            continue
        sx, sy = np.std(x), np.std(y)
        if sx <= 0 or sy <= 0:
            continue
        lags.append(k)
        values.append(float(np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy)))
    if not values:
        return float("nan"), float("nan")

    best = int(np.argmax(values))
    r = values[best]
    if best == 0 or best == len(values) - 1:
        return r, float("nan")
    y0, y1, y2 = values[best - 1], values[best], values[best + 1]
    curvature = y0 - 2.0 * y1 + y2
    shift = 0.5 * (y0 - y2) / curvature if curvature < 0 else 0.0
    return r, float((lags[best] + shift) / rate)


# ---------------------------------------------------------------------------
# Index/time conversion
#
# The pipeline converts an index to seconds as idx / fs, so the exact inverse is
# round(t * fs).  Truncating instead shifts the window by one sample, which
# changed the lumbar sway variance by ~0.8 % -- small enough to miss, large
# enough to matter.  Every conversion in the editor goes through these two.
# ---------------------------------------------------------------------------


def sec_to_idx(seconds: float, fs: float, n_samples: int) -> int:
    return int(np.clip(round(float(seconds) * fs), 0, n_samples - 1))


def idx_to_sec(index: int, fs: float) -> float:
    return float(index) / fs
