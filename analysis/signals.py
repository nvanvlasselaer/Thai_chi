"""Signal-processing primitives shared by the detectors, metrics and figures."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, correlate, correlation_lags, sosfiltfilt

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


def cross_correlation_lag(a: np.ndarray, b: np.ndarray, fs: float, max_lag_s: float = 2.0) -> tuple[float, float]:
    a = lowpass_signal(a, fs, cutoff_hz=6.0)
    b = lowpass_signal(b, fs, cutoff_hz=6.0)
    a = (a - np.mean(a)) / (np.std(a) + 1e-12)
    b = (b - np.mean(b)) / (np.std(b) + 1e-12)
    max_lag = int(max_lag_s * fs)
    corr = correlate(a, b, mode="full", method="fft")
    lags = correlation_lags(len(a), len(b), mode="full")
    overlap = correlate(np.ones_like(a), np.ones_like(b), mode="full", method="direct")
    values = corr / np.maximum(overlap, 1.0)
    keep = np.abs(lags) <= max_lag
    lags = lags[keep]
    values = values[keep]
    idx = int(np.argmax(np.abs(values)))
    return float(values[idx]), float(lags[idx] / fs)


# ---------------------------------------------------------------------------
# Index/time conversion
#
# The pipeline converts an index to seconds as idx / fs, so the exact inverse is
# round(t * fs).  Truncating instead shifts the window by one sample, which
# changes lumbar_ap_acc_variance_g2 by ~0.8 % -- small enough to miss, large
# enough to matter.  Every conversion in the editor goes through these two.
# ---------------------------------------------------------------------------


def sec_to_idx(seconds: float, fs: float, n_samples: int) -> int:
    return int(np.clip(round(float(seconds) * fs), 0, n_samples - 1))


def idx_to_sec(index: int, fs: float) -> float:
    return float(index) / fs
