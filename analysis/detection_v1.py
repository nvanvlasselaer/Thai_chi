"""The original fixed-window detectors, kept so earlier results stay reproducible.

``config.DETECTOR = "v1"`` selects them for trunk rotation: every event is the
8 s sliding window with the largest trunk-pelvis yaw excursion, and each novice
event is matched to the trained recording by banded dynamic time warping.
:mod:`analysis.detection` replaces both with boundaries derived from the motion
and pairing through a whole-recording alignment.

The v1 stabilization search, :func:`find_stabilization`, places the settling
window of a v1 session's single-leg stances too.

Everything here reads ``Kinematics.omega_raw_mag``, the gyroscope magnitude with
its bias still in, because that is what these detectors were designed and run
on: a v1 session detected today gets exactly the original windows.  The
metrics computed on those windows are the current ones; the original numbers
come from the git tag ``metrics-v1``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import zscore

from analysis.kinematics import Kinematics
from analysis.signals import highpass_detrend, lowpass_signal


# ---------------------------------------------------------------------------
# Trunk rotation: fixed 8 s windows
# ---------------------------------------------------------------------------


def detect_novice_trunk_rotation_events(kin: Kinematics, fs: float, num_events: int = 3) -> tuple[list[tuple[int, int, int]], str]:
    z = highpass_detrend(kin.trunk_rel_euler_deg[:, 2], fs, cutoff_hz=0.05)
    z = lowpass_signal(z, fs, cutoff_hz=4.0)
    chest_omega = lowpass_signal(kin.omega_raw_mag["chestbone"], fs, cutoff_hz=6.0)
    ignore = int(15 * fs)
    window = int(8 * fs)
    step = int(1 * fs)

    scores = []
    for start in range(ignore, len(z) - window - ignore, step):
        end = start + window
        z_range = np.ptp(z[start:end])
        omega_peak = np.percentile(chest_omega[start:end], 95)
        score = z_range * np.log1p(omega_peak)
        scores.append((score, start, end))

    scores.sort(key=lambda x: x[0], reverse=True)

    selected_events = []
    for score, start, end in scores:
        overlap = False
        for sel_start, sel_end, _ in selected_events:
            if not (end < sel_start or start > sel_end):
                overlap = True
                break
        if not overlap:
            peak = start + int(np.argmax(np.abs(z[start:end] - np.median(z[start:end]))))
            selected_events.append((start, end, peak))
        if len(selected_events) == num_events:
            break

    selected_events.sort(key=lambda x: x[0])
    rationale = f"Selected {num_events} 8 s trunk-rotation transitions based on combined trunk-pelvis relative-z excursion and chest angular-velocity peaks."
    return selected_events, rationale


# ---------------------------------------------------------------------------
# Stabilization: longest run below the local 25th percentile
# ---------------------------------------------------------------------------


def find_stabilization(kin: Kinematics, fs: float, after_end: int) -> tuple[int, int]:
    lumbar = lowpass_signal(kin.omega_raw_mag["lumbar"], fs, cutoff_hz=4.0)
    chest = lowpass_signal(kin.omega_raw_mag["chestbone"], fs, cutoff_hz=4.0)
    combined = lumbar + chest
    search_start = after_end
    search_end = min(len(combined), after_end + int(10 * fs))
    threshold = np.percentile(combined[search_start:search_end], 25)
    min_len = int(2 * fs)
    mask = combined[search_start:search_end] < threshold
    best_start = search_start
    best_len = 0
    run_start = None
    for i, value in enumerate(mask):
        if value and run_start is None:
            run_start = i
        if (not value or i == len(mask) - 1) and run_start is not None:
            run_end = i if not value else i + 1
            run_len = run_end - run_start
            if run_len >= min_len:
                best_start = search_start + run_start
                best_len = run_len
                break
            if run_len > best_len:
                best_len = run_len
                best_start = search_start + run_start
            run_start = None
    if best_len < min_len:
        center = search_start + int(np.argmin(pd.Series(combined[search_start:search_end]).rolling(min_len, min_periods=1).mean()))
        best_start = max(search_start, center - min_len // 2)
        best_len = min_len
    return best_start, min(best_start + best_len, len(combined))


# ----------------------------------------------------------------------------
# Matching novice events in the trained recording by DTW
#
# Each novice window is described by seven z-scored channels resampled to 20 Hz,
# and the trained recording is searched in 1 s steps for the window with the
# smallest banded DTW distance.  The channels are low-pass filtered once, up
# front, and sliced per candidate window; filtering the full signal again for
# every candidate gives identical numbers and took minutes per event.
# ----------------------------------------------------------------------------


def dtw_distance(a: np.ndarray, b: np.ndarray, band: int = 25) -> float:
    n, m = len(a), len(b)
    inf = np.inf
    prev = np.full(m + 1, inf)
    curr = np.full(m + 1, inf)
    prev[0] = 0.0
    for i in range(1, n + 1):
        curr[:] = inf
        j_start = max(1, i - band)
        j_end = min(m, i + band)
        for j in range(j_start, j_end + 1):
            cost = np.linalg.norm(a[i - 1] - b[j - 1])
            curr[j] = cost + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev
    return float(prev[m] / (n + m))


SIGNATURE_CUTOFF_HZ = 6.0
"""The anti-alias cutoff ``resample_filtered_window`` uses for a 20 Hz target rate."""


@dataclass
class SignatureChannels:
    """Pre-filtered channels for the DTW signature, in source-rate samples."""

    time_s: np.ndarray
    channels: list[np.ndarray] = field(default_factory=list)


def precompute_signature_channels(kin: Kinematics, fs: float) -> SignatureChannels:
    """Filter the seven DTW signature channels once, at the source rate."""
    raw = [
        highpass_detrend(kin.trunk_rel_euler_deg[:, 2], fs, cutoff_hz=0.05),
        kin.eulers_deg["lumbar"][:, 0],
        kin.omega_raw_mag["chestbone"],
        kin.left_knee_deg[:, 0],
        kin.right_knee_deg[:, 0],
        kin.omega_raw_mag["lhand"],
        kin.omega_raw_mag["rhand"],
    ]
    return SignatureChannels(
        time_s=kin.t,
        channels=[lowpass_signal(channel, fs, cutoff_hz=SIGNATURE_CUTOFF_HZ) for channel in raw],
    )


def signature_matrix(
    precomputed: SignatureChannels, fs: float, start: int, end: int, target_fs: float = 20.0
) -> np.ndarray:
    """Resample the pre-filtered channels over a window and z-score them."""
    time = precomputed.time_s
    n_target = max(2, int(round((end - start) / fs * target_fs)))
    target_time = time[start] + np.arange(n_target) / target_fs
    matrix = np.column_stack(
        [np.interp(target_time, time, channel) for channel in precomputed.channels]
    )
    return np.nan_to_num(zscore(matrix, axis=0, nan_policy="omit"))


def find_trained_match(
    novice: SignatureChannels,
    trained: SignatureChannels,
    fs: float,
    novice_start: int,
    novice_end: int,
    min_start: int = 0,
    target_fs: float = 20.0,
    ignore_s: float = 15.0,
    step_s: float = 1.0,
) -> tuple[int, int, float]:
    """Locate the trained window best matching a novice window, by banded DTW."""
    template = signature_matrix(novice, fs, novice_start, novice_end, target_fs)
    window = novice_end - novice_start
    ignore = int(ignore_s * fs)
    start_search = max(ignore, min_start)
    step = max(1, int(step_s * fs))
    band = int(1.5 * target_fs)

    stop = len(trained.time_s) - window - ignore
    if start_search >= stop:
        # No admissible window remains -- the caller's monotonic constraint has
        # run past the end of the recording.  Report it rather than returning a
        # silently meaningless window at an infinite distance.
        return start_search, min(start_search + window, len(trained.time_s) - 1), float("nan")

    best = (start_search, start_search + window, np.inf)
    for start in range(start_search, stop, step):
        end = start + window
        distance = dtw_distance(template, signature_matrix(trained, fs, start, end, target_fs), band=band)
        if distance < best[2]:
            best = (start, end, distance)
    return best


def pair_trunk_events_v1(
    novice_kin: Kinematics,
    trained_kin: Kinematics,
    novice_fs: float,
    trained_fs: float,
    num_events: int = 6,
) -> tuple[dict[str, list[tuple[int, int, int, int, float]]], list[float]]:
    """Detect v1 novice events and match each one in the trained recording.

    Returns the same ``{"Novice": [...], "Trained": [...]}`` layout as
    :func:`analysis.detection.pair_trunk_events` -- each entry
    ``(event_start, event_end, stab_start, stab_end, peak)`` -- plus the DTW
    distance of each match.  The search only moves forward through the trained
    recording, so one mismatch shifts every later event.
    """
    windows: dict[str, list[tuple[int, int, int, int, float]]] = {"Novice": [], "Trained": []}
    dtw_dists: list[float] = []
    novice_events, _ = detect_novice_trunk_rotation_events(novice_kin, novice_fs, num_events)

    # Window lengths are carried from one recording to the other in samples,
    # so v1 matches both at one rate (the two are identical in this dataset).
    novice_signature = precompute_signature_channels(novice_kin, novice_fs)
    trained_signature = precompute_signature_channels(trained_kin, novice_fs)

    last_trained_end = 0
    for novice_start, novice_end, novice_peak in novice_events:
        trained_start, trained_end, dtw_dist = find_trained_match(
            novice_signature, trained_signature, novice_fs, novice_start, novice_end, min_start=last_trained_end
        )
        last_trained_end = trained_end
        dtw_dists.append(dtw_dist)

        novice_stab_start, novice_stab_end = find_stabilization(novice_kin, novice_fs, novice_end)
        trained_stab_start, trained_stab_end = find_stabilization(trained_kin, trained_fs, trained_end)

        windows["Novice"].append((novice_start, novice_end, novice_stab_start, novice_stab_end, novice_peak))
        windows["Trained"].append((trained_start, trained_end, trained_stab_start, trained_stab_end, np.nan))

    return windows, dtw_dists
