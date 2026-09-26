"""Balance metrics for trunk-rotation and single-leg-stance events.

Each event has a movement window and a stabilization window that follows it.
The movement window is scored for trunk-pelvis coordination and smoothness --
for a single-leg stance it is the support phase itself, lift-off to
touch-down, and its sway is scored too -- and the stabilization window for
postural sway, tilt and angular activity.  The third family, the turns of the
sequence, is scored in :mod:`analysis.smoothness_metrics`.

Sway is the lumbar acceleration with gravity removed, in the pelvis-heading
frame (:func:`analysis.gravity.linear_acceleration`), reported as RMS about the
window mean in m/s^2 -- x mediolateral, y anteroposterior.  Tilt is read from
the vertical in the lumbar frame (:func:`analysis.gravity.tilt_deg`), not from
Euler angles, so it contains no yaw.  Angular speeds have the gyroscope bias
removed.  Each of those three was different before (Review.md): the
acceleration variance was taken in sensor axes, where tilt-projected gravity
was 2-30 times the sway, with its AP and ML labels swapped; the orientation
"variability" was dominated by yaw; and the raw gyroscope magnitude carried
6.6 deg/s of lumbar bias, enough to reverse which participant looked busier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from analysis import detection
from analysis.data_io import TrialData
from analysis.gravity import leg_lift_index, linear_acceleration, tilt_deg
from analysis.kinematics import Kinematics
from analysis.signals import lowpass_signal, pearson_lag
from analysis.smoothness_metrics import sparc, turn_signals, yaw_lag


# ---------------------------------------------------------------------------
# Per-recording signals, computed once and shared by every event
# ---------------------------------------------------------------------------


def balance_signals(
    kin: Kinematics, trial: TrialData, fs: float, stab_params: detection.StabilizationParams | None = None
) -> dict[str, object]:
    """Everything the balance metrics slice, for one recording."""
    lumbar_sagittal, lumbar_frontal = tilt_deg(kin, "lumbar")
    _, chest_frontal = tilt_deg(kin, "chestbone")
    signals: dict[str, object] = dict(turn_signals(kin, trial, fs))
    signals.update({
        "sway": linear_acceleration(kin, trial, "lumbar"),
        "lumbar_sagittal_tilt": lowpass_signal(lumbar_sagittal, fs, cutoff_hz=4.0),
        "lumbar_frontal_tilt": lowpass_signal(lumbar_frontal, fs, cutoff_hz=4.0),
        "chest_frontal_tilt": lowpass_signal(chest_frontal, fs, cutoff_hz=4.0),
        "lumbar_omega": kin.omega_mag["lumbar"],
        "lift": lowpass_signal(leg_lift_index(kin), fs, cutoff_hz=detection.LIFT_CUTOFF_HZ),
        "stability": detection.combined_omega(kin, fs, stab_params),
        "corrective_threshold": corrective_threshold(kin.omega_mag["lumbar"], fs),
    })
    return signals


# ---------------------------------------------------------------------------
# Window measures
# ---------------------------------------------------------------------------


def corrective_threshold(omega: np.ndarray, fs: float) -> float:
    """Tukey outlier threshold on the recording's angular speed: q75 + 1.5 IQR."""
    smoothed = lowpass_signal(omega, fs, cutoff_hz=6.0)
    q25, q75 = np.percentile(smoothed, [25, 75])
    return float(q75 + 1.5 * (q75 - q25))


def count_corrective_peaks(omega: np.ndarray, fs: float, threshold: float) -> int:
    """Angular-speed bursts above a recording-wide threshold within a window."""
    smoothed = lowpass_signal(omega, fs, cutoff_hz=6.0)
    peaks, _ = find_peaks(smoothed, height=threshold, distance=max(1, int(0.3 * fs)))
    return int(len(peaks))


def _rms_about_mean(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean((values - np.mean(values)) ** 2))) if len(values) else float("nan")


def sway_metrics(signals: dict[str, object], fs: float, window: slice, prefix: str) -> dict[str, float]:
    """Sway, tilt and angular activity of the lumbar segment over one window.

    Acceleration RMS is taken about the window mean: a constant residue of the
    orientation estimate is not sway.  ``corrective_peak_rate_hz`` is kept as a
    secondary measure -- it was zero in 15 of 20 windows -- and
    ``rms_angular_velocity_dps``, which needs no threshold, is the primary one.
    """
    sway = signals["sway"][window]
    omega = signals["lumbar_omega"][window]
    duration_s = max((window.stop - window.start) / fs, 1e-9)
    return {
        f"{prefix}ml_acc_rms_mps2": _rms_about_mean(sway[:, 0]),
        f"{prefix}ap_acc_rms_mps2": _rms_about_mean(sway[:, 1]),
        f"{prefix}frontal_tilt_sd_deg": float(np.std(signals["lumbar_frontal_tilt"][window])),
        f"{prefix}sagittal_tilt_sd_deg": float(np.std(signals["lumbar_sagittal_tilt"][window])),
        f"{prefix}rms_angular_velocity_dps": float(np.sqrt(np.mean(omega ** 2))) if len(omega) else float("nan"),
        f"{prefix}corrective_peak_rate_hz": (
            count_corrective_peaks(omega, fs, signals["corrective_threshold"]) / duration_s
        ),
    }


def _quiet_z(signals: dict[str, object], start: int, end: int) -> float:
    return detection.window_quietness(signals["stability"], start, end)[1]


# ---------------------------------------------------------------------------
# Trunk rotation
# ---------------------------------------------------------------------------


def compute_trunk_rotation_balance_metrics(
    label: str, fs: float, event_id: str, start: int, end: int, stab_start: int, stab_end: int,
    signals: dict[str, object], lag_pad_s: float | None = None,
) -> dict[str, float | str]:
    """Balance metrics for one trunk-rotation event.

    ``lag_pad_s`` widens *only* the window of the trunk-pelvis lag, by that many
    seconds on each side: velocity-segmented events are routinely 2-3 s, too
    short to support the +/-1 s search on their own.

    Smoothness is SPARC of the trunk-on-pelvis turning speed over the event.
    It replaces the dimensionless jerk of the lumbar Euler x angle -- which was
    pelvic pitch, not the lateral weight shift it was named for, and dominated
    by noise.
    """
    n = len(signals["chest_rate"])
    if lag_pad_s is None:
        lag_window = slice(start, end)
    else:
        pad = int(round(lag_pad_s * fs))
        lag_window = slice(max(0, start - pad), min(n, end + pad))
    r, lag = yaw_lag(signals, fs, lag_window)
    relative_rate = signals["chest_rate"][start:end] - signals["pelvis_rate"][start:end]
    relative_angle = np.cumsum(relative_rate) / fs

    return {
        "trial": label,
        "event_id": event_id,
        "event_start_s": start / fs,
        "event_end_s": end / fs,
        "event_duration_s": (end - start) / fs,
        "stabilization_start_s": stab_start / fs,
        "stabilization_end_s": stab_end / fs,
        "trunk_pelvis_yaw_r": r,
        "trunk_pelvis_lag_s": lag,
        "trunk_yaw_sparc": sparc(relative_rate, fs),
        "trunk_yaw_excursion_deg": float(np.ptp(relative_angle)) if len(relative_angle) else float("nan"),
        "trunk_yaw_peak_rate_dps": float(np.max(np.abs(relative_rate))) if len(relative_rate) else float("nan"),
        **sway_metrics(signals, fs, slice(stab_start, stab_end), "lumbar_"),
        "stabilization_quiet_z": _quiet_z(signals, stab_start, stab_end),
    }


# ---------------------------------------------------------------------------
# Single-leg stance
# ---------------------------------------------------------------------------


def _leg_flexion(kin: Kinematics, joint: str, leg: str, fs: float) -> np.ndarray:
    angles = getattr(kin, f"{leg.lower()}_{joint}_deg")[:, 0]
    return np.abs(lowpass_signal(angles, fs, cutoff_hz=6.0))


def summarize_single_leg_event(
    label: str, kin: Kinematics, fs: float, event_id: str, lifted_leg: str, start: int, end: int,
    lift: np.ndarray,
) -> dict[str, float | str]:
    """Where and how high the foot went, and what the knee did: the event-window row."""
    peak = detection.peak_lift_index(lift, start, end, lifted_leg)
    knee = _leg_flexion(kin, "knee", lifted_leg, fs)[start:end]
    return {
        "trial": label,
        "event_id": event_id,
        "lifted_leg": lifted_leg,
        "stance_leg": "Right" if lifted_leg == "Left" else "Left",
        "window_start_s": start / fs,
        "window_end_s": end / fs,
        "peak_time_s": peak / fs,
        "peak_lift_index": float(abs(lift[peak])),
        "peak_knee_flexion_deg": float(np.max(knee)) if len(knee) else float("nan"),
    }


def compute_single_leg_metrics(
    label: str, kin: Kinematics, fs: float, event_id: str, lifted_leg: str,
    start: int, end: int, stab_start: int, stab_end: int, signals: dict[str, object],
    lag_window_s: tuple[float, float] = (2.0, 3.0),
) -> dict[str, float | str]:
    """Balance metrics for one single-leg stance.

    The event window is the support phase, lift-off to touch-down, and is
    where the base of support is one foot: its sway, tilt and angular activity
    are the single-leg balance measures (``support_*``).  The stabilization
    window after touch-down gives the same measures for settling (``settle_*``),
    and ``time_to_stabilization_s`` is counted from touch-down.

    The knee is described rather than thresholded: ``knee_extension_while_lifted_deg``
    is how far it straightened after its deepest flexion while the foot was
    still at least half as high as it got -- large for a kick, small for a knee
    lift, whose knee only straightens as the foot comes down.
    The two lags are reported with their correlation, over ``lag_window_s``
    around the highest lift.  Neither is reliable on these windows -- the
    trunk barely turns during a stance, and before the lag estimator was fixed
    they picked anti-phase peaks -- so they are kept out of the figures.
    """
    lift = signals["lift"]
    peak = detection.peak_lift_index(lift, start, end, lifted_leg)
    knee = _leg_flexion(kin, "knee", lifted_leg, fs)[start:end]
    hip = _leg_flexion(kin, "hip", lifted_leg, fs)[start:end]
    deepest = int(np.argmax(knee)) if len(knee) else 0
    still_up = (np.abs(lift[start:end]) >= 0.5 * abs(lift[peak])) & (np.arange(len(knee)) >= deepest)

    n = len(lift)
    lag_window = slice(max(0, peak - int(round(lag_window_s[0] * fs))),
                       min(n, peak + int(round(lag_window_s[1] * fs))))
    yaw_r, yaw_lag_s = yaw_lag(signals, fs, lag_window)
    frontal_r, frontal_lag = pearson_lag(signals["lumbar_frontal_tilt"][lag_window],
                                         signals["chest_frontal_tilt"][lag_window], fs)

    return {
        "trial": label,
        "event_id": event_id,
        "lifted_leg": lifted_leg,
        "stance_leg": "Right" if lifted_leg == "Left" else "Left",
        "support_duration_s": (end - start) / fs,
        "peak_lift_index": float(abs(lift[peak])),
        "peak_knee_flexion_deg": float(knee[deepest]) if len(knee) else float("nan"),
        "knee_extension_while_lifted_deg": (
            float(knee[deepest] - np.min(knee[still_up])) if still_up.any() else 0.0
        ),
        "peak_hip_flexion_deg": float(np.max(hip)) if len(hip) else float("nan"),
        "lumbar_frontal_tilt_at_peak_deg": float(signals["lumbar_frontal_tilt"][peak]),
        **sway_metrics(signals, fs, slice(start, end), "support_"),
        **sway_metrics(signals, fs, slice(stab_start, stab_end), "settle_"),
        "time_to_stabilization_s": (stab_start - end) / fs,
        "stabilization_quiet_z": _quiet_z(signals, stab_start, stab_end),
        "trunk_pelvis_yaw_r": yaw_r,
        "trunk_pelvis_lag_s": yaw_lag_s,
        "trunk_pelvis_frontal_r": frontal_r,
        "trunk_pelvis_frontal_lag_s": frontal_lag,
    }


ASYMMETRY_METRICS = (
    "support_duration_s",
    "peak_lift_index",
    "peak_knee_flexion_deg",
    "support_ml_acc_rms_mps2",
    "support_frontal_tilt_sd_deg",
    "support_rms_angular_velocity_dps",
    "time_to_stabilization_s",
)


def mirrored_pairs(events: list[dict]) -> list[tuple[dict, dict]]:
    """Left and right versions of the same movement, as ``(left, right)``.

    Events carrying the same ``movement`` label pair with each other; the rest
    pair the k-th left lift with the k-th right lift in time order -- on these
    recordings a knee lift on each side, then a lift-and-kick on each side.
    """
    pairs: list[tuple[dict, dict]] = []
    labelled = [e for e in events if e.get("movement")]
    for name in dict.fromkeys(e["movement"] for e in labelled):
        sides = {e["lifted_leg"]: e for e in labelled if e["movement"] == name}
        if "Left" in sides and "Right" in sides:
            pairs.append((sides["Left"], sides["Right"]))
    rest = [e for e in events if not e.get("movement")]
    lefts = [e for e in rest if e["lifted_leg"] == "Left"]
    rights = [e for e in rest if e["lifted_leg"] == "Right"]
    pairs += list(zip(lefts, rights))
    return pairs


def compute_asymmetry_metrics(single_leg: pd.DataFrame, events: list[dict]) -> pd.DataFrame:
    """Left-minus-right differences between mirrored stances, per recording.

    One row per mirrored pair with the signed difference of each metric, and a
    ``mean |L-R|`` row per recording.  Pooling every left and every right event
    instead -- as before -- compared a knee lift with a kick, and put the right
    kick on the left side because it had been labelled with the wrong leg.
    """
    if len(single_leg) == 0:
        return pd.DataFrame()
    rows = []
    for label, group in single_leg.groupby("trial", sort=False):
        by_id = {row["event_id"]: row for _, row in group.iterrows()}
        present = [e for e in events if e["event_id"] in by_id]
        pair_rows = []
        for index, (left, right) in enumerate(mirrored_pairs(present), 1):
            a, b = by_id[left["event_id"]], by_id[right["event_id"]]
            row = {"trial": label, "pair": str(index), "left_event_id": left["event_id"],
                   "right_event_id": right["event_id"],
                   "movement": left.get("movement") or right.get("movement") or ""}
            row.update({f"{m}_l_minus_r": float(a[m] - b[m]) for m in ASYMMETRY_METRICS})
            pair_rows.append(row)
        if pair_rows:
            summary = {"trial": label, "pair": "mean |L-R|", "left_event_id": "", "right_event_id": "",
                       "movement": ""}
            summary.update({f"{m}_l_minus_r": float(np.mean([abs(r[f"{m}_l_minus_r"]) for r in pair_rows]))
                            for m in ASYMMETRY_METRICS})
            rows += pair_rows + [summary]
    return pd.DataFrame(rows)
