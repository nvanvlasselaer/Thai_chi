"""Balance metrics for trunk-rotation and monopodal-stance events.

Each event has a movement window and a stabilization window that follows it.
The movement window is scored for trunk-pelvis coordination and smoothness; the
stabilization window for postural sway, orientation variability and corrective
activity.  The third family, sequence segments, is scored in
:mod:`analysis.smoothness_metrics`.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from analysis.data_io import ACC_COLUMNS, TrialData
from analysis.detection_v1 import find_stabilization as find_stabilization_v1
from analysis.kinematics import Kinematics
from analysis.orientation import mounting_matrix
from analysis.signals import cross_correlation_lag, highpass_detrend, lowpass_signal
from analysis.smoothness_metrics import dimensionless_jerk


# ---------------------------------------------------------------------------
# Stabilization-window measures
# ---------------------------------------------------------------------------


def corrective_threshold(omega: np.ndarray, fs: float) -> float:
    """Tukey outlier threshold on angular velocity: q75 + 1.5 IQR."""
    smoothed = lowpass_signal(omega, fs, cutoff_hz=6.0)
    q25, q75 = np.percentile(smoothed, [25, 75])
    return float(q75 + 1.5 * (q75 - q25))


def count_corrective_peaks(omega: np.ndarray, fs: float, threshold: float | None = None) -> tuple[int, float]:
    """Count angular-velocity bursts in a window.

    With ``threshold=None`` the bar is computed from the window's own
    distribution, which is self-referential: a window twice as busy gets a
    threshold more than twice as high, so the count reports shape rather than
    magnitude.  Passing a threshold derived from the whole recording
    (``corrective_threshold`` over the full signal) makes counts comparable
    between windows, trials and participants.  The default preserves the
    original behaviour.
    """
    smoothed = lowpass_signal(omega, fs, cutoff_hz=6.0)
    if threshold is None:
        threshold = np.percentile(smoothed, 75) + 1.5 * (np.percentile(smoothed, 75) - np.percentile(smoothed, 25))
    peaks, props = find_peaks(smoothed, height=threshold, distance=int(0.3 * fs))
    if len(peaks) == 0:
        return 0, float(np.max(smoothed))
    return int(len(peaks)), float(np.max(props["peak_heights"]))


def corrective_activity(kin: Kinematics, fs: float, stab: slice) -> dict[str, float]:
    """Corrective-activity measures that are comparable across windows.

    The raw peak count over a 2 s window is both duration-dependent and scored
    against a bar that moves with the window, which makes it the least reliable
    metric in the set (ICC 0.77 against +/-0.25 s boundary jitter, and it
    resolves the novice/trained difference at only 0.56x its own noise).  A rate
    measured against a recording-wide threshold reaches ICC 0.98 and 4.2x, and
    RMS angular velocity -- which needs no threshold at all -- reaches 0.98 and
    4.0x.
    """
    omega = kin.omega_mag["lumbar"]
    threshold = corrective_threshold(omega, fs)
    count, _ = count_corrective_peaks(omega[stab], fs, threshold=threshold)
    duration_s = max((stab.stop - stab.start) / fs, 1e-9)
    return {
        "corrective_peak_rate_hz": count / duration_s,
        "lumbar_rms_angular_velocity_dps": float(np.sqrt(np.mean(omega[stab] ** 2))),
    }


def lumbar_body_acceleration(kin: Kinematics, trial: TrialData) -> np.ndarray:
    """Lumbar acceleration (g) in the body frame: x = forward (AP), y = right (ML), z = down."""
    lumbar_acc = trial.data["lumbar"][ACC_COLUMNS].to_numpy()[: len(kin.t)]
    return lumbar_acc @ mounting_matrix("lumbar").T


# ---------------------------------------------------------------------------
# Trunk rotation
# ---------------------------------------------------------------------------


def compute_trunk_rotation_balance_metrics(label: str, kin: Kinematics, trial: TrialData, start: int, end: int, stab_start: int, stab_end: int, lag_pad_s: float | None = None) -> dict[str, float | str]:
    """Balance metrics for one trunk-rotation event.

    ``lag_pad_s`` widens *only* the window used for the trunk-pelvis
    cross-correlation, by that many seconds on each side.  The lag search spans
    +/-2 s (``cross_correlation_lag``), so an event shorter than about 4 s
    cannot support it and the estimate pins to the bound.  Velocity-segmented
    events are routinely 2-3 s, so they need the padding; the fixed 8 s windows
    did not, and the default of None reproduces the original behaviour exactly.
    The same decoupling is used by ``compute_knee_balance_metrics``, which
    scores its lag over a fixed window around peak flexion.
    """
    fs = trial.fs
    event = slice(start, end)
    stab = slice(stab_start, stab_end)

    if lag_pad_s is None:
        lag_event = event
    else:
        pad = int(round(lag_pad_s * fs))
        lag_event = slice(max(0, start - pad), min(len(kin.t), end + pad))

    # Postural sway: AP (x) and ML (y) acceleration variance while stabilizing.
    body_acc = lumbar_body_acceleration(kin, trial)
    ap_acc_var = float(np.var(body_acc[stab, 0]))
    ml_acc_var = float(np.var(body_acc[stab, 1]))

    # Detrend yaw over the full signal before slicing the event window so that
    # slow drift accumulated before the event does not corrupt the correlation.
    lumbar_z_detrended = highpass_detrend(kin.eulers_deg["lumbar"][:, 2], fs, cutoff_hz=0.05)
    chest_z_detrended = highpass_detrend(kin.eulers_deg["chestbone"][:, 2], fs, cutoff_hz=0.05)
    corr, lag = cross_correlation_lag(lumbar_z_detrended[lag_event], chest_z_detrended[lag_event], fs)
    dj = dimensionless_jerk(kin.eulers_deg["lumbar"][event, 0], fs)
    orientation = kin.eulers_deg["lumbar"][stab, :]
    orient_var = float(np.sqrt(np.mean(np.var(orientation, axis=0))))
    peak_count, peak_height = count_corrective_peaks(kin.omega_mag["lumbar"][stab], fs)
    return {
        "trial": label,
        "event_start_s": start / fs,
        "event_end_s": end / fs,
        "stabilization_start_s": stab_start / fs,
        "stabilization_end_s": stab_end / fs,
        "trunk_pelvis_peak_cross_correlation": corr,
        "trunk_pelvis_lag_s": lag,
        "weight_shift_dimensionless_jerk": dj,
        "weight_shift_log10_dimensionless_jerk": math.log10(dj) if dj > 0 else float("nan"),
        "lumbar_orientation_variability_deg": orient_var,
        "lumbar_ap_acc_variance_g2": ap_acc_var,
        "lumbar_ml_acc_variance_g2": ml_acc_var,
        "corrective_lumbar_angular_velocity_peak_count": peak_count,
        "largest_corrective_lumbar_angular_velocity_dps": peak_height,
        **corrective_activity(kin, fs, stab),
    }


# ---------------------------------------------------------------------------
# Monopodal stance
# ---------------------------------------------------------------------------


def summarize_knee_flexion_event(
    label: str,
    kin: Kinematics,
    fs: float,
    side: str,
    start: int,
    end: int,
    threshold_deg: float = 60.0,
) -> dict[str, float | str]:
    knee_signal = kin.left_knee_deg[:, 0] if side.lower().startswith("l") else kin.right_knee_deg[:, 0]
    knee_filtered = lowpass_signal(knee_signal, fs, cutoff_hz=6.0)
    knee_abs = np.abs(knee_filtered)

    event = slice(start, end)
    peak_rel = int(np.argmax(knee_abs[event]))
    peak_idx = start + peak_rel

    trunk_rel = np.column_stack(
        [
            lowpass_signal(kin.trunk_rel_euler_deg[:, i], fs, cutoff_hz=4.0)[event]
            for i in range(3)
        ]
    )
    lumbar = np.column_stack(
        [
            lowpass_signal(kin.eulers_deg["lumbar"][:, i], fs, cutoff_hz=4.0)[event]
            for i in range(3)
        ]
    )
    chest_omega = lowpass_signal(kin.omega_mag["chestbone"], fs, cutoff_hz=6.0)[event]
    lumbar_omega = lowpass_signal(kin.omega_mag["lumbar"], fs, cutoff_hz=6.0)[event]

    trunk_rel_mean = np.nanmean(trunk_rel, axis=0)
    lumbar_mean = np.nanmean(lumbar, axis=0)

    return {
        "trial": label,
        "flexed_leg": side,
        "stance_leg": "Right" if side.lower().startswith("l") else "Left",
        "threshold_deg": threshold_deg,
        "window_start_s": start / fs,
        "window_end_s": end / fs,
        "window_duration_s": (end - start) / fs,
        "peak_time_s": peak_idx / fs,
        "peak_signed_knee_flexion_deg": float(knee_filtered[peak_idx]),
        "peak_abs_knee_flexion_deg": float(knee_abs[peak_idx]),
        "knee_flexion_range_deg": float(np.ptp(knee_abs[event])),
        "trunk_rel_x_mean_deg": float(trunk_rel_mean[0]),
        "trunk_rel_y_mean_deg": float(trunk_rel_mean[1]),
        "trunk_rel_z_mean_deg": float(trunk_rel_mean[2]),
        "trunk_rel_x_range_deg": float(np.ptp(trunk_rel[:, 0])),
        "trunk_rel_y_range_deg": float(np.ptp(trunk_rel[:, 1])),
        "trunk_rel_z_range_deg": float(np.ptp(trunk_rel[:, 2])),
        "lumbar_x_mean_deg": float(lumbar_mean[0]),
        "lumbar_y_mean_deg": float(lumbar_mean[1]),
        "lumbar_z_mean_deg": float(lumbar_mean[2]),
        "lumbar_x_range_deg": float(np.ptp(lumbar[:, 0])),
        "lumbar_y_range_deg": float(np.ptp(lumbar[:, 1])),
        "lumbar_z_range_deg": float(np.ptp(lumbar[:, 2])),
        "peak_lumbar_angular_velocity_dps": float(np.max(lumbar_omega)),
        "peak_chest_angular_velocity_dps": float(np.max(chest_omega)),
        "peak_trunk_rel_x_deg": float(trunk_rel[peak_rel, 0]),
        "peak_trunk_rel_y_deg": float(trunk_rel[peak_rel, 1]),
        "peak_trunk_rel_z_deg": float(trunk_rel[peak_rel, 2]),
    }


def compute_knee_balance_metrics(
    label: str,
    kin: Kinematics,
    trial: TrialData,
    side: str,
    start: int,
    end: int,
    lag_pre_s: float = 1.0,
    lag_post_s: float = 2.0,
    stab_start: int | None = None,
    stab_end: int | None = None,
) -> dict[str, float | str]:
    fs = trial.fs
    knee_signal = kin.left_knee_deg[:, 0] if side.lower().startswith("l") else kin.right_knee_deg[:, 0]
    knee_abs = np.abs(lowpass_signal(knee_signal, fs, cutoff_hz=6.0))
    peak_idx = start + int(np.argmax(knee_abs[start:end]))

    # Cross-correlation over a fixed window centred on peak knee flexion so
    # that the window length is consistent across events and participants,
    # and directly targets the coordination response at maximum challenge.
    lag_start = max(0, peak_idx - int(lag_pre_s * fs))
    lag_end = min(len(knee_abs), peak_idx + int(lag_post_s * fs))
    # A caller that has curated the stabilization window (the event editor)
    # passes it in; otherwise fall back to the v1 search, as before.
    if stab_start is None or stab_end is None:
        stab_start, stab_end = find_stabilization_v1(kin, fs, peak_idx)
    corr, lag = cross_correlation_lag(
        highpass_detrend(kin.eulers_deg["lumbar"][:, 2], fs)[lag_start:lag_end],
        highpass_detrend(kin.eulers_deg["chestbone"][:, 2], fs)[lag_start:lag_end],
        fs,
    )

    corr_pitch, lag_pitch = cross_correlation_lag(
        highpass_detrend(kin.eulers_deg["lumbar"][:, 1], fs)[lag_start:lag_end],
        highpass_detrend(kin.eulers_deg["chestbone"][:, 1], fs)[lag_start:lag_end],
        fs,
    )

    orientation = kin.eulers_deg["lumbar"][stab_start:stab_end,:]
    orient_var = float(np.sqrt(np.mean(np.var(orientation, axis=0))))

    body_acc = lumbar_body_acceleration(kin, trial)
    ap_acc_var = float(np.var(body_acc[stab_start:stab_end, 0]))
    ml_acc_var = float(np.var(body_acc[stab_start:stab_end, 1]))
    peak_count, peak_height = count_corrective_peaks(
        kin.omega_mag["lumbar"][stab_start:stab_end], fs
    )

    trunk_y = lowpass_signal(kin.eulers_deg["lumbar"][:,1], fs, cutoff_hz=4.0)

    return {
        "trial": label,
        "flexed_leg": side,
        "stance_leg": "Right" if side.lower().startswith("l") else "Left",
        "peak_knee_flexion_deg": float(knee_abs[peak_idx]),
        "peak_trunk_y_deg": float(trunk_y[peak_idx]),
        "time_to_stabilization_s": float((stab_start - peak_idx) / fs),
        "lumbar_orientation_variability_deg": orient_var,
        "lumbar_ap_acc_variance_g2": ap_acc_var,
        "lumbar_ml_acc_variance_g2": ml_acc_var,
        "corrective_peak_count": peak_count,
        "largest_corrective_peak_dps": peak_height,
        "trunk_pelvis_lag_s": lag,
        "trunk_pelvis_pitch_lag_s": lag_pitch,
        **corrective_activity(kin, fs, slice(stab_start, stab_end)),
    }


def compute_asymmetry_metrics(knee_metrics: pd.DataFrame) -> pd.DataFrame:
    """Absolute left-vs-right differences per trial, averaged over events.

    Returns an empty frame when a trial does not have both stance legs
    represented, which is the same behaviour as the inline version this was
    extracted from.
    """
    if len(knee_metrics) == 0:
        return pd.DataFrame()

    agg = knee_metrics.groupby(["trial", "stance_leg"]).mean(numeric_only=True).reset_index()
    asym_rows = []
    for tr in ["Novice", "Trained"]:
        tr_data = agg[agg.trial == tr]
        if len(tr_data) == 2:
            left = tr_data[tr_data.stance_leg == "Left"].iloc[0]
            right = tr_data[tr_data.stance_leg == "Right"].iloc[0]
            asym_rows.append({
                "trial": tr,
                "peak_knee_flexion_diff_deg": abs(left["peak_knee_flexion_deg"] - right["peak_knee_flexion_deg"]),
                "time_to_stabilization_diff_s": abs(left["time_to_stabilization_s"] - right["time_to_stabilization_s"]),
                "lumbar_ml_acc_variance_diff_g2": abs(left["lumbar_ml_acc_variance_g2"] - right["lumbar_ml_acc_variance_g2"]),
            })
    return pd.DataFrame(asym_rows)
