#!/usr/bin/env python3
"""Kinematic comparison of novice and trained Tai Chi IMU recordings.

The script parses Delsys Trigno Discover CSV exports, estimates segment
orientations with a 6-axis Madgwick filter, derives balance-relevant trunk and
lower-limb kinematics, aligns a selected novice movement module to the trained
recording, and writes the required figure/tables/report.
"""

from __future__ import annotations

import csv
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, correlate, correlation_lags, find_peaks, sosfiltfilt
from scipy.spatial.transform import Rotation
from scipy.stats import zscore


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
NOVICE_CSV = DATA_DIR / "IMU_Trial_1_RC_Novice.csv"
TRAINED_CSV = DATA_DIR / "IMU_Trial_3_RC_Trained.csv"
FS = 370.3704

SENSOR_MAP = {
    "chestbone": "Thorax",
    "lumbar": "Pelvis/Lower Trunk Proxy",
    "lthigh": "Left Thigh",
    "rthigh": "Right Thigh",
    "ltibia": "Left Shank",
    "rtibia": "Right Shank",
    "lfoot": "Left Foot",
    "rfoot": "Right Foot",
    "lhumerus": "Left Upper Arm",
    "rhumerus": "Right Upper Arm",
    "lulna": "Left Forearm",
    "rulna": "Right Forearm",
    "lhand": "Left Hand",
    "rhand": "Right Hand",
}


@dataclass
class TrialData:
    label: str
    path: Path
    duration_meta_s: float
    fs: float
    sensors: list[str]
    data: dict[str, pd.DataFrame]


@dataclass
class Kinematics:
    t: np.ndarray
    quaternions: dict[str, np.ndarray]
    eulers_deg: dict[str, np.ndarray]
    omega_mag: dict[str, np.ndarray]
    trunk_rel_euler_deg: np.ndarray
    left_knee_deg: np.ndarray
    right_knee_deg: np.ndarray
    left_ankle_deg: np.ndarray
    right_ankle_deg: np.ndarray


def canonical_sensor(name: str) -> str:
    stem = re.sub(r"\s*\(\d+\)\s*$", "", name.strip())
    return stem.lower().replace("_", "")


def parse_trigno_csv(path: Path, label: str) -> TrialData:
    with path.open(newline="") as f:
        rows = [next(csv.reader(f)) for _ in range(9)]

    duration_meta_s = float(rows[2][1].strip())
    sensor_row = rows[3]
    channel_row = rows[5]
    fs_row = rows[6]

    sensor_starts = [i for i, value in enumerate(sensor_row) if value.strip()]
    sensors = [canonical_sensor(sensor_row[i]) for i in sensor_starts]
    n_sensor_cols = len(sensors) * 12

    block_fs = []
    for start in sensor_starts:
        hz_text = fs_row[start + 1].strip()
        block_fs.append(float(hz_text.replace("Hz", "").strip()))
    fs = float(np.median(block_fs))

    raw = pd.read_csv(path, skiprows=8, header=None, usecols=range(n_sensor_cols), low_memory=False)
    raw = raw.apply(pd.to_numeric, errors="coerce")

    data: dict[str, pd.DataFrame] = {}
    for block_index, sensor in enumerate(sensors):
        start = block_index * 12
        cols = raw.iloc[:, start : start + 12].copy()
        names = channel_row[start : start + 12]
        channel_map = {}
        for idx, name in enumerate(names):
            cleaned = name.strip().lower()
            if "acc x time" in cleaned:
                channel_map["time"] = cols.iloc[:, idx]
            elif cleaned == "acc x (g)":
                channel_map["acc_x_g"] = cols.iloc[:, idx]
            elif cleaned == "acc y (g)":
                channel_map["acc_y_g"] = cols.iloc[:, idx]
            elif cleaned == "acc z (g)":
                channel_map["acc_z_g"] = cols.iloc[:, idx]
            elif cleaned == "gyro x (deg/s)":
                channel_map["gyro_x_dps"] = cols.iloc[:, idx]
            elif cleaned == "gyro y (deg/s)":
                channel_map["gyro_y_dps"] = cols.iloc[:, idx]
            elif cleaned == "gyro z (deg/s)":
                channel_map["gyro_z_dps"] = cols.iloc[:, idx]

        sensor_df = pd.DataFrame(channel_map).dropna(how="all")
        sensor_df = sensor_df.dropna(subset=["time"])
        data[sensor] = sensor_df.reset_index(drop=True)

    return TrialData(label=label, path=path, duration_meta_s=duration_meta_s, fs=fs, sensors=sensors, data=data)


def make_sensor_inventory(trials: list[TrialData]) -> pd.DataFrame:
    rows = []
    required = ["acc_x_g", "acc_y_g", "acc_z_g", "gyro_x_dps", "gyro_y_dps", "gyro_z_dps"]
    for trial in trials:
        for sensor in sorted(trial.sensors):
            df = trial.data[sensor]
            time = df["time"].to_numpy()
            diffs = np.diff(time)
            rows.append(
                {
                    "trial": trial.label,
                    "sensor": sensor,
                    "body_segment": SENSOR_MAP.get(sensor, "Unmapped"),
                    "samples": len(df),
                    "duration_s": round(float(time[-1] - time[0]), 3),
                    "fs_hz": round(trial.fs, 4),
                    "median_dt_s": round(float(np.nanmedian(diffs)), 6),
                    "max_dt_s": round(float(np.nanmax(diffs)), 6),
                    "missing_channel_values": int(df[required].isna().sum().sum()),
                    "complete_accel_gyro": all(col in df.columns for col in required),
                }
            )
    return pd.DataFrame(rows)


def initial_quaternion_from_acc(acc: np.ndarray) -> np.ndarray:
    a = acc[np.all(np.isfinite(acc), axis=1)]
    if len(a) == 0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    ax, ay, az = np.median(a[: min(len(a), int(FS))], axis=0)
    norm = math.sqrt(ax * ax + ay * ay + az * az)
    if norm == 0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    ax, ay, az = ax / norm, ay / norm, az / norm
    roll = math.atan2(ay, az)
    pitch = math.atan2(-ax, math.sqrt(ay * ay + az * az))
    quat_xyzw = Rotation.from_euler("xyz", [roll, pitch, 0.0]).as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


def madgwick_imu(acc_g: np.ndarray, gyro_dps: np.ndarray, fs: float, beta: float = 0.05) -> np.ndarray:
    gyro = np.deg2rad(gyro_dps.copy())
    gyro_bias = np.nanmedian(gyro[: max(1, int(fs))], axis=0)
    gyro = gyro - gyro_bias

    q = initial_quaternion_from_acc(acc_g)
    q = q / np.linalg.norm(q)
    quats = np.zeros((len(acc_g), 4), dtype=float)
    dt = 1.0 / fs

    for i, (acc, gyr) in enumerate(zip(acc_g, gyro)):
        q1, q2, q3, q4 = q
        gx, gy, gz = gyr

        q_dot = 0.5 * np.array(
            [
                -q2 * gx - q3 * gy - q4 * gz,
                q1 * gx + q3 * gz - q4 * gy,
                q1 * gy - q2 * gz + q4 * gx,
                q1 * gz + q2 * gy - q3 * gx,
            ]
        )

        if np.all(np.isfinite(acc)):
            ax, ay, az = acc
            norm = math.sqrt(ax * ax + ay * ay + az * az)
            if norm > 1e-12:
                ax, ay, az = ax / norm, ay / norm, az / norm

                s1 = 4 * q1 * q3 * q3 + 2 * q3 * ax + 4 * q1 * q2 * q2 - 2 * q2 * ay
                s2 = (
                    4 * q2 * q4 * q4
                    - 2 * q4 * ax
                    + 4 * q1 * q1 * q2
                    - 2 * q1 * ay
                    - 4 * q2
                    + 8 * q2 * q2 * q2
                    + 8 * q2 * q3 * q3
                    + 4 * q2 * az
                )
                s3 = (
                    4 * q1 * q1 * q3
                    + 2 * q1 * ax
                    + 4 * q3 * q4 * q4
                    - 2 * q4 * ay
                    - 4 * q3
                    + 8 * q3 * q2 * q2
                    + 8 * q3 * q3 * q3
                    + 4 * q3 * az
                )
                s4 = 4 * q2 * q2 * q4 - 2 * q2 * ax + 4 * q3 * q3 * q4 - 2 * q3 * ay
                step = np.array([s1, s2, s3, s4])
                step_norm = np.linalg.norm(step)
                if step_norm > 1e-12:
                    q_dot -= beta * step / step_norm

        q = q + q_dot * dt
        q = q / np.linalg.norm(q)
        if i > 0 and np.dot(q, quats[i - 1]) < 0:
            q = -q
        quats[i] = q

    return quats


def quat_inverse(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] *= -1
    return out


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def quat_to_euler_deg(q_wxyz: np.ndarray) -> np.ndarray:
    q_xyzw = np.column_stack([q_wxyz[:, 1], q_wxyz[:, 2], q_wxyz[:, 3], q_wxyz[:, 0]])
    e = Rotation.from_quat(q_xyzw).as_euler("xyz", degrees=True)
    return np.rad2deg(np.unwrap(np.deg2rad(e), axis=0))


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


def find_neutral_pose_window(trial: TrialData, min_duration_s: float = 2.0) -> tuple[int, int]:
    lumbar_acc_x = trial.data["lumbar"]["acc_x_g"].to_numpy()
    lumbar_gyro = trial.data["lumbar"][["gyro_x_dps", "gyro_y_dps", "gyro_z_dps"]].to_numpy()
    lumbar_omega = np.linalg.norm(lumbar_gyro, axis=1)

    search_window = int(10 * trial.fs)
    search_window = min(search_window, len(lumbar_acc_x))
    peak_idx = int(np.nanargmax(np.abs(lumbar_acc_x[:search_window])))

    settle_time = int(0.5 * trial.fs)
    start_search = peak_idx + settle_time
    window_size = int(trial.fs * min_duration_s)

    if start_search + window_size >= len(lumbar_omega):
        return 0, len(lumbar_omega)

    search_end = min(len(lumbar_omega), start_search + int(15 * trial.fs))
    omega_search = lumbar_omega[start_search:search_end]

    rolling_mean_omega = pd.Series(omega_search).rolling(window_size).mean().to_numpy()
    best_end_rel = int(np.nanargmin(rolling_mean_omega))

    best_end = start_search + best_end_rel
    best_start = best_end - window_size + 1

    return best_start, best_end


def mean_quaternion(quats: np.ndarray) -> np.ndarray:
    q = np.mean(quats, axis=0)
    return q / np.linalg.norm(q)


def get_nominal_sensor_quaternion(sensor_name: str) -> np.ndarray:
    if sensor_name in ["chestbone", "lulna", "rulna"]:
        R = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    elif sensor_name == "lumbar":
        R = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])
    elif sensor_name in ["lhumerus", "lhand", "lthigh", "ltibia"]:
        R = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])
    elif sensor_name in ["rhumerus", "rhand", "rthigh", "rtibia"]:
        R = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    elif sensor_name in ["lfoot", "rfoot"]:
        R = np.eye(3)
    else:
        R = np.eye(3)
    q = Rotation.from_matrix(R).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def compute_kinematics(trial: TrialData) -> Kinematics:
    n = min(len(df) for df in trial.data.values())
    t = trial.data[trial.sensors[0]]["time"].to_numpy()[:n]
    quats = {}
    eulers = {}
    omega_mag = {}

    neutral_start, neutral_end = find_neutral_pose_window(trial)

    for sensor in trial.sensors:
        df = trial.data[sensor].iloc[:n]
        acc = df[["acc_x_g", "acc_y_g", "acc_z_g"]].to_numpy()
        gyro = df[["gyro_x_dps", "gyro_y_dps", "gyro_z_dps"]].to_numpy()
        
        raw_quats = madgwick_imu(acc, gyro, trial.fs)
        q_offset = mean_quaternion(raw_quats[neutral_start:neutral_end])
        q_nom = get_nominal_sensor_quaternion(sensor)
        
        q_O_inv = quat_inverse(q_offset)
        q_nom_inv = quat_inverse(q_nom)
        
        step1 = quat_multiply(q_nom, q_O_inv)
        step2 = quat_multiply(step1, raw_quats)
        quats[sensor] = quat_multiply(step2, q_nom_inv)
        
        eulers[sensor] = quat_to_euler_deg(quats[sensor])
        omega_mag[sensor] = np.linalg.norm(gyro, axis=1)

    trunk_rel_q = quat_multiply(quat_inverse(quats["lumbar"]), quats["chestbone"])
    trunk_rel_euler = quat_to_euler_deg(trunk_rel_q)

    left_knee = quat_to_euler_deg(quat_multiply(quat_inverse(quats["lthigh"]), quats["ltibia"]))[:, 0]
    right_knee = quat_to_euler_deg(quat_multiply(quat_inverse(quats["rthigh"]), quats["rtibia"]))[:, 0]
    left_ankle = quat_to_euler_deg(quat_multiply(quat_inverse(quats["ltibia"]), quats["lfoot"]))[:, 0]
    right_ankle = quat_to_euler_deg(quat_multiply(quat_inverse(quats["rtibia"]), quats["rfoot"]))[:, 0]

    return Kinematics(
        t=t,
        quaternions=quats,
        eulers_deg=eulers,
        omega_mag=omega_mag,
        trunk_rel_euler_deg=trunk_rel_euler,
        left_knee_deg=left_knee,
        right_knee_deg=right_knee,
        left_ankle_deg=left_ankle,
        right_ankle_deg=right_ankle,
    )


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


def select_novice_events(kin: Kinematics, fs: float, num_events: int = 3) -> tuple[list[tuple[int, int, int]], str]:
    yaw = lowpass_signal(kin.trunk_rel_euler_deg[:, 2], fs, cutoff_hz=4.0)
    chest_omega = lowpass_signal(kin.omega_mag["chestbone"], fs, cutoff_hz=6.0)
    ignore = int(15 * fs)
    window = int(8 * fs)
    step = int(1 * fs)
    
    scores = []
    for start in range(ignore, len(yaw) - window - ignore, step):
        end = start + window
        yaw_range = np.ptp(yaw[start:end])
        omega_peak = np.percentile(chest_omega[start:end], 95)
        score = yaw_range * np.log1p(omega_peak)
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
            peak = start + int(np.argmax(np.abs(yaw[start:end] - np.median(yaw[start:end]))))
            selected_events.append((start, end, peak))
        if len(selected_events) == num_events:
            break
            
    selected_events.sort(key=lambda x: x[0])
    rationale = f"Selected {num_events} 8 s trunk-rotation transitions based on combined trunk-pelvis relative-yaw excursion and chest angular-velocity peaks."
    return selected_events, rationale


def signature_matrix(kin: Kinematics, fs: float, start: int, end: int, target_fs: float = 20.0) -> np.ndarray:
    sig = np.column_stack(
        [
            resample_filtered_window(kin.t, kin.trunk_rel_euler_deg[:, 2], fs, start, end, target_fs),
            resample_filtered_window(kin.t, kin.eulers_deg["lumbar"][:, 0], fs, start, end, target_fs),
            resample_filtered_window(kin.t, kin.omega_mag["chestbone"], fs, start, end, target_fs),
            resample_filtered_window(kin.t, kin.left_knee_deg, fs, start, end, target_fs),
            resample_filtered_window(kin.t, kin.right_knee_deg, fs, start, end, target_fs),
            resample_filtered_window(kin.t, kin.omega_mag["lhand"], fs, start, end, target_fs),
            resample_filtered_window(kin.t, kin.omega_mag["rhand"], fs, start, end, target_fs),
        ]
    )
    sig = zscore(sig, axis=0, nan_policy="omit")
    return np.nan_to_num(sig)


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


def find_trained_match(novice: Kinematics, trained: Kinematics, fs: float, novice_start: int, novice_end: int, min_start: int = 0) -> tuple[int, int, float]:
    target_fs = 20.0
    template = signature_matrix(novice, fs, novice_start, novice_end, target_fs=target_fs)
    window = novice_end - novice_start
    search_step = int(1.0 * fs)
    ignore = int(15 * fs)
    start_search = max(ignore, min_start)
    best = (start_search, start_search + window, np.inf)
    for start in range(start_search, len(trained.t) - window - ignore, search_step):
        end = start + window
        candidate = signature_matrix(trained, fs, start, end, target_fs=target_fs)
        dist = dtw_distance(template, candidate, band=int(1.5 * target_fs))
        if dist < best[2]:
            best = (start, end, dist)
    return best


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


def dimensionless_jerk(angle_deg: np.ndarray, fs: float) -> float:
    angle = np.deg2rad(lowpass_signal(angle_deg, fs, cutoff_hz=6.0, order=4))
    dt = 1.0 / fs
    vel = np.gradient(angle, dt)
    acc = np.gradient(vel, dt)
    jerk = np.gradient(acc, dt)
    duration = len(angle) / fs
    amplitude = np.ptp(angle)
    if amplitude < 1e-9:
        return float("nan")
    return float((duration**5 / amplitude**2) * np.trapezoid(jerk**2, dx=dt))


def find_stabilization(kin: Kinematics, fs: float, after_end: int) -> tuple[int, int]:
    lumbar = lowpass_signal(kin.omega_mag["lumbar"], fs, cutoff_hz=4.0)
    chest = lowpass_signal(kin.omega_mag["chestbone"], fs, cutoff_hz=4.0)
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


def count_corrective_peaks(omega: np.ndarray, fs: float) -> tuple[int, float]:
    smoothed = lowpass_signal(omega, fs, cutoff_hz=6.0)
    threshold = np.percentile(smoothed, 75) + 1.5 * (np.percentile(smoothed, 75) - np.percentile(smoothed, 25))
    peaks, props = find_peaks(smoothed, height=threshold, distance=int(0.3 * fs))
    if len(peaks) == 0:
        return 0, float(np.max(smoothed))
    return int(len(peaks)), float(np.max(props["peak_heights"]))


def compute_metrics(label: str, kin: Kinematics, fs: float, start: int, end: int, stab_start: int, stab_end: int) -> dict[str, float | str]:
    event = slice(start, end)
    stab = slice(stab_start, stab_end)
    corr, lag = cross_correlation_lag(kin.eulers_deg["lumbar"][event, 2], kin.eulers_deg["chestbone"][event, 2], fs)
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
        "corrective_lumbar_angular_velocity_peak_count": peak_count,
        "largest_corrective_lumbar_angular_velocity_dps": peak_height,
    }


def save_orientation_npz(label: str, kin: Kinematics) -> None:
    payload = {"time_s": kin.t}
    for sensor, q in kin.quaternions.items():
        payload[f"{sensor}_q_wxyz"] = q
    np.savez_compressed(OUTPUT_DIR / f"orientation_{label.lower()}.npz", **payload)


def save_kinematic_variables(label: str, kin: Kinematics, fs: float) -> None:
    target_fs = 50.0
    target_time, trunk_roll = resample_filtered_full(kin.t, kin.trunk_rel_euler_deg[:, 0], fs, target_fs)
    _, trunk_pitch = resample_filtered_full(kin.t, kin.trunk_rel_euler_deg[:, 1], fs, target_fs)
    _, trunk_yaw = resample_filtered_full(kin.t, kin.trunk_rel_euler_deg[:, 2], fs, target_fs)
    _, lumbar_roll = resample_filtered_full(kin.t, kin.eulers_deg["lumbar"][:, 0], fs, target_fs)
    _, lumbar_pitch = resample_filtered_full(kin.t, kin.eulers_deg["lumbar"][:, 1], fs, target_fs)
    _, lumbar_yaw = resample_filtered_full(kin.t, kin.eulers_deg["lumbar"][:, 2], fs, target_fs)
    _, chest_yaw = resample_filtered_full(kin.t, kin.eulers_deg["chestbone"][:, 2], fs, target_fs)
    _, lumbar_omega = resample_filtered_full(kin.t, kin.omega_mag["lumbar"], fs, target_fs)
    _, chest_omega = resample_filtered_full(kin.t, kin.omega_mag["chestbone"], fs, target_fs)
    _, left_knee = resample_filtered_full(kin.t, kin.left_knee_deg, fs, target_fs)
    _, right_knee = resample_filtered_full(kin.t, kin.right_knee_deg, fs, target_fs)
    _, left_ankle = resample_filtered_full(kin.t, kin.left_ankle_deg, fs, target_fs)
    _, right_ankle = resample_filtered_full(kin.t, kin.right_ankle_deg, fs, target_fs)
    df = pd.DataFrame(
        {
            "time_s": target_time,
            "trunk_pelvis_roll_deg": trunk_roll,
            "trunk_pelvis_pitch_deg": trunk_pitch,
            "trunk_pelvis_yaw_deg": trunk_yaw,
            "lumbar_roll_deg": lumbar_roll,
            "lumbar_pitch_deg": lumbar_pitch,
            "lumbar_yaw_deg": lumbar_yaw,
            "chest_yaw_deg": chest_yaw,
            "lumbar_omega_dps": lumbar_omega,
            "chest_omega_dps": chest_omega,
            "left_knee_flexion_est_deg": left_knee,
            "right_knee_flexion_est_deg": right_knee,
            "left_ankle_flexion_est_deg": left_ankle,
            "right_ankle_flexion_est_deg": right_ankle,
        }
    )
    df.to_csv(OUTPUT_DIR / f"kinematic_variables_{label.lower()}_50hz.csv", index=False)


def make_traceability_figure(
    novice: Kinematics,
    trained: Kinematics,
    metrics: pd.DataFrame,
    windows: dict[str, list[tuple[int, int, int, int, float]]],
    fs: float,
) -> None:
    fig = plt.figure(figsize=(11, 8.2), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.15, 1.15, 0.9])
    axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]

    for ax, label, kin in [(axes[0], "Novice", novice), (axes[1], "Trained", trained)]:
        t = kin.t
        yaw = lowpass_signal(kin.trunk_rel_euler_deg[:, 2], fs, cutoff_hz=4.0)
        roll = lowpass_signal(kin.trunk_rel_euler_deg[:, 0], fs, cutoff_hz=4.0)
        pitch = lowpass_signal(kin.trunk_rel_euler_deg[:, 1], fs, cutoff_hz=4.0)
        yaw_vel = np.gradient(yaw, 1.0 / fs)
        
        ax.plot(t, yaw, color="#1f77b4", lw=1.2, label="rel yaw")
        ax.plot(t, roll, color="#9467bd", lw=1.2, label="rel roll")
        ax.plot(t, pitch, color="#8c564b", lw=1.2, label="rel pitch")
        
        ax2 = ax.twinx()
        ax2.plot(t, yaw_vel, color="#e377c2", lw=1.0, linestyle="--", label="yaw vel")
        ax2.set_ylabel("vel (deg/s)", color="#e377c2", fontsize=9)
        ax2.tick_params(axis='y', labelcolor="#e377c2", labelsize=8)
        
        for i, (event_start, event_end, stab_start, stab_end, peak) in enumerate(windows[label]):
            ax.axvspan(event_start / fs, event_end / fs, color="#f2b134", alpha=0.22, label="aligned event" if i == 0 else "")
            ax.axvspan(stab_start / fs, stab_end / fs, color="#57a773", alpha=0.18, label="stabilization" if i == 0 else "")
            if label == "Novice":
                ax.scatter([t[peak]], [yaw[peak]], s=30, color="#d1495b", zorder=4, label="yaw peak" if i == 0 else "")
        
        ax.set_ylabel(f"{label}\nangle (deg)")
        ax.grid(True, color="#dddddd", lw=0.6)
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8, frameon=False, ncol=4)

    metric_names = [
        "trunk_pelvis_lag_s",
        "weight_shift_log10_dimensionless_jerk",
        "lumbar_orientation_variability_deg",
        "corrective_lumbar_angular_velocity_peak_count",
    ]
    pretty = ["coordination lag (s)", "log10 smoothness jerk", "orientation variability (deg)", "corrective peaks"]
    x = np.arange(len(metric_names))
    width = 0.34
    novice_vals = [metrics.loc[metrics.trial == "Novice", m].mean() for m in metric_names]
    trained_vals = [metrics.loc[metrics.trial == "Trained", m].mean() for m in metric_names]
    novice_bars = axes[2].bar(x - width / 2, novice_vals, width, color="#d1495b", label="Novice")
    trained_bars = axes[2].bar(x + width / 2, trained_vals, width, color="#2a9d8f", label="Trained")
    axes[2].bar_label(novice_bars, fmt="%.2f", padding=3, fontsize=8)
    axes[2].bar_label(trained_bars, fmt="%.2f", padding=3, fontsize=8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(pretty, rotation=12, ha="right")
    axes[2].set_ylabel("metric value")
    axes[2].set_title("Raw IMU -> Madgwick orientation -> kinematic event -> balance-relevant metrics")
    axes[2].grid(True, axis="y", color="#dddddd", lw=0.6)
    axes[2].legend(frameon=False)

    fig.suptitle("Kinematic Traceability: Novice vs Trained Tai Chi", fontsize=14, fontweight="bold")
    fig.savefig(OUTPUT_DIR / "traceability_figure.png", dpi=220)
    plt.close(fig)


def make_orientation_validation_figure(novice: Kinematics, trained: Kinematics, fs: float) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.8), sharex=False, constrained_layout=True)
    for row, (label, kin) in enumerate([("Novice", novice), ("Trained", trained)]):
        for col, sensor in enumerate(["lumbar", "chestbone"]):
            ax = axes[row, col]
            target_time, roll = resample_filtered_full(kin.t, kin.eulers_deg[sensor][:, 0], fs, 10.0)
            _, pitch = resample_filtered_full(kin.t, kin.eulers_deg[sensor][:, 1], fs, 10.0)
            _, yaw = resample_filtered_full(kin.t, kin.eulers_deg[sensor][:, 2], fs, 10.0)
            ax.plot(target_time, roll, lw=0.9, label="roll")
            ax.plot(target_time, pitch, lw=0.9, label="pitch")
            ax.plot(target_time, yaw, lw=0.9, label="yaw")
            ax.set_title(f"{label} {sensor} orientation")
            ax.set_xlabel("time (s)")
            ax.set_ylabel("angle (deg)")
            ax.grid(True, color="#dddddd", lw=0.6)
            ax.legend(frameon=False, fontsize=8, ncol=3)
    fig.suptitle("Orientation Validation: Lumbar and Chestbone", fontsize=13, fontweight="bold")
    fig.savefig(OUTPUT_DIR / "orientation_validation.png", dpi=220)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    novice_trial = parse_trigno_csv(NOVICE_CSV, "Novice")
    trained_trial = parse_trigno_csv(TRAINED_CSV, "Trained")
    inventory = make_sensor_inventory([novice_trial, trained_trial])
    inventory.to_csv(OUTPUT_DIR / "sensor_inventory.csv", index=False)

    novice_kin = compute_kinematics(novice_trial)
    trained_kin = compute_kinematics(trained_trial)
    save_orientation_npz("novice", novice_kin)
    save_orientation_npz("trained", trained_kin)
    save_kinematic_variables("novice", novice_kin, novice_trial.fs)
    save_kinematic_variables("trained", trained_kin, trained_trial.fs)

    novice_events, rationale = select_novice_events(novice_kin, novice_trial.fs, 6)
    
    all_metrics = []
    windows = {"Novice": [], "Trained": []}
    dtw_dists = []
    
    last_trained_end = 0
    for idx, (novice_start, novice_end, novice_peak) in enumerate(novice_events):
        trained_start, trained_end, dtw_dist = find_trained_match(
            novice_kin, trained_kin, novice_trial.fs, novice_start, novice_end, min_start=last_trained_end
        )
        last_trained_end = trained_end
        dtw_dists.append(dtw_dist)
        
        novice_stab_start, novice_stab_end = find_stabilization(novice_kin, novice_trial.fs, novice_end)
        trained_stab_start, trained_stab_end = find_stabilization(trained_kin, trained_trial.fs, trained_end)
        
        windows["Novice"].append((novice_start, novice_end, novice_stab_start, novice_stab_end, novice_peak))
        windows["Trained"].append((trained_start, trained_end, trained_stab_start, trained_stab_end, np.nan))
        
        all_metrics.append(compute_metrics("Novice", novice_kin, novice_trial.fs, novice_start, novice_end, novice_stab_start, novice_stab_end))
        all_metrics.append(compute_metrics("Trained", trained_kin, trained_trial.fs, trained_start, trained_end, trained_stab_start, trained_stab_end))

    event_rows = []
    for label, events in windows.items():
        for i, (start, end, stab_start, stab_end, peak) in enumerate(events):
            event_rows.append({
                "trial": label,
                "event_index": i + 1,
                "event_start_s": start / FS,
                "event_end_s": end / FS,
                "event_peak_s": (peak / FS if label == "Novice" else np.nan),
                "stabilization_start_s": stab_start / FS,
                "stabilization_end_s": stab_end / FS,
            })
    pd.DataFrame(event_rows).to_csv(OUTPUT_DIR / "event_windows.csv", index=False)

    metrics = pd.DataFrame(all_metrics)
    metrics.to_csv(OUTPUT_DIR / "balance_metrics.csv", index=False)

    make_traceability_figure(novice_kin, trained_kin, metrics, windows, novice_trial.fs)
    make_orientation_validation_figure(novice_kin, trained_kin, novice_trial.fs)
    avg_dtw_dist = sum(dtw_dists) / len(dtw_dists)

    print("Analysis complete.")
    print(metrics.to_string(index=False))
    print(f"Orientation validation figure: {OUTPUT_DIR / 'orientation_validation.png'}")
    print(f"Traceability figure: {OUTPUT_DIR / 'traceability_figure.png'}")

if __name__ == "__main__":
    main()
