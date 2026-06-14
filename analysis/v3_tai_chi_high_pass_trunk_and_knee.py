#!/usr/bin/env python3
"""Kinematic comparison of novice and trained Tai Chi IMU recordings.

The script parses Delsys Trigno Discover CSV exports, estimates segment
orientations with a 6-axis Madgwick filter, derives balance-relevant trunk and
lower-limb kinematics, aligns a selected novice movement module to the trained
recording and performs a knee-flexion (>60°) balance analysis
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

Ignore_high_pass_filter = False  # Set to True to skip the high-pass filter (for yaw drift)

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
    left_shoulder_deg: np.ndarray
    right_shoulder_deg: np.ndarray
    left_elbow_deg: np.ndarray
    right_elbow_deg: np.ndarray
    left_hip_deg: np.ndarray
    right_hip_deg: np.ndarray


def canonical_sensor(name: str) -> str:
    stem = re.sub(r"\s*\(\d+\)\s*$", "", name.strip())
    return stem.lower().replace("_", "")


def parse_IMU_csv(path: Path, label: str) -> TrialData:
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


def highpass_detrend(x: np.ndarray, fs: float, cutoff_hz: float = 0.05, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth high-pass filter to remove yaw drift trends.

    A 0.05 Hz cutoff removes slow drift (period > 20 s) while preserving all
    Tai Chi trunk-rotation content (typically 0.2–1 Hz).  The filter is applied
    with sosfiltfilt so it introduces no phase distortion.
    """
    if Ignore_high_pass_filter:
        return x.copy()
    
    else:

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



def find_neutral_pose_window(trial: TrialData, min_duration_s: float = 2.0) -> tuple[int, int]:

    """ After the sync signal, the participant stands still for a moment to establish a neutral pose.  
    This function finds a window of at least 2 s where the trunk is relatively still.  
    This is used to compute the neutral orientation of each sensor.  
    The search starts after the first peak in lumbar acceleration and looks for a window with minimal lumbar angular velocity.  
    If no suitable window is found, the function returns the full signal range.
    """
        
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
    
## because the sensors are mounted in different orientations on the body, we need to apply a rotation to align them to a common reference frame.
    if sensor_name in ["chestbone", "lulna", "rulna"]:
        R = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]) # internal orientation of these sensors is x = forward, y = left, and z = upward. R transforms local coordinate system to common reference frame: x = forward, y = right, and z = downward
    elif sensor_name == "lumbar":
        R = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]]) # internal orientation of this sensor is x = backward, y = left, and z = upward. R transforms local coordinate system to common reference frame: x = forward, y = right, and z = downward
    elif sensor_name in ["lhumerus", "lhand", "lthigh", "ltibia"]:
        R = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]]) # internal orientation of these sensors is x = downward, y = left, and z = backward. R transforms local coordinate system to common reference frame: x = forward, y = right, and z = downward
    elif sensor_name in ["rhumerus", "rhand", "rthigh", "rtibia"]:
        R = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]) # internal orientation of these sensors is x = upward, y = right, and z = forward. R transforms local coordinate system to common reference frame: x = forward, y = right, and z = downward
    elif sensor_name in ["lfoot", "rfoot"]:
        R = np.eye(3) # internal orientation of these sensors is already aligned with the common reference frame: x = forward, y = right, and z = downward
    else:
        R = np.eye(3) # default to identity if sensor name is not recognized
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

    left_hip = quat_to_euler_deg(quat_multiply(quat_inverse(quats["lumbar"]), quats["lthigh"]))
    right_hip = quat_to_euler_deg(quat_multiply(quat_inverse(quats["lumbar"]), quats["rthigh"]))

    left_knee = quat_to_euler_deg(quat_multiply(quat_inverse(quats["lthigh"]), quats["ltibia"]))
    right_knee = quat_to_euler_deg(quat_multiply(quat_inverse(quats["rthigh"]), quats["rtibia"]))

    left_ankle = quat_to_euler_deg(quat_multiply(quat_inverse(quats["ltibia"]), quats["lfoot"]))
    right_ankle = quat_to_euler_deg(quat_multiply(quat_inverse(quats["rtibia"]), quats["rfoot"]))

    left_shoulder = quat_to_euler_deg(quat_multiply(quat_inverse(quats["chestbone"]), quats["lhumerus"]))
    right_shoulder = quat_to_euler_deg(quat_multiply(quat_inverse(quats["chestbone"]), quats["rhumerus"]))

    left_elbow = quat_to_euler_deg(quat_multiply(quat_inverse(quats["lhumerus"]), quats["lulna"]))
    right_elbow = quat_to_euler_deg(quat_multiply(quat_inverse(quats["rhumerus"]), quats["rulna"]))


    return Kinematics(
        t=t,
        quaternions=quats,
        eulers_deg=eulers,
        omega_mag=omega_mag,
        trunk_rel_euler_deg=trunk_rel_euler,
        left_hip_deg=left_hip,
        right_hip_deg=right_hip,
        left_knee_deg=left_knee,
        right_knee_deg=right_knee,
        left_ankle_deg=left_ankle,
        right_ankle_deg=right_ankle,
        left_shoulder_deg=left_shoulder,
        right_shoulder_deg=right_shoulder,
        left_elbow_deg=left_elbow,
        right_elbow_deg=right_elbow,
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



def detect_novice_trunk_rotation_events(kin: Kinematics, fs: float, num_events: int = 3) -> tuple[list[tuple[int, int, int]], str]:
    yaw = highpass_detrend(kin.trunk_rel_euler_deg[:, 2], fs, cutoff_hz=0.05)
    yaw = lowpass_signal(yaw, fs, cutoff_hz=4.0)
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
    # Detrend yaw over the full signal before slicing the window so that slow
    # drift accumulated before the window does not shift the baseline.
    yaw_detrended = highpass_detrend(kin.trunk_rel_euler_deg[:, 2], fs, cutoff_hz=0.05)
    sig = np.column_stack(
        [
            resample_filtered_window(kin.t, yaw_detrended, fs, start, end, target_fs),
            resample_filtered_window(kin.t, kin.eulers_deg["lumbar"][:, 0], fs, start, end, target_fs),
            resample_filtered_window(kin.t, kin.omega_mag["chestbone"], fs, start, end, target_fs),
            resample_filtered_window(kin.t, kin.left_knee_deg[:, 0], fs, start, end, target_fs),
            resample_filtered_window(kin.t, kin.right_knee_deg[:, 0], fs, start, end, target_fs),
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


def find_knee_flexion_windows(
    knee_angle_deg: np.ndarray,
    fs: float,
    threshold_deg: float = 60.0,
    min_duration_s: float = 0.4,
    merge_gap_s: float = 0.2,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Return windows where the knee flexion magnitude exceeds a threshold."""
    filtered = lowpass_signal(knee_angle_deg, fs, cutoff_hz=6.0)
    flexion_mag = np.abs(filtered)
    above = flexion_mag >= threshold_deg

    runs = extract_contiguous_runs(above)
    min_len = max(1, int(round(min_duration_s * fs)))
    gap_len = max(1, int(round(merge_gap_s * fs)))

    windows: list[tuple[int, int]] = []
    for start, end in runs:
        if end - start < min_len:
            continue
        if windows and start - windows[-1][1] <= gap_len:
            windows[-1] = (windows[-1][0], end)
        else:
            windows.append((start, end))

    return windows, flexion_mag


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
        "side": side,
        "threshold_deg": threshold_deg,
        "window_start_s": start / fs,
        "window_end_s": end / fs,
        "window_duration_s": (end - start) / fs,
        "peak_time_s": peak_idx / fs,
        "peak_signed_knee_flexion_deg": float(knee_filtered[peak_idx]),
        "peak_abs_knee_flexion_deg": float(knee_abs[peak_idx]),
        "knee_flexion_range_deg": float(np.ptp(knee_abs[event])),
        "trunk_rel_roll_mean_deg": float(trunk_rel_mean[0]),
        "trunk_rel_pitch_mean_deg": float(trunk_rel_mean[1]),
        "trunk_rel_yaw_mean_deg": float(trunk_rel_mean[2]),
        "trunk_rel_roll_range_deg": float(np.ptp(trunk_rel[:, 0])),
        "trunk_rel_pitch_range_deg": float(np.ptp(trunk_rel[:, 1])),
        "trunk_rel_yaw_range_deg": float(np.ptp(trunk_rel[:, 2])),
        "lumbar_roll_mean_deg": float(lumbar_mean[0]),
        "lumbar_pitch_mean_deg": float(lumbar_mean[1]),
        "lumbar_yaw_mean_deg": float(lumbar_mean[2]),
        "lumbar_roll_range_deg": float(np.ptp(lumbar[:, 0])),
        "lumbar_pitch_range_deg": float(np.ptp(lumbar[:, 1])),
        "lumbar_yaw_range_deg": float(np.ptp(lumbar[:, 2])),
        "peak_lumbar_angular_velocity_dps": float(np.max(lumbar_omega)),
        "peak_chest_angular_velocity_dps": float(np.max(chest_omega)),
        "peak_trunk_rel_roll_deg": float(trunk_rel[peak_rel, 0]),
        "peak_trunk_rel_pitch_deg": float(trunk_rel[peak_rel, 1]),
        "peak_trunk_rel_yaw_deg": float(trunk_rel[peak_rel, 2]),
    }


def compute_trunk_rotation_balance_metrics(label: str, kin: Kinematics, fs: float, start: int, end: int, stab_start: int, stab_end: int) -> dict[str, float | str]:
    event = slice(start, end)
    stab = slice(stab_start, stab_end)
    # Detrend yaw over the full signal before slicing the event window so that
    # slow drift accumulated before the event does not corrupt the correlation.
    lumbar_yaw_detrended = highpass_detrend(kin.eulers_deg["lumbar"][:, 2], fs, cutoff_hz=0.05)
    chest_yaw_detrended = highpass_detrend(kin.eulers_deg["chestbone"][:, 2], fs, cutoff_hz=0.05)
    corr, lag = cross_correlation_lag(lumbar_yaw_detrended[event], chest_yaw_detrended[event], fs)
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
    # Detrend yaw channels before resampling to remove drift trend from exported CSV.
    trunk_yaw_detrended = highpass_detrend(kin.trunk_rel_euler_deg[:, 2], fs, cutoff_hz=0.05)
    lumbar_yaw_raw = highpass_detrend(kin.eulers_deg["lumbar"][:, 2], fs, cutoff_hz=0.05)
    chest_yaw_raw = highpass_detrend(kin.eulers_deg["chestbone"][:, 2], fs, cutoff_hz=0.05)
    _, trunk_yaw = resample_filtered_full(kin.t, trunk_yaw_detrended, fs, target_fs)
    _, lumbar_roll = resample_filtered_full(kin.t, kin.eulers_deg["lumbar"][:, 0], fs, target_fs)
    _, lumbar_pitch = resample_filtered_full(kin.t, kin.eulers_deg["lumbar"][:, 1], fs, target_fs)
    _, lumbar_yaw = resample_filtered_full(kin.t, lumbar_yaw_raw, fs, target_fs)
    _, chest_yaw = resample_filtered_full(kin.t, chest_yaw_raw, fs, target_fs)
    _, lumbar_omega = resample_filtered_full(kin.t, kin.omega_mag["lumbar"], fs, target_fs)
    _, chest_omega = resample_filtered_full(kin.t, kin.omega_mag["chestbone"], fs, target_fs)
    _, left_knee_roll = resample_filtered_full(kin.t, highpass_detrend(kin.left_knee_deg[:, 0], fs, cutoff_hz=0.05), fs, target_fs)
    _, left_knee_pitch = resample_filtered_full(kin.t, highpass_detrend(kin.left_knee_deg[:, 1], fs, cutoff_hz=0.05), fs, target_fs)
    _, left_knee_yaw = resample_filtered_full(kin.t, highpass_detrend(kin.left_knee_deg[:, 2], fs, cutoff_hz=0.05), fs, target_fs)
    _, right_knee_roll = resample_filtered_full(kin.t, highpass_detrend(kin.right_knee_deg[:, 0], fs, cutoff_hz=0.05), fs, target_fs)
    _, right_knee_pitch = resample_filtered_full(kin.t, highpass_detrend(kin.right_knee_deg[:, 1], fs, cutoff_hz=0.05), fs, target_fs)
    _, right_knee_yaw = resample_filtered_full(kin.t, highpass_detrend(kin.right_knee_deg[:, 2], fs, cutoff_hz=0.05), fs, target_fs)
    _, left_ankle_roll = resample_filtered_full(kin.t, highpass_detrend(kin.left_ankle_deg[:, 0], fs, cutoff_hz=0.05), fs, target_fs)
    _, left_ankle_pitch = resample_filtered_full(kin.t, highpass_detrend(kin.left_ankle_deg[:, 1], fs, cutoff_hz=0.05), fs, target_fs)
    _, left_ankle_yaw = resample_filtered_full(kin.t, highpass_detrend(kin.left_ankle_deg[:, 2], fs, cutoff_hz=0.05), fs, target_fs)
    _, right_ankle_roll = resample_filtered_full(kin.t, highpass_detrend(kin.right_ankle_deg[:, 0], fs, cutoff_hz=0.05), fs, target_fs)
    _, right_ankle_pitch = resample_filtered_full(kin.t, highpass_detrend(kin.right_ankle_deg[:, 1], fs, cutoff_hz=0.05), fs, target_fs)
    _, right_ankle_yaw = resample_filtered_full(kin.t, highpass_detrend(kin.right_ankle_deg[:, 2], fs, cutoff_hz=0.05), fs, target_fs)
    _, left_shoulder_roll = resample_filtered_full(kin.t, highpass_detrend(kin.left_shoulder_deg[:, 0], fs, cutoff_hz=0.05), fs, target_fs)
    _, left_shoulder_pitch = resample_filtered_full(kin.t, highpass_detrend(kin.left_shoulder_deg[:, 1], fs, cutoff_hz=0.05), fs, target_fs)
    _, left_shoulder_yaw = resample_filtered_full(kin.t, highpass_detrend(kin.left_shoulder_deg[:, 2], fs, cutoff_hz=0.05), fs, target_fs)
    _, right_shoulder_roll = resample_filtered_full(kin.t, highpass_detrend(kin.right_shoulder_deg[:, 0], fs, cutoff_hz=0.05), fs, target_fs)
    _, right_shoulder_pitch = resample_filtered_full(kin.t, highpass_detrend(kin.right_shoulder_deg[:, 1], fs, cutoff_hz=0.05), fs, target_fs)
    _, right_shoulder_yaw = resample_filtered_full(kin.t, highpass_detrend(kin.right_shoulder_deg[:, 2], fs, cutoff_hz=0.05), fs, target_fs)
    _, left_elbow_roll = resample_filtered_full(kin.t, highpass_detrend(kin.left_elbow_deg[:, 0], fs, cutoff_hz=0.05), fs, target_fs)
    _, left_elbow_pitch = resample_filtered_full(kin.t, highpass_detrend(kin.left_elbow_deg[:, 1], fs, cutoff_hz=0.05), fs, target_fs)
    _, left_elbow_yaw = resample_filtered_full(kin.t, highpass_detrend(kin.left_elbow_deg[:, 2], fs, cutoff_hz=0.05), fs, target_fs)
    _, right_elbow_roll = resample_filtered_full(kin.t, highpass_detrend(kin.right_elbow_deg[:, 0], fs, cutoff_hz=0.05), fs, target_fs)
    _, right_elbow_pitch = resample_filtered_full(kin.t, highpass_detrend(kin.right_elbow_deg[:, 1], fs, cutoff_hz=0.05), fs, target_fs)
    _, right_elbow_yaw = resample_filtered_full(kin.t, highpass_detrend(kin.right_elbow_deg[:, 2], fs, cutoff_hz=0.05), fs, target_fs)
    _, left_hip_roll = resample_filtered_full(kin.t, highpass_detrend(kin.left_hip_deg[:, 0], fs, cutoff_hz=0.05), fs, target_fs)
    _, left_hip_pitch = resample_filtered_full(kin.t, highpass_detrend(kin.left_hip_deg[:, 1], fs, cutoff_hz=0.05), fs, target_fs)
    _, left_hip_yaw = resample_filtered_full(kin.t, highpass_detrend(kin.left_hip_deg[:, 2], fs, cutoff_hz=0.05), fs, target_fs)
    _, right_hip_roll = resample_filtered_full(kin.t, highpass_detrend(kin.right_hip_deg[:, 0], fs, cutoff_hz=0.05), fs, target_fs)
    _, right_hip_pitch = resample_filtered_full(kin.t, highpass_detrend(kin.right_hip_deg[:, 1], fs, cutoff_hz=0.05), fs, target_fs)
    _, right_hip_yaw = resample_filtered_full(kin.t, highpass_detrend(kin.right_hip_deg[:, 2], fs, cutoff_hz=0.05), fs, target_fs)
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

            "left_knee_roll_est_deg": left_knee_roll,
            "left_knee_pitch_est_deg": left_knee_pitch,
            "left_knee_yaw_est_deg": left_knee_yaw,
            "right_knee_roll_est_deg": right_knee_roll,
            "right_knee_pitch_est_deg": right_knee_pitch,
            "right_knee_yaw_est_deg": right_knee_yaw,
            "left_ankle_roll_est_deg": left_ankle_roll,
            "left_ankle_pitch_est_deg": left_ankle_pitch,
            "left_ankle_yaw_est_deg": left_ankle_yaw,
            "right_ankle_roll_est_deg": right_ankle_roll,
            "right_ankle_pitch_est_deg": right_ankle_pitch,
            "right_ankle_yaw_est_deg": right_ankle_yaw,
            "left_shoulder_roll_est_deg": left_shoulder_roll,
            "left_shoulder_pitch_est_deg": left_shoulder_pitch,
            "left_shoulder_yaw_est_deg": left_shoulder_yaw,
            "right_shoulder_roll_est_deg": right_shoulder_roll,
            "right_shoulder_pitch_est_deg": right_shoulder_pitch,
            "right_shoulder_yaw_est_deg": right_shoulder_yaw,
            "left_elbow_roll_est_deg": left_elbow_roll,
            "left_elbow_pitch_est_deg": left_elbow_pitch,
            "left_elbow_yaw_est_deg": left_elbow_yaw,
            "right_elbow_roll_est_deg": right_elbow_roll,
            "right_elbow_pitch_est_deg": right_elbow_pitch,
            "right_elbow_yaw_est_deg": right_elbow_yaw,
            "left_hip_roll_est_deg": left_hip_roll,
            "left_hip_pitch_est_deg": left_hip_pitch,
            "left_hip_yaw_est_deg": left_hip_yaw,
            "right_hip_roll_est_deg": right_hip_roll,
            "right_hip_pitch_est_deg": right_hip_pitch,
            "right_hip_yaw_est_deg": right_hip_yaw,
        }
    )
    df.to_csv(OUTPUT_DIR / f"kinematic_variables_{label.lower()}_50hz.csv", index=False)


def make_trunk_traceability_figure(
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
        yaw = highpass_detrend(kin.trunk_rel_euler_deg[:, 2], fs, cutoff_hz=0.05)
        yaw = lowpass_signal(yaw, fs, cutoff_hz=4.0)
        # roll = lowpass_signal(kin.trunk_rel_euler_deg[:, 0], fs, cutoff_hz=4.0)
        # pitch = lowpass_signal(kin.trunk_rel_euler_deg[:, 1], fs, cutoff_hz=4.0)
        # yaw_vel = np.gradient(yaw, 1.0 / fs)
        
        ax.plot(t, yaw, color="#1f77b4", lw=1.2, label="rel yaw")
        # ax.plot(t, roll, color="#9467bd", lw=1.2, label="rel roll")
        # ax.plot(t, pitch, color="#8c564b", lw=1.2, label="rel pitch")
        
        # ax2 = ax.twinx()
        # ax2.plot(t, yaw_vel, color="#e377c2", lw=1.0, linestyle="--", label="yaw vel")
        # ax2.set_ylabel("vel (deg/s)", color="#e377c2", fontsize=9)
        # ax2.tick_params(axis='y', labelcolor="#e377c2", labelsize=8)
        
        for i, (event_start, event_end, stab_start, stab_end, peak) in enumerate(windows[label]):
            ax.axvspan(event_start / fs, event_end / fs, color="#f2b134", alpha=0.22, label="aligned event" if i == 0 else "")
            ax.axvspan(stab_start / fs, stab_end / fs, color="#57a773", alpha=0.18, label="stabilization" if i == 0 else "")
            if label == "Novice":
                ax.scatter([t[peak]], [yaw[peak]], s=30, color="#d1495b", zorder=4, label="yaw peak" if i == 0 else "")
        
        ax.set_ylabel(f"{label}\nangle (deg)")
        ax.grid(True, color="#dddddd", lw=0.6)
        
        lines1, labels1 = ax.get_legend_handles_labels()
        # lines2, labels2 = ax2.get_legend_handles_labels()
        # ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8, frameon=False, ncol=4)

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
    fig.savefig(OUTPUT_DIR / "trunk_traceability_figure.png", dpi=220)
    plt.close(fig)



def compute_knee_balance_metrics(
    label: str,
    kin: Kinematics,
    fs: float,
    side: str,
    start: int,
    end: int,
    lag_pre_s: float = 1.0,
    lag_post_s: float = 2.0,
) -> dict[str, float | str]:
    knee_signal = kin.left_knee_deg[:, 0] if side.lower().startswith("l") else kin.right_knee_deg[:, 0]
    knee_abs = np.abs(lowpass_signal(knee_signal, fs, cutoff_hz=6.0))
    peak_idx = start + int(np.argmax(knee_abs[start:end]))

    # Cross-correlation over a fixed window centred on peak knee flexion so
    # that the window length is consistent across events and participants,
    # and directly targets the coordination response at maximum challenge.
    lag_start = max(0, peak_idx - int(lag_pre_s * fs))
    lag_end = min(len(knee_abs), peak_idx + int(lag_post_s * fs))
    stab_start, stab_end = find_stabilization(kin, fs, peak_idx)
    corr, lag = cross_correlation_lag(
        highpass_detrend(kin.eulers_deg["lumbar"][:, 2], fs)[lag_start:lag_end],
        highpass_detrend(kin.eulers_deg["chestbone"][:, 2], fs)[lag_start:lag_end],
        fs,
    )

    orientation = kin.eulers_deg["lumbar"][stab_start:stab_end,:]
    orient_var = float(np.sqrt(np.mean(np.var(orientation, axis=0))))
    peak_count, peak_height = count_corrective_peaks(
        kin.omega_mag["lumbar"][stab_start:stab_end], fs
    )

    trunk_pitch = lowpass_signal(kin.eulers_deg["lumbar"][:,1], fs, cutoff_hz=4.0)

    return {
        "trial": label,
        "side": side,
        "peak_knee_flexion_deg": float(knee_abs[peak_idx]),
        "peak_trunk_pitch_deg": float(trunk_pitch[peak_idx]),
        "time_to_stabilization_s": float((stab_start - peak_idx) / fs),
        "lumbar_orientation_variability_deg": orient_var,
        "corrective_peak_count": peak_count,
        "largest_corrective_peak_dps": peak_height,
        "trunk_pelvis_lag_s": lag,
    }



def make_knee_flexion_overview_figure(
    novice: Kinematics,
    trained: Kinematics,
    knee_events: dict[str, list[dict[str, float | str]]],
    fs: float,
) -> None:
    fig = plt.figure(figsize=(11, 9.2), constrained_layout=True)
    gs = fig.add_gridspec(4, 1, height_ratios=[1.0, 0.9, 1.0, 0.9])
    axes = [fig.add_subplot(gs[i, 0]) for i in range(4)]

    trial_specs = [
        (axes[0], axes[1], "Novice", novice),
        (axes[2], axes[3], "Trained", trained),
    ]

    for knee_ax, trunk_ax, label, kin in trial_specs:
        t = kin.t
        left_knee = np.abs(lowpass_signal(kin.left_knee_deg[:, 0], fs, cutoff_hz=6.0))
        right_knee = np.abs(lowpass_signal(kin.right_knee_deg[:, 0], fs, cutoff_hz=6.0))
        trunk_pitch = lowpass_signal(kin.eulers_deg["lumbar"][:, 1], fs, cutoff_hz=4.0)
        trunk_roll = lowpass_signal(kin.eulers_deg["lumbar"][:, 0], fs, cutoff_hz=4.0)
        trunk_yaw = lowpass_signal(highpass_detrend(kin.trunk_rel_euler_deg[:, 2], fs, cutoff_hz=0.05), fs, cutoff_hz=4.0)

        knee_ax.plot(t, left_knee, lw=1.0, label="left knee |flexion|")
        knee_ax.plot(t, right_knee, lw=1.0, label="right knee |flexion|")
        knee_ax.axhline(60.0, lw=1.0, linestyle="--", color="#444444", label="60° threshold")
        knee_ax.set_ylabel(f"{label}\nknee flexion (deg)")
        knee_ax.grid(True, color="#dddddd", lw=0.6)

        seen_knee_labels: set[str] = set()
        for ev in knee_events[label]:
            color = "#f2b134" if ev["side"] == "Left" else "#8ecae6"
            knee_label = f"{ev['side']} flexion >60°"
            knee_ax.axvspan(
                ev["window_start_s"],
                ev["window_end_s"],
                color=color,
                alpha=0.18,
                label=knee_label if knee_label not in seen_knee_labels else "",
            )
            seen_knee_labels.add(knee_label)
            knee_ax.scatter([ev["peak_time_s"]], [ev["peak_abs_knee_flexion_deg"]], s=18, color="#d1495b", zorder=4)

        trunk_ax.plot(t, trunk_pitch, lw=1.0, label="lumbar pitch")
        trunk_ax.plot(t, trunk_roll, lw=1.0, label="lumbar roll")
        trunk_ax.plot(t, trunk_yaw, lw=1.0, label="trunk-relative yaw")
        trunk_ax.set_ylabel("trunk angle (deg)")
        trunk_ax.set_xlabel("time (s)")
        trunk_ax.grid(True, color="#dddddd", lw=0.6)

        seen_trunk_labels: set[str] = set()
        for ev in knee_events[label]:
            color = "#f2b134" if ev["side"] == "Left" else "#8ecae6"
            knee_label = f"{ev['side']} knee window"
            trunk_ax.axvspan(
                ev["window_start_s"],
                ev["window_end_s"],
                color=color,
                alpha=0.10,
                label=knee_label if knee_label not in seen_trunk_labels else "",
            )
            seen_trunk_labels.add(knee_label)

        knee_ax.legend(frameon=False, fontsize=8, ncol=3)
        trunk_ax.legend(frameon=False, fontsize=8, ncol=3)

    fig.suptitle("Knee Flexion > 60° and Trunk Balance Response", fontsize=14, fontweight="bold")
    fig.savefig(OUTPUT_DIR / "knee_flexion_balance_figure.png", dpi=220)
    plt.close(fig)



def make_knee_traceability_figure(
    novice: Kinematics,
    trained: Kinematics,
    knee_metrics: pd.DataFrame,
    knee_events: dict[str, list[dict[str, float | str]]],
    fs: float,
) -> None:
    fig = plt.figure(figsize=(11, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.15, 1.15, 0.9])
    axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]

    for ax, label, kin in [
        (axes[0], "Novice", novice),
        (axes[1], "Trained", trained),
    ]:
        t = kin.t
        knee = np.maximum(
            np.abs(lowpass_signal(kin.left_knee_deg[:, 0], fs, cutoff_hz=6.0)),
            np.abs(lowpass_signal(kin.right_knee_deg[:, 0], fs, cutoff_hz=6.0)),
        )

        ax.plot(t, knee, color="#1f77b4", lw=1.2, label="max knee flexion")
        ax.axhline(60.0, color="#d1495b", linestyle="--", alpha=0.7, label="60° threshold")

        for i, ev in enumerate(knee_events[label]):
            ax.axvspan(
                ev["window_start_s"],
                ev["window_end_s"],
                color="#f2b134",
                alpha=0.22,
                label="knee-flexion event" if i == 0 else "",
            )

            peak_idx = int(ev["peak_time_s"] * fs)
            stab_start, stab_end = find_stabilization(kin, fs, peak_idx)

            ax.axvspan(
                stab_start / fs,
                stab_end / fs,
                color="#57a773",
                alpha=0.18,
                label="stabilization" if i == 0 else "",
            )

            ax.scatter(
                [ev["peak_time_s"]],
                [ev["peak_abs_knee_flexion_deg"]],
                s=30,
                color="#d1495b",
                zorder=4,
                label="peak flexion" if i == 0 else "",
            )

        ax.set_ylabel(f"{label}\nknee flexion (deg)")
        ax.grid(True, color="#dddddd", lw=0.6)
        ax.legend(frameon=False, fontsize=8)

    metric_names = [
        "time_to_stabilization_s",
        "lumbar_orientation_variability_deg",
        "corrective_peak_count",
        "trunk_pelvis_lag_s",
        "peak_trunk_pitch_deg",
    ]

    pretty = [
        "stabilization time (s)",
        "orientation variability (deg)",
        "corrective peaks",
        "trunk-pelvis lag (s)",
        "peak trunk pitch (deg)",
    ]

    novice_vals = [
        knee_metrics.loc[knee_metrics.trial == "Novice", m].mean()
        for m in metric_names
    ]
    trained_vals = [
        knee_metrics.loc[knee_metrics.trial == "Trained", m].mean()
        for m in metric_names
    ]

    x = np.arange(len(metric_names))
    width = 0.34

    novice_bars = axes[2].bar(
        x - width / 2, novice_vals, width, color="#d1495b", label="Novice"
    )
    trained_bars = axes[2].bar(
        x + width / 2, trained_vals, width, color="#2a9d8f", label="Trained"
    )

    axes[2].bar_label(novice_bars, fmt="%.2f", padding=3, fontsize=8)
    axes[2].bar_label(trained_bars, fmt="%.2f", padding=3, fontsize=8)

    axes[2].set_xticks(x)
    axes[2].set_xticklabels(pretty, rotation=12, ha="right")
    axes[2].set_ylabel("metric value")
    axes[2].set_title("Balance metrics during deep knee flexion (>60°)")
    axes[2].grid(True, axis="y", color="#dddddd", lw=0.6)
    axes[2].legend(frameon=False)

    fig.suptitle(
        "Knee Flexion Balance Traceability: Novice vs Trained Tai Chi",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(OUTPUT_DIR / "knee_traceability_figure.png", dpi=220)
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
    # ------------------------------------------------------------------
    # 1. Load and parse IMU data
    # ------------------------------------------------------------------
    OUTPUT_DIR.mkdir(exist_ok=True)

    novice_trial = parse_IMU_csv(NOVICE_CSV, "Novice")
    trained_trial = parse_IMU_csv(TRAINED_CSV, "Trained")
    inventory = make_sensor_inventory([novice_trial, trained_trial])
    inventory.to_csv(OUTPUT_DIR / "sensor_inventory.csv", index=False)

    # ------------------------------------------------------------------
    # 2. Compute kinematics and save orientation outputs
    # ------------------------------------------------------------------
    novice_kin = compute_kinematics(novice_trial)
    trained_kin = compute_kinematics(trained_trial)
    save_orientation_npz("novice", novice_kin)
    save_orientation_npz("trained", trained_kin)
    save_kinematic_variables("novice", novice_kin, novice_trial.fs)
    save_kinematic_variables("trained", trained_kin, trained_trial.fs)

    # ------------------------------------------------------------------
    # 3. Trunk rotation balance analysis
    #    - Detect characteristic trunk-rotation events in the novice
    #    - Match each event to the best corresponding window in trained
    #    - Compute balance metrics (coordination, smoothness, variability)
    # ------------------------------------------------------------------
    novice_trunk_rotation_events, rationale = detect_novice_trunk_rotation_events(novice_kin, novice_trial.fs, 6)

    trunk_metric_rows: list[dict] = []
    trunk_event_windows: dict[str, list[tuple[int, int, int, int, float]]] = {"Novice": [], "Trained": []}
    dtw_dists = []

    last_trained_end = 0
    for idx, (novice_start, novice_end, novice_peak) in enumerate(novice_trunk_rotation_events):
        trained_start, trained_end, dtw_dist = find_trained_match(
            novice_kin, trained_kin, novice_trial.fs, novice_start, novice_end, min_start=last_trained_end
        )
        last_trained_end = trained_end
        dtw_dists.append(dtw_dist)

        novice_stab_start, novice_stab_end = find_stabilization(novice_kin, novice_trial.fs, novice_end)
        trained_stab_start, trained_stab_end = find_stabilization(trained_kin, trained_trial.fs, trained_end)

        trunk_event_windows["Novice"].append((novice_start, novice_end, novice_stab_start, novice_stab_end, novice_peak))
        trunk_event_windows["Trained"].append((trained_start, trained_end, trained_stab_start, trained_stab_end, np.nan))

        trunk_metric_rows.append(compute_trunk_rotation_balance_metrics("Novice", novice_kin, novice_trial.fs, novice_start, novice_end, novice_stab_start, novice_stab_end))
        trunk_metric_rows.append(compute_trunk_rotation_balance_metrics("Trained", trained_kin, trained_trial.fs, trained_start, trained_end, trained_stab_start, trained_stab_end))

    trunk_event_rows = []
    for label, events in trunk_event_windows.items():
        for i, (start, end, stab_start, stab_end, peak) in enumerate(events):
            trunk_event_rows.append({
                "trial": label,
                "event_index": i + 1,
                "event_start_s": (start / novice_trial.fs if label == "Novice" else start / trained_trial.fs),
                "event_end_s": (end / novice_trial.fs if label == "Novice" else end / trained_trial.fs),
                "event_peak_s": (peak / novice_trial.fs if label == "Novice" else np.nan),
                "stabilization_start_s": (stab_start / novice_trial.fs if label == "Novice" else stab_start / trained_trial.fs),
                "stabilization_end_s": (stab_end / novice_trial.fs if label == "Novice" else stab_end / trained_trial.fs),
            })
    pd.DataFrame(trunk_event_rows).to_csv(OUTPUT_DIR / "trunk_rotation_event_windows.csv", index=False)

    trunk_metrics = pd.DataFrame(trunk_metric_rows)
    trunk_metrics.to_csv(OUTPUT_DIR / "trunk_rotation_balance_metrics.csv", index=False)

    # ------------------------------------------------------------------
    # 4. Knee flexion balance analysis
    #    - Detect windows where knee flexion exceeds 60°
    #    - Summarise the trunk response during each knee-flexion window
    #    - Compute balance metrics (stabilization, orientation variability)
    # ------------------------------------------------------------------
    knee_event_rows = []
    knee_events: dict[str, list[dict]] = {"Novice": [], "Trained": []}
    for label, kin, fs in [("Novice", novice_kin, novice_trial.fs), ("Trained", trained_kin, trained_trial.fs)]:
        for side, angle in [("Left", kin.left_knee_deg[:, 0]), ("Right", kin.right_knee_deg[:, 0])]:
            windows_60, _ = find_knee_flexion_windows(angle, fs, threshold_deg=60.0, min_duration_s=0.4, merge_gap_s=0.2)
            for start, end in windows_60:
                event = summarize_knee_flexion_event(label, kin, fs, side, start, end, threshold_deg=60.0)
                knee_event_rows.append(event)
                knee_events[label].append(event)

    knee_balance_rows = []
    for label, kin, fs in [("Novice", novice_kin, novice_trial.fs), ("Trained", trained_kin, trained_trial.fs)]:
        for ev in knee_events[label]:
            start = int(ev["window_start_s"] * fs)
            end = int(ev["window_end_s"] * fs)
            knee_balance_rows.append(
                compute_knee_balance_metrics(label, kin, fs, ev["side"], start, end)
            )

    knee_metrics = pd.DataFrame(knee_balance_rows)
    knee_metrics.to_csv(OUTPUT_DIR / "knee_flexion_balance_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "trial": ev["trial"],
                "side": ev["side"],
                "window_start_s": ev["window_start_s"],
                "window_end_s": ev["window_end_s"],
                "peak_time_s": ev["peak_time_s"],
                "peak_abs_knee_flexion_deg": ev["peak_abs_knee_flexion_deg"],
            }
            for ev in knee_event_rows
        ]
    ).to_csv(OUTPUT_DIR / "knee_flexion_event_windows.csv", index=False)

    # ------------------------------------------------------------------
    # 5. Generate figures
    # ------------------------------------------------------------------
    make_trunk_traceability_figure(novice_kin, trained_kin, trunk_metrics, trunk_event_windows, novice_trial.fs)
    make_knee_flexion_overview_figure(novice_kin, trained_kin, knee_events, novice_trial.fs)
    make_knee_traceability_figure(novice_kin, trained_kin, knee_metrics, knee_events, novice_trial.fs)
    make_orientation_validation_figure(novice_kin, trained_kin, novice_trial.fs)

    # ------------------------------------------------------------------
    # 6. Print summary
    # ------------------------------------------------------------------
    avg_dtw_dist = sum(dtw_dists) / len(dtw_dists)

    print("Analysis complete.")
    print("\nTrunk rotation balance metrics:")
    print(trunk_metrics.to_string(index=False))
    if len(knee_metrics) > 0:
        print("\nKnee-flexion balance metrics:")
        print(knee_metrics.to_string(index=False))
    else:
        print("\nKnee-flexion balance analysis: no windows exceeded 60°.")
    print(f"\nOrientation validation figure: {OUTPUT_DIR / 'orientation_validation.png'}")
    print(f"Trunk traceability figure:     {OUTPUT_DIR / 'trunk_traceability_figure.png'}")
    print(f"Knee flexion overview figure:  {OUTPUT_DIR / 'knee_flexion_balance_figure.png'}")
    print(f"Knee traceability figure:      {OUTPUT_DIR / 'knee_traceability_figure.png'}")

if __name__ == "__main__":
    main()
