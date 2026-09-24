"""Segment and joint kinematics of one recording.

:func:`compute_kinematics` runs the full chain from raw IMU data: Madgwick
orientation, calibration to the neutral pose, and alignment of each sensor to
the body frame.  :func:`kinematics_from_quaternions` then derives every segment
and joint angle from the quaternions.  That second step takes well under a
second, which is what lets :func:`load_trial` rebuild a recording from the
orientation cache instead of re-running the filter for minutes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analysis.data_io import ACC_COLUMNS, GYRO_COLUMNS, TrialData, load_orientation_npz, parse_IMU_csv
from analysis.orientation import (
    get_nominal_sensor_quaternion,
    madgwick_imu,
    mean_quaternion,
    quat_inverse,
    quat_multiply,
    quat_to_euler_deg,
)
from analysis.recordings import Recording, Selection, active


@dataclass
class Kinematics:
    """Orientation-derived kinematics of one recording, at the source sampling rate.

    Per sensor: body-frame quaternions (wxyz), xyz Euler angles and the
    gyroscope magnitude (deg/s).  ``trunk_rel_euler_deg`` is the chest relative
    to the pelvis (lumbar sensor); each joint field is its distal segment
    relative to its proximal one, per :data:`JOINT_SENSOR_PAIRS`.
    """

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


# Joint angles are relative orientations between a proximal and a distal sensor.
# Every joint field of Kinematics is filled from this table.
JOINT_SENSOR_PAIRS = {
    "left_hip_deg": ("lumbar", "lthigh"),
    "right_hip_deg": ("lumbar", "rthigh"),
    "left_knee_deg": ("lthigh", "ltibia"),
    "right_knee_deg": ("rthigh", "rtibia"),
    "left_ankle_deg": ("ltibia", "lfoot"),
    "right_ankle_deg": ("rtibia", "rfoot"),
    "left_shoulder_deg": ("chestbone", "lhumerus"),
    "right_shoulder_deg": ("chestbone", "rhumerus"),
    "left_elbow_deg": ("lhumerus", "lulna"),
    "right_elbow_deg": ("rhumerus", "rulna"),
}


# ---------------------------------------------------------------------------
# From raw IMU data
# ---------------------------------------------------------------------------


def find_neutral_pose_window(trial: TrialData, min_duration_s: float = 2.0) -> tuple[int, int]:
    """Locate the still standing pose held after the sync movement.

    After the sync signal, the participant stands still for a moment to establish
    a neutral pose.  This function finds a window of at least 2 s where the trunk
    is relatively still, which is used to compute the neutral orientation of each
    sensor.  The search starts after the first peak in lumbar acceleration and
    looks for a window with minimal lumbar angular velocity.  If no suitable
    window is found, the function returns the full signal range.
    """
    lumbar_acc_x = trial.data["lumbar"]["acc_x_g"].to_numpy()
    lumbar_gyro = trial.data["lumbar"][GYRO_COLUMNS].to_numpy()
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


def sensor_signals(trial: TrialData) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Accelerometer (g) and gyroscope (deg/s) samples per sensor, cut to a common length.

    These are the inputs of the orientation filter, one independent run per
    sensor, which is what lets :mod:`analysis.pipeline` run them in parallel.
    """
    n = min(len(df) for df in trial.data.values())
    signals = {}
    for sensor in trial.sensors:
        df = trial.data[sensor].iloc[:n]
        signals[sensor] = (df[ACC_COLUMNS].to_numpy(), df[GYRO_COLUMNS].to_numpy())
    return signals


def compute_kinematics(trial: TrialData, raw_quats: dict[str, np.ndarray] | None = None) -> Kinematics:
    """Run the orientation filter on every sensor and derive all joint angles.

    Each sensor's Madgwick output is re-referenced to the neutral pose, then
    rotated from the sensor's own axes into the body frame:
    ``q_body = q_mount * q_neutral^-1 * q_raw * q_mount^-1``.

    ``raw_quats`` supplies the filter output per sensor when it has already
    been computed from :func:`sensor_signals`; otherwise it is computed here,
    one sensor after another.
    """
    n = min(len(df) for df in trial.data.values())
    t = trial.data[trial.sensors[0]]["time"].to_numpy()[:n]
    quats = {}
    omega_mag = {}

    neutral_start, neutral_end = find_neutral_pose_window(trial)

    for sensor, (acc, gyro) in sensor_signals(trial).items():
        if raw_quats is None:
            sensor_quats = madgwick_imu(acc, gyro, trial.fs)
        else:
            sensor_quats = raw_quats[sensor]
        q_offset = mean_quaternion(sensor_quats[neutral_start:neutral_end])
        q_nom = get_nominal_sensor_quaternion(sensor)

        q_O_inv = quat_inverse(q_offset)
        q_nom_inv = quat_inverse(q_nom)

        step1 = quat_multiply(q_nom, q_O_inv)
        step2 = quat_multiply(step1, sensor_quats)
        quats[sensor] = quat_multiply(step2, q_nom_inv)
        omega_mag[sensor] = np.linalg.norm(gyro, axis=1)

    return kinematics_from_quaternions(t, quats, omega_mag)


# ---------------------------------------------------------------------------
# From body-frame quaternions
# ---------------------------------------------------------------------------


def relative_euler(quats: dict[str, np.ndarray], proximal: str, distal: str) -> np.ndarray:
    """Euler angles of the distal segment expressed in the proximal frame."""
    return quat_to_euler_deg(quat_multiply(quat_inverse(quats[proximal]), quats[distal]))


def kinematics_from_quaternions(
    t: np.ndarray, quats: dict[str, np.ndarray], omega_mag: dict[str, np.ndarray]
) -> Kinematics:
    """Derive every segment and joint angle from body-frame quaternions."""
    return Kinematics(
        t=t,
        quaternions=quats,
        eulers_deg={sensor: quat_to_euler_deg(q) for sensor, q in quats.items()},
        omega_mag=omega_mag,
        trunk_rel_euler_deg=relative_euler(quats, "lumbar", "chestbone"),
        **{
            field: relative_euler(quats, proximal, distal)
            for field, (proximal, distal) in JOINT_SENSOR_PAIRS.items()
        },
    )


# ---------------------------------------------------------------------------
# Loading a recording
# ---------------------------------------------------------------------------


@dataclass
class LoadedTrial:
    """One recording loaded in one role, parsed and with its kinematics."""

    label: str
    """The role: ``"Novice"`` or ``"Trained"``."""
    kin: Kinematics
    trial: TrialData
    fs: float
    n_samples: int
    recording: Recording

    @property
    def duration_s(self) -> float:
        return float(self.kin.t[-1])


def load_trial(recording: Recording, label: str) -> LoadedTrial:
    """Load one recording in a role, rebuilding its kinematics from the orientation cache.

    Parsing the raw CSV takes ~1.6 s and rederiving every angle from the cache
    ~0.35 s, against much longer for the Madgwick filter, which is why the
    cache (written by :func:`analysis.pipeline.preprocess`) is required.
    """
    if not recording.path.exists():
        raise FileNotFoundError(f"Raw recording not found: {recording.path}")
    if not recording.orientation_path.exists():
        raise FileNotFoundError(
            f"{recording.name} has not been preprocessed yet (no {recording.orientation_path}). "
            "Run the pipeline first."
        )

    trial = parse_IMU_csv(recording.path, label)
    n = min(len(df) for df in trial.data.values())

    time_s, quats = load_orientation_npz(recording.orientation_path)
    quats = {sensor: q[:n] for sensor, q in quats.items()}

    if len(time_s) != n:
        raise ValueError(
            f"{recording.orientation_path} holds {len(time_s)} samples but {recording.name} parses "
            f"to {n}. The orientation cache is out of sync with the recording: preprocess it again."
        )

    omega = {
        sensor: np.linalg.norm(trial.data[sensor][GYRO_COLUMNS].to_numpy()[:n], axis=1)
        for sensor in quats
    }
    kin = kinematics_from_quaternions(time_s, quats, omega)
    return LoadedTrial(label=label, kin=kin, trial=trial, fs=trial.fs, n_samples=n, recording=recording)


def load_all_trials(selection: Selection | None = None) -> dict[str, LoadedTrial]:
    """Every selected recording, keyed by role (the active selection by default)."""
    selection = selection or active()
    problem = selection.problem()
    if problem:
        raise ValueError(f"Cannot load the recordings: {problem}.")
    return {label: load_trial(recording, label) for label, recording in selection.recordings().items()}
