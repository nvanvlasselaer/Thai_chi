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
    mounting_matrix,
    quat_inverse,
    quat_multiply,
    quat_to_euler_deg,
)
from analysis.recordings import Recording, Selection, active


@dataclass
class Calibration:
    """What a recording's own still moments say about its sensors.

    Worked out from the raw data every time a recording is loaded -- it takes a
    fraction of a second -- so it needs no cache and is always consistent with
    the file it came from.
    """

    neutral: tuple[int, int]
    """The neutral-pose window every orientation is referenced to (sample indices)."""
    quiet_spans: list[tuple[int, int]]
    """The stillest 10 s near the start and near the end, where the gyroscope bias is read."""
    gyro_bias_dps: dict[str, np.ndarray]
    """Per sensor, the gyroscope reading while standing still, subtracted from every angular speed."""
    bias_drift_dps: dict[str, float]
    """Per sensor, how far the start and end estimates of the bias are apart."""
    bias_scatter_dps: dict[str, float]
    """Per sensor, the scatter of 1 s medians within the quiet spans: what a difference is judged against."""
    up_ref: dict[str, np.ndarray]
    """Per sensor, the upward direction in its neutral body frame (unit vector)."""
    g_ref: dict[str, float]
    """Per sensor, the size of the accelerometer reading in the neutral pose (g)."""

    def warnings(self, sensors: tuple[str, ...] = ("lumbar", "chestbone")) -> list[str]:
        """Sensors whose start and end bias disagree by more than the scatter allows."""
        return [
            f"{sensor}: gyroscope bias differs by {self.bias_drift_dps[sensor]:.2f} deg/s between the "
            f"start and the end of the recording (scatter {self.bias_scatter_dps[sensor]:.2f} deg/s); "
            "it may not start and end with the participant standing still"
            for sensor in sensors
            if sensor in self.bias_drift_dps and self.bias_drift_dps[sensor] > 2.0 * self.bias_scatter_dps[sensor]
        ]


@dataclass
class Kinematics:
    """Orientation-derived kinematics of one recording, at the source sampling rate.

    Per sensor: body-frame quaternions (wxyz), xyz Euler angles and the
    gyroscope magnitude (deg/s).  ``omega_mag`` has the gyroscope bias removed
    (:class:`Calibration`); ``omega_raw_mag`` keeps it, for the v1 detectors
    only, whose windows were found on it.  ``trunk_rel_euler_deg`` is the chest
    relative to the pelvis (lumbar sensor); each joint field is its distal
    segment relative to its proximal one, per :data:`JOINT_SENSOR_PAIRS`.

    The body frame is x = mediolateral (the flexion/extension axis), y =
    anteroposterior, z = vertical: see :func:`analysis.orientation.mounting_matrix`.
    """

    t: np.ndarray
    quaternions: dict[str, np.ndarray]
    eulers_deg: dict[str, np.ndarray]
    omega_mag: dict[str, np.ndarray]
    omega_raw_mag: dict[str, np.ndarray]
    calibration: Calibration
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


QUIET_SPAN_S = 10.0
"""Length of each span the gyroscope bias is read from.  The 2 s neutral window
is too short: the sternum sensor moves with breathing (0.13-0.33 Hz here), and
a 2 s median caught 2.9 deg/s of it on the chest's vertical axis.  10 s spans
one to three breaths."""

QUIET_SEARCH_S = 60.0
"""How far into the start and the end of a recording the quiet span is sought."""


def quiet_standing_spans(trial: TrialData, n: int) -> list[tuple[int, int]]:
    """The stillest :data:`QUIET_SPAN_S` near the start and near the end of a recording.

    Stillness is the rolling mean of the raw angular power summed over every
    sensor.  A constant bias only adds a constant to that, so the quietest span
    is found correctly before the bias is known.  Both recordings here stand
    still for well over 10 s before and after the form.
    """
    fs = trial.fs
    width = int(round(QUIET_SPAN_S * fs))
    if n < 2 * width:
        return [find_neutral_pose_window(trial)]
    power = np.zeros(n)
    for sensor in trial.sensors:
        gyro = trial.data[sensor][GYRO_COLUMNS].to_numpy()[:n]
        power += np.nansum(gyro ** 2, axis=1)
    rolling = pd.Series(power).rolling(width).mean().to_numpy()

    def stillest(lo: int, hi: int) -> tuple[int, int]:
        ends = np.arange(lo + width - 1, hi)
        end = int(ends[np.nanargmin(rolling[ends])])
        return end - width + 1, end + 1

    search = min(n, int(round(QUIET_SEARCH_S * fs)))
    first = stillest(0, max(search, width))
    last = stillest(min(n - search, n - width), n)
    return [first] if last[0] < first[1] else [first, last]


def calibrate(trial: TrialData, n: int) -> Calibration:
    """Gyroscope bias from the quiet spans, and each sensor's vertical in the neutral pose."""
    fs = trial.fs
    neutral = find_neutral_pose_window(trial)
    spans = quiet_standing_spans(trial, n)
    block = max(1, int(round(fs)))

    bias, drift, scatter, up_ref, g_ref = {}, {}, {}, {}, {}
    for sensor in trial.sensors:
        gyro = trial.data[sensor][GYRO_COLUMNS].to_numpy()[:n]
        pieces = [gyro[start:end] for start, end in spans]
        bias[sensor] = np.nanmedian(np.vstack(pieces), axis=0)
        per_span = [np.nanmedian(piece, axis=0) for piece in pieces]
        drift[sensor] = float(np.linalg.norm(per_span[-1] - per_span[0]))
        medians = np.vstack([
            np.nanmedian(piece[i:i + block], axis=0)
            for piece in pieces for i in range(0, len(piece) - block + 1, block)
        ])
        scatter[sensor] = float(np.linalg.norm(np.nanstd(medians, axis=0)))

        acc = trial.data[sensor][ACC_COLUMNS].to_numpy()[:n] @ mounting_matrix(sensor).T
        mean_acc = np.nanmean(acc[neutral[0]:neutral[1]], axis=0)
        g_ref[sensor] = float(np.linalg.norm(mean_acc))
        up_ref[sensor] = mean_acc / g_ref[sensor]

    return Calibration(neutral=neutral, quiet_spans=spans, gyro_bias_dps=bias, bias_drift_dps=drift,
                       bias_scatter_dps=scatter, up_ref=up_ref, g_ref=g_ref)


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
    gyros = {}

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
        gyros[sensor] = gyro

    return kinematics_from_quaternions(t, quats, gyros, calibrate(trial, n))


# ---------------------------------------------------------------------------
# From body-frame quaternions
# ---------------------------------------------------------------------------


def relative_euler(quats: dict[str, np.ndarray], proximal: str, distal: str) -> np.ndarray:
    """Euler angles of the distal segment expressed in the proximal frame."""
    return quat_to_euler_deg(quat_multiply(quat_inverse(quats[proximal]), quats[distal]))


def kinematics_from_quaternions(
    t: np.ndarray, quats: dict[str, np.ndarray], gyro_dps: dict[str, np.ndarray], calibration: Calibration
) -> Kinematics:
    """Derive every segment and joint angle from body-frame quaternions.

    ``gyro_dps`` is the raw gyroscope signal per sensor; its magnitude is taken
    with and without the calibration's bias.
    """
    return Kinematics(
        t=t,
        quaternions=quats,
        eulers_deg={sensor: quat_to_euler_deg(q) for sensor, q in quats.items()},
        omega_mag={
            sensor: np.linalg.norm(gyro - calibration.gyro_bias_dps[sensor], axis=1)
            for sensor, gyro in gyro_dps.items()
        },
        omega_raw_mag={sensor: np.linalg.norm(gyro, axis=1) for sensor, gyro in gyro_dps.items()},
        calibration=calibration,
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

    gyros = {sensor: trial.data[sensor][GYRO_COLUMNS].to_numpy()[:n] for sensor in quats}
    kin = kinematics_from_quaternions(time_s, quats, gyros, calibrate(trial, n))
    return LoadedTrial(label=label, kin=kin, trial=trial, fs=trial.fs, n_samples=n, recording=recording)


def load_all_trials(selection: Selection | None = None) -> dict[str, LoadedTrial]:
    """Every selected recording, keyed by role (the active selection by default)."""
    selection = selection or active()
    problem = selection.problem()
    if problem:
        raise ValueError(f"Cannot load the recordings: {problem}.")
    return {label: load_trial(recording, label) for label, recording in selection.recordings().items()}
