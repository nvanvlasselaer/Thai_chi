"""Gravity-referenced quantities: tilt, the vertical rotation rate, sway, foot lift.

Everything here is built on one primitive, :func:`up_in_body`: the upward
direction expressed in a sensor's own body frame at each sample.  Tilt is
anchored by gravity in a 6-axis orientation estimate, so everything derived
from it is free of the yaw drift the missing magnetometer leaves:

* **tilt** -- where the vertical points in the segment frame, so the segment's
  sagittal and frontal tilt with no Euler decomposition;
* **vertical angular velocity** -- the bias-corrected gyroscope projected on the
  vertical, i.e. the turning rate.  Unlike the derivative of Euler yaw it
  needs neither the 0.05 Hz high-pass nor differentiation, which on turns of
  8-14 s had shrunk the excursion by a quarter;
* **linear acceleration** -- the accelerometer rotated into the pelvis-heading
  frame (the smallest rotation that makes the segment's z axis vertical) with
  gravity removed.  In the sensor frame a 1 deg tilt alone puts 0.017 g on a
  "horizontal" axis, and on these recordings the tilt-projected gravity was
  2-30 times the actual sway;
* **leg lift** -- how much higher one ankle is than the other, from the tilt of
  both thighs and shanks.

The body frame is x = mediolateral, y = anteroposterior, z = vertical
(:func:`analysis.orientation.mounting_matrix`).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from analysis.data_io import ACC_COLUMNS, GYRO_COLUMNS, TrialData
from analysis.kinematics import Kinematics
from analysis.orientation import mounting_matrix
from analysis.signals import lowpass_signal

STANDARD_GRAVITY = 9.80665
"""m/s^2 per g."""


def up_in_body(kin: Kinematics, sensor: str) -> np.ndarray:
    """Unit upward vector in the sensor's body frame, one row per sample.

    The body quaternion takes body-frame vectors into the neutral body frame,
    where the calibration found the vertical; its inverse brings that vertical
    back into the frame of each sample.
    """
    rotation = Rotation.from_quat(kin.quaternions[sensor][:, [1, 2, 3, 0]])
    return rotation.inv().apply(kin.calibration.up_ref[sensor])


def body_gyro_dps(kin: Kinematics, trial: TrialData, sensor: str) -> np.ndarray:
    """Bias-corrected gyroscope signal in body axes (deg/s)."""
    gyro = trial.data[sensor][GYRO_COLUMNS].to_numpy()[: len(kin.t)]
    return (gyro - kin.calibration.gyro_bias_dps[sensor]) @ mounting_matrix(sensor).T


def vertical_angular_velocity(kin: Kinematics, trial: TrialData, sensor: str) -> np.ndarray:
    """Rotation rate about the true vertical (deg/s): how fast the segment turns."""
    return np.sum(body_gyro_dps(kin, trial, sensor) * up_in_body(kin, sensor), axis=1)


def tilt_deg(kin: Kinematics, sensor: str) -> tuple[np.ndarray, np.ndarray]:
    """Sagittal and frontal tilt of the segment from the neutral pose (deg).

    Sagittal is rotation about the mediolateral axis (forward-backward lean),
    frontal about the anteroposterior axis (sideways lean), read from where the
    vertical sits in the segment frame.  The sign of each has not been checked
    against a known movement (that needs a functional-calibration trial), so
    only spread and range are interpreted.
    """
    up = up_in_body(kin, sensor)
    ref = kin.calibration.up_ref[sensor]
    # Measured from where the vertical sat in the neutral pose, so the sensor's
    # own mounting tilt (the lumbar sensor sits ~17 deg off vertical on the
    # lordosis) is not counted as lean.
    sagittal = np.degrees(np.arctan2(up[:, 1], up[:, 2]) - np.arctan2(ref[1], ref[2]))
    frontal = np.degrees(np.arctan2(up[:, 0], up[:, 2]) - np.arctan2(ref[0], ref[2]))
    return sagittal, frontal


def linear_acceleration(
    kin: Kinematics, trial: TrialData, sensor: str = "lumbar", cutoff_hz: float = 10.0
) -> np.ndarray:
    """Acceleration without gravity, in the heading frame of the segment (m/s^2).

    Columns are x = mediolateral, y = anteroposterior, z = vertical.  Each sample
    is turned by the smallest rotation that takes the segment's z axis to the
    vertical, which keeps the segment's heading, and the neutral-pose gravity
    reading is subtracted.  No yaw enters, so the heading drift of a 6-axis
    estimate does not either.  Low-passed at ``cutoff_hz``: the raw signal is
    sampled at 370 Hz and sensor noise would otherwise enter the sway.
    """
    n = len(kin.t)
    acc = trial.data[sensor][ACC_COLUMNS].to_numpy()[:n] @ mounting_matrix(sensor).T
    up = up_in_body(kin, sensor)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(up, z)
    sin = np.linalg.norm(axis, axis=1)
    angle = np.arctan2(sin, up @ z)
    scale = np.divide(angle, sin, out=np.ones_like(angle), where=sin > 1e-12)
    levelled = Rotation.from_rotvec(axis * scale[:, None]).apply(acc)
    levelled[:, 2] -= kin.calibration.g_ref[sensor]
    return np.column_stack([
        lowpass_signal(levelled[:, i] * STANDARD_GRAVITY, trial.fs, cutoff_hz=cutoff_hz) for i in range(3)
    ])


def leg_lift_index(kin: Kinematics) -> np.ndarray:
    """How much higher the left ankle is than the right, in thigh lengths.

    Each leg's vertical reach is the cosine of its thigh's tilt from vertical
    plus that of its shank, i.e. the hip-to-ankle height with both segments
    taken as one length (they are within 1 % of each other anthropometrically).
    With both feet on the floor the two reaches are equal; lifting a foot
    shortens that leg's.  Positive means the left foot is up, negative the
    right.  This catches a lift whatever the knee does -- a kick with the knee
    extended included -- and says which leg is lifted, where knee flexion
    cannot tell a lifted leg from a deeply bent stance leg.
    """
    def reach(thigh: str, shank: str) -> np.ndarray:
        return up_in_body(kin, thigh)[:, 2] + up_in_body(kin, shank)[:, 2]

    return reach("rthigh", "rtibia") - reach("lthigh", "ltibia")
