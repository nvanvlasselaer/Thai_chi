"""Orientation estimation: quaternion algebra, the Madgwick filter, sensor mounting.

Quaternions are scalar-first ``(w, x, y, z)`` throughout.  scipy's
:class:`~scipy.spatial.transform.Rotation` is scalar-last, so the conversion
happens at the boundary, inside the functions that call it.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Quaternion algebra
# ---------------------------------------------------------------------------


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


def mean_quaternion(quats: np.ndarray) -> np.ndarray:
    q = np.mean(quats, axis=0)
    return q / np.linalg.norm(q)


# ---------------------------------------------------------------------------
# Madgwick filter
# ---------------------------------------------------------------------------


def initial_quaternion_from_acc(acc: np.ndarray, fs: float) -> np.ndarray:
    """Tilt-only starting orientation from the median accelerometer reading of the first second."""
    a = acc[np.all(np.isfinite(acc), axis=1)]
    if len(a) == 0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    ax, ay, az = np.median(a[: min(len(a), int(fs))], axis=0)
    norm = math.sqrt(ax * ax + ay * ay + az * az)
    if norm == 0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    ax, ay, az = ax / norm, ay / norm, az / norm
    roll = math.atan2(ay, az)
    pitch = math.atan2(-ax, math.sqrt(ay * ay + az * az))
    quat_xyzw = Rotation.from_euler("xyz", [roll, pitch, 0.0]).as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


def madgwick_imu(acc_g: np.ndarray, gyro_dps: np.ndarray, fs: float, beta: float = 0.05) -> np.ndarray:
    """6-axis (accelerometer + gyroscope) Madgwick filter: one unit quaternion per sample.

    The gyroscope bias is taken as the median over the first second.  With no
    magnetometer, heading (yaw) is unobservable and drifts; tilt is corrected
    towards gravity with gain ``beta``.  A pure-Python per-sample loop, and the
    only slow step of the pipeline -- see :mod:`analysis.data_io` for the cache.
    """
    gyro = np.deg2rad(gyro_dps.copy())
    gyro_bias = np.nanmedian(gyro[: max(1, int(fs))], axis=0)
    gyro = gyro - gyro_bias

    q = initial_quaternion_from_acc(acc_g, fs)
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


# ---------------------------------------------------------------------------
# Sensor mounting
# ---------------------------------------------------------------------------


def mounting_matrix(sensor_name: str) -> np.ndarray:
    """Rotation from a sensor's own axes to the common body frame.

    The sensors are mounted in different orientations on the body, so each is
    rotated into one frame before anything is compared: x = forward,
    y = right, z = downward.
    """
    if sensor_name in ["chestbone", "lulna", "rulna"]:
        return np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    if sensor_name == "lumbar":
        return np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])
    if sensor_name in ["lhumerus", "lhand", "lthigh", "ltibia"]:
        return np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])
    if sensor_name in ["rhumerus", "rhand", "rthigh", "rtibia"]:
        return np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    # The foot sensors are already aligned with the body frame; an unrecognised
    # sensor is left as it is.
    return np.eye(3)


def get_nominal_sensor_quaternion(sensor_name: str) -> np.ndarray:
    """:func:`mounting_matrix` as a wxyz quaternion."""
    q = Rotation.from_matrix(mounting_matrix(sensor_name)).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])
