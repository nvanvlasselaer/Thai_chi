"""Checking stage 1: are the sensors where the analysis assumes, and do the kinematics look right?

Three views of one recording, shown on the dashboard's Kinematics check page
and drawn by :mod:`analysis.tools.plot_kinematics`:

* **A table per sensor** (:func:`sensor_check`), which stage 1 also writes to
  each recording's ``sensor_check.csv``: how far the sensor sits from the
  orientation its mounting assumes, whether its accelerometer reads 1 g at
  rest, how still it is while the participant stands still, missing samples,
  for each knee how far its flexion axis lies from the one assumed, and for
  each leg segment whether it moves the one way its joint allows.
* **Joint angles** (:func:`panel_series`), grouped by joint with left and right
  overlaid and each component named by its plane.
* **A stick figure** (:mod:`analysis.skeleton`) at any moment, with a list of
  moments worth comparing with the video (:func:`check_moments`).  Whether the
  sensors are on the segments and sides they are labelled with is judged here,
  by eye: at the highest left-foot lift the figure's left leg must be the one
  up.  (An automatic version was tried -- same-side segments should move
  together -- and separated the arms but not the legs, whose segments move
  differently in Tai Chi.)

Nothing here changes the analysis: a sensor marked ``check`` is for a person
to look at.  Axes: x = right (mediolateral), y = forward (anteroposterior),
z = up.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from analysis import config
from analysis.data_io import ACC_COLUMNS, GYRO_COLUMNS, TrialData
from analysis.gravity import leg_lift_index, tilt_deg
from analysis.kinematics import Kinematics
from analysis.signals import highpass_detrend, lowpass_signal, resample_filtered_full


# ---------------------------------------------------------------------------
# Sensor table
# ---------------------------------------------------------------------------


MOUNTING_LIMIT_DEG = 45.0
"""In the neutral pose a sensor's assumed vertical axis should point roughly up.
Past 45 deg it is nearer another axis than the one its mounting assumes: the
sensor is rotated or turned over on the segment, or the segment was not upright.
(On these recordings: trunk 13-18 deg, on the lordosis and the sternum; feet
~40 deg, on the slope of the instep; the rest 4-27 deg.)"""

GRAVITY_TOLERANCE_G = 0.05
"""A still, calibrated accelerometer reads 1 g; these read 0.97-1.02 g."""

STILLNESS_Z = 3.0
"""A sensor whose stillness is more than this many robust standard deviations
(1.4826 x the median absolute deviation) above the median sensor is moving
while the participant stands still: a loose strap, or that segment moved."""

KNEE_AXIS_LIMIT_DEG = 45.0
"""A knee whose flexion axis is more than this from x is read mostly on another
axis: a thigh or shank sensor is rotated on the leg."""

KNEE_AXIS_NOTE_DEG = 20.0
"""Below the limit but above this, flexion leaks into the ab/adduction and
rotation components: the sensor sits turned on the leg, which the neutral pose
cannot see.  A functional calibration (a pure knee bend) would remove it."""

KNEE_FLEXED_DEG = 30.0
"""The knee's flexion axis is read from the samples flexed past this."""

KNEES = {"left knee": ("lthigh", "ltibia"), "right knee": ("rthigh", "rtibia")}

THIGH_RAISED_DEG = 45.0
"""A thigh raised this far from hanging is past the hip's extension range
(20-30 deg) unless it points forward."""

LEG_DIRECTIONS = {
    # segment: (the segment it moves on, how far it must move, which way along y the joint allows)
    "lthigh": ("lumbar", THIGH_RAISED_DEG, +1),
    "rthigh": ("lumbar", THIGH_RAISED_DEG, +1),
    "ltibia": ("lthigh", KNEE_FLEXED_DEG, -1),
    "rtibia": ("rthigh", KNEE_FLEXED_DEG, -1),
}
"""A raised thigh points forward and a bent knee swings the shank back.  A leg
sensor turned around on its segment, or the left and right ones swapped, makes
it the other way -- which the neutral pose cannot see, since turning a sensor
about the vertical leaves its tilt as it was.  (On these recordings every leg
segment moves the allowed way 100 % of the time.)"""

DOWN = np.array([0.0, 0.0, -1.0])
"""A segment's long axis: in the neutral pose the limbs hang straight down, and
the trunk's axis points down to the pelvis."""


def allowed_direction_share(rotations: dict[str, Rotation], sensor: str, fs: float) -> float:
    """Of the time a leg segment is clearly raised or bent on its parent, the share it points the way the joint allows."""
    parent, threshold_deg, sign = LEG_DIRECTIONS[sensor]
    axis = (rotations[parent].inv() * rotations[sensor]).apply(DOWN)
    moved = np.degrees(np.arccos(np.clip(-axis[:, 2], -1.0, 1.0))) > threshold_deg
    if moved.sum() < fs:  # less than a second: no direction to judge
        return float("nan")
    return float(np.mean(sign * axis[moved, 1] > 0))


def knee_axis_offset(kin: Kinematics, thigh: str, shank: str, fs: float) -> tuple[float, float]:
    """``(angle of the knee's flexion axis from x, largest knee rotation)``, in degrees.

    The shank's rotation relative to the thigh, as a rotation vector in the
    thigh frame, points along the axis it turned about.  The principal
    direction of those vectors while the knee is clearly flexed is the knee's
    functional flexion axis.
    """
    thigh_rot = Rotation.from_quat(kin.quaternions[thigh][:, [1, 2, 3, 0]])
    shank_rot = Rotation.from_quat(kin.quaternions[shank][:, [1, 2, 3, 0]])
    rotvec = (thigh_rot.inv() * shank_rot).as_rotvec()
    angle = np.linalg.norm(rotvec, axis=1)
    flexed = angle > np.radians(KNEE_FLEXED_DEG)
    largest = float(np.degrees(angle.max())) if len(angle) else float("nan")
    if flexed.sum() < fs:  # less than a second of clear flexion: no axis to read
        return float("nan"), largest
    axis = np.linalg.eigh(rotvec[flexed].T @ rotvec[flexed])[1][:, -1]
    return float(np.degrees(np.arccos(min(1.0, abs(axis[0]))))), largest


def sensor_check(kin: Kinematics, trial: TrialData, dedrifted: dict[str, Rotation] | None = None) -> pd.DataFrame:
    """One row per sensor: where it sits, how it reads at rest, and a status.

    ``dedrifted`` is :func:`dedrifted_rotations` of the recording, computed here
    when not given.
    """
    calibration, fs, n = kin.calibration, trial.fs, len(kin.t)
    if dedrifted is None:
        dedrifted = dedrifted_rotations(kin, fs)
    still = {}
    for sensor in trial.sensors:
        gyro = trial.data[sensor][GYRO_COLUMNS].to_numpy()[:n] - calibration.gyro_bias_dps[sensor]
        # The stiller of the quiet spans: at the end of a recording feet shuffle.
        still[sensor] = min(float(np.sqrt(np.nanmean(np.sum(gyro[a:b] ** 2, axis=1))))
                            for a, b in calibration.quiet_spans)
    levels = np.array(list(still.values()))
    median = float(np.median(levels))
    spread = max(1.4826 * float(np.median(np.abs(levels - median))), 1e-9)

    knees = {shank: (joint, *knee_axis_offset(kin, thigh, shank, fs)) for joint, (thigh, shank) in KNEES.items()
             if thigh in kin.quaternions and shank in kin.quaternions}
    directions = {sensor: allowed_direction_share(dedrifted, sensor, fs) for sensor, (parent, _, _)
                  in LEG_DIRECTIONS.items() if sensor in dedrifted and parent in dedrifted}

    order = {sensor: i for i, sensor in enumerate(config.SENSOR_MAP)}
    rows = []
    for sensor in sorted(trial.sensors, key=lambda s: order.get(s, len(order))):
        tilt = float(np.degrees(np.arccos(np.clip(calibration.up_ref[sensor][2], -1.0, 1.0))))
        gravity = calibration.g_ref[sensor]
        z = (still[sensor] - median) / spread
        missing = int(trial.data[sensor][ACC_COLUMNS + GYRO_COLUMNS].iloc[:n].isna().sum().sum())
        checks, notes = [], []
        if tilt > MOUNTING_LIMIT_DEG:
            checks.append(f"sits {tilt:.0f} deg from the orientation its mounting assumes: rotated or turned "
                          "over on the segment, or the segment was not upright in the neutral pose")
        if abs(gravity - 1.0) > GRAVITY_TOLERANCE_G:
            checks.append(f"reads {gravity:.2f} g at rest")
        if z > STILLNESS_Z:
            checks.append(f"moves {still[sensor] / median:.1f}x as much as the typical sensor while the "
                          "participant stands still: loose, or that segment moved")
        if missing:
            checks.append(f"{missing} missing values")
        joint, offset, largest = knees.get(sensor, (None, float("nan"), float("nan")))
        if joint and np.isfinite(offset):
            if offset > KNEE_AXIS_LIMIT_DEG:
                checks.append(f"the {joint} flexes {offset:.0f} deg away from the assumed axis: a thigh or "
                              "shank sensor is rotated on the leg")
            elif offset > KNEE_AXIS_NOTE_DEG:
                notes.append(f"{joint} flexion axis {offset:.0f} deg from x: part of the flexion shows in "
                             "the other components (a functional calibration would remove it)")
        elif joint:
            notes.append(f"the {joint} never flexed past {KNEE_FLEXED_DEG:.0f} deg for a second: axis not read")
        share = directions.get(sensor, float("nan"))
        if np.isfinite(share) and share < 0.5:
            if LEG_DIRECTIONS[sensor][2] > 0:
                checks.append(f"raised, it points backward {1 - share:.0%} of the time, which the hip cannot do: "
                              "turned around on the thigh, or the left and right thigh sensors swapped")
            else:
                checks.append(f"bent, the knee swings the shank forward {1 - share:.0%} of the time, which it "
                              "cannot do: the thigh or shank sensor turned around, or left and right swapped")
        rows.append({
            "sensor": sensor,
            "segment": config.SENSOR_MAP.get(sensor, "unmapped"),
            "mounting_tilt_deg": round(tilt, 1),
            "gravity_at_rest_g": round(gravity, 3),
            "gyro_bias_dps": round(float(np.linalg.norm(calibration.gyro_bias_dps[sensor])), 2),
            "still_rms_dps": round(still[sensor], 2),
            "still_robust_z": round(z, 1),
            "knee": joint or "",
            "knee_flexion_axis_offset_deg": round(offset, 1) if np.isfinite(offset) else np.nan,
            "allowed_direction_pct": round(100 * share, 1) if np.isfinite(share) else np.nan,
            "missing_values": missing,
            "status": "check" if checks else "ok",
            "notes": "; ".join(checks + notes),
        })
    return pd.DataFrame(rows)


def summary(check: pd.DataFrame) -> str:
    """One line for a status card: which sensors, if any, need a look."""
    flagged = check[check.status == "check"]
    if len(flagged) == 0:
        return f"all {len(check)} sensors look as assumed"
    return f"{len(flagged)} of {len(check)} sensors to check: " + ", ".join(flagged.sensor)


def summary_of(path: Path) -> str | None:
    """:func:`summary` of a written ``sensor_check.csv``, or None if there is none."""
    try:
        return summary(pd.read_csv(path))
    except (OSError, ValueError, KeyError):
        return None


# ---------------------------------------------------------------------------
# Joint angles
# ---------------------------------------------------------------------------


LEFT, RIGHT = "Left", "Right"


def _hp(signal: np.ndarray, fs: float) -> np.ndarray:
    return highpass_detrend(signal, fs, cutoff_hz=0.05)


def axis_angle_deg(rotations: dict[str, Rotation], proximal: str, distal: str) -> np.ndarray:
    """Angle between two segments' long axes (deg), 0 as in the neutral pose.

    How far the joint between them is bent, whichever way: it ignores rotation
    about either segment's long axis, and with it forearm pronation and the
    humerus's axial rotation.
    """
    a, b = rotations[proximal].apply(DOWN), rotations[distal].apply(DOWN)
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=1), -1.0, 1.0)))


PANELS = (
    ("Trunk on pelvis (deg)", (
        ("flexion-extension (x)", None, lambda k, fs, r: k.trunk_rel_euler_deg[:, 0]),
        ("lateral bending (y)", None, lambda k, fs, r: k.trunk_rel_euler_deg[:, 1]),
        ("axial rotation (z, de-drifted)", None, lambda k, fs, r: _hp(k.trunk_rel_euler_deg[:, 2], fs)),
    )),
    ("Pelvis tilt against gravity (deg)", (
        ("sagittal", None, lambda k, fs, r: tilt_deg(k, "lumbar")[0]),
        ("frontal", None, lambda k, fs, r: tilt_deg(k, "lumbar")[1]),
    )),
    ("Hip flexion-extension (x, deg; flexion > 0)", (
        ("left", LEFT, lambda k, fs, r: k.left_hip_deg[:, 0]),
        ("right", RIGHT, lambda k, fs, r: k.right_hip_deg[:, 0]),
    )),
    ("Hip ab/adduction (y, deg; abduction > 0, right side mirrored)", (
        ("left", LEFT, lambda k, fs, r: k.left_hip_deg[:, 1]),
        ("right", RIGHT, lambda k, fs, r: -k.right_hip_deg[:, 1]),
    )),
    ("Knee flexion-extension (x, deg; flexion < 0)", (
        ("left", LEFT, lambda k, fs, r: k.left_knee_deg[:, 0]),
        ("right", RIGHT, lambda k, fs, r: k.right_knee_deg[:, 0]),
    )),
    ("Ankle (x, deg; dorsiflexion > 0)", (
        ("left", LEFT, lambda k, fs, r: k.left_ankle_deg[:, 0]),
        ("right", RIGHT, lambda k, fs, r: k.right_ankle_deg[:, 0]),
    )),
    ("Leg lift (thigh lengths, + left foot up)", (
        ("lift index", None, lambda k, fs, r: leg_lift_index(k)),
    )),
    ("Shoulder elevation: upper arm from the trunk's axis (deg)", (
        ("left", LEFT, lambda k, fs, r: axis_angle_deg(r, "chestbone", "lhumerus")),
        ("right", RIGHT, lambda k, fs, r: axis_angle_deg(r, "chestbone", "rhumerus")),
    )),
    ("Elbow bend: angle between upper arm and forearm (deg)", (
        ("left", LEFT, lambda k, fs, r: axis_angle_deg(r, "lhumerus", "lulna")),
        ("right", RIGHT, lambda k, fs, r: axis_angle_deg(r, "rhumerus", "rulna")),
    )),
)
"""The panels of the joint-angle view: a title, and per trace a label, a side
and the signal, from the kinematics, the sampling rate and
:func:`dedrifted_rotations`.

Leg and trunk joint angles are the analysis' own: the distal segment's Euler
angles relative to the proximal one, without the axial (z) component, where
the heading drift of two sensors lands.  For the legs the signs are fixed by
anatomy: in both recordings the hips reach 86-111 deg of positive x and the
knees 85-116 deg of negative x, which only flexion can, so x points right and
y forward (:data:`LEG_DIRECTIONS` checks the same per sensor).  The arms are
shown as angles between long axes instead: the elbow's Euler angles pass
through their singularity in Tai Chi's arm positions (the analysis' elbow
angle runs to -1000 deg), and the upper-arm sensor, on soft tissue, misses
much of the humerus's axial rotation, so no ordering of the angles gives a
clean elbow flexion."""


def panel_series(
    kin: Kinematics, fs: float, target_fs: float = 50.0, dedrifted: dict[str, Rotation] | None = None
) -> tuple[np.ndarray, list[dict]]:
    """Every panel's traces resampled to ``target_fs``: ``(time, [{title, traces: [(label, side, values)]}])``."""
    if dedrifted is None:
        dedrifted = dedrifted_rotations(kin, fs)
    time = None
    panels = []
    for title, traces in PANELS:
        resampled = []
        for label, side, signal in traces:
            time, values = resample_filtered_full(kin.t, signal(kin, fs, dedrifted), fs, target_fs)
            resampled.append((label, side, values))
        panels.append({"title": title, "traces": resampled})
    return time, panels


# ---------------------------------------------------------------------------
# Moments to compare with the video
# ---------------------------------------------------------------------------


def check_moments(kin: Kinematics, fs: float) -> list[tuple[str, float]]:
    """Moments where a misplaced sensor shows: ``(what should be seen, time_s)``.

    At the neutral pose the figure must stand upright; at each leg's deepest
    knee bend and highest foot lift that leg, and only that one, must bend or
    rise; at the largest trunk turns the chest must turn on the pelvis.
    """
    start, end = kin.calibration.neutral
    moments = [("neutral pose: standing upright", 0.5 * (start + end) / fs)]
    lift = lowpass_signal(leg_lift_index(kin), fs, cutoff_hz=2.0)
    for side, angles in ((LEFT, kin.left_knee_deg), (RIGHT, kin.right_knee_deg)):
        knee = np.abs(lowpass_signal(angles[:, 0], fs, cutoff_hz=6.0))
        moments.append((f"deepest {side.lower()} knee bend: the {side.lower()} knee bent",
                        float(np.argmax(knee)) / fs))
    moments.append(("highest left-foot lift: the left leg up", float(np.argmax(lift)) / fs))
    moments.append(("highest right-foot lift: the right leg up", float(np.argmin(lift)) / fs))
    twist = lowpass_signal(_hp(kin.trunk_rel_euler_deg[:, 2], fs), fs, cutoff_hz=2.0)
    moments.append(("largest trunk turn one way", float(np.argmax(twist)) / fs))
    moments.append(("largest trunk turn the other way", float(np.argmin(twist)) / fs))
    return moments


# ---------------------------------------------------------------------------
# Stick figure
# ---------------------------------------------------------------------------


DEDRIFT_HZ = 0.05
"""Heading changes slower than this (periods over 20 s) are taken for drift, as
in every yaw channel of the analysis."""


def rotations(kin: Kinematics) -> dict[str, Rotation]:
    """Each sensor's body-frame orientation over the recording."""
    return {sensor: Rotation.from_quat(q[:, [1, 2, 3, 0]]) for sensor, q in kin.quaternions.items()}


def heading_phasor(kin: Kinematics, sensor: str) -> np.ndarray:
    """The segment's turn about the true vertical since the neutral pose, as ``w * exp(i * heading)``.

    Of the body quaternion split into a tilt and a turn about the vertical,
    the turn: ``(q_w + i q_up)^2`` is ``cos^2(tilt / 2) * exp(i * turn)``.  The
    weight fades to zero where the heading has no meaning -- a segment turned
    upside down, as a hand held fingers-up is -- so, unlike an Euler angle or
    an unwrapped twist, it never jumps.
    """
    q = kin.quaternions[sensor]
    return (q[:, 0] + 1j * (q[:, 1:] @ kin.calibration.up_ref[sensor])) ** 2


def dedrifted_rotations(kin: Kinematics, fs: float) -> dict[str, Rotation]:
    """Each sensor's orientation with the slow part of its heading turned back.

    Tilt is anchored by gravity and exact; heading is not, and each sensor's
    drifts on its own -- over these four-minute recordings the upper-body
    sensors' by up to 115 deg, two of them ending up 130 deg apart -- so
    without this the stick figure twists apart and the angle between two
    segments drifts with it.  The slow heading is the direction of the
    low-passed :func:`heading_phasor` (below :data:`DEDRIFT_HZ`); it is turned
    back about the vertical, which leaves the tilt as it was.
    """
    result = {}
    for sensor, rotation in rotations(kin).items():
        phasor = heading_phasor(kin, sensor)
        slow = np.angle(lowpass_signal(phasor.real, fs, cutoff_hz=DEDRIFT_HZ)
                        + 1j * lowpass_signal(phasor.imag, fs, cutoff_hz=DEDRIFT_HZ))
        result[sensor] = Rotation.from_rotvec(-slow[:, None] * kin.calibration.up_ref[sensor]) * rotation
    return result


def segment_orientations(
    kin: Kinematics, fs: float, target_fs: float = 50.0, dedrifted: dict[str, Rotation] | None = None
) -> tuple[np.ndarray, dict[str, Rotation], dict[str, Rotation]]:
    """Each sensor's orientation at ``target_fs``, as measured and de-drifted: ``(time, raw, dedrifted)``.

    The grid is that of :func:`analysis.signals.resample_filtered_full`, so it
    lines up with :func:`panel_series`.
    """
    if dedrifted is None:
        dedrifted = dedrifted_rotations(kin, fs)
    count = max(2, int(round((kin.t[-1] - kin.t[0]) * target_fs)) + 1)
    time = kin.t[0] + np.arange(count) / target_fs
    time = time[time <= kin.t[-1]]
    index = np.clip(np.searchsorted(kin.t, time), 0, len(kin.t) - 1)
    raw = {sensor: rotation[index] for sensor, rotation in rotations(kin).items()}
    return time, raw, {sensor: rotation[index] for sensor, rotation in dedrifted.items()}


def rotations_at(orientations: dict[str, Rotation], index: int) -> dict[str, Rotation]:
    """The sensors' rotations at one sample of :func:`segment_orientations`' grid."""
    return {sensor: rotation[index] for sensor, rotation in orientations.items()}
