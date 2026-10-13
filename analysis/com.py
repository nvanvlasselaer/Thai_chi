"""The whole-body centre of mass, from the orientation of every segment.

Shifting the weight is half of Tai Chi, and the lumbar accelerometer cannot
see it.  A weight shift of 10 cm over 4-8 s is a movement at 0.06-0.12 Hz,
and the 6-axis orientation filter takes a horizontal acceleration that slow
for a tilt: a pure 10 cm translation fed through the pipeline's own filter
and gravity removal comes out with a gain of 0.003 at 0.1 Hz, 0.01 at 0.2 Hz
and 0.05 at 0.3 Hz, against 0.90 at 0.5 Hz (:mod:`analysis.tools.method_checks`
re-runs this; at half or twice the sensor noise the gain stays below 0.1 up
to 0.3 Hz).  The sway columns are therefore what moves faster than about
0.4 Hz -- corrections, not the weight shift itself.

What the filter does get right is each segment's tilt: a slow horizontal
acceleration tilts the apparent vertical by a / g, about 0.5 deg here.  So
the centre of mass is built the way a motion-capture system builds it, from
the segments: each one's long axis from its orientation, chained from the
hips to the feet and up the trunk, with Winter's (2009) table of segment
masses and centres of mass (Dempster's data) and lengths in proportion to
body height (Drillis and Contini, from the same book).  Every length is a
fraction of height H, so positions are in body heights and no participant
height is needed; multiply by H for metres.  Three things come out of it:

* **height** -- the centre of mass above the floor, which is under the lower
  ankle.  Only tilt enters a height, so no heading does.  With both feet still
  on the floor through a weight shift, the two ankles' heights stay within
  1.2-1.3 % of H of each other (median range, about 2 cm), and a single-leg
  stance raises the centre of mass by 2-5 % of H as the leg goes up.
* **weight index** -- where the centre of mass lies along the line from the
  left ankle to the right one, 0 over the left and 1 over the right: by the
  lever rule, the share of the weight on the right foot while both are down.
  It needs no reference on the ground, so neither stepping nor the heading
  of the whole body enters it.  Through every single-leg support, where all
  the weight is on one foot, it stays within 0.03-0.08 (median) of 0 or 1.
* **where the weight index is valid** -- with both feet down, by the lever
  rule, and with the feet at least as far apart as the hip joints: closer
  together they are one base of support, and the index divides by their
  small distance.  A foot is taken to be off the floor while the two ankles
  move relative to each other as in a step (:func:`steps`), which needs no
  foot sensor.

Heading enters only the horizontal placement of the segments, through the
relative heading of the thighs, shanks and pelvis, and is de-drifted as for
the stick figure (:func:`analysis.kinematics_check.dedrifted_rotations`).  The
arms are included with their measured orientation: they are a tenth of the
mass, and holding them hanging instead moves the weight index by 0.01-0.02
(median; 0.04-0.06 at the 95th percentile) and the height by up to 1.5 % of
H, the height change being real -- it is the arms being raised, which tilt
alone measures.

What this is not.  An absolute horizontal position: during single-leg support
the centre of mass comes out about 3 % of H (novice) and 5-7.5 % (trained) from
the stance ankle, mostly to its outside, where it cannot be for seconds,
because the stance leg reads as leaning further in than it does -- for the
trained participant more on the left leg than the right, so that recording's
weight index reads 0.33-0.41 in symmetric standing (the novice's 0.49-0.51).
Such an offset cancels in a transfer, which is a difference within a window.
Nor a path over the ground: anchored to whichever foot is planted, the
excursion measured from the left and from the right foot disagrees by half
its size in the windows where both stay down.  Both stay out of the metrics.
The video-based body models planned for later would give a full trajectory to
check this against.

Without some of the sensors.  A missing foot moves with its shank and a missing
hand along its forearm; a missing forearm or upper arm takes the orientation of
its arm's nearest measured segment, which leaves the weight transfer within the
tolerance but not the heights (an arm's elevation moves the centre of mass up
and down).  Without the pelvis, the chest or a leg segment nothing stands in
and nothing is reported.  :func:`analysis.coverage.com_coverage` holds the
rules and ``docs/missing_sensors.md`` the simulations behind them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation

from analysis.coverage import COM_HEIGHTS, COM_WEIGHT_TRANSFER, ComCoverage, com_coverage
from analysis.kinematics import Kinematics
from analysis.kinematics_check import dedrifted_rotations
from analysis.signals import lowpass_signal


# ---------------------------------------------------------------------------
# The segment model, in body heights
# ---------------------------------------------------------------------------

# Lengths: Drillis and Contini (1966) as drawn in Winter (2009), Fig. 4.1 -- joint
# heights 0.818 H (shoulder), 0.530 H (greater trochanter), 0.285 H (knee) and
# 0.039 H (ankle); upper arm 0.186 H, forearm 0.146 H, hand 0.108 H, foot 0.152 H,
# shoulder width 0.259 H.
THIGH = 0.530 - 0.285
SHANK = 0.285 - 0.039
ANKLE_HEIGHT = 0.039
TRUNK = 0.818 - 0.530
"""Hip joints to shoulder joints."""
UPPER_ARM, FOREARM, HAND = 0.186, 0.146, 0.108
SHOULDER_HALF_WIDTH = 0.259 / 2
FOOT_TO_TOES = np.array([0.0, 0.48 * 0.152, -ANKLE_HEIGHT])
"""Ankle to the metatarsal heads in the neutral pose: 0.48 of the foot length
forward (the ankle sits about a quarter of it in front of the heel, the heads
near three quarters) and down to the floor.  The feet are 3 % of the mass, so
this barely moves the centre of mass."""

HIP_JOINT_SEPARATION = 0.105
"""Between the two hip-joint centres: 0.33 x the pelvic width + 7 mm either side
of the midline (Harrington et al. 2007), about 18 cm for a typical adult pelvis
of 25 cm, 0.105 of a 1.70 m height."""

TRUNK_SPLIT = 0.5
"""Where along the trunk the lumbar sensor's segment ends and the chest sensor's
begins, as a fraction from the hips: the thoracolumbar junction, taken half-way.
The trunk is upright in Tai Chi, so where exactly matters little."""

MASS = {
    "thigh": 0.100, "shank": 0.0465, "foot": 0.0145,
    "lower_trunk": 0.281,  # abdomen and pelvis
    "upper_trunk": 0.216 + 0.081,  # thorax, head and neck: no sensor on the head
    "upper_arm": 0.028, "forearm": 0.016, "hand": 0.006,
}
"""Fraction of body mass per segment, Winter (2009), Table 4.1: twice each limb
segment plus the two trunk parts makes 1.000."""

LOWER_TRUNK_COM = 0.73 * TRUNK_SPLIT * TRUNK
"""Abdomen and pelvis: 0.27 of the way down from the thoracolumbar junction to
the hips (Winter's 0.27 from T12-L1 to the greater trochanter)."""

UPPER_TRUNK_COM = (
    (MASS["lower_trunk"] + MASS["upper_trunk"]) * 0.66 * TRUNK - MASS["lower_trunk"] * LOWER_TRUNK_COM
) / MASS["upper_trunk"] - TRUNK_SPLIT * TRUNK
"""Thorax, head and neck, above the thoracolumbar junction: placed where it puts
the trunk-head-neck centre of mass at Winter's 0.66 of the way from the hips to
the shoulders in the neutral pose -- near the shoulders, the head being in it."""

LIMB_COM = {"thigh": 0.433, "shank": 0.433, "upper_arm": 0.436, "forearm": 0.430, "hand": 0.506}
"""Each limb segment's centre of mass, as a fraction of its length from its proximal joint."""

DOWN = np.array([0.0, 0.0, -1.0])


def levelled_rotations(kin: Kinematics, rotations: dict[str, Rotation]) -> dict[str, Rotation]:
    """Each segment's rotation from the neutral pose, in one frame whose z is the true vertical.

    A body quaternion is expressed in its own sensor's neutral frame, whose z
    axis is the sensor's, 4-27 deg off vertical on the limbs and 40 deg on the
    feet.  Conjugating it with the rotation that levels that frame puts every
    segment in the same upright frame, so that a segment's long axis -- vertical
    in the neutral pose -- is ``rotation.apply(DOWN)`` whatever the sensor's
    mounting tilt: turning a segment about the vertical then leaves it upright.
    """
    z = np.array([0.0, 0.0, 1.0])
    out = {}
    for sensor, rotation in rotations.items():
        up = kin.calibration.up_ref[sensor]
        axis = np.cross(up, z)
        sin = float(np.linalg.norm(axis))
        level = (Rotation.from_rotvec(axis / sin * np.arctan2(sin, float(up @ z)))
                 if sin > 1e-12 else Rotation.identity())
        out[sensor] = level * rotation * level.inv()
    return out


# ---------------------------------------------------------------------------
# The centre of mass of one recording
# ---------------------------------------------------------------------------


COM_CUTOFF_HZ = 2.0
"""Smoothing of the height and the weight index: the centre of mass moves over
seconds, and above ~2 Hz the segment chain carries soft-tissue and filter noise
-- the same signal, and cutoff, as the leg lift index."""

STEP_MIN_PEAK = 0.10
"""How fast the ankles must move relative to each other for a step, in body
heights per second (17 cm/s at 1.70 m).  With both feet planted the segment
chain reads at most 0.07 H/s through a weight shift here, and 90 % of the
steps peak above 0.10 H/s (median 0.30)."""

STEP_REL_THRESHOLD = 0.15
"""A step's start and end, as a fraction of its own peak speed -- the rule the
trunk and single-leg detectors use -- never below the recording's median speed,
the both-feet-down level."""


@dataclass
class CentreOfMass:
    """The centre of mass of one recording, sample by sample, in body heights."""

    height: np.ndarray
    """Above the floor, which is under the lower ankle."""
    neutral_height: float
    """In the neutral pose: 0.58 H by construction, every segment upright."""
    weight_index: np.ndarray
    """Along the line between the ankles: 0 over the left, 1 over the right."""
    feet_apart: np.ndarray
    """The ankles at least :data:`HIP_JOINT_SEPARATION` apart."""
    stepping: np.ndarray
    """A foot is moving over the floor: :func:`steps`."""
    position: np.ndarray
    """Relative to the midpoint of the hip joints, in the levelled frame (n, 3)."""
    ankles: dict[str, np.ndarray]
    """``{"Left": (n, 3), "Right": (n, 3)}``, in the same frame."""
    coverage: ComCoverage = ComCoverage()
    """Which segments a neighbour stood in for, and which metrics that leaves valid."""

    @property
    def weight_valid(self) -> np.ndarray:
        """Where the weight index means the share of the weight on the right foot.

        The lever rule needs both feet on the floor: while one is in the air the
        line between the ankles is not the edge of a base of support, and the
        index runs anywhere -- more than 0.15 outside 0-1 for 2.2-3.5 % of the
        time the feet are apart, against 0.1-0.5 % once steps are left out.
        """
        return self.feet_apart & ~self.stepping


def steps(between: np.ndarray, fs: float) -> np.ndarray:
    """Where a foot is moving over the floor, from the ankle-to-ankle vector ``(n, 2)``.

    With both feet planted the vector between the ankles does not change, so
    its speed is the chain's noise; a step moves it at 0.1-0.4 H/s.  Every
    peak above :data:`STEP_MIN_PEAK` is a step, walked outward to
    :data:`STEP_REL_THRESHOLD` of its peak, never below the recording's median
    speed.  The foot sensors are not used: a loose one, like the trained
    participant's right, would read as stepping in quiet standing.
    """
    speed = np.linalg.norm(np.gradient(between, 1.0 / fs, axis=0), axis=1)
    floor = float(np.median(speed))
    stepping = np.zeros(len(speed), dtype=bool)
    peaks, _ = find_peaks(speed, height=STEP_MIN_PEAK)
    for peak in peaks:
        if stepping[peak]:
            continue
        threshold = max(STEP_REL_THRESHOLD * speed[peak], floor)
        start = end = int(peak)
        while start > 0 and speed[start] > threshold:
            start -= 1
        while end < len(speed) - 1 and speed[end] > threshold:
            end += 1
        stepping[start:end + 1] = True
    return stepping


def segments(G: dict[str, Rotation]) -> tuple[list[tuple[float, np.ndarray]], dict[str, np.ndarray]]:
    """Every segment's ``(mass fraction, centre of mass)``, and both ankles, relative to the hips' midpoint."""
    parts: list[tuple[float, np.ndarray]] = []
    ankles = {}
    pelvis_right = G["lumbar"].apply([1.0, 0.0, 0.0])
    for side, sign, prefix in (("Left", -1.0, "l"), ("Right", 1.0, "r")):
        hip = sign * 0.5 * HIP_JOINT_SEPARATION * pelvis_right
        thigh, shank = G[f"{prefix}thigh"].apply(DOWN), G[f"{prefix}tibia"].apply(DOWN)
        knee = hip + THIGH * thigh
        ankle = knee + SHANK * shank
        parts += [
            (MASS["thigh"], hip + LIMB_COM["thigh"] * THIGH * thigh),
            (MASS["shank"], knee + LIMB_COM["shank"] * SHANK * shank),
            (MASS["foot"], ankle + 0.5 * G[f"{prefix}foot"].apply(FOOT_TO_TOES)),
        ]
        ankles[side] = ankle

    lumbar_up, chest_up = G["lumbar"].apply(-DOWN), G["chestbone"].apply(-DOWN)
    junction = TRUNK_SPLIT * TRUNK * lumbar_up
    parts += [(MASS["lower_trunk"], LOWER_TRUNK_COM * lumbar_up),
              (MASS["upper_trunk"], junction + UPPER_TRUNK_COM * chest_up)]
    neck = junction + (1.0 - TRUNK_SPLIT) * TRUNK * chest_up
    chest_right = G["chestbone"].apply([1.0, 0.0, 0.0])
    for sign, prefix in ((-1.0, "l"), (1.0, "r")):
        shoulder = neck + sign * SHOULDER_HALF_WIDTH * chest_right
        upper, fore, hand = (G[f"{prefix}{name}"].apply(DOWN) for name in ("humerus", "ulna", "hand"))
        elbow = shoulder + UPPER_ARM * upper
        wrist = elbow + FOREARM * fore
        parts += [
            (MASS["upper_arm"], shoulder + LIMB_COM["upper_arm"] * UPPER_ARM * upper),
            (MASS["forearm"], elbow + LIMB_COM["forearm"] * FOREARM * fore),
            (MASS["hand"], wrist + LIMB_COM["hand"] * HAND * hand),
        ]
    return parts, ankles


def centre_of_mass(kin: Kinematics, fs: float) -> CentreOfMass | None:
    """Height, weight index and position of the centre of mass over a whole recording.

    None when a sensor it cannot do without is missing (:func:`analysis.coverage.com_coverage`).
    """
    coverage = com_coverage(kin.sensors)
    if not coverage.computed:
        return None
    G = levelled_rotations(kin, dedrifted_rotations(kin, fs))
    G.update({segment: G[donor] for segment, donor in coverage.stand_ins.items()})
    return centre_of_mass_from(kin, fs, G, coverage)


def centre_of_mass_from(kin: Kinematics, fs: float, G: dict[str, Rotation],
                        coverage: ComCoverage = ComCoverage()) -> CentreOfMass:
    """The centre of mass from every segment's levelled rotation ``G`` (one per sensor name, stand-ins included)."""
    parts, ankles = segments(G)
    total = sum(mass for mass, _ in parts)
    position = sum(mass * point for mass, point in parts) / total

    floor = np.minimum(ankles["Left"][:, 2], ankles["Right"][:, 2]) - ANKLE_HEIGHT
    height = lowpass_signal(position[:, 2] - floor, fs, cutoff_hz=COM_CUTOFF_HZ)
    start, end = kin.calibration.neutral

    between = ankles["Right"][:, :2] - ankles["Left"][:, :2]
    distance = np.linalg.norm(between, axis=1)
    along = np.sum((position[:, :2] - ankles["Left"][:, :2]) * between, axis=1)
    index = lowpass_signal(along / np.maximum(distance, 1e-9) ** 2, fs, cutoff_hz=COM_CUTOFF_HZ)
    smooth_between = np.column_stack([lowpass_signal(between[:, i], fs, cutoff_hz=COM_CUTOFF_HZ) for i in range(2)])
    return CentreOfMass(
        height=height,
        neutral_height=float(np.mean(height[start:end])),
        weight_index=index,
        feet_apart=np.linalg.norm(smooth_between, axis=1) >= HIP_JOINT_SEPARATION,
        stepping=steps(smooth_between, fs),
        position=position,
        ankles=ankles,
        coverage=coverage,
    )


# ---------------------------------------------------------------------------
# Window measures
# ---------------------------------------------------------------------------


def com_signals(kin: Kinematics, fs: float) -> dict[str, object]:
    """What :func:`com_metrics` slices, for one recording: None for what the sensors present do not support."""
    centre = centre_of_mass(kin, fs)
    heights = centre is not None and centre.coverage.valid(COM_HEIGHTS)
    transfer = centre is not None and centre.coverage.valid(COM_WEIGHT_TRANSFER)
    return {
        "com_height": centre.height if heights else None,
        "com_neutral_height": centre.neutral_height if heights else None,
        "weight_index": centre.weight_index if transfer else None,
        "weight_valid": centre.weight_valid if transfer else None,
    }


MIN_VALID_SHARE = 0.5
"""A window's weight transfer is reported when the weight index is valid -- both
feet down and apart -- for at least this share of it, as a lag needs that much
overlap (:data:`analysis.signals.LAG_MIN_OVERLAP`).  A window that is mostly a
step is mostly unmeasured."""


def com_metrics(
    signals: dict[str, object], window: slice, prefix: str = "", weight_transfer: bool = True
) -> dict[str, float]:
    """How much weight moved between the feet, and how level and how low the centre of mass stayed.

    ``com_weight_transfer`` is the range of the weight index over the part of
    the window with both feet down and apart, clipped to 0-1: the share of the
    body weight moved from one foot to the other, 1 for a full shift.  Outside
    0-1 the centre of mass would lie beyond an ankle, which a quasi-static
    stance cannot hold.  Heights are in percent of body height:
    ``com_height_range_pct_height`` is how far the centre of mass rose and sank
    within the window, ``com_lowering_pct_height`` how far below upright
    standing it was on average -- how low the participant sat into the stance.
    ``weight_transfer=False`` leaves the transfer out, for a window with one
    foot off the floor throughout.  A metric whose signal is None -- the
    recording lacks a sensor it needs (:func:`com_signals`) -- is NaN.
    """
    metrics = {}
    if weight_transfer:
        if signals["weight_index"] is None:
            metrics[f"{prefix}com_weight_transfer"] = float("nan")
        else:
            index = signals["weight_index"][window]
            valid = signals["weight_valid"][window]
            metrics[f"{prefix}com_weight_transfer"] = (
                float(np.ptp(np.clip(index[valid], 0.0, 1.0)))
                if len(index) and valid.sum() >= MIN_VALID_SHARE * len(index) else float("nan")
            )
    height = signals["com_height"][window] if signals["com_height"] is not None else np.array([])
    metrics[f"{prefix}com_height_range_pct_height"] = 100.0 * float(np.ptp(height)) if len(height) else float("nan")
    metrics[f"{prefix}com_lowering_pct_height"] = (
        100.0 * (signals["com_neutral_height"] - float(np.mean(height))) if len(height) else float("nan")
    )
    return metrics
