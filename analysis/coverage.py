"""What a recording's sensors allow: the missing-sensor strategy, in one place.

A recording does not always have all 14 sensors.  A missing one never stops
the analysis: every metric is measured as with all sensors, measured with a
stand-in that the simulations showed does the job, or left empty (NaN).  This
module holds those rules as data, so the metric code, the stand-ins and the
``sensor_coverage.csv`` report cannot drift apart:

* :data:`METRICS` -- per family and metric column, the sensors without which it
  is left empty;
* :func:`com_coverage` -- which centre-of-mass segments a neighbour stands in
  for, and which centre-of-mass metrics that leaves valid;
* :func:`lift_stand_ins` -- the thigh standing in for a missing shank, for
  placing single-leg windows only;
* :data:`DETECTOR_SENSORS` and :func:`stabilization_sensors` -- what each
  detector needs.

``docs/missing_sensors.md`` gives the reason for every rule and the numbers
behind it; ``analysis/tools/missing_sensor_checks.py`` re-runs the simulations
on two complete recordings.  The tolerance throughout is the "rule of ten": a
stand-in may move a metric by at most a tenth of its interquartile range across
the windows, in 95 % of them, in every family.  A recording with all 14
sensors goes through none of this.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from analysis import config

ALL = tuple(config.SENSOR_MAP)
"""Every sensor the analysis knows, in the order it lists them."""
TRUNK = ("lumbar", "chestbone")
LEGS = ("lthigh", "rthigh", "ltibia", "rtibia")

LIFTED_THIGH, LIFTED_SHANK = "<lifted thigh>", "<lifted shank>"
"""Placeholders in :data:`METRICS` for the thigh and shank of whichever leg a stance lifts."""


def missing(present) -> tuple[str, ...]:
    """The analysis's sensors not in ``present``, in the order it lists them."""
    present = set(present)
    return tuple(sensor for sensor in ALL if sensor not in present)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


SWAY = ("ml_acc_rms_mps2", "ap_acc_rms_mps2", "frontal_tilt_sd_deg", "sagittal_tilt_sd_deg",
        "rms_angular_velocity_dps", "corrective_peak_rate_hz")
"""The lumbar sensor's sway, tilt and angular-speed columns, after a prefix (:func:`analysis.balance_metrics.sway_metrics`)."""

COM_WEIGHT_TRANSFER = "weight transfer"
COM_HEIGHTS = "heights"
"""The two groups of centre-of-mass metrics, whose stand-ins are judged separately (:func:`com_coverage`)."""

NEUTRAL_POSE = ("lumbar",)
"""Peak knee flexion needs the neutral pose found from the lumbar sensor: from the other sensors it is
another still moment, and the knee's zero moves by up to 8 deg (6.2x the tolerance)."""


@dataclass(frozen=True)
class Requirement:
    columns: tuple[str, ...]
    sensors: tuple[str, ...] = ()
    """Sensors without which the columns are empty; may include :data:`LIFTED_THIGH`/:data:`LIFTED_SHANK`."""
    what: str = ""
    """What the columns are computed from, for the report."""
    com: str | None = None
    """:data:`COM_WEIGHT_TRANSFER` or :data:`COM_HEIGHTS` for centre-of-mass columns, judged by :func:`com_coverage`."""


METRICS: dict[str, tuple[Requirement, ...]] = {
    "trunk": (
        Requirement(("trunk_pelvis_yaw_r", "trunk_pelvis_lag_s", "trunk_yaw_sparc", "trunk_yaw_excursion_deg",
                     "trunk_yaw_peak_rate_dps", "event_peak_s"), TRUNK, "the chest's turning on the pelvis"),
        Requirement(("com_weight_transfer",), what="the centre of mass", com=COM_WEIGHT_TRANSFER),
        Requirement(("com_height_range_pct_height", "com_lowering_pct_height"), what="the centre of mass",
                    com=COM_HEIGHTS),
        Requirement(tuple(f"lumbar_{name}" for name in SWAY), ("lumbar",),
                    "the lumbar sensor's sway, tilt and angular speed"),
        Requirement(("stabilization_quiet_z",), TRUNK, "the angular speed of both trunk sensors"),
    ),
    "knee": (
        Requirement(("peak_lift_index", "peak_time_s"), LEGS, "the lift index (both thighs and shanks)"),
        Requirement(("peak_knee_flexion_deg",), (LIFTED_THIGH, LIFTED_SHANK) + NEUTRAL_POSE,
                    "the lifted leg's knee, referenced to the neutral pose found from the lumbar sensor"),
        Requirement(("knee_extension_while_lifted_deg",), LEGS, "the lifted leg's knee and the lift index"),
        Requirement(("peak_hip_flexion_deg",), ("lumbar", LIFTED_THIGH), "the lifted leg's hip"),
        Requirement(("lumbar_frontal_tilt_at_peak_deg",), ("lumbar",) + LEGS,
                    "the lumbar tilt at the highest lift of the lift index"),
        Requirement(tuple(f"support_{name}" for name in SWAY), ("lumbar",),
                    "the lumbar sensor's sway, tilt and angular speed"),
        Requirement(("support_com_height_range_pct_height", "support_com_lowering_pct_height"),
                    what="the centre of mass", com=COM_HEIGHTS),
        Requirement(tuple(f"settle_{name}" for name in SWAY), ("lumbar",),
                    "the lumbar sensor's sway, tilt and angular speed"),
        Requirement(("time_to_stabilization_s", "stabilization_quiet_z"), TRUNK,
                    "the angular speed of both trunk sensors"),
        Requirement(("trunk_pelvis_yaw_r", "trunk_pelvis_lag_s", "trunk_pelvis_frontal_r", "trunk_pelvis_frontal_lag_s"),
                    TRUNK + LEGS, "chest and pelvis, around the highest lift of the lift index"),
    ),
    "smooth": (
        Requirement(("chest_yaw_excursion_deg", "chest_yaw_peak_rate_dps", "chest_yaw_sparc",
                     "chest_yaw_submovement_count", "chest_yaw_submovement_rate_hz",
                     "chest_yaw_log10_dimensionless_jerk"), ("chestbone",), "the chest's turning"),
        Requirement(("pelvis_yaw_excursion_deg",), ("lumbar",), "the pelvis's turning"),
        Requirement(("chest_pelvis_yaw_r", "chest_pelvis_lag_s", "chest_pelvis_gain", "relative_yaw_range_deg",
                     "relative_yaw_sd_deg"), TRUNK, "the chest's and the pelvis's turning"),
        Requirement(("com_weight_transfer",), what="the centre of mass", com=COM_WEIGHT_TRANSFER),
        Requirement(("com_height_range_pct_height", "com_lowering_pct_height"), what="the centre of mass",
                    com=COM_HEIGHTS),
    ),
}
"""Per family, every sensor-dependent metric column and what it needs.  Columns not listed (window
times, durations, ids) need no sensor."""


def resolve(sensors: tuple[str, ...], lifted_leg: str | None) -> tuple[str, ...]:
    """``sensors`` with the lifted-leg placeholders replaced by that leg's sensors."""
    side = {"Left": "l", "Right": "r"}.get(lifted_leg or "")
    out = []
    for sensor in sensors:
        if sensor in (LIFTED_THIGH, LIFTED_SHANK):
            if side:
                out.append(f"{side}{'thigh' if sensor == LIFTED_THIGH else 'tibia'}")
        else:
            out.append(sensor)
    return tuple(out)


def has(present, *sensors: str) -> bool:
    """Whether every one of ``sensors`` is in ``present``."""
    present = set(present)
    return all(sensor in present for sensor in sensors)


def unmeasured(present, family: str, lifted_leg: str | None = None) -> set[str]:
    """The columns of ``family`` a recording cannot measure; for one stance, given the leg it lifts."""
    present = set(present)
    if not missing(present):
        return set()
    out: set[str] = set()
    com = None
    for requirement in METRICS[family]:
        if requirement.com:
            com = com or com_coverage(present)
            if not com.valid(requirement.com):
                out.update(requirement.columns)
        elif not has(present, *resolve(requirement.sensors, lifted_leg)):
            out.update(requirement.columns)
    return out


def blank_unmeasured(row: dict, present, family: str, lifted_leg: str | None = None) -> dict:
    """``row`` with every column the recording cannot measure set to NaN.

    The metric functions never compute on a missing sensor's signal; this
    also catches what depends on one indirectly -- peak knee flexion, whose
    zero comes from the neutral pose the lumbar sensor finds -- so that what
    is reported is always what :data:`METRICS` says.
    """
    for column in unmeasured(present, family, lifted_leg):
        if column in row:
            row[column] = float("nan")
    return row


# ---------------------------------------------------------------------------
# Centre of mass
# ---------------------------------------------------------------------------


COM_REQUIRED = TRUNK + LEGS
"""No stand-in passes for these: the chest for the pelvis or the reverse (rigid trunk) is 1.4-36x the
tolerance, a thigh for a shank or the reverse 20-52x."""

ARM_NEIGHBOURS = {"hand": ("ulna", "humerus"), "ulna": ("hand", "humerus"), "humerus": ("ulna", "hand")}
"""Which measured segment of the same arm stands in for a missing one, nearest first: a hand along its
forearm, a forearm along its hand or else its upper arm, an upper arm along its forearm or else its hand."""

ARM_SEGMENTS = ("lhumerus", "rhumerus", "lulna", "rulna")


@dataclass(frozen=True)
class ComCoverage:
    """What the centre of mass of a recording can be built from."""

    stand_ins: dict[str, str] = field(default_factory=dict)
    """``{missing segment: the sensor whose orientation it takes}``."""
    weight_transfer: bool = True
    heights: bool = True
    """Whether the height range and the lowering are valid."""
    reason: str = ""
    """Why a group is not valid, for the report."""

    @property
    def computed(self) -> bool:
        return self.weight_transfer or self.heights

    def valid(self, group: str) -> bool:
        return self.heights if group == COM_HEIGHTS else self.weight_transfer


def com_coverage(present) -> ComCoverage:
    """The centre-of-mass stand-ins a recording needs, and which metrics they leave valid.

    Checked on every one of the 255 combinations of missing hands, forearms,
    upper arms and feet (exact, because the centre of mass is linear in the
    segment positions and the ankles do not move with them):

    * a foot moves with its shank and a hand along its forearm: every metric
      stays within the tolerance as long as only hands, or only feet, are
      missing (6 combinations, worst 0.86 of the tolerance);
    * weight transfer, a horizontal position divided by the distance between
      the ankles, also stays within it with one forearm or upper arm standing in
      as well (79 combinations, worst 0.90); the heights do not -- raising an
      arm raises the centre of mass -- and two or more such segments fail in
      120 of 176 combinations.
    """
    absent = set(missing(present))
    lost = [sensor for sensor in COM_REQUIRED if sensor in absent]
    if lost:
        return ComCoverage({}, False, False, f"no stand-in for the {', '.join(lost)} sensor"
                                             + "s" * (len(lost) > 1))
    stand_ins = {}
    for side in "lr":
        for segment, donors in ARM_NEIGHBOURS.items():
            if f"{side}{segment}" not in absent:
                continue
            donor = next((f"{side}{d}" for d in donors if f"{side}{d}" not in absent), None)
            if donor is None:
                return ComCoverage({}, False, False, f"the whole {'left' if side == 'l' else 'right'} arm is missing")
            stand_ins[f"{side}{segment}"] = donor
        if f"{side}foot" in absent:
            stand_ins[f"{side}foot"] = f"{side}tibia"
    arm = sorted(absent & set(ARM_SEGMENTS))
    hands, feet = absent & {"lhand", "rhand"}, absent & {"lfoot", "rfoot"}
    weight_transfer = len(arm) <= 1
    heights = not arm and not (hands and feet)
    reasons = []
    if arm:
        reasons.append(f"heights: no stand-in for the {', '.join(arm)} sensor" + "s" * (len(arm) > 1)
                       + (" (an arm's elevation moves the centre of mass's height)" if weight_transfer else ""))
        if not weight_transfer:
            reasons.append("weight transfer: two or more forearms or upper arms missing")
    elif hands and feet:
        reasons.append("heights: a missing hand and a missing foot together exceed the tolerance")
    return ComCoverage(stand_ins, weight_transfer, heights, "; ".join(reasons))


# ---------------------------------------------------------------------------
# Detection and window placement
# ---------------------------------------------------------------------------


DETECTOR_SENSORS = {"trunk": TRUNK, "knee": LEGS, "smooth": ("chestbone",)}
"""What each family's detector works on: trunk-pelvis yaw, the lift index, the chest yaw."""


def lift_stand_ins(present) -> dict[str, str] | None:
    """For placing single-leg windows: ``{missing shank: its thigh}``, or None if a thigh is missing.

    With a shank missing, its thigh's tilt stands in for the shank's in the
    lift index: in the June recordings the detector then finds the same stances
    with every boundary within 0.25 s (one extra stance in one recording).  The
    value of the lift index is not reported with it (18-19x the tolerance).  A
    thigh cannot be stood in for: a knee lift raises the thigh while the shank
    hangs, and stances are lost.
    """
    present = set(present)
    if not {"lthigh", "rthigh"} <= present:
        return None
    return {f"{side}tibia": f"{side}thigh" for side in "lr" if f"{side}tibia" not in present}


def can_detect(present, family: str, knee_method: str = "lift") -> bool:
    """Whether a recording's own sensors can find a family's windows."""
    if family == "knee":
        if knee_method == "knee_threshold":
            return any(has(present, f"{side}thigh", f"{side}tibia") for side in "lr")
        return lift_stand_ins(present) is not None
    return has(present, *DETECTOR_SENSORS[family])


def stabilization_sensors(present) -> tuple[str, ...]:
    """The trunk sensors the stabilization search can use: normally both.

    With one, its angular speed alone places the window (95 % within 0.25 s
    of the two-sensor placement in the June recordings), but the quietness
    score and the time to stabilization are then a different quantity and are
    not reported (:data:`METRICS`).
    """
    return tuple(sensor for sensor in TRUNK if sensor in set(present))


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


FAMILY_NAMES = {"trunk": "trunk rotation", "knee": "single-leg stance", "smooth": "sequence turn"}


def metric_status(present, family: str, requirement: Requirement) -> tuple[str, str]:
    """``(status, detail)`` of one requirement for a recording: measured, stand-in or not calculated."""
    if requirement.com:
        coverage = com_coverage(present)
        if not coverage.valid(requirement.com):
            return "not calculated", coverage.reason
        if coverage.stand_ins:
            return "stand-in", "centre of mass with " + ", ".join(
                f"the {segment} along the {donor}" for segment, donor in coverage.stand_ins.items())
        return "measured", ""
    absent = set(missing(present))
    if family == "knee" and (LIFTED_THIGH in requirement.sensors or LIFTED_SHANK in requirement.sensors):
        common = [s for s in resolve(requirement.sensors, None) if s in absent]
        legs = {leg: [s for s in resolve(requirement.sensors, leg) if s in absent] for leg in ("Left", "Right")}
        if common or (legs["Left"] and legs["Right"]):
            lost = sorted(set(common) | set(legs["Left"]) | set(legs["Right"]), key=ALL.index)
            return "not calculated", f"needs {requirement.what}; missing: {', '.join(lost)}"
        for leg, lost in legs.items():
            if lost:
                return ("not calculated", f"for stances lifting the {leg.lower()} leg: needs {requirement.what}; "
                                          f"missing: {', '.join(lost)}")
        return "measured", ""
    lost = [s for s in requirement.sensors if s in absent]
    if lost:
        return "not calculated", f"needs {requirement.what}; missing: {', '.join(lost)}"
    return "measured", ""


def coverage_table(present_by_role: dict[str, set], sources: dict[str, str],
                   windows: dict[str, dict[str, str]] | None = None) -> pd.DataFrame:
    """``sensor_coverage.csv``: for every recording that lacks a sensor, what each metric is.

    One row per recording and family per requirement, with ``status`` measured,
    stand-in or not calculated, and the reason.  ``windows`` adds a row per
    family saying where that recording's windows came from when not from its
    own detector (``{role: {family: description}}``).  Empty when every
    recording has all its sensors.
    """
    rows = []
    for role, present in present_by_role.items():
        absent = missing(present)
        if not absent:
            continue
        base = {"trial": role, "recording": sources.get(role, ""), "missing_sensors": " ".join(absent)}
        for family, requirements in METRICS.items():
            note = (windows or {}).get(role, {}).get(family)
            if note:
                rows.append({**base, "family": FAMILY_NAMES[family], "metrics": "(windows)",
                             "status": "placement", "detail": note})
            for requirement in requirements:
                status, detail = metric_status(present, family, requirement)
                rows.append({**base, "family": FAMILY_NAMES[family], "metrics": " ".join(requirement.columns),
                             "status": status, "detail": detail})
    return pd.DataFrame(rows)


def summary(present) -> list[str]:
    """What a recording's missing sensors cost, one line each, for logs and status cards."""
    absent = missing(present)
    if not absent:
        return []
    lines = [f"no {', '.join(absent)} sensor{'s' * (len(absent) > 1)}: reduced analysis (docs/missing_sensors.md)"]
    lost = []
    for family, requirements in METRICS.items():
        empty = [column for requirement in requirements
                 if metric_status(present, family, requirement)[0] == "not calculated"
                 for column in requirement.columns]
        if empty:
            lost.append(f"{len(empty)} {FAMILY_NAMES[family]} metric{'s' * (len(empty) > 1)}")
    if lost:
        lines.append(", ".join(lost[:-1]) + (" and " if len(lost) > 1 else "") + lost[-1]
                     + " not calculated (sensor_coverage.csv lists them)")
    coverage = com_coverage(present)
    if coverage.computed and coverage.stand_ins:
        lines.append("centre of mass with " + ", ".join(f"the {s} along the {d}" for s, d in coverage.stand_ins.items()))
    for family, sensors in DETECTOR_SENSORS.items():
        if not can_detect(present, family):
            lines.append(f"{FAMILY_NAMES[family]} windows cannot be detected (no {', '.join(s for s in sensors if s in absent)}): "
                         "carried over from the other recording when two are selected")
    stand_ins = lift_stand_ins(present)
    if stand_ins:
        lines.append("single-leg stances found with " + ", ".join(f"the {d} standing in for the {s}" for s, d in stand_ins.items()))
    trunk = stabilization_sensors(present)
    if len(trunk) == 1:
        lines.append(f"stabilization windows placed on the {trunk[0]} sensor alone")
    elif not trunk:
        lines.append("no stabilization search: no trunk sensor")
    return lines
