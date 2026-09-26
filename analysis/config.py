"""Paths, recordings and analysis-wide switches.

The two switches, ``DETECTOR`` and ``IGNORE_HIGH_PASS_FILTER``, are read as
``config.NAME`` at the moment they are used, so setting one from a script or a
notebook takes effect everywhere::

    from analysis import config
    config.DETECTOR = "v1"
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
"""Every CSV in here, including in subfolders, is offered as a recording."""
OUTPUT_DIR = ROOT / "outputs"
RECORDINGS_DIR = OUTPUT_DIR / "recordings"
"""One folder per recording, for what depends on that recording alone."""
ANALYSES_DIR = OUTPUT_DIR / "analyses"
"""One folder per selection of recordings: the session, the metrics and the figures."""

TRIALS = ("Novice", "Trained")
"""The two roles a recording can be selected for.  Either may be left empty."""

DEFAULT_RECORDINGS = {"Novice": "IMU_Trial_1_RC_Novice.csv", "Trained": "IMU_Trial_3_RC_Trained.csv"}
"""What is selected before anything has been chosen: file names relative to DATA_DIR."""

# Default False: the 0.05 Hz high-pass is applied, so yaw is de-drifted.  Set to
# True to make highpass_detrend a pass-through.  This changes every yaw-derived
# number in all three families, so it is recorded into the session file on every
# save and shown in the editor header.
IGNORE_HIGH_PASS_FILTER = False

# Which trunk-event detector the pipeline uses.  "v2" derives each event's
# duration and boundaries from the trunk yaw velocity (analysis/detection.py);
# "v1" is the original fixed 8 s sliding window (analysis/detection_v1.py),
# kept so the committed results stay reproducible.
DETECTOR = "v2"

METRICS_VERSION = 2
"""Which definition of the metrics produced the numbers, stamped into every
session save, every metric CSV and metrics_provenance.json.

1. Everything up to the git tag ``metrics-v1``: overlap-normalised cross-
   correlation lag, sensor-axis acceleration variance with gravity in it and
   the AP/ML labels swapped, raw gyroscope magnitude (bias included), knee
   flexion > 60 deg for single-leg stance, full-cycle sequence segments paired
   by order.
2. After the external review (Review.md), verified on the recordings: per-lag
   Pearson lag, gravity-compensated sway in the pelvis-heading frame with the
   axes named correctly (x = mediolateral, y = anteroposterior), gyroscope bias
   removed, single-leg stance from leg geometry, single-direction turns, and
   every family paired through a whole-recording alignment.
"""

# Body segment each sensor is strapped to.  Placement is described in
# docs/notes.txt.
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

EXPORT_VERSION = 2
"""Format of the 50 Hz kinematic export, recorded in each recording's
``recording.json``; an older one marks the export out of date, and it is
rebuilt from the orientation cache without re-running the filter.

1. Every joint angle high-passed at 0.05 Hz, so each joint had zero mean -- a
   knee held flexed for tens of seconds lost its flexion -- and the angular
   speeds included the gyroscope bias.
2. Joint x (flexion, the mediolateral axis) and y are gravity-referenced and
   exported as they are; only the axial z channels, which carry the heading
   drift of two sensors, are high-passed.  Angular speeds are bias-corrected.
"""
