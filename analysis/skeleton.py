"""Stick-figure skeleton: the segments, and a pose from the segment orientations.

Each joint is placed at its parent plus a neutral-pose offset (metres) rotated
by the orientation of the sensor that drives the segment.  Segment lengths are
nominal; no anthropometric scaling is applied.  The offsets use the body frame
of :func:`analysis.orientation.mounting_matrix`: x = right (mediolateral),
y = forward (anteroposterior), z = up.

Used by the kinematics check of the dashboard and by the standalone viewers in
:mod:`analysis.tools`.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

SEGMENTS = {
    # joint:          (parent,          sensor,      neutral offset [x, y, z])
    # ── Pelvis hips (rigid offsets from pelvis_center; lumbar drives both) ──
    "pelvis_right":   ("pelvis_center", "lumbar",    [ 0.12,  0,     0   ]),
    "pelvis_left":    ("pelvis_center", "lumbar",    [-0.12,  0,     0   ]),

    # ── Spine / thorax ──
    "neck":           ("pelvis_center", "chestbone", [ 0,     0,     0.5 ]),

    # Shoulders are rigid extensions of the thorax; sensor=None means they
    # inherit the neck's world rotation instead of re-applying chestbone.
    "shoulder_right": ("neck",          None,        [ 0.18,  0,     0   ]),
    "shoulder_left":  ("neck",          None,        [-0.18,  0,     0   ]),

    # ── Right arm ──
    "elbow_right":    ("shoulder_right","rhumerus",  [ 0,     0,    -0.28]),
    "wrist_right":    ("elbow_right",   "rulna",     [ 0,     0,    -0.26]),
    "hand_right":     ("wrist_right",   "rhand",     [ 0,     0,    -0.1 ]),

    # ── Left arm ──
    "elbow_left":     ("shoulder_left", "lhumerus",  [ 0,     0,    -0.28]),
    "wrist_left":     ("elbow_left",    "lulna",     [ 0,     0,    -0.26]),
    "hand_left":      ("wrist_left",    "lhand",     [ 0,     0,    -0.1 ]),

    # ── Right leg ──
    "knee_right":     ("pelvis_right",  "rthigh",    [ 0,     0,    -0.42]),
    "ankle_right":    ("knee_right",    "rtibia",    [ 0,     0,    -0.4 ]),
    "toe_right":      ("ankle_right",   "rfoot",     [ 0,     0.18, -0.05]),

    # ── Left leg ──
    "knee_left":      ("pelvis_left",   "lthigh",    [ 0,     0,    -0.42]),
    "ankle_left":     ("knee_left",     "ltibia",    [ 0,     0,    -0.4 ]),
    "toe_left":       ("ankle_left",    "lfoot",     [ 0,     0.18, -0.05]),
}

CONNECTIONS = [
    ("pelvis_center",  "pelvis_right"),
    ("pelvis_center",  "pelvis_left"),
    ("pelvis_center",  "neck"),
    ("neck",           "shoulder_right"),
    ("neck",           "shoulder_left"),
    ("shoulder_right", "elbow_right"),
    ("shoulder_left",  "elbow_left"),
    ("elbow_right",    "wrist_right"),
    ("elbow_left",     "wrist_left"),
    ("wrist_right",    "hand_right"),
    ("wrist_left",     "hand_left"),
    ("pelvis_right",   "knee_right"),
    ("pelvis_left",    "knee_left"),
    ("knee_right",     "ankle_right"),
    ("knee_left",      "ankle_left"),
    ("ankle_right",    "toe_right"),
    ("ankle_left",     "toe_left"),
]
"""Bones to draw, as (joint, joint) pairs."""


def _verify_topological_order(segments: dict) -> None:
    """Raise ValueError if any joint appears before its parent in SEGMENTS."""
    seen = {"pelvis_center"}
    for joint, (parent, _, _) in segments.items():
        if parent not in seen:
            raise ValueError(
                f"Joint '{joint}' references parent '{parent}' which has not "
                f"been defined yet. Fix the ordering in SEGMENTS."
            )
        seen.add(joint)


_verify_topological_order(SEGMENTS)


def pose(rotations: dict[str, Rotation]) -> tuple[dict[str, np.ndarray], dict[str, Rotation]]:
    """Joint positions and segment rotations for one frame.

    ``rotations`` holds each sensor's body-frame orientation.  That is already
    the segment's orientation relative to the neutral pose, in the one body
    frame every sensor is expressed in, so it is used as it is rather than
    composed with the parent's -- composing is right only at the neutral pose.
    A segment with no sensor (the shoulders) turns with its parent.
    """
    positions = {"pelvis_center": np.zeros(3)}
    world = {"pelvis_center": Rotation.identity()}
    for joint, (parent, sensor, offset) in SEGMENTS.items():
        rotation = rotations.get(sensor) if sensor is not None else None
        if rotation is None:
            rotation = world[parent]
        positions[joint] = positions[parent] + rotation.apply(offset)
        world[joint] = rotation
    return positions, world
