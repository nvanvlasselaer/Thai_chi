#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# ----------------------------------------------------------------------
# Skeleton definition
# ----------------------------------------------------------------------
SEGMENTS = {
    "pelvis_right":   ("pelvis_center", "lumbar",    [ 0.12,  0.0,   0.0 ]),
    "pelvis_left":    ("pelvis_center", "lumbar",    [-0.12,  0.0,   0.0 ]),

    "spine":          ("pelvis_center", "chestbone", [ 0.0,   0.0,   0.5 ]),

    # rigid extensions (no extra local sensor rotation applied)
    "shoulder_right": ("spine",         None,        [ 0.18,  0.0,   0.0 ]),
    "shoulder_left":  ("spine",         None,        [-0.18,  0.0,   0.0 ]),

    "elbow_right":    ("shoulder_right","rhumerus",   [ 0.0,   0.0,  -0.28]),
    "wrist_right":    ("elbow_right",   "rulna",      [ 0.0,   0.0,  -0.26]),
    "hand_right":     ("wrist_right",   "rhand",      [ 0.0,   0.0,  -0.10]),

    "elbow_left":     ("shoulder_left", "lhumerus",   [ 0.0,   0.0,  -0.28]),
    "wrist_left":     ("elbow_left",    "lulna",      [ 0.0,   0.0,  -0.26]),
    "hand_left":      ("wrist_left",    "lhand",      [ 0.0,   0.0,  -0.10]),

    "knee_right":     ("pelvis_right",  "rthigh",     [ 0.0,   0.0,  -0.42]),
    "ankle_right":    ("knee_right",    "rtibia",     [ 0.0,   0.0,  -0.40]),
    "toe_right":      ("ankle_right",   "rfoot",      [ 0.0,   0.18, -0.05]),

    "knee_left":      ("pelvis_left",   "lthigh",     [ 0.0,   0.0,  -0.42]),
    "ankle_left":     ("knee_left",     "ltibia",     [ 0.0,   0.0,  -0.40]),
    "toe_left":       ("ankle_left",    "lfoot",      [ 0.0,   0.18, -0.05]),
}


def _verify_topological_order(segments: dict) -> None:
    seen = {"pelvis_center"}
    for joint, (parent, _, _) in segments.items():
        if parent not in seen:
            raise ValueError(
                f"Joint '{joint}' references parent '{parent}' before it is defined."
            )
        seen.add(joint)


_verify_topological_order(SEGMENTS)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def quat_wxyz_to_rot(q_wxyz):
    """Convert [w, x, y, z] to scipy Rotation."""
    q_wxyz = np.asarray(q_wxyz)
    return Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])


def load_first_frame(npz_path):
    """
    Load the first frame of all *_q_wxyz entries from an npz file.
    Returns:
        time_s: array
        quats: dict {sensor_name: (4,) wxyz}
    """
    data = np.load(npz_path)
    time_s = data["time_s"]
    quats = {}

    for k in data.files:
        if k.endswith("_q_wxyz"):
            sensor = k.replace("_q_wxyz", "")
            quats[sensor] = data[k][0]  # first frame

    return time_s, quats


def compute_positions_and_rotations_first_frame(quats):
    """
    Forward kinematics for a single frame.
    Returns:
        positions: dict joint -> xyz
        rotations: dict joint -> world Rotation
    """
    positions = {"pelvis_center": np.zeros(3)}
    rotations = {"pelvis_center": Rotation.identity()}

    for joint_name, (parent, sensor, neutral_vec) in SEGMENTS.items():
        parent_rot = rotations[parent]

        if sensor is not None and sensor in quats:
            local_rot = quat_wxyz_to_rot(quats[sensor])
        else:
            local_rot = Rotation.identity()

        world_rot = parent_rot * local_rot
        positions[joint_name] = positions[parent] + world_rot.apply(neutral_vec)
        rotations[joint_name] = world_rot

    return positions, rotations


def draw_local_axes(ax, origin, rot, scale=0.08, lw=2.0):
    """
    Draw xyz axes of a local reference frame at origin using a world rotation.
    x=red, y=green, z=blue
    """
    origin = np.asarray(origin)
    axes = np.eye(3) * scale
    world_axes = rot.apply(axes)  # 3 vectors

    colors = ["r", "g", "b"]
    labels = ["x", "y", "z"]
    for i in range(3):
        v = world_axes[i]
        ax.quiver(
            origin[0], origin[1], origin[2],
            v[0], v[1], v[2],
            color=colors[i],
            linewidth=lw,
            arrow_length_ratio=0.18,
            normalize=False,
        )
        # optional small label at arrow tip
        tip = origin + v
        ax.text(tip[0], tip[1], tip[2], labels[i], color=colors[i], fontsize=8)


def plot_first_frame(npz_path, axis_scale=0.10):
    time_s, quats = load_first_frame(npz_path)
    positions, rotations = compute_positions_and_rotations_first_frame(quats)

    # Connections for a stick figure
    connections = [
        ("pelvis_center", "pelvis_right"),
        ("pelvis_center", "pelvis_left"),
        ("pelvis_center", "spine"),
        ("spine", "shoulder_right"),
        ("spine", "shoulder_left"),
        ("shoulder_right", "elbow_right"),
        ("elbow_right", "wrist_right"),
        ("wrist_right", "hand_right"),
        ("shoulder_left", "elbow_left"),
        ("elbow_left", "wrist_left"),
        ("wrist_left", "hand_left"),
        ("pelvis_right", "knee_right"),
        ("knee_right", "ankle_right"),
        ("ankle_right", "toe_right"),
        ("pelvis_left", "knee_left"),
        ("knee_left", "ankle_left"),
        ("ankle_left", "toe_left"),
    ]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Draw stick figure
    for a, b in connections:
        p1 = positions[a]
        p2 = positions[b]
        ax.plot([p1[0], p2[0]],
                [p1[1], p2[1]],
                [p1[2], p2[2]],
                "k-", lw=2)

    # Draw joints
    pts = np.array(list(positions.values()))
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=20, c="k")

    # Draw local xyz axes at each joint that has a rotation definition
    for joint_name, pos in positions.items():
        rot = rotations.get(joint_name, Rotation.identity())
        draw_local_axes(ax, pos, rot, scale=axis_scale, lw=1.5)

    # Make plot readable
    all_pts = np.array(list(positions.values()))
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = (maxs - mins).max() / 2.0
    radius = max(radius, 0.35)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

    ax.set_xlabel("X (Right)")
    ax.set_ylabel("Y (Forward)")
    ax.set_zlabel("Z (Up)")
    ax.set_title("First Frame Stick Figure with Local XYZ Axes")

    # Equal-ish aspect in 3D
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass

    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot the first frame of the skeleton with local xyz axes."
    )
    parser.add_argument(
        "npz_path",
        nargs="?",
        default="outputs/orientation_novice.npz",
        help="Path to orientation .npz file",
    )
    parser.add_argument(
        "--axis-scale",
        type=float,
        default=0.10,
        help="Length of each local axis arrow",
    )
    args = parser.parse_args()

    plot_first_frame(args.npz_path, axis_scale=args.axis_scale)