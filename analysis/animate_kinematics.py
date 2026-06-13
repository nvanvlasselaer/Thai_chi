import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from scipy.spatial.transform import Rotation

# Define segment properties: (parent_joint, sensor_name, neutral_vector_xyz)
#
# Root is 'pelvis_center' at (0,0,0) with identity rotation.
#
# sensor_name may be None for joints that are rigid offsets of their parent
# (no additional local rotation). In that case the parent's world rotation is
# inherited directly and only the neutral_vector offset is applied.
#
# IMPORTANT: entries MUST appear after their parent in this dict so that the
# forward-kinematics traversal always finds a parent's world rotation before
# computing a child's. Topological order is verified at runtime.
SEGMENTS = {
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

# Validate once at import time so any reordering mistake is caught immediately.
_verify_topological_order(SEGMENTS)


def load_kinematics(npz_path):
    data = np.load(npz_path)
    time_s = data["time_s"]
    quats = {}
    for k in data.files:
        if k.endswith("_q_wxyz"):
            sensor = k.replace("_q_wxyz", "")
            quats[sensor] = data[k]
    return time_s, quats


def compute_joint_positions(quats: dict, frame_idx: int) -> dict:
    """
    Forward kinematics with accumulated rotations.

    For every joint the world rotation is:
        world_rot[joint] = world_rot[parent] * local_rot[sensor]

    When sensor is None the joint is a rigid offset: it simply inherits the
    parent's world rotation without adding any new local rotation.

    The joint's world position is:
        pos[joint] = pos[parent] + world_rot[joint].apply(neutral_vec)
    """
    positions = {"pelvis_center": np.zeros(3)}
    rotations = {"pelvis_center": Rotation.identity()}

    for joint_name, (parent, sensor, neutral_vec) in SEGMENTS.items():
        parent_rot = rotations[parent]

        if sensor is not None:
            # Data saved as WXYZ → convert to scipy's XYZW convention.
            q_wxyz = quats[sensor][frame_idx]
            q_xyzw = [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]
            local_rot = Rotation.from_quat(q_xyzw)
        else:
            # Rigid offset: no additional rotation beyond the parent's.
            local_rot = Rotation.identity()

        # Accumulate: world rotation = parent world rotation * local rotation.
        world_rot = parent_rot * local_rot

        # Rotate the segment's local offset vector into world space.
        rotated_vec = world_rot.apply(neutral_vec)

        positions[joint_name] = positions[parent] + rotated_vec
        rotations[joint_name] = world_rot

    return positions


def create_animation(npz_path, output_path, fps=30, start_time=None, end_time=None):
    time_s, quats = load_kinematics(npz_path)

    # Original sampling rate
    orig_dt = time_s[1] - time_s[0]
    orig_fs = 1.0 / orig_dt

    # Downsample to target fps
    step = max(1, int(orig_fs / fps))

    start_idx = 0
    end_idx = len(time_s)
    if start_time is not None:
        start_idx = int(np.searchsorted(time_s, start_time))
    if end_time is not None:
        end_idx = int(np.searchsorted(time_s, end_time))

    frame_indices = list(range(start_idx, end_idx, step))

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Skeleton connections to draw (parent joint → child joint)
    connections = [
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

    lines = []
    for _ in connections:
        line, = ax.plot([], [], [], 'o-', lw=2, markersize=4)
        lines.append(line)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel('X (Right)')
    ax.set_ylabel('Y (Forward)')
    ax.set_zlabel('Z (Up)')
    ax.set_title("Tai Chi Kinematics")

    def update(frame_idx):
        positions = compute_joint_positions(quats, frame_idx)
        for i, (j1, j2) in enumerate(connections):
            p1 = positions[j1]
            p2 = positions[j2]
            lines[i].set_data([p1[0], p2[0]], [p1[1], p2[1]])
            lines[i].set_3d_properties([p1[2], p2[2]])
            if "right" in j2:
                lines[i].set_color('red')
            elif "left" in j2:
                lines[i].set_color('blue')
            else:
                lines[i].set_color('black')
        return lines

    ani = animation.FuncAnimation(fig, update, frames=frame_indices, blit=False)

    print(f"Saving animation to {output_path} (Frames: {len(frame_indices)})...")
    writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
    ani.save(output_path, writer=writer)
    plt.close(fig)
    print("Done!")


DEFAULT_NPZ    = "outputs/orientation_novice.npz"
DEFAULT_OUTPUT = "outputs/animation_novice.mp4"
DEFAULT_START  = 0.0
DEFAULT_END    = 60.0
DEFAULT_FPS    = 60

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="animate_kinematics.py",
        description="Animate Tai Chi Kinematics from IMU orientation data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # shows defaults in --help
    )
    parser.add_argument(
        "npz_path",
        type=str,
        nargs="?",                  # makes the positional optional
        default=DEFAULT_NPZ,
        help="Path to orientation .npz file",
    )
    parser.add_argument(
        "output_path",
        type=str,
        nargs="?",                  # makes the positional optional
        default=DEFAULT_OUTPUT,
        help="Path to output .mp4 file",
    )
    parser.add_argument("--start", type=float, default=DEFAULT_START, help="Start time in seconds")
    parser.add_argument("--end",   type=float, default=DEFAULT_END,   help="End time in seconds")
    parser.add_argument("--fps",   type=int,   default=DEFAULT_FPS,   help="Frames per second")
    args = parser.parse_args()

    print(
        f"Running with:\n"
        f"  npz_path    : {args.npz_path}\n"
        f"  output_path : {args.output_path}\n"
        f"  start       : {args.start} s\n"
        f"  end         : {args.end} s\n"
        f"  fps         : {args.fps}\n"
    )

    create_animation(
        args.npz_path,
        args.output_path,
        fps=args.fps,
        start_time=args.start,
        end_time=args.end,
    )
