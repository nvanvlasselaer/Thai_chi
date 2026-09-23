#!/usr/bin/env python3
"""3D stick-figure animation of a recording from its orientation cache (needs ffmpeg).

    python3 analysis/tools/animate_kinematics.py [npz] [out.mp4] [--start S --end E --fps N]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation
from scipy.signal import butter, sosfiltfilt

if __package__ in (None, ""):
    # Run as a script rather than with -m: make the ``analysis`` package importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis import config
from analysis.data_io import load_orientation_npz
from analysis.tools.skeleton import CONNECTIONS, SEGMENTS


# ----------------------------------------------------------------------
# Quaternion filtering (zero‑phase high‑pass)
# ----------------------------------------------------------------------
def highpass_quat_sos(time_s, quat_wxyz, cutoff=0.05, order=4):
    """
    Zero‑phase high‑pass filter a quaternion time series (N x 4, wxyz order).

    Filtering is applied in the rotation‑vector domain to preserve valid
    orientations after re‑normalisation.

    Parameters
    ----------
    time_s : (N,) array, monotonically increasing time stamps.
    quat_wxyz : (N,4) array, quaternions as [w, x, y, z].
    cutoff : float, high‑pass cutoff frequency in Hz.
    order : int, Butterworth filter order (effective order doubled by sosfiltfilt).

    Returns
    -------
    filtered_wxyz : (N,4) array.
    """
    # Sampling rate
    fs = 1.0 / np.median(np.diff(time_s))
    nyq = 0.5 * fs
    Wn = cutoff / nyq

    # Design high‑pass Butterworth filter (SOS form)
    sos = butter(order, Wn, btype='high', output='sos')

    # Convert to rotation vectors (N x 3)
    # scipy expects (x, y, z, w) order internally
    rotvec = Rotation.from_quat(quat_wxyz[:, [1, 2, 3, 0]]).as_rotvec()

    # Padding to reduce filter transient at edges
    padlen = 3 * len(sos) * 4   # generous symmetric padding
    rv_padded = np.pad(rotvec, ((padlen, padlen), (0, 0)), mode='reflect')
    rv_filt_padded = sosfiltfilt(sos, rv_padded, axis=0)
    rv_filt = rv_filt_padded[padlen:-padlen, :]

    # Back to quaternions (xyzw), then reorder to wxyz
    quat_xyzw = Rotation.from_rotvec(rv_filt).as_quat()   # [x, y, z, w]
    quat_wxyz = np.column_stack([quat_xyzw[:, 3], quat_xyzw[:, :3]])
    return quat_wxyz


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_kinematics(npz_path, filter_hp=False, hp_cutoff=0.05, hp_order=4):
    """
    Load IMU orientations from an .npz file and optionally high‑pass filter them.

    Returns
    -------
    time_s : (N,) array
    quats : dict {sensor_name: (N,4) array of quaternions in wxyz order}
    """
    time_s, quats = load_orientation_npz(npz_path)

    if filter_hp:
        for sensor, q in quats.items():
            print(f"  High‑pass filtering sensor '{sensor}' ...")
            quats[sensor] = highpass_quat_sos(time_s, q, cutoff=hp_cutoff, order=hp_order)

    if filter_hp:
        # Warn about possible edge transients
        edge_sec = 1.0  # rough estimate
        print(f"  Note: filter transients may affect ~{edge_sec} s at start/end.")

    return time_s, quats


# ----------------------------------------------------------------------
# Forward kinematics
# ----------------------------------------------------------------------
# def compute_joint_positions(quats: dict, frame_idx: int) -> dict:
#     """
#     Compute joint world positions for a single frame using accumulated rotations.

#     For each joint:
#         world_rot = parent_world_rot * local_rot
#         position  = parent_position + world_rot.apply(neutral_vector)
#     """
#     positions = {"pelvis_center": np.zeros(3)}
#     rotations = {"pelvis_center": Rotation.identity()}

#     for joint_name, (parent, sensor, neutral_vec) in SEGMENTS.items():
#         parent_rot = rotations[parent]

#         if sensor is not None:
#             q_wxyz = quats[sensor][frame_idx]
#             # Convert wxyz → xyzw for scipy
#             q_xyzw = [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]
#             local_rot = Rotation.from_quat(q_xyzw)
#         else:
#             local_rot = Rotation.identity()

#         world_rot = parent_rot * local_rot
#         rotated_vec = world_rot.apply(neutral_vec)
#         positions[joint_name] = positions[parent] + rotated_vec
#         rotations[joint_name] = world_rot

#     return positions


def compute_joint_positions(quats: dict, frame_idx: int) -> tuple:
    """
    Compute joint world positions for a single frame
    IMU's output GLOBAL orientation in world frame, so we can use it directly to compute the position of each joint in the world frame.
    
    Returns
    -------
    positions : dict {joint_name: (3,) array}
    rotations : dict {joint_name: Rotation object}
    """
    positions = {"pelvis_center": np.zeros(3)}
    rotations = {"pelvis_center": Rotation.identity()}

    for joint_name, (parent, sensor, neutral_vec) in SEGMENTS.items():
        parent_rot = rotations[parent]

        if sensor is not None:
            q_wxyz = quats[sensor][frame_idx]
            # Convert wxyz → xyzw for scipy
            q_xyzw = [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]
            world_rot = Rotation.from_quat(q_xyzw)
        else:
            world_rot = parent_rot

        rotated_vec = world_rot.apply(neutral_vec)
        positions[joint_name] = positions[parent] + rotated_vec
        rotations[joint_name] = world_rot

    return positions, rotations

# ----------------------------------------------------------------------
# Animation
# ----------------------------------------------------------------------
def create_animation(npz_path, output_path, fps=30, start_time=None, end_time=None,
                     filter_hp=False, hp_cutoff=0.05, quiver_scale=0.1, show_quivers=True):
    """Load data, optionally filter, and render a skeleton animation with reference frames."""
    time_s, quats = load_kinematics(
        npz_path, filter_hp=filter_hp, hp_cutoff=hp_cutoff
    )

    orig_fs = 1.0 / np.median(np.diff(time_s))
    step = max(1, int(orig_fs / fps))

    start_idx = 0
    end_idx = len(time_s)
    if start_time is not None:
        start_idx = int(np.searchsorted(time_s, start_time))
    if end_time is not None:
        end_idx = int(np.searchsorted(time_s, end_time))

    frame_indices = list(range(start_idx, end_idx, step))
    print(f"Animating {len(frame_indices)} frames at {fps} fps...")

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    connections = CONNECTIONS

    lines = []
    for _ in connections:
        line, = ax.plot([], [], [], 'o-', lw=2, markersize=4)
        lines.append(line)

    # Create line segments for reference frame axes (X, Y, Z)
    axis_lines = {}
    
    if show_quivers:
        for joint_name in SEGMENTS.keys():
            axis_lines[joint_name] = {
                'x': ax.plot([], [], [], color='red', lw=1.5, alpha=0.7)[0],    # X-axis (red)
                'y': ax.plot([], [], [], color='green', lw=1.5, alpha=0.7)[0],  # Y-axis (green)
                'z': ax.plot([], [], [], color='blue', lw=1.5, alpha=0.7)[0],   # Z-axis (blue)
            }

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel('X (Right)')
    ax.set_ylabel('Y (Forward)')
    ax.set_zlabel('Z (Up)')
    filter_str = " (filtered)" if filter_hp else ""
    ax.set_title(f"Tai Chi Kinematics{filter_str}")

    def update(frame_idx):
        positions, rotations = compute_joint_positions(quats, frame_idx)
        
        # Update skeleton lines
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
        
        # Update reference frame axes as line segments
        if show_quivers:
            for joint_name in SEGMENTS.keys():
                pos = positions[joint_name]
                rot = rotations[joint_name]
                
                # Get unit vectors for X, Y, Z axes in world frame
                x_axis = rot.apply([1, 0, 0])
                y_axis = rot.apply([0, 1, 0])
                z_axis = rot.apply([0, 0, 1])
                
                # Update X-axis line (red)
                x_end = pos + quiver_scale * x_axis
                axis_lines[joint_name]['x'].set_data([pos[0], x_end[0]], [pos[1], x_end[1]])
                axis_lines[joint_name]['x'].set_3d_properties([pos[2], x_end[2]])
                
                # Update Y-axis line (green)
                y_end = pos + quiver_scale * y_axis
                axis_lines[joint_name]['y'].set_data([pos[0], y_end[0]], [pos[1], y_end[1]])
                axis_lines[joint_name]['y'].set_3d_properties([pos[2], y_end[2]])
                
                # Update Z-axis line (blue)
                z_end = pos + quiver_scale * z_axis
                axis_lines[joint_name]['z'].set_data([pos[0], z_end[0]], [pos[1], z_end[1]])
                axis_lines[joint_name]['z'].set_3d_properties([pos[2], z_end[2]])
        
        return lines

    ani = animation.FuncAnimation(fig, update, frames=frame_indices, blit=False)

    writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
    ani.save(output_path, writer=writer)
    # plt.close(fig)
    plt.show()
    print(f"Saved animation to {output_path}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Animate Tai Chi Kinematics from IMU orientation data, "
                    "optionally high‑pass filtered to remove slow drift.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "npz_path", nargs="?", default=str(config.OUTPUT_DIR / "orientation_trained.npz"),
        help="Path to orientation .npz file"
    )
    parser.add_argument(
        "output_path", nargs="?", default=str(config.OUTPUT_DIR / "animation_trained.mp4"),
        help="Path to output .mp4 file"
    )
    parser.add_argument("--start", type=float, default=0.0, help="Start time (s)")
    parser.add_argument("--end", type=float, default=60.0, help="End time (s)")
    parser.add_argument("--fps", type=int, default=60, help="Output frames per second")
    parser.add_argument(
        "--filter-highpass", action="store_true",
        help="Apply a 0.05 Hz high‑pass filter to quaternions before animation"
    )
    parser.add_argument(
        "--hp-cutoff", type=float, default=0.05,
        help="High‑pass cutoff frequency in Hz (used with --filter-highpass)"
    )
    args = parser.parse_args()

    print(
        f"Running with:\n"
        f"  npz_path      : {args.npz_path}\n"
        f"  output_path   : {args.output_path}\n"
        f"  start         : {args.start} s\n"
        f"  end           : {args.end} s\n"
        f"  fps           : {args.fps}\n"
        f"  filter-highpass: {args.filter_highpass}\n"
        f"  hp-cutoff     : {args.hp_cutoff} Hz\n"
    )

    create_animation(
        args.npz_path,
        args.output_path,
        fps=args.fps,
        start_time=args.start,
        end_time=args.end,
        filter_hp=args.filter_highpass,
        hp_cutoff=args.hp_cutoff,
    )