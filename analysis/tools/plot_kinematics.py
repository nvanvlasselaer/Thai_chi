#!/usr/bin/env python3
"""The kinematics check's sensor table and joint-angle panels, outside the dashboard.

    python3 analysis/tools/plot_kinematics.py [--role Novice|Trained] [--recording FILE] [--save PNG]

Shows one recording -- by default the selection's novice one, else the other
-- as the dashboard's Kinematics check page does (see
:mod:`analysis.kinematics_check`): prints the sensor table and plots the joint
angles, left blue and right red, with the neutral pose (green) and the quiet
standing spans (grey) shaded.  Needs the recording's orientation cache, i.e.
stage 1 run once.  The stick figure is in ``animate_kinematics`` and
``plot_frame``.
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

if __package__ in (None, ""):
    # Run as a script rather than with -m: make the ``analysis`` package importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis import kinematics_check, recordings
from analysis.kinematics import load_trial

SIDE_COLOURS = {kinematics_check.LEFT: "#3a86c8", kinematics_check.RIGHT: "#d1495b"}
COMPONENT_COLOURS = ("#1f77b4", "#e76f51", "#2a9d8f")


def main() -> None:
    parser = argparse.ArgumentParser(description="The kinematics check of one recording, outside the dashboard.")
    parser.add_argument("--role", choices=["Novice", "Trained"], default="Novice",
                        help="which recording of the selection to show (default: the novice one, else the other)")
    parser.add_argument("--recording", help="a file name in data/, instead of one of the selection")
    parser.add_argument("--save", type=Path, help="write the figure to this file instead of showing it")
    args = parser.parse_args()

    recording = recordings.find(args.recording) if args.recording else recordings.selected_recording(args.role)
    if recording is None:
        sys.exit("No recording to show: select one in the dashboard, or pass --recording with a file in data/.")
    if not recording.orientation_current():
        sys.exit(f"{recording.name} has no current orientation cache: run stage 1 of the pipeline first.")
    loaded = load_trial(recording, args.role)
    kin, fs = loaded.kin, loaded.fs

    rotations = kinematics_check.dedrifted_rotations(kin, fs)
    check = kinematics_check.sensor_check(kin, loaded.trial, rotations)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(check.drop(columns=["notes"]).to_string(index=False))
    for row in check.itertuples():
        if row.notes:
            print(f"  {row.sensor}: {row.notes}")
    print(f"\n{recording.name}: {kinematics_check.summary(check)}.")

    time, panels = kinematics_check.panel_series(kin, fs, 50.0, rotations)
    neutral = [index / fs for index in kin.calibration.neutral]
    quiet = [(start / fs, end / fs) for start, end in kin.calibration.quiet_spans]
    fig, axes = plt.subplots(len(panels), 1, sharex=True, figsize=(14, 1.9 * len(panels)))
    fig.suptitle(f"Kinematics check: {recording.name}", fontweight="bold")
    for ax, panel in zip(axes, panels):
        for start, end in quiet:
            ax.axvspan(start, end, color="#9aa0a6", alpha=0.15, lw=0)
        ax.axvspan(*neutral, color="#57a773", alpha=0.35, lw=0)
        for k, (label, side, values) in enumerate(panel["traces"]):
            ax.plot(time, values, lw=0.9, label=label,
                    color=SIDE_COLOURS.get(side, COMPONENT_COLOURS[k % len(COMPONENT_COLOURS)]))
        ax.set_title(panel["title"], fontsize=9, loc="left")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=7, ncol=3)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=110)
        print(f"Saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
