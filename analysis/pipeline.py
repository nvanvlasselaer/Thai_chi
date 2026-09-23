#!/usr/bin/env python3
"""Batch analysis of the novice and trained Tai Chi recordings.

Parses both Delsys recordings, estimates segment orientations with a 6-axis
Madgwick filter, derives balance-relevant trunk and lower-limb kinematics,
segments trunk-rotation events and monopodal stances (knee flexion > 60°),
computes their balance metrics, and writes every CSV and figure to outputs/.
Takes about a minute, almost all of it in the orientation filter.

Run from the repository root with either of::

    python3 analysis/pipeline.py
    python3 -m analysis.pipeline

The switches that change the results are in :mod:`analysis.config`.  Curated
windows, and the sequence-smoothness family, come from the event editor
(:mod:`analysis.editor.app`) instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    # Run as a script rather than with -m: make the ``analysis`` package importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from analysis import config, detection, detection_v1
from analysis.balance_metrics import (
    compute_asymmetry_metrics,
    compute_knee_balance_metrics,
    compute_trunk_rotation_balance_metrics,
    summarize_knee_flexion_event,
)
from analysis.data_io import make_sensor_inventory, parse_IMU_csv, save_kinematic_variables, save_orientation_npz
from analysis.detection import find_knee_flexion_windows
from analysis.figures import (
    make_knee_flexion_overview_figure,
    make_knee_traceability_figure,
    make_orientation_validation_figure,
    make_trunk_traceability_figure,
)
from analysis.kinematics import Kinematics, compute_kinematics


def detect_trunk_event_windows(
    novice_kin: Kinematics,
    trained_kin: Kinematics,
    novice_fs: float,
    trained_fs: float,
    num_events: int = 6,
) -> tuple[dict[str, list[tuple[int, int, int, int, float]]], list[float]]:
    """Segment trunk-rotation events in both trials and pair them.

    Returns ``{"Novice": [...], "Trained": [...]}`` where each entry is
    ``(event_start, event_end, stab_start, stab_end, peak)`` in sample indices,
    plus the DTW distance of each pairing (v1 only).

    ``config.DETECTOR`` selects the method.  "v2" takes the event boundaries
    from the yaw-velocity segmentation and pairs events by order
    (:func:`analysis.detection.pair_trunk_events`); "v1" uses the original
    fixed 8 s window and matches each novice event in the trained recording by
    banded DTW (:func:`analysis.detection_v1.pair_trunk_events_v1`).
    """
    if config.DETECTOR == "v2":
        paired, novice_events, _ = detection.pair_trunk_events(
            novice_kin, trained_kin, novice_fs, trained_fs,
            detection.TrunkDetectorParams(n_events=num_events),
        )
        windows = {
            "Novice": [
                (start, end, stab_start, stab_end, event.peak)
                for (start, end, stab_start, stab_end, _), event in zip(paired["Novice"], novice_events)
            ],
            "Trained": [
                (start, end, stab_start, stab_end, np.nan)
                for start, end, stab_start, stab_end, _ in paired["Trained"]
            ],
        }
        return windows, []

    return detection_v1.pair_trunk_events_v1(novice_kin, trained_kin, novice_fs, trained_fs, num_events)


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load and parse IMU data
    # ------------------------------------------------------------------
    config.OUTPUT_DIR.mkdir(exist_ok=True)

    novice_trial = parse_IMU_csv(config.NOVICE_CSV, "Novice")
    trained_trial = parse_IMU_csv(config.TRAINED_CSV, "Trained")
    inventory = make_sensor_inventory([novice_trial, trained_trial])
    inventory.to_csv(config.OUTPUT_DIR / "sensor_inventory.csv", index=False)

    # ------------------------------------------------------------------
    # 2. Compute kinematics and save orientation outputs
    # ------------------------------------------------------------------
    novice_kin = compute_kinematics(novice_trial)
    trained_kin = compute_kinematics(trained_trial)
    save_orientation_npz("novice", novice_kin)
    save_orientation_npz("trained", trained_kin)
    save_kinematic_variables("novice", novice_kin, novice_trial.fs)
    save_kinematic_variables("trained", trained_kin, trained_trial.fs)

    # ------------------------------------------------------------------
    # 3. Trunk rotation balance analysis
    #    - Detect characteristic trunk-rotation events in the novice
    #    - Match each event to the best corresponding window in trained
    #    - Compute balance metrics (coordination, smoothness, variability)
    # ------------------------------------------------------------------
    trunk_event_windows, _ = detect_trunk_event_windows(
        novice_kin, trained_kin, novice_trial.fs, trained_trial.fs, num_events=6
    )

    trunk_metric_rows: list[dict] = []
    for (novice_start, novice_end, novice_stab_start, novice_stab_end, _), (
        trained_start,
        trained_end,
        trained_stab_start,
        trained_stab_end,
        _,
    ) in zip(trunk_event_windows["Novice"], trunk_event_windows["Trained"]):
        trunk_metric_rows.append(compute_trunk_rotation_balance_metrics("Novice", novice_kin, novice_trial, novice_start, novice_end, novice_stab_start, novice_stab_end))
        trunk_metric_rows.append(compute_trunk_rotation_balance_metrics("Trained", trained_kin, trained_trial, trained_start, trained_end, trained_stab_start, trained_stab_end))

    trunk_event_rows = []
    for label, events in trunk_event_windows.items():
        for i, (start, end, stab_start, stab_end, peak) in enumerate(events):
            trunk_event_rows.append({
                "trial": label,
                "event_index": i + 1,
                "event_start_s": (start / novice_trial.fs if label == "Novice" else start / trained_trial.fs),
                "event_end_s": (end / novice_trial.fs if label == "Novice" else end / trained_trial.fs),
                "event_peak_s": (peak / novice_trial.fs if label == "Novice" else np.nan),
                "stabilization_start_s": (stab_start / novice_trial.fs if label == "Novice" else stab_start / trained_trial.fs),
                "stabilization_end_s": (stab_end / novice_trial.fs if label == "Novice" else stab_end / trained_trial.fs),
            })
    pd.DataFrame(trunk_event_rows).to_csv(config.OUTPUT_DIR / "trunk_rotation_event_windows.csv", index=False)

    trunk_metrics = pd.DataFrame(trunk_metric_rows)
    trunk_metrics.to_csv(config.OUTPUT_DIR / "trunk_rotation_balance_metrics.csv", index=False)

    # ------------------------------------------------------------------
    # 4. Knee flexion balance analysis
    #    - Detect windows where knee flexion exceeds 60°
    #    - Summarise the trunk response during each knee-flexion window
    #    - Compute balance metrics (stabilization, orientation variability)
    # ------------------------------------------------------------------
    knee_event_rows = []
    knee_events: dict[str, list[dict]] = {"Novice": [], "Trained": []}
    for label, kin, fs in [("Novice", novice_kin, novice_trial.fs), ("Trained", trained_kin, trained_trial.fs)]:
        for side, angle in [("Left", kin.left_knee_deg[:, 0]), ("Right", kin.right_knee_deg[:, 0])]:
            windows_60, _ = find_knee_flexion_windows(angle, fs, threshold_deg=60.0, min_duration_s=0.4, merge_gap_s=0.2)
            for start, end in windows_60:
                event = summarize_knee_flexion_event(label, kin, fs, side, start, end, threshold_deg=60.0)
                knee_event_rows.append(event)
                knee_events[label].append(event)

    knee_balance_rows = []
    for label, kin, trial in [("Novice", novice_kin, novice_trial), ("Trained", trained_kin, trained_trial)]:
        fs = trial.fs
        for ev in knee_events[label]:
            start = int(ev["window_start_s"] * fs)
            end = int(ev["window_end_s"] * fs)
            knee_balance_rows.append(
                compute_knee_balance_metrics(label, kin, trial, ev.get("flexed_leg", ev.get("side")), start, end)
            )

    knee_metrics = pd.DataFrame(knee_balance_rows)
    knee_metrics.to_csv(config.OUTPUT_DIR / "monopodal_stance_balance_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "trial": ev["trial"],
                "flexed_leg": ev.get("flexed_leg", ev.get("side")),
                "stance_leg": ev.get("stance_leg", "Unknown"),
                "window_start_s": ev["window_start_s"],
                "window_end_s": ev["window_end_s"],
                "peak_time_s": ev["peak_time_s"],
                "peak_abs_knee_flexion_deg": ev["peak_abs_knee_flexion_deg"],
            }
            for ev in knee_event_rows
        ]
    ).to_csv(config.OUTPUT_DIR / "monopodal_stance_event_windows.csv", index=False)

    # ------------------------------------------------------------------
    # 5. Generate figures
    # ------------------------------------------------------------------
    make_trunk_traceability_figure(novice_kin, trained_kin, trunk_metrics, trunk_event_windows, novice_trial.fs)
    make_knee_flexion_overview_figure(novice_kin, trained_kin, knee_events, novice_trial.fs)
    make_knee_traceability_figure(novice_kin, trained_kin, knee_metrics, knee_events, novice_trial.fs)
    make_orientation_validation_figure(novice_kin, trained_kin, novice_trial.fs)

    # ------------------------------------------------------------------
    # 6. Print summary
    # ------------------------------------------------------------------
    print("Analysis complete.")
    print("\nTrunk rotation balance metrics:")
    print(trunk_metrics.to_string(index=False))
    if len(knee_metrics) > 0:
        print("\nMonopodal stance balance metrics:")
        print(knee_metrics.to_string(index=False))

        # Calculate Asymmetry
        asym_df = compute_asymmetry_metrics(knee_metrics)
        if len(asym_df) > 0:
            asym_df.to_csv(config.OUTPUT_DIR / "monopodal_stance_asymmetry_metrics.csv", index=False)
            print("\nLeft-Right Asymmetry (Absolute Difference between Stance Legs):")
            print(asym_df.to_string(index=False))
    else:
        print("\nMonopodal stance balance analysis: no windows exceeded 60°.")
    print(f"\nOrientation validation figure: {config.OUTPUT_DIR / 'orientation_validation.png'}")
    print(f"Trunk traceability figure:     {config.OUTPUT_DIR / 'trunk_traceability_figure.png'}")
    print(f"Monopodal stance overview figure: {config.OUTPUT_DIR / 'monopodal_stance_overview_figure.png'}")
    print(f"Monopodal stance traceability figure: {config.OUTPUT_DIR / 'monopodal_stance_traceability_figure.png'}")


if __name__ == "__main__":
    main()
