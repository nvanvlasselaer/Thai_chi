#!/usr/bin/env python3
"""How much each metric moves when the window boundaries are placed slightly differently.

    python3 analysis/tools/boundary_robustness.py [--jitter 0.25] [--replicates 20] [--seed 0]

Every boundary of every event in the active selection's working session is
moved by a random amount within +/-``jitter`` seconds -- about the precision
of a manual drag -- ``replicates`` times, and each metric is recomputed on the
moved windows.  Robustness is reported as ICC(1,1), events (per recording) as
subjects and jittered replicates as repeated measurements: the share of the
metric's variance that is between events rather than due to where exactly the
boundaries fell.

This is **boundary robustness**, not reliability.  It says how much a number
depends on curation, not how much it would change if the participant
performed the form again; that needs a test-retest session.  Reads the
recordings and the session, and writes nothing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from analysis import detection, recordings, sessions
from analysis.balance_metrics import (
    balance_signals,
    compute_single_leg_metrics,
    compute_trunk_rotation_balance_metrics,
)
from analysis.comparison import PAIRED_METRICS
from analysis.kinematics import load_all_trials
from analysis.smoothness_metrics import compute_turn_metrics


def icc_1_1(values: np.ndarray) -> float:
    """ICC(1,1) of a subjects x replicates table."""
    values = values[np.all(np.isfinite(values), axis=1)]
    n, k = values.shape
    if n < 2 or k < 2:
        return float("nan")
    grand = values.mean()
    between = k * np.sum((values.mean(axis=1) - grand) ** 2) / (n - 1)
    within = np.sum((values - values.mean(axis=1, keepdims=True)) ** 2) / (n * (k - 1))
    denominator = between + (k - 1) * within
    return float((between - within) / denominator) if denominator > 0 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--jitter", type=float, default=0.25, help="largest boundary shift, seconds")
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    trials = load_all_trials(recordings.active())
    session = sessions.load_session()
    if session is None:
        sys.exit("No session saved for the active selection.")
    _, stab_params, _, _ = detection.params_from_dict(session.get("detector") or {})
    signals = {label: balance_signals(t.kin, t.trial, t.fs, stab_params) for label, t in trials.items()}
    lag_pad = session.get("lag_pad_s")
    knee_lag = tuple(session.get("knee_lag_window_s") or (1.0, 2.0))

    def metrics_for(family: str, event: dict, label: str, bounds: list[int]) -> dict:
        loaded = trials[label]
        if family == "trunk":
            return compute_trunk_rotation_balance_metrics(label, loaded.fs, event["event_id"], *bounds,
                                                          signals[label], lag_pad_s=lag_pad)
        if family == "knee":
            return compute_single_leg_metrics(label, loaded.kin, loaded.fs, event["event_id"],
                                              event.get("lifted_leg"), *bounds, signals[label],
                                              lag_window_s=knee_lag)
        return compute_turn_metrics(label, loaded.fs, event["event_id"], *bounds, signals[label],
                                    direction=event.get("direction"), lag_pad_s=lag_pad)

    rows = []
    for family in sessions.FAMILIES:
        keys = ("event_start", "event_end", "stab_start", "stab_end") if family != "smooth" else ("event_start", "event_end")
        names = PAIRED_METRICS[family]
        table: dict[str, list[list[float]]] = {name: [] for name in names}
        for event in session.get(sessions.FAMILY_KEY[family], []):
            if not event.get("enabled", True):
                continue
            for label, window in event.get("windows", {}).items():
                if label not in trials:
                    continue
                fs, n = trials[label].fs, trials[label].n_samples
                shift = int(round(args.jitter * fs))
                samples = {name: [] for name in names}
                for _ in range(args.replicates):
                    bounds = [int(np.clip(window[key] + rng.integers(-shift, shift + 1), 0, n - 1)) for key in keys]
                    bounds[1] = max(bounds[1], bounds[0] + 2)
                    if len(bounds) == 4:
                        bounds[3] = max(bounds[3], bounds[2] + 2)
                    result = metrics_for(family, event, label, bounds)
                    for name in names:
                        samples[name].append(float(result.get(name, np.nan)))
                for name in names:
                    table[name].append(samples[name])
        for name in names:
            values = np.array(table[name], dtype=float)
            if values.size:
                rows.append({"family": family, "metric": name, "subjects": len(values),
                             "icc": icc_1_1(values)})

    report = pd.DataFrame(rows)
    print(f"Boundary robustness: every boundary jittered by up to +/-{args.jitter:g} s, "
          f"{args.replicates} replicates, ICC(1,1)\n")
    print(report.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
