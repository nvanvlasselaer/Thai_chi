"""Novice versus trained, event by event.

Each family's events are paired through the whole-recording alignment, so a
pair is the same movement of the form performed by both participants.  That
makes the paired difference the meaningful comparison here -- not the spread
across events, which (every event being a different movement, performed once)
describes the choreography rather than either person's consistency.

The table is descriptive.  With one participant per group, nothing in it
separates a training effect from two people simply moving differently, and
consecutive events of one performance are not independent samples, so no
p-values are given.  What it does say is how consistently one performer
differs from the other across the matched parts of the same form: the median
paired difference, its spread, and in what fraction of pairs the trained value
is the higher.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PAIRED_METRICS = {
    "trunk": (
        "trunk_pelvis_lag_s",
        "trunk_yaw_sparc",
        "trunk_yaw_excursion_deg",
        "lumbar_ml_acc_rms_mps2",
        "lumbar_ap_acc_rms_mps2",
        "lumbar_frontal_tilt_sd_deg",
        "lumbar_sagittal_tilt_sd_deg",
        "lumbar_rms_angular_velocity_dps",
    ),
    "knee": (
        "support_duration_s",
        "peak_lift_index",
        "peak_knee_flexion_deg",
        "support_ml_acc_rms_mps2",
        "support_ap_acc_rms_mps2",
        "support_frontal_tilt_sd_deg",
        "support_sagittal_tilt_sd_deg",
        "support_rms_angular_velocity_dps",
        "settle_ml_acc_rms_mps2",
        "time_to_stabilization_s",
    ),
    "smooth": (
        "turn_duration_s",
        "chest_yaw_excursion_deg",
        "chest_yaw_peak_rate_dps",
        "chest_yaw_sparc",
        "chest_yaw_submovement_rate_hz",
        "chest_pelvis_lag_s",
        "chest_pelvis_gain",
        "relative_yaw_range_deg",
    ),
}
"""Per family, the metrics compared.  Lags of the single-leg family are left
out: on those windows the correlation behind them is weak."""

FAMILY_TITLE = {"trunk": "trunk rotation", "knee": "single-leg stance", "smooth": "turn"}


def paired_comparison(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per family and metric, over the events present in both recordings."""
    rows = []
    for family, frame in frames.items():
        if frame is None or len(frame) == 0 or "event_id" not in frame:
            continue
        novice = frame[frame.trial == "Novice"].set_index("event_id")
        trained = frame[frame.trial == "Trained"].set_index("event_id")
        common = [event_id for event_id in novice.index if event_id in trained.index]
        if not common:
            continue
        for metric in PAIRED_METRICS.get(family, ()):
            if metric not in frame:
                continue
            a = novice.loc[common, metric].astype(float).to_numpy()
            b = trained.loc[common, metric].astype(float).to_numpy()
            keep = np.isfinite(a) & np.isfinite(b)
            a, b = a[keep], b[keep]
            difference = b - a
            q25, q75 = (np.percentile(difference, [25, 75]) if len(difference) else (np.nan, np.nan))
            rows.append({
                "family": FAMILY_TITLE.get(family, family),
                "metric": metric,
                "n_pairs": int(len(a)),
                "novice_median": float(np.median(a)) if len(a) else float("nan"),
                "trained_median": float(np.median(b)) if len(b) else float("nan"),
                "median_difference": float(np.median(difference)) if len(difference) else float("nan"),
                "difference_q25": float(q25),
                "difference_q75": float(q75),
                "trained_higher_fraction": float(np.mean(difference > 0)) if len(difference) else float("nan"),
            })
    return pd.DataFrame(rows)
