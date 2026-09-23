"""The output figures, written to ``config.OUTPUT_DIR``.

The traceability figures draw the signal the event boundaries were taken from,
with the bands on it, above a bar chart of the metrics computed over those
bands, so every number can be traced back to a stretch of the recording.
"""

from __future__ import annotations

import os
from pathlib import Path

# Matplotlib needs a writable config directory, which the default may not be.
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import config
from analysis.detection_v1 import find_stabilization as find_stabilization_v1
from analysis.kinematics import Kinematics
from analysis.signals import highpass_detrend, lowpass_signal, resample_filtered_full


TRIAL_COLOURS = {"Novice": "#d1495b", "Trained": "#2a9d8f"}


def _metric_bars(fig: plt.Figure, cell, metrics: pd.DataFrame, metric_names: list[str], pretty: list[str]) -> None:
    """One small novice-vs-trained bar chart per metric, side by side in ``cell``."""
    gs_bars = cell.subgridspec(1, len(metric_names))
    for i, (m, p) in enumerate(zip(metric_names, pretty)):
        ax = fig.add_subplot(gs_bars[0, i])
        nov_val = metrics.loc[metrics.trial == "Novice", m].mean()
        train_val = metrics.loc[metrics.trial == "Trained", m].mean()

        bars = ax.bar([0, 1], [nov_val, train_val], color=[TRIAL_COLOURS["Novice"], TRIAL_COLOURS["Trained"]])
        ax.bar_label(bars, fmt="%.3g", padding=3, fontsize=8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Nov", "Train"], fontsize=9)
        ax.set_title(p, fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


# ---------------------------------------------------------------------------
# Trunk rotation
# ---------------------------------------------------------------------------


def make_trunk_traceability_figure(
    novice: Kinematics,
    trained: Kinematics,
    metrics: pd.DataFrame,
    windows: dict[str, list[tuple[int, int, int, int, float]]],
    fs: float,
) -> None:
    fig = plt.figure(figsize=(12, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.15, 1.15, 1.0])
    axes = [fig.add_subplot(gs[i, 0]) for i in range(2)]

    for ax, label, kin in [(axes[0], "Novice", novice), (axes[1], "Trained", trained)]:
        t = kin.t
        z = highpass_detrend(kin.trunk_rel_euler_deg[:, 2], fs, cutoff_hz=0.05)
        z = lowpass_signal(z, fs, cutoff_hz=4.0)

        ax.plot(t, z, color="#1f77b4", lw=1.2, label="rel z")

        for i, (event_start, event_end, stab_start, stab_end, peak) in enumerate(windows[label]):
            ax.axvspan(event_start / fs, event_end / fs, color="#f2b134", alpha=0.22, label="aligned event" if i == 0 else "")
            ax.axvspan(stab_start / fs, stab_end / fs, color="#57a773", alpha=0.18, label="stabilization" if i == 0 else "")
            if label == "Novice":
                ax.scatter([t[peak]], [z[peak]], s=30, color="#d1495b", zorder=4, label="z peak" if i == 0 else "")

        ax.set_ylabel(f"{label}\nangle (deg)")
        ax.grid(True, color="#dddddd", lw=0.6)

        lines1, labels1 = ax.get_legend_handles_labels()
        ax.legend(lines1, labels1, loc="upper right", fontsize=8, frameon=False, ncol=4)

    metric_names = [
        "trunk_pelvis_lag_s",
        "weight_shift_log10_dimensionless_jerk",
        "lumbar_orientation_variability_deg",
        "corrective_peak_rate_hz",
        "lumbar_ml_acc_variance_g2",
        "lumbar_ap_acc_variance_g2",
    ]
    pretty = ["yaw lag (s)", "log10 jerk", "orient var (deg)", "corr rate (Hz)", "ML sway (g²)", "AP sway (g²)"]

    _metric_bars(fig, gs[2], metrics, metric_names, pretty)

    fig.suptitle("Trunk Rotation Balance Traceability: Novice vs Trained Tai Chi", fontsize=14, fontweight="bold")
    fig.savefig(config.OUTPUT_DIR / "trunk_traceability_figure.png", dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Monopodal stance
# ---------------------------------------------------------------------------


def make_knee_flexion_overview_figure(
    novice: Kinematics,
    trained: Kinematics,
    knee_events: dict[str, list[dict[str, float | str]]],
    fs: float,
) -> None:
    fig = plt.figure(figsize=(11, 9.2), constrained_layout=True)
    gs = fig.add_gridspec(4, 1, height_ratios=[1.0, 0.9, 1.0, 0.9])
    axes = [fig.add_subplot(gs[i, 0]) for i in range(4)]

    trial_specs = [
        (axes[0], axes[1], "Novice", novice),
        (axes[2], axes[3], "Trained", trained),
    ]

    for knee_ax, trunk_ax, label, kin in trial_specs:
        t = kin.t
        left_knee = np.abs(lowpass_signal(kin.left_knee_deg[:, 0], fs, cutoff_hz=6.0))
        right_knee = np.abs(lowpass_signal(kin.right_knee_deg[:, 0], fs, cutoff_hz=6.0))
        trunk_y = lowpass_signal(kin.eulers_deg["lumbar"][:, 1], fs, cutoff_hz=4.0)
        trunk_x = lowpass_signal(kin.eulers_deg["lumbar"][:, 0], fs, cutoff_hz=4.0)
        trunk_z = lowpass_signal(highpass_detrend(kin.trunk_rel_euler_deg[:, 2], fs, cutoff_hz=0.05), fs, cutoff_hz=4.0)

        knee_ax.plot(t, left_knee, lw=1.0, label="left knee |flexion|")
        knee_ax.plot(t, right_knee, lw=1.0, label="right knee |flexion|")
        knee_ax.axhline(60.0, lw=1.0, linestyle="--", color="#444444", label="60° threshold")
        knee_ax.set_ylabel(f"{label}\nknee flexion (deg)")
        knee_ax.grid(True, color="#dddddd", lw=0.6)

        seen_knee_labels: set[str] = set()
        for ev in knee_events[label]:
            side = ev.get("flexed_leg", ev.get("side", "Unknown"))
            color = "#f2b134" if side == "Left" else "#8ecae6"
            knee_label = f"{side} flexion >60°"
            knee_ax.axvspan(
                ev["window_start_s"],
                ev["window_end_s"],
                color=color,
                alpha=0.18,
                label=knee_label if knee_label not in seen_knee_labels else "",
            )
            seen_knee_labels.add(knee_label)
            knee_ax.scatter([ev["peak_time_s"]], [ev["peak_abs_knee_flexion_deg"]], s=18, color="#d1495b", zorder=4)

        trunk_ax.plot(t, trunk_y, lw=1.0, label="lumbar y")
        trunk_ax.plot(t, trunk_x, lw=1.0, label="lumbar x")
        trunk_ax.plot(t, trunk_z, lw=1.0, label="trunk-relative z")
        trunk_ax.set_ylabel("trunk angle (deg)")
        trunk_ax.set_xlabel("time (s)")
        trunk_ax.grid(True, color="#dddddd", lw=0.6)

        seen_trunk_labels: set[str] = set()
        for ev in knee_events[label]:
            side = ev.get("flexed_leg", ev.get("side", "Unknown"))
            color = "#f2b134" if side == "Left" else "#8ecae6"
            knee_label = f"{side} knee window"
            trunk_ax.axvspan(
                ev["window_start_s"],
                ev["window_end_s"],
                color=color,
                alpha=0.10,
                label=knee_label if knee_label not in seen_trunk_labels else "",
            )
            seen_trunk_labels.add(knee_label)

        knee_ax.legend(frameon=False, fontsize=8, ncol=3)
        trunk_ax.legend(frameon=False, fontsize=8, ncol=3)

    fig.suptitle("Monopodal Stance (> 60° Flexion) and Trunk Balance Response", fontsize=14, fontweight="bold")
    fig.savefig(config.OUTPUT_DIR / "monopodal_stance_overview_figure.png", dpi=220)
    plt.close(fig)


def make_knee_traceability_figure(
    novice: Kinematics,
    trained: Kinematics,
    knee_metrics: pd.DataFrame,
    knee_events: dict[str, list[dict[str, float | str]]],
    fs: float,
    stab_overrides: dict[str, list[tuple[int, int]]] | None = None,
) -> None:
    fig = plt.figure(figsize=(13, 8.8), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.15, 1.15, 1.0])
    axes = [fig.add_subplot(gs[i, 0]) for i in range(2)]

    for ax, label, kin in [
        (axes[0], "Novice", novice),
        (axes[1], "Trained", trained),
    ]:
        t = kin.t
        knee = np.maximum(
            np.abs(lowpass_signal(kin.left_knee_deg[:, 0], fs, cutoff_hz=6.0)),
            np.abs(lowpass_signal(kin.right_knee_deg[:, 0], fs, cutoff_hz=6.0)),
        )

        ax.plot(t, knee, color="#1f77b4", lw=1.2, label="max knee flexion")
        ax.axhline(60.0, color="#d1495b", linestyle="--", alpha=0.7, label="60° threshold")

        for i, ev in enumerate(knee_events[label]):
            ax.axvspan(
                ev["window_start_s"],
                ev["window_end_s"],
                color="#f2b134",
                alpha=0.22,
                label="knee-flexion event" if i == 0 else "",
            )

            if stab_overrides is not None:
                stab_start, stab_end = stab_overrides[label][i]
            else:
                peak_idx = int(ev["peak_time_s"] * fs)
                stab_start, stab_end = find_stabilization_v1(kin, fs, peak_idx)

            ax.axvspan(
                stab_start / fs,
                stab_end / fs,
                color="#57a773",
                alpha=0.18,
                label="stabilization" if i == 0 else "",
            )

            ax.scatter(
                [ev["peak_time_s"]],
                [ev["peak_abs_knee_flexion_deg"]],
                s=30,
                color="#d1495b",
                zorder=4,
                label="peak flexion" if i == 0 else "",
            )

        ax.set_ylabel(f"{label}\nknee flexion (deg)")
        ax.grid(True, color="#dddddd", lw=0.6)
        ax.legend(frameon=False, fontsize=8)

    metric_names = [
        "time_to_stabilization_s",
        "lumbar_ml_acc_variance_g2",
        "lumbar_orientation_variability_deg",
        "corrective_peak_rate_hz",
        "trunk_pelvis_lag_s",
        "trunk_pelvis_pitch_lag_s",
    ]

    pretty = [
        "stab. time (s)",
        "ML sway (g²)",
        "orient var (deg)",
        "corr rate (Hz)",
        "yaw lag (s)",
        "pitch lag (s)",
    ]

    _metric_bars(fig, gs[2], knee_metrics, metric_names, pretty)

    fig.suptitle(
        "Monopodal Stance Balance Traceability: Novice vs Trained Tai Chi",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(config.OUTPUT_DIR / "monopodal_stance_traceability_figure.png", dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sequence segments
# ---------------------------------------------------------------------------


def make_sequence_smoothness_figure(
    yaw: dict[str, tuple[np.ndarray, np.ndarray]],
    fs: dict[str, float],
    metrics: pd.DataFrame,
    waveforms: dict[str, dict[str, object]],
) -> None:
    """Traceability figure: where the segments fell, and how alike they were.

    The left column is the same "show your working" view the other two families
    get -- the signal the boundaries were taken from, with the bands drawn on
    it, so a number can always be traced back to a piece of the recording.  The
    right column is the corridor that the consistency metrics summarise.
    """
    trials = [label for label in ("Novice", "Trained") if label in yaw]
    fig = plt.figure(figsize=(13, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(len(trials) + 1, 2, width_ratios=[2.0, 1.0],
                          height_ratios=[1.15] * len(trials) + [1.0])

    for row, label in enumerate(trials):
        chest_yaw, pelvis_yaw = yaw[label]
        rate = fs[label]
        time = np.arange(len(chest_yaw)) / rate
        rows = metrics[metrics.trial == label]

        ax = fig.add_subplot(gs[row, 0])
        ax.plot(time, chest_yaw, color="#1f77b4", lw=1.1, label="chest yaw")
        ax.plot(time, pelvis_yaw, color="#8ecae6", lw=1.0, label="pelvis yaw")
        ax.axhline(0.0, color="#888888", lw=0.8, ls=":")
        # Segments abut, so a single shade would read as one long block; the
        # alternating alpha is what makes the individual parts countable.
        for i, (_, segment) in enumerate(rows.iterrows()):
            ax.axvspan(segment["segment_start_s"], segment["segment_end_s"],
                       color="#f2b134", alpha=0.28 if i % 2 else 0.12,
                       label="sequence segment" if i == 0 else "")
        ax.set_ylabel(f"{label}\nyaw (deg)")
        ax.grid(True, color="#dddddd", lw=0.6)
        ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=3)

        ax = fig.add_subplot(gs[row, 1])
        waveform = waveforms.get(label, {})
        mean = np.asarray(waveform.get("mean_waveform_deg", []), dtype=float)
        sd = np.asarray(waveform.get("sd_waveform_deg", []), dtype=float)
        if mean.size and not np.all(np.isnan(mean)):
            percent = np.linspace(0.0, 100.0, len(mean))
            colour = TRIAL_COLOURS.get(label, "#444444")
            ax.fill_between(percent, mean - sd, mean + sd, color=colour, alpha=0.25,
                            label="+/- 1 SD")
            ax.plot(percent, mean, color=colour, lw=1.6, label="mean")
            ax.set_title(
                f"VR {waveform.get('waveform_variance_ratio', float('nan')):.2f}  "
                f"SD {waveform.get('waveform_mean_sd_deg', float('nan')):.1f} deg",
                fontsize=10,
            )
            ax.legend(loc="upper right", fontsize=8, frameon=False)
        ax.set_ylabel("aligned yaw (deg)")
        ax.grid(True, color="#dddddd", lw=0.6)

    fig.axes[-2].set_xlabel("time (s)")
    fig.axes[-1].set_xlabel("percent of segment")

    metric_names = [
        "chest_yaw_log10_dimensionless_jerk",
        "chest_yaw_sparc",
        "chest_yaw_submovement_rate_hz",
        "chest_pelvis_lag_s",
        "chest_pelvis_gain",
        "relative_yaw_range_deg",
    ]
    pretty = ["log10 jerk", "SPARC", "submov (Hz)", "lag (s)", "gain", "rel ROM (deg)"]

    bars_gs = gs[len(trials), :].subgridspec(1, len(metric_names))
    for i, (name, title) in enumerate(zip(metric_names, pretty)):
        ax = fig.add_subplot(bars_gs[0, i])
        means = [metrics.loc[metrics.trial == label, name].mean() for label in trials]
        sds = [metrics.loc[metrics.trial == label, name].std(ddof=1) for label in trials]
        bars = ax.bar(range(len(trials)), means, yerr=sds, capsize=3,
                      color=[TRIAL_COLOURS.get(label, "#444444") for label in trials])
        ax.bar_label(bars, fmt="%.3g", padding=3, fontsize=8)
        ax.set_xticks(range(len(trials)))
        ax.set_xticklabels([label[:5] for label in trials], fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Sequence smoothness and chest-pelvis coordination, per part of the form",
                 fontsize=14, fontweight="bold")
    fig.savefig(config.OUTPUT_DIR / "sequence_smoothness_figure.png", dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------------


def make_orientation_validation_figure(novice: Kinematics, trained: Kinematics, fs: float) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.8), sharex=False, constrained_layout=True)
    for row, (label, kin) in enumerate([("Novice", novice), ("Trained", trained)]):
        for col, sensor in enumerate(["lumbar", "chestbone"]):
            ax = axes[row, col]
            target_time, x = resample_filtered_full(kin.t, kin.eulers_deg[sensor][:, 0], fs, 10.0)
            _, y = resample_filtered_full(kin.t, kin.eulers_deg[sensor][:, 1], fs, 10.0)
            _, z = resample_filtered_full(kin.t, kin.eulers_deg[sensor][:, 2], fs, 10.0)
            ax.plot(target_time, x, lw=0.9, label="x")
            ax.plot(target_time, y, lw=0.9, label="y")
            ax.plot(target_time, z, lw=0.9, label="z")
            ax.set_title(f"{label} {sensor} orientation")
            ax.set_xlabel("time (s)")
            ax.set_ylabel("angle (deg)")
            ax.grid(True, color="#dddddd", lw=0.6)
            ax.legend(frameon=False, fontsize=8, ncol=3)
    fig.suptitle("Orientation Validation: Lumbar and Chestbone", fontsize=13, fontweight="bold")
    fig.savefig(config.OUTPUT_DIR / "orientation_validation.png", dpi=220)
    plt.close(fig)
