"""The output figures.

The traceability figures draw the signal the event boundaries were taken from,
with the bands on it, above a bar chart of the metrics computed over those
bands, so every number can be traced back to a stretch of the recording.  Each
takes the recordings to draw as ``{role: ...}`` dicts -- one row per recording,
so a novice-only or trained-only analysis gets a figure of its own -- and names
the source files underneath, so a figure copied into a report still says what
it was made from.

Figures are built with :class:`matplotlib.figure.Figure` rather than pyplot, so
they always render off-screen through Agg.  With pyplot they would go through
whatever interactive backend is active -- on macOS that one shrinks a figure
taller than the screen before saving it, so the same analysis wrote a
different PNG from the command line than from the dashboard.  It also keeps
pyplot's global state out of the dashboard's worker threads.
"""

from __future__ import annotations

import os
from pathlib import Path

# Matplotlib needs a writable config directory, which the default may not be.
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))

import numpy as np
from matplotlib.figure import Figure
import pandas as pd

from analysis.config import TRIALS
from analysis.detection_v1 import find_stabilization as find_stabilization_v1
from analysis.kinematics import Kinematics
from analysis.signals import highpass_detrend, lowpass_signal, resample_filtered_full


TRIAL_COLOURS = {"Novice": "#d1495b", "Trained": "#2a9d8f"}
SHORT = {"Novice": "Nov", "Trained": "Train"}


def _roles(recordings: dict) -> list[str]:
    return [label for label in TRIALS if label in recordings]


def _comparison(labels: list[str]) -> str:
    return " vs ".join(labels) if len(labels) > 1 else f"{labels[0]} only"


def _sources(fig: Figure, sources: dict[str, str]) -> None:
    """Name the recording files under the figure."""
    fig.supxlabel("    ".join(f"{label}: {name}" for label, name in sources.items()),
                  fontsize=8, color="#666666")


def _metric_bars(fig: Figure, cell, metrics: pd.DataFrame, metric_names: list[str], pretty: list[str],
                 labels: list[str]) -> None:
    """One small bar chart per metric, one bar per recording, side by side in ``cell``."""
    gs_bars = cell.subgridspec(1, len(metric_names))
    positions = list(range(len(labels)))
    for i, (m, p) in enumerate(zip(metric_names, pretty)):
        ax = fig.add_subplot(gs_bars[0, i])
        values = [metrics.loc[metrics.trial == label, m].mean() for label in labels]

        bars = ax.bar(positions, values, color=[TRIAL_COLOURS[label] for label in labels])
        ax.bar_label(bars, fmt="%.3g", padding=3, fontsize=8)
        ax.set_xticks(positions)
        ax.set_xticklabels([SHORT[label] for label in labels], fontsize=9)
        ax.set_title(p, fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


# ---------------------------------------------------------------------------
# Trunk rotation
# ---------------------------------------------------------------------------


def make_trunk_traceability_figure(
    kins: dict[str, Kinematics],
    fs_of: dict[str, float],
    metrics: pd.DataFrame,
    windows: dict[str, list[tuple[int, int, int, int, float]]],
    sources: dict[str, str],
    out_dir: Path,
) -> None:
    labels = _roles(kins)
    fig = Figure(figsize=(12, 3.35 * len(labels) + 1.8), constrained_layout=True)
    gs = fig.add_gridspec(len(labels) + 1, 1, height_ratios=[1.15] * len(labels) + [1.0])

    for row, label in enumerate(labels):
        ax, kin, fs = fig.add_subplot(gs[row, 0]), kins[label], fs_of[label]
        t = kin.t
        z = highpass_detrend(kin.trunk_rel_euler_deg[:, 2], fs, cutoff_hz=0.05)
        z = lowpass_signal(z, fs, cutoff_hz=4.0)

        ax.plot(t, z, color="#1f77b4", lw=1.2, label="rel z")

        for i, (event_start, event_end, stab_start, stab_end, peak) in enumerate(windows[label]):
            ax.axvspan(event_start / fs, event_end / fs, color="#f2b134", alpha=0.22, label="aligned event" if i == 0 else "")
            ax.axvspan(stab_start / fs, stab_end / fs, color="#57a773", alpha=0.18, label="stabilization" if i == 0 else "")
            if not np.isnan(peak):
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

    _metric_bars(fig, gs[len(labels)], metrics, metric_names, pretty, labels)

    fig.suptitle(f"Trunk Rotation Balance Traceability: {_comparison(labels)} Tai Chi", fontsize=14, fontweight="bold")
    _sources(fig, sources)
    fig.savefig(out_dir / "trunk_traceability_figure.png", dpi=220)


# ---------------------------------------------------------------------------
# Monopodal stance
# ---------------------------------------------------------------------------


def make_knee_flexion_overview_figure(
    kins: dict[str, Kinematics],
    fs_of: dict[str, float],
    knee_events: dict[str, list[dict[str, float | str]]],
    sources: dict[str, str],
    out_dir: Path,
) -> None:
    labels = _roles(kins)
    fig = Figure(figsize=(11, 4.6 * len(labels) + 0.4), constrained_layout=True)
    gs = fig.add_gridspec(2 * len(labels), 1, height_ratios=[1.0, 0.9] * len(labels))

    for row, label in enumerate(labels):
        knee_ax, trunk_ax = fig.add_subplot(gs[2 * row, 0]), fig.add_subplot(gs[2 * row + 1, 0])
        kin, fs = kins[label], fs_of[label]
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
    _sources(fig, sources)
    fig.savefig(out_dir / "monopodal_stance_overview_figure.png", dpi=220)


def make_knee_traceability_figure(
    kins: dict[str, Kinematics],
    fs_of: dict[str, float],
    knee_metrics: pd.DataFrame,
    knee_events: dict[str, list[dict[str, float | str]]],
    sources: dict[str, str],
    out_dir: Path,
    stab_overrides: dict[str, list[tuple[int, int]]] | None = None,
) -> None:
    labels = _roles(kins)
    fig = Figure(figsize=(13, 3.45 * len(labels) + 1.9), constrained_layout=True)
    gs = fig.add_gridspec(len(labels) + 1, 1, height_ratios=[1.15] * len(labels) + [1.0])

    for row, label in enumerate(labels):
        ax, kin, fs = fig.add_subplot(gs[row, 0]), kins[label], fs_of[label]
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

    _metric_bars(fig, gs[len(labels)], knee_metrics, metric_names, pretty, labels)

    fig.suptitle(
        f"Monopodal Stance Balance Traceability: {_comparison(labels)} Tai Chi",
        fontsize=14,
        fontweight="bold",
    )
    _sources(fig, sources)
    fig.savefig(out_dir / "monopodal_stance_traceability_figure.png", dpi=220)


# ---------------------------------------------------------------------------
# Sequence segments
# ---------------------------------------------------------------------------


def make_sequence_smoothness_figure(
    yaw: dict[str, tuple[np.ndarray, np.ndarray]],
    fs: dict[str, float],
    metrics: pd.DataFrame,
    waveforms: dict[str, dict[str, object]],
    sources: dict[str, str],
    out_dir: Path,
) -> None:
    """Traceability figure: where the segments fell, and how alike they were.

    The left column is the same "show your working" view the other two families
    get -- the signal the boundaries were taken from, with the bands drawn on
    it, so a number can always be traced back to a piece of the recording.  The
    right column is the corridor that the consistency metrics summarise.
    """
    trials = _roles(yaw)
    fig = Figure(figsize=(13, 3.35 * len(trials) + 1.8), constrained_layout=True)
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
    _sources(fig, sources)
    fig.savefig(out_dir / "sequence_smoothness_figure.png", dpi=220)


# ---------------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------------


def make_orientation_validation_figure(kin: Kinematics, fs: float, source: str, path: Path) -> None:
    """Lumbar and chest orientation of one recording, for sanity-checking the filter."""
    fig = Figure(figsize=(11, 3.8), constrained_layout=True)
    axes = fig.subplots(1, 2, sharex=False)
    for ax, sensor in zip(axes, ["lumbar", "chestbone"]):
        target_time, x = resample_filtered_full(kin.t, kin.eulers_deg[sensor][:, 0], fs, 10.0)
        _, y = resample_filtered_full(kin.t, kin.eulers_deg[sensor][:, 1], fs, 10.0)
        _, z = resample_filtered_full(kin.t, kin.eulers_deg[sensor][:, 2], fs, 10.0)
        ax.plot(target_time, x, lw=0.9, label="x")
        ax.plot(target_time, y, lw=0.9, label="y")
        ax.plot(target_time, z, lw=0.9, label="z")
        ax.set_title(f"{sensor} orientation")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("angle (deg)")
        ax.grid(True, color="#dddddd", lw=0.6)
        ax.legend(frameon=False, fontsize=8, ncol=3)
    fig.suptitle(f"Orientation Validation: {source}", fontsize=13, fontweight="bold")
    fig.savefig(path, dpi=220)
