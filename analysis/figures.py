"""The output figures.

The traceability figures draw the signal the event boundaries were taken from,
with the bands on it, above small panels of the metrics computed over those
bands, so every number can be traced back to a stretch of the recording.  Each
takes the recordings to draw as ``{role: ...}`` dicts -- one row per recording,
so a novice-only or trained-only analysis gets a figure of its own -- and names
the source files underneath, so a figure copied into a report still says what
it was made from.

The metric panels are paired: every event is the same movement in both
recordings (the pairs come from the whole-recording alignment), so each is
drawn as a line from its novice value to its trained value, with the medians on
top.  A bar of means would hide which way each pair goes.

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

from analysis import config
from analysis.config import TRIALS
from analysis.kinematics import Kinematics
from analysis.signals import highpass_detrend, lowpass_signal, resample_filtered_full, sec_to_idx


TRIAL_COLOURS = {"Novice": "#d1495b", "Trained": "#2a9d8f"}
SHORT = {"Novice": "Nov", "Trained": "Train"}
LEG_COLOURS = {"Left": "#f2b134", "Right": "#8ecae6"}


def _roles(recordings: dict) -> list[str]:
    return [label for label in TRIALS if label in recordings]


def _comparison(labels: list[str]) -> str:
    return " vs ".join(labels) if len(labels) > 1 else f"{labels[0]} only"


def _sources(fig: Figure, sources: dict[str, str]) -> None:
    """Name the recording files, and the metric definitions, under the figure."""
    names = "    ".join(f"{label}: {name}" for label, name in sources.items())
    fig.supxlabel(f"{names}    metrics v{config.METRICS_VERSION}", fontsize=8, color="#666666")


def _paired_panels(fig: Figure, cell, metrics: pd.DataFrame, metric_names: list[str], pretty: list[str],
                   labels: list[str]) -> None:
    """One small panel per metric: each event a line from novice to trained, medians on top."""
    grid = cell.subgridspec(1, len(metric_names))
    positions = {label: i for i, label in enumerate(labels)}
    for i, (name, title) in enumerate(zip(metric_names, pretty)):
        ax = fig.add_subplot(grid[0, i])
        if name not in metrics:
            ax.set_visible(False)
            continue
        table = metrics.pivot_table(index="event_id", columns="trial", values=name, aggfunc="first")
        if len(labels) == 2 and all(label in table for label in labels):
            both = table[labels].dropna()
            for _, row in both.iterrows():
                ax.plot([0, 1], row.to_numpy(), color="#b0b0b0", lw=0.9, zorder=1)
            n = len(both)
        else:
            n = int(table.notna().sum().max()) if len(table) else 0
        for label in labels:
            if label not in table:
                continue
            values = table[label].dropna().to_numpy()
            ax.scatter(np.full(len(values), positions[label]), values, s=10, color=TRIAL_COLOURS[label],
                       alpha=0.55, zorder=2)
            if len(values):
                ax.scatter([positions[label]], [np.median(values)], s=70, marker="_", linewidths=2.5,
                           color=TRIAL_COLOURS[label], zorder=3)
        ax.set_xticks(list(positions.values()))
        ax.set_xticklabels([SHORT[label] for label in labels], fontsize=9)
        ax.set_xlim(-0.4, len(labels) - 0.6)
        ax.set_title(f"{title}\n(n = {n})", fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


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
    fig = Figure(figsize=(12, 3.35 * len(labels) + 2.1), constrained_layout=True)
    gs = fig.add_gridspec(len(labels) + 1, 1, height_ratios=[1.15] * len(labels) + [1.2])

    for row, label in enumerate(labels):
        ax, kin, fs = fig.add_subplot(gs[row, 0]), kins[label], fs_of[label]
        t = kin.t
        z = highpass_detrend(kin.trunk_rel_euler_deg[:, 2], fs, cutoff_hz=0.05)
        z = lowpass_signal(z, fs, cutoff_hz=4.0)

        ax.plot(t, z, color="#1f77b4", lw=1.2, label="trunk-pelvis yaw")

        for i, (event_start, event_end, stab_start, stab_end, peak) in enumerate(windows[label]):
            ax.axvspan(event_start / fs, event_end / fs, color="#f2b134", alpha=0.22, label="rotation" if i == 0 else "")
            ax.axvspan(stab_start / fs, stab_end / fs, color="#57a773", alpha=0.18,
                       label="post-rotation window" if i == 0 else "")
            if not np.isnan(peak):
                ax.scatter([t[int(peak)]], [z[int(peak)]], s=30, color="#d1495b", zorder=4,
                           label="yaw peak" if i == 0 else "")

        ax.set_ylabel(f"{label}\nangle (deg)")
        ax.grid(True, color="#dddddd", lw=0.6)
        ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=4)

    metric_names = [
        "trunk_pelvis_lag_s",
        "trunk_yaw_sparc",
        "lumbar_ml_acc_rms_mps2",
        "lumbar_ap_acc_rms_mps2",
        "lumbar_frontal_tilt_sd_deg",
        "lumbar_rms_angular_velocity_dps",
    ]
    pretty = ["pelvis-chest lag (s)", "rotation SPARC", "ML sway RMS (m/s²)", "AP sway RMS (m/s²)",
              "frontal tilt SD (deg)", "lumbar ω RMS (deg/s)"]
    _paired_panels(fig, gs[len(labels)], metrics, metric_names, pretty, labels)

    fig.suptitle(f"Trunk Rotation Balance Traceability: {_comparison(labels)} Tai Chi", fontsize=14, fontweight="bold")
    _sources(fig, sources)
    fig.savefig(out_dir / "trunk_traceability_figure.png", dpi=220)


# ---------------------------------------------------------------------------
# Single-leg stance
# ---------------------------------------------------------------------------


def _stance_bands(ax, events: list[dict], alpha: float, with_labels: bool = True) -> None:
    seen: set[str] = set()
    for ev in events:
        leg = ev.get("lifted_leg", "Left")
        name = f"{leg} leg up"
        ax.axvspan(ev["window_start_s"], ev["window_end_s"], color=LEG_COLOURS.get(leg, "#cccccc"), alpha=alpha,
                   label=name if with_labels and name not in seen else "")
        seen.add(name)


def make_knee_flexion_overview_figure(
    kins: dict[str, Kinematics],
    fs_of: dict[str, float],
    knee_events: dict[str, list[dict[str, float | str]]],
    signals: dict[str, dict],
    sources: dict[str, str],
    out_dir: Path,
) -> None:
    """Which foot was up, what the knees did, and how the pelvis tilted, per recording."""
    labels = _roles(kins)
    fig = Figure(figsize=(11, 6.3 * len(labels) + 0.4), constrained_layout=True)
    gs = fig.add_gridspec(3 * len(labels), 1, height_ratios=[1.0, 0.9, 0.9] * len(labels))

    for row, label in enumerate(labels):
        lift_ax = fig.add_subplot(gs[3 * row, 0])
        knee_ax = fig.add_subplot(gs[3 * row + 1, 0], sharex=lift_ax)
        tilt_ax = fig.add_subplot(gs[3 * row + 2, 0], sharex=lift_ax)
        kin, fs = kins[label], fs_of[label]
        t = kin.t
        lift = signals[label]["lift"]

        lift_ax.plot(t, lift, color="#444444", lw=1.0, label="lift index (+ left foot up)")
        lift_ax.axhline(0.0, color="#888888", lw=0.8, ls=":")
        _stance_bands(lift_ax, knee_events[label], 0.25)
        for ev in knee_events[label]:
            peak = sec_to_idx(ev["peak_time_s"], fs, len(t))
            lift_ax.scatter([ev["peak_time_s"]], [lift[peak]], s=18, color="#d1495b", zorder=4)
        lift_ax.set_ylabel(f"{label}\nlift (thigh lengths)")
        lift_ax.grid(True, color="#dddddd", lw=0.6)
        lift_ax.legend(frameon=False, fontsize=8, ncol=3)

        knee_ax.plot(t, np.abs(lowpass_signal(kin.left_knee_deg[:, 0], fs, cutoff_hz=6.0)), lw=1.0,
                     color=LEG_COLOURS["Left"], label="left knee flexion")
        knee_ax.plot(t, np.abs(lowpass_signal(kin.right_knee_deg[:, 0], fs, cutoff_hz=6.0)), lw=1.0,
                     color="#3a86c8", label="right knee flexion")
        _stance_bands(knee_ax, knee_events[label], 0.10, with_labels=False)
        knee_ax.set_ylabel("knee (deg)")
        knee_ax.grid(True, color="#dddddd", lw=0.6)
        knee_ax.legend(frameon=False, fontsize=8, ncol=2)

        tilt_ax.plot(t, signals[label]["lumbar_frontal_tilt"], lw=1.0, label="lumbar frontal tilt")
        tilt_ax.plot(t, signals[label]["lumbar_sagittal_tilt"], lw=1.0, label="lumbar sagittal tilt")
        _stance_bands(tilt_ax, knee_events[label], 0.10, with_labels=False)
        tilt_ax.set_ylabel("pelvis tilt (deg)")
        tilt_ax.set_xlabel("time (s)")
        tilt_ax.grid(True, color="#dddddd", lw=0.6)
        tilt_ax.legend(frameon=False, fontsize=8, ncol=2)

    fig.suptitle("Single-Leg Stance: Which Foot Was Up, the Knees, and the Pelvis", fontsize=14, fontweight="bold")
    _sources(fig, sources)
    fig.savefig(out_dir / "monopodal_stance_overview_figure.png", dpi=220)


def make_knee_traceability_figure(
    kins: dict[str, Kinematics],
    fs_of: dict[str, float],
    knee_metrics: pd.DataFrame,
    knee_events: dict[str, list[dict[str, float | str]]],
    signals: dict[str, dict],
    sources: dict[str, str],
    out_dir: Path,
) -> None:
    labels = _roles(kins)
    fig = Figure(figsize=(13, 3.45 * len(labels) + 2.2), constrained_layout=True)
    gs = fig.add_gridspec(len(labels) + 1, 1, height_ratios=[1.15] * len(labels) + [1.2])

    for row, label in enumerate(labels):
        ax, kin, fs = fig.add_subplot(gs[row, 0]), kins[label], fs_of[label]
        t = kin.t
        magnitude = np.abs(signals[label]["lift"])
        ax.plot(t, magnitude, color="#1f77b4", lw=1.2, label="|lift index|")

        for i, ev in enumerate(knee_events[label]):
            ax.axvspan(ev["window_start_s"], ev["window_end_s"], color="#f2b134", alpha=0.22,
                       label="single-leg support" if i == 0 else "")
            ax.axvspan(ev["stabilization_start_s"], ev["stabilization_end_s"], color="#57a773", alpha=0.18,
                       label="settling after touch-down" if i == 0 else "")
            peak = sec_to_idx(ev["peak_time_s"], fs, len(t))
            ax.scatter([ev["peak_time_s"]], [magnitude[peak]], s=30, color="#d1495b", zorder=4,
                       label="highest lift" if i == 0 else "")

        ax.set_ylabel(f"{label}\nlift (thigh lengths)")
        ax.grid(True, color="#dddddd", lw=0.6)
        ax.legend(frameon=False, fontsize=8)

    metric_names = [
        "support_duration_s",
        "support_ml_acc_rms_mps2",
        "support_frontal_tilt_sd_deg",
        "support_rms_angular_velocity_dps",
        "settle_ml_acc_rms_mps2",
        "time_to_stabilization_s",
    ]
    pretty = ["support time (s)", "support ML sway (m/s²)", "support frontal tilt SD (deg)",
              "support lumbar ω RMS (deg/s)", "settling ML sway (m/s²)", "time to settle (s)"]
    _paired_panels(fig, gs[len(labels)], knee_metrics, metric_names, pretty, labels)

    fig.suptitle(f"Single-Leg Stance Balance Traceability: {_comparison(labels)} Tai Chi",
                 fontsize=14, fontweight="bold")
    _sources(fig, sources)
    fig.savefig(out_dir / "monopodal_stance_traceability_figure.png", dpi=220)


# ---------------------------------------------------------------------------
# Sequence turns
# ---------------------------------------------------------------------------


def make_sequence_smoothness_figure(
    signals: dict[str, dict],
    fs: dict[str, float],
    metrics: pd.DataFrame,
    corridors: dict[str, dict],
    sources: dict[str, str],
    out_dir: Path,
) -> None:
    """Traceability figure: where the turns fell, their typical shape, and the paired metrics.

    The left column is the signal the turn boundaries were taken from, with the
    turns drawn on it.  The right column is the typical time course of a turn:
    the fraction of its own excursion completed at each percent of its
    duration, whose spread shows how varied the form's turns are -- every turn
    being a different movement, not how consistently one is repeated.
    """
    trials = _roles(signals)
    fig = Figure(figsize=(13, 3.35 * len(trials) + 2.1), constrained_layout=True)
    gs = fig.add_gridspec(len(trials) + 1, 2, width_ratios=[2.0, 1.0],
                          height_ratios=[1.15] * len(trials) + [1.2])

    for row, label in enumerate(trials):
        chest_yaw, pelvis_yaw = signals[label]["chest_yaw"], signals[label]["pelvis_yaw"]
        rate = fs[label]
        time = np.arange(len(chest_yaw)) / rate
        rows = metrics[metrics.trial == label]

        ax = fig.add_subplot(gs[row, 0])
        ax.plot(time, chest_yaw, color="#1f77b4", lw=1.1, label="chest yaw")
        ax.plot(time, pelvis_yaw, color="#8ecae6", lw=1.0, label="pelvis yaw")
        ax.axhline(0.0, color="#888888", lw=0.8, ls=":")
        # Turns abut, so a single shade would read as one long block; the
        # alternating alpha is what makes the individual turns countable.
        for i, (_, turn) in enumerate(rows.iterrows()):
            ax.axvspan(turn["turn_start_s"], turn["turn_end_s"],
                       color="#f2b134", alpha=0.28 if i % 2 else 0.12,
                       label="turn" if i == 0 else "")
        ax.set_ylabel(f"{label}\nyaw (deg)")
        ax.grid(True, color="#dddddd", lw=0.6)
        ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=3)

        ax = fig.add_subplot(gs[row, 1])
        corridor = corridors.get(label, {})
        mean = np.asarray(corridor.get("mean", []), dtype=float)
        sd = np.asarray(corridor.get("sd", []), dtype=float)
        if mean.size and not np.all(np.isnan(mean)):
            percent = np.linspace(0.0, 100.0, len(mean))
            colour = TRIAL_COLOURS.get(label, "#444444")
            ax.fill_between(percent, mean - sd, mean + sd, color=colour, alpha=0.25, label="+/- 1 SD")
            ax.plot(percent, mean, color=colour, lw=1.6, label="mean")
            ax.set_title(f"turn time course, {corridor.get('n', 0)} turns\n"
                         "(the form's turns differ; not repetition variability)", fontsize=9)
            ax.legend(loc="lower right", fontsize=8, frameon=False)
        ax.set_ylabel("fraction of turn")
        ax.grid(True, color="#dddddd", lw=0.6)

    fig.axes[-2].set_xlabel("time (s)")
    fig.axes[-1].set_xlabel("percent of turn")

    metric_names = [
        "chest_yaw_sparc",
        "chest_yaw_peak_rate_dps",
        "chest_yaw_submovement_rate_hz",
        "chest_pelvis_lag_s",
        "chest_pelvis_gain",
        "relative_yaw_range_deg",
    ]
    pretty = ["SPARC (less neg. = smoother)", "peak turning rate (deg/s)", "submovements (Hz)",
              "pelvis-chest lag (s)", "pelvis/chest gain", "trunk twist range (deg)"]
    _paired_panels(fig, gs[len(trials), :], metrics, metric_names, pretty, trials)

    fig.suptitle("Chest turns of the whole form: smoothness and chest-pelvis coordination",
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
