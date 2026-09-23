#!/usr/bin/env python3
"""Smoothness and coordination metrics for parts of the whole Tai Chi sequence.

The trunk-rotation and monopodal-stance families measure balance on a handful of
hand-picked events.  Between them the recording is unmeasured, so nothing
describes how the *form as a whole* is performed.  This module scores the parts
that :func:`event_detection.detect_yaw_cycle_segments` cuts the recording into —
one back-and-forth turn of the chest each — on three questions:

* **How smooth is the turn?**  Dimensionless jerk and spectral arc length both
  answer this, from opposite directions: jerk in the time domain, SPARC from the
  shape of the speed spectrum.  The submovement count is the crude version of
  the same idea and is the easiest of the three to check against the plot.
* **Do chest and pelvis turn together?**  A lag and a peak correlation say *when*
  they move relative to each other; the gain and the relative-yaw range say *how
  much* they dissociate.
* **Is the same movement repeated?**  The spread of all of the above across
  segments, plus a waveform corridor built by time-normalising every segment.

The per-segment functions deliberately mirror
``compute_trunk_rotation_balance_metrics`` in shape, and reuse its primitives
rather than reimplementing them, so numbers stay comparable across families.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from tai_chi_trunk_and_knee import (
    Kinematics,
    OUTPUT_DIR,
    cross_correlation_lag,
    dimensionless_jerk,
    highpass_detrend,
    lowpass_signal,
)

WAVEFORM_POINTS = 101
"""Samples per time-normalised segment: 0-100 % of the part, in 1 % steps."""


# ---------------------------------------------------------------------------
# Smoothness primitives
# ---------------------------------------------------------------------------


def spectral_arc_length(
    angle_deg: np.ndarray,
    fs: float,
    cutoff_hz: float = 10.0,
    amplitude_threshold: float = 0.05,
) -> float:
    """Spectral arc length of an angular movement's speed profile.

    SPARC (Balasubramanian et al., *J NeuroEng Rehabil* 2015) measures the
    arc length of the normalised magnitude spectrum of the speed profile: a
    movement made of one smooth acceleration has a compact, smooth spectrum and
    a short arc, whereas hesitations add spectral content and lengthen it.
    Unlike dimensionless jerk it needs no amplitude or duration normalisation,
    so it does not inherit their sensitivity to the window edges.

    Returns a negative number; **less negative is smoother**.
    """
    speed = np.abs(np.gradient(np.asarray(angle_deg, dtype=float), 1.0 / fs))
    if len(speed) < 4 or not np.any(speed):
        return float("nan")

    # Zero-padding to several times the next power of two interpolates the
    # spectrum, so the arc length does not jump with the segment length.
    n_fft = int(2 ** math.ceil(math.log2(len(speed)))) * 4
    freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
    magnitudes = np.abs(np.fft.rfft(speed, n_fft))
    peak = magnitudes.max()
    if peak <= 0:
        return float("nan")
    magnitudes = magnitudes / peak

    within = freqs <= cutoff_hz
    freqs, magnitudes = freqs[within], magnitudes[within]
    # Cut the spectrum back to the band that actually carries the movement,
    # otherwise the arc length counts the noise floor out to the cutoff.
    carried = np.where(magnitudes >= amplitude_threshold)[0]
    if len(carried) < 2:
        return float("nan")
    freqs = freqs[carried[0]:carried[-1] + 1]
    magnitudes = magnitudes[carried[0]:carried[-1] + 1]

    span = freqs[-1] - freqs[0]
    if span <= 0:
        return float("nan")
    d_freq = np.diff(freqs) / span
    return float(-np.sum(np.hypot(d_freq, np.diff(magnitudes))))


SUBMOVEMENT_CUTOFF_HZ = 2.0
"""Smoothing applied before counting speed peaks.  The chest yaw carrier sits
near 0.1 Hz and a hesitation within a turn shows up below ~2 Hz, so anything
above that is ripple: at the pipeline's usual 6 Hz the count rises by half again
without a corresponding change in the movement."""


def count_submovements(
    angle_deg: np.ndarray,
    fs: float,
    cutoff_hz: float = SUBMOVEMENT_CUTOFF_HZ,
    relative_height: float = 0.2,
    min_separation_s: float = 0.3,
) -> tuple[int, float]:
    """Peaks in the turning speed, and the fastest of them.

    One continuous turn has a single speed peak; a turn made in two goes, or one
    interrupted by a hesitation, has more.  The threshold is a fraction of the
    segment's *own* peak speed, as the trunk detector does, so a gentle turn and
    a vigorous one are judged at the same relative level.

    Report the **rate** rather than this raw count when comparing segments: the
    equivalent stabilization metric was a count until boundary-jitter testing
    put its ICC at 0.41, against 0.97 for the rate, purely because segments
    differ in length.
    """
    speed = np.abs(np.gradient(
        lowpass_signal(np.asarray(angle_deg, dtype=float), fs, cutoff_hz=cutoff_hz), 1.0 / fs
    ))
    if not len(speed) or not np.any(speed):
        return 0, float("nan")
    peaks, _ = find_peaks(
        speed,
        height=relative_height * float(speed.max()),
        distance=max(1, int(min_separation_s * fs)),
    )
    return len(peaks), float(speed.max())


# ---------------------------------------------------------------------------
# Waveform consistency
# ---------------------------------------------------------------------------


def time_normalise(signal: np.ndarray, n_points: int = WAVEFORM_POINTS) -> np.ndarray:
    """Resample one segment onto a 0-100 % grid so segments can be averaged."""
    signal = np.asarray(signal, dtype=float)
    if len(signal) < 2:
        return np.full(n_points, np.nan)
    source = np.linspace(0.0, 1.0, len(signal))
    return np.interp(np.linspace(0.0, 1.0, n_points), source, signal)


def waveform_consistency(segments: list[np.ndarray]) -> dict[str, object]:
    """Mean +/- SD corridor over time-normalised segments, and its spread.

    Segments alternate in direction — one starts by turning left, the next by
    turning right — so they are sign-aligned before averaging, otherwise the
    mean waveform would cancel to nothing.  Each is also centred, so the result
    describes the *shape* of the turn rather than where neutral happened to sit.

    ``variance_ratio`` is the Kadaba ratio: within-cycle variance about the mean
    waveform over total variance about the grand mean.  **Lower is more
    repeatable**; it is dimensionless, so it compares across participants where
    the SD in degrees does not.
    """
    curves = []
    for segment in segments:
        curve = time_normalise(segment)
        if np.any(np.isnan(curve)):
            continue
        curve = curve - np.mean(curve)
        # Align on direction: every curve is made to start on the negative lobe.
        lead = curve[: max(1, len(curve) // 4)]
        if np.mean(lead) > 0:
            curve = -curve
        curves.append(curve)

    if len(curves) < 2:
        return {
            "mean_waveform_deg": np.full(WAVEFORM_POINTS, np.nan),
            "sd_waveform_deg": np.full(WAVEFORM_POINTS, np.nan),
            "waveform_mean_sd_deg": float("nan"),
            "waveform_variance_ratio": float("nan"),
            "n_waveforms": len(curves),
        }

    stack = np.vstack(curves)
    mean_waveform = stack.mean(axis=0)
    sd_waveform = stack.std(axis=0, ddof=1)

    n_cycles, n_points = stack.shape
    within = np.sum((stack - mean_waveform) ** 2) / (n_points * (n_cycles - 1))
    total = np.sum((stack - stack.mean()) ** 2) / (n_points * n_cycles - 1)

    return {
        "mean_waveform_deg": mean_waveform,
        "sd_waveform_deg": sd_waveform,
        "waveform_mean_sd_deg": float(np.mean(sd_waveform)),
        "waveform_variance_ratio": float(within / total) if total > 0 else float("nan"),
        "n_waveforms": n_cycles,
    }


# ---------------------------------------------------------------------------
# Per-segment metrics
# ---------------------------------------------------------------------------


def detrended_yaw(kin: Kinematics, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Chest and pelvis yaw, high-passed over the *whole* recording.

    Detrending before slicing, as the trunk family does, keeps drift accumulated
    earlier in the recording out of a segment's correlation.
    """
    chest = highpass_detrend(kin.eulers_deg["chestbone"][:, 2], fs, cutoff_hz=0.05)
    pelvis = highpass_detrend(kin.eulers_deg["lumbar"][:, 2], fs, cutoff_hz=0.05)
    return chest, pelvis


def compute_sequence_smoothness_metrics(
    label: str,
    kin: Kinematics,
    fs: float,
    segment_id: str,
    start: int,
    end: int,
    lag_pad_s: float | None = None,
    yaw: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, float | str]:
    """Smoothness and chest-pelvis coordination for one part of the sequence.

    ``lag_pad_s`` widens only the cross-correlation window, for the same reason
    it does in ``compute_trunk_rotation_balance_metrics``: the lag search spans
    +/-2 s and pins to that bound on a window too short to support it.

    Pass ``yaw`` to reuse the detrended traces across segments of the same
    recording; it is recomputed per call otherwise.
    """
    chest_yaw, pelvis_yaw = yaw if yaw is not None else detrended_yaw(kin, fs)
    segment = slice(start, end)

    if lag_pad_s is None:
        lag_window = segment
    else:
        pad = int(round(lag_pad_s * fs))
        lag_window = slice(max(0, start - pad), min(len(kin.t), end + pad))

    corr, lag = cross_correlation_lag(pelvis_yaw[lag_window], chest_yaw[lag_window], fs)

    chest_excursion = float(np.ptp(chest_yaw[segment]))
    pelvis_excursion = float(np.ptp(pelvis_yaw[segment]))
    relative_yaw = kin.trunk_rel_euler_deg[segment, 2]

    jerk = dimensionless_jerk(chest_yaw[segment], fs)
    submovements, peak_rate = count_submovements(chest_yaw[segment], fs)
    duration_s = (end - start) / fs

    return {
        "trial": label,
        "segment_id": segment_id,
        "segment_start_s": start / fs,
        "segment_end_s": end / fs,
        "segment_duration_s": duration_s,
        "chest_yaw_excursion_deg": chest_excursion,
        "pelvis_yaw_excursion_deg": pelvis_excursion,
        "chest_yaw_dimensionless_jerk": jerk,
        "chest_yaw_log10_dimensionless_jerk": (
            math.log10(jerk) if jerk and jerk > 0 else float("nan")
        ),
        "chest_yaw_sparc": spectral_arc_length(chest_yaw[segment], fs),
        "chest_yaw_submovement_count": submovements,
        "chest_yaw_submovement_rate_hz": (
            submovements / duration_s if duration_s > 0 else float("nan")
        ),
        "chest_yaw_peak_rate_dps": peak_rate,
        "chest_pelvis_peak_cross_correlation": corr,
        "chest_pelvis_lag_s": lag,
        # 1 = the trunk turns as a unit; below 1 the chest turns on a
        # comparatively still pelvis.
        "chest_pelvis_gain": (
            pelvis_excursion / chest_excursion if chest_excursion > 1e-9 else float("nan")
        ),
        "relative_yaw_range_deg": float(np.ptp(relative_yaw)),
        "relative_yaw_rms_deg": float(np.std(relative_yaw)),
    }


def summarize_sequence_variability(
    metrics: pd.DataFrame, waveforms: dict[str, dict[str, object]]
) -> pd.DataFrame:
    """One row per recording: how much each metric varies across segments.

    This is the movement-variability answer.  The per-segment table is what you
    read when one of these numbers looks wrong.
    """

    def spread(values: pd.Series) -> tuple[float, float, float]:
        mean = float(values.mean())
        sd = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        cv = sd / abs(mean) if mean and abs(mean) > 1e-9 else float("nan")
        return mean, sd, cv

    rows = []
    for label, group in metrics.groupby("trial", sort=False):
        duration_mean, duration_sd, duration_cv = spread(group["segment_duration_s"])
        excursion_mean, excursion_sd, excursion_cv = spread(group["chest_yaw_excursion_deg"])
        jerk_mean, jerk_sd, _ = spread(group["chest_yaw_log10_dimensionless_jerk"])
        sparc_mean, sparc_sd, _ = spread(group["chest_yaw_sparc"])
        sub_mean, sub_sd, _ = spread(group["chest_yaw_submovement_rate_hz"])
        lag_mean, lag_sd, _ = spread(group["chest_pelvis_lag_s"])
        gain_mean, gain_sd, gain_cv = spread(group["chest_pelvis_gain"])
        waveform = waveforms.get(label, {})
        rows.append({
            "trial": label,
            "n_segments": int(len(group)),
            "total_segmented_s": float(group["segment_duration_s"].sum()),
            "duration_mean_s": duration_mean,
            "duration_sd_s": duration_sd,
            "duration_cv": duration_cv,
            "chest_yaw_excursion_mean_deg": excursion_mean,
            "chest_yaw_excursion_cv": excursion_cv,
            "log10_dimensionless_jerk_mean": jerk_mean,
            "log10_dimensionless_jerk_sd": jerk_sd,
            "sparc_mean": sparc_mean,
            "sparc_sd": sparc_sd,
            "submovement_rate_mean_hz": sub_mean,
            "submovement_rate_sd_hz": sub_sd,
            "chest_pelvis_lag_mean_s": lag_mean,
            "chest_pelvis_lag_sd_s": lag_sd,
            "chest_pelvis_peak_cross_correlation_mean": float(
                group["chest_pelvis_peak_cross_correlation"].mean()
            ),
            "chest_pelvis_gain_mean": gain_mean,
            "chest_pelvis_gain_cv": gain_cv,
            "relative_yaw_range_mean_deg": float(group["relative_yaw_range_deg"].mean()),
            "waveform_mean_sd_deg": waveform.get("waveform_mean_sd_deg", float("nan")),
            "waveform_variance_ratio": waveform.get("waveform_variance_ratio", float("nan")),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


TRIAL_COLOURS = {"Novice": "#d1495b", "Trained": "#2a9d8f"}


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
    fig.savefig(OUTPUT_DIR / "sequence_smoothness_figure.png", dpi=220)
    plt.close(fig)
