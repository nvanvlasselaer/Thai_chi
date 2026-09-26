#!/usr/bin/env python3
"""Smoothness and coordination of the chest turns that carry the whole form.

The trunk-rotation and single-leg families measure balance on a handful of
hand-picked events.  Between them the recording is unmeasured, so nothing
describes how the *form as a whole* is performed.  This module scores the turns
that :func:`analysis.detection.detect_yaw_turns` cuts the recording into -- one
single-direction turn of the chest each, from one moment of stillness to the
next -- on two questions:

* **How smooth is the turn?**  SPARC, the spectral arc length of the turning
  speed, is the primary measure; the submovement rate is the crude version of
  the same idea and the easiest to check against the plot.  Dimensionless jerk
  is kept as a secondary column only: on these recordings its log is
  explained almost entirely by the turn's duration and size (R^2 = 0.97, with
  the exponents a noise floor predicts), and 86-93 % of its jerk power lies
  above 1 Hz, while the turns themselves are ~0.1 Hz.
* **Do chest and pelvis turn together?**  A lag says *when* they move relative
  to each other; the gain and the relative-yaw range say *how much* they
  dissociate.

Turning speed and excursion come from the bias-corrected gyroscope projected on
the vertical (:func:`analysis.gravity.vertical_angular_velocity`), integrated
where an angle is needed.  The high-passed Euler yaw the earlier version used
shrank turns of 8-14 s by about a quarter.

Every turn is a different movement of the form, performed once, so the spread
of these numbers across turns describes the choreography, not how consistently
a movement is repeated.  The comparison that does mean something here is
between the two performers on the *same* turn: :mod:`analysis.comparison`.
"""


from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from analysis.data_io import TrialData
from analysis.gravity import vertical_angular_velocity
from analysis.kinematics import Kinematics
from analysis.signals import highpass_detrend, lowpass_signal, pearson_lag


WAVEFORM_POINTS = 101
"""Samples per time-normalised turn: 0-100 % of the turn, in 1 % steps."""


# ---------------------------------------------------------------------------
# Smoothness primitives
# ---------------------------------------------------------------------------


def dimensionless_jerk(angle_deg: np.ndarray, fs: float) -> float:
    """Duration^5 / amplitude^2 times the integrated squared jerk; lower is smoother.

    Kept for comparability with earlier results only.  For slow, long
    movements the integral is dominated by noise near the 6 Hz cutoff, whose
    contribution grows like duration^6 / amplitude^2 -- which is what the
    values on these recordings follow -- so it measures the window more than
    the movement.
    """
    angle = np.deg2rad(lowpass_signal(angle_deg, fs, cutoff_hz=6.0, order=4))
    dt = 1.0 / fs
    vel = np.gradient(angle, dt)
    acc = np.gradient(lowpass_signal(vel, fs, cutoff_hz=6.0), dt)
    jerk = np.gradient(lowpass_signal(acc, fs, cutoff_hz=6.0), dt)
    duration = len(angle) / fs
    amplitude = np.ptp(angle)
    if amplitude < 1e-9:
        return float("nan")
    return float((duration**5 / amplitude**2) * np.trapezoid(jerk**2, dx=dt))


def sparc(
    speed: np.ndarray,
    fs: float,
    padlevel: int = 4,
    cutoff_hz: float = 10.0,
    amplitude_threshold: float = 0.05,
) -> float:
    """Spectral arc length of a speed profile (Balasubramanian et al. 2012, 2015).

    The arc length of the normalised magnitude spectrum: a movement made of one
    smooth acceleration has a compact, smooth spectrum and a short arc, whereas
    hesitations add spectral content and lengthen it.  It is valid on the
    angular velocity a gyroscope measures (Melendez-Calderon et al. 2021) and
    assumes a discrete movement, from rest to rest.  The defaults are the
    published ones, zero-padding included (padlevel 4; this module used 2
    before, which shifts values by ~0.1).

    Returns a negative number; **less negative is smoother**.
    """
    speed = np.abs(np.asarray(speed, dtype=float))
    if len(speed) < 4 or not np.any(speed):
        return float("nan")

    n_fft = int(2 ** (math.ceil(math.log2(len(speed))) + padlevel))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
    magnitudes = np.abs(np.fft.rfft(speed, n_fft))
    magnitudes = magnitudes / magnitudes.max()

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
    return float(-np.sum(np.hypot(np.diff(freqs) / span, np.diff(magnitudes))))


SUBMOVEMENT_CUTOFF_HZ = 2.0
"""Smoothing applied before counting speed peaks.  The chest yaw carrier sits
near 0.1 Hz and a hesitation within a turn shows up below ~2 Hz, so anything
above that is ripple: at the pipeline's usual 6 Hz the count rises by half again
without a corresponding change in the movement."""


def count_submovements(
    rate_dps: np.ndarray,
    fs: float,
    cutoff_hz: float = SUBMOVEMENT_CUTOFF_HZ,
    relative_height: float = 0.2,
    min_separation_s: float = 0.3,
) -> tuple[int, float]:
    """Peaks in the turning speed, and the fastest of them.

    One continuous turn has a single speed peak; a turn made in two goes, or one
    interrupted by a hesitation, has more.  The threshold is a fraction of the
    turn's *own* peak speed, as the trunk detector does, so a gentle turn and a
    vigorous one are judged at the same relative level.  Report the **rate**
    rather than this raw count when comparing turns of different length.
    """
    speed = np.abs(lowpass_signal(np.asarray(rate_dps, dtype=float), fs, cutoff_hz=cutoff_hz))
    if not len(speed) or not np.any(speed):
        return 0, float("nan")
    peaks, _ = find_peaks(
        speed,
        height=relative_height * float(speed.max()),
        distance=max(1, int(min_separation_s * fs)),
    )
    return len(peaks), float(speed.max())


# ---------------------------------------------------------------------------
# Turn shape
# ---------------------------------------------------------------------------


def time_normalise(signal: np.ndarray, n_points: int = WAVEFORM_POINTS) -> np.ndarray:
    """Resample one turn onto a 0-100 % grid so turns can be overlaid."""
    signal = np.asarray(signal, dtype=float)
    if len(signal) < 2:
        return np.full(n_points, np.nan)
    source = np.linspace(0.0, 1.0, len(signal))
    return np.interp(np.linspace(0.0, 1.0, n_points), source, signal)


def turn_shape_corridor(progress: list[np.ndarray]) -> dict[str, np.ndarray | int]:
    """Mean and SD of the turns' progress curves, each running from 0 to 1.

    Each turn is expressed as the fraction of its own excursion completed at
    each percent of its duration, so the corridor shows the typical time course
    of a turn -- an even S for a smooth one -- independent of how far or how
    long each turn went.  The turns are different movements of the form, so the
    spread is how varied the form's turns are, not repetition variability.
    """
    curves = [time_normalise(curve) for curve in progress if len(curve) >= 2]
    curves = [curve for curve in curves if np.all(np.isfinite(curve))]
    if len(curves) < 2:
        empty = np.full(WAVEFORM_POINTS, np.nan)
        return {"mean": empty, "sd": empty, "n": len(curves)}
    stack = np.vstack(curves)
    return {"mean": stack.mean(axis=0), "sd": stack.std(axis=0, ddof=1), "n": len(curves)}


# ---------------------------------------------------------------------------
# Per-turn metrics
# ---------------------------------------------------------------------------


def turn_signals(kin: Kinematics, trial: TrialData, fs: float) -> dict[str, np.ndarray]:
    """The per-recording signals every turn is measured on, computed once.

    ``chest_rate`` and ``pelvis_rate`` are the vertical angular velocities (deg/s),
    low-passed at 6 Hz; ``chest_yaw`` is the high-passed Euler yaw, used only for
    the secondary jerk column and for display.
    """
    return {
        "chest_rate": lowpass_signal(vertical_angular_velocity(kin, trial, "chestbone"), fs, cutoff_hz=6.0),
        "pelvis_rate": lowpass_signal(vertical_angular_velocity(kin, trial, "lumbar"), fs, cutoff_hz=6.0),
        "chest_yaw": highpass_detrend(kin.eulers_deg["chestbone"][:, 2], fs, cutoff_hz=0.05),
        "pelvis_yaw": highpass_detrend(kin.eulers_deg["lumbar"][:, 2], fs, cutoff_hz=0.05),
    }


def yaw_lag(signals: dict[str, np.ndarray], fs: float, window: slice) -> tuple[float, float]:
    """Pelvis-to-chest turning lag over a window: ``(r, lag_s)``, negative when the pelvis leads.

    Correlates the two segments' yaw *angles*, each integrated from its
    vertical angular velocity over the window -- drift-free over the window and
    untouched by the high-pass.  Angles rather than velocities, because the two
    do not turn as shifted copies: velocities time the peak turning speed and
    give lags a third as long, angles time the turn as a whole, which is what
    earlier results and the literature report.
    """
    pelvis = np.cumsum(signals["pelvis_rate"][window]) / fs
    chest = np.cumsum(signals["chest_rate"][window]) / fs
    return pearson_lag(pelvis, chest, fs)


def turn_progress(signals: dict[str, np.ndarray], fs: float, start: int, end: int) -> np.ndarray:
    """Fraction of the turn's excursion completed at each sample (0 to 1)."""
    angle = np.cumsum(signals["chest_rate"][start:end]) / fs
    total = angle[-1] if len(angle) else 0.0
    return angle / total if abs(total) > 1e-9 else np.full(len(angle), np.nan)


def compute_turn_metrics(
    label: str,
    fs: float,
    event_id: str,
    start: int,
    end: int,
    signals: dict[str, np.ndarray],
    direction: int | None = None,
    lag_pad_s: float | None = None,
) -> dict[str, float | str]:
    """Smoothness and chest-pelvis coordination for one turn.

    ``lag_pad_s`` widens only the lag window, as for the trunk family: the lag
    is a property of the coordination around the turn, and a turn of 2-3 s
    cannot support a +/-1 s search on its own.
    """
    chest, pelvis = signals["chest_rate"], signals["pelvis_rate"]
    turn = slice(start, end)
    duration_s = (end - start) / fs

    if lag_pad_s is None:
        lag_window = turn
    else:
        pad = int(round(lag_pad_s * fs))
        lag_window = slice(max(0, start - pad), min(len(chest), end + pad))
    r, lag = yaw_lag(signals, fs, lag_window)

    chest_angle = np.cumsum(chest[turn]) / fs
    pelvis_angle = np.cumsum(pelvis[turn]) / fs
    relative = chest_angle - pelvis_angle
    chest_excursion = float(abs(chest_angle[-1])) if len(chest_angle) else float("nan")
    pelvis_excursion = float(abs(pelvis_angle[-1])) if len(pelvis_angle) else float("nan")
    submovements, peak_rate = count_submovements(chest[turn], fs)
    jerk = dimensionless_jerk(signals["chest_yaw"][turn], fs)

    return {
        "trial": label,
        "event_id": event_id,
        "direction": direction if direction is not None else int(np.sign(chest_angle[-1])) if len(chest_angle) else 0,
        "turn_start_s": start / fs,
        "turn_end_s": end / fs,
        "turn_duration_s": duration_s,
        "chest_yaw_excursion_deg": chest_excursion,
        "pelvis_yaw_excursion_deg": pelvis_excursion,
        "chest_yaw_peak_rate_dps": float(np.max(np.abs(chest[turn]))) if duration_s > 0 else float("nan"),
        "chest_yaw_sparc": sparc(chest[turn], fs),
        "chest_yaw_submovement_count": submovements,
        "chest_yaw_submovement_rate_hz": submovements / duration_s if duration_s > 0 else float("nan"),
        "chest_pelvis_yaw_r": r,
        "chest_pelvis_lag_s": lag,
        # 1 = the trunk turns as a unit; below 1 the chest turns on a
        # comparatively still pelvis.
        "chest_pelvis_gain": (
            pelvis_excursion / chest_excursion if chest_excursion > 1e-9 else float("nan")
        ),
        "relative_yaw_range_deg": float(np.ptp(relative)) if len(relative) else float("nan"),
        "relative_yaw_sd_deg": float(np.std(relative)) if len(relative) else float("nan"),
        "chest_yaw_log10_dimensionless_jerk": math.log10(jerk) if jerk and jerk > 0 else float("nan"),
    }


SUMMARY_METRICS = (
    "turn_duration_s",
    "chest_yaw_excursion_deg",
    "chest_yaw_peak_rate_dps",
    "chest_yaw_sparc",
    "chest_yaw_submovement_rate_hz",
    "chest_pelvis_lag_s",
    "chest_pelvis_gain",
    "relative_yaw_range_deg",
)


def sequence_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """One row per recording: how many turns, and the median of each metric.

    Medians, not spreads: the turns are different movements, so their spread
    measures the choreography rather than consistency (see the module note).
    """
    rows = []
    for label, group in metrics.groupby("trial", sort=False):
        row = {
            "trial": label,
            "n_turns": int(len(group)),
            "total_turn_s": float(group["turn_duration_s"].sum()),
        }
        row.update({f"{name}_median": float(group[name].median()) for name in SUMMARY_METRICS})
        rows.append(row)
    return pd.DataFrame(rows)
