#!/usr/bin/env python3
"""Motion-driven detection of Tai Chi balance events.

The original detectors (:mod:`analysis.detection_v1`) score a fixed-width 8 s
sliding window, so every trunk-rotation event comes out exactly 8.000 s long
regardless of how long the movement actually took, and the stabilization search
thresholds at the 25th percentile of a local window, which by construction lets
~25 % of samples through whether or not the participant ever settled.

The detectors here derive the boundaries from the motion instead:

* :func:`detect_trunk_rotation_events` finds rotations from the trunk yaw
  *velocity* envelope and walks outward from each velocity peak to the moment
  the trunk starts and stops turning, so the duration is measured rather than
  assumed.  Differentiating also removes the constant yaw drift that this
  6-axis (no magnetometer) dataset suffers from.
* :func:`find_single_leg_phases` marks a single-leg stance wherever one foot is
  clearly higher than the other, from the tilt of both thighs and shanks
  (:func:`analysis.gravity.leg_lift_index`).  It finds the whole support phase
  of a lift with the knee extended as well as flexed, and which leg is up.
  The original knee-flexion > 60 deg rule, :func:`find_knee_flexion_windows`,
  is still selectable: it split every kick in two where the knee extended.
* :func:`detect_yaw_turns` cuts the whole form into single-direction turns of
  the chest, from one turning point of the yaw to the next, for the
  sequence-smoothness family.  A turn starts and ends at rest, which is what
  SPARC and the submovement count assume.
* :func:`find_stabilization` scores candidate quiet windows by how quiet they
  are *relative to the whole recording* and how soon they follow the event, and
  reports how quiet the window is so one that is not actually quiet can be
  flagged rather than silently used.

Novice and trained events are paired through the whole-recording alignment
(:mod:`analysis.alignment`), never by their order.  The signals the detectors
threshold on are exposed (:func:`trunk_yaw_envelope`, :func:`combined_omega`,
:func:`chest_yaw_signals`, :func:`lift_signal`), so a user interface can draw
the signal and the threshold next to the detected band and show *why* a
boundary landed where it did.
"""


from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from analysis.alignment import Alignment, match_windows
from analysis.gravity import leg_lift_index
from analysis.kinematics import Kinematics
from analysis.signals import extract_contiguous_runs, highpass_detrend, lowpass_signal


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass
class TrunkDetectorParams:
    """Tunable parameters for :func:`detect_trunk_rotation_events`."""

    n_events: int = 6
    rel_threshold: float = 0.15
    """Fraction of a peak's own velocity used as its onset/offset threshold."""
    floor_percentile: float = 40.0
    """Recording-wide percentile of the envelope used as an absolute floor."""
    peak_percentile: float = 90.0
    """Envelope percentile a candidate must exceed to count as a peak."""
    peak_separation_s: float = 2.0
    min_duration_s: float = 1.5
    max_duration_s: float = 20.0
    min_excursion_deg: float = 8.0
    envelope_cutoff_hz: float = 1.5
    yaw_cutoff_hz: float = 4.0


@dataclass
class KneeDetectorParams:
    """Tunable parameters for single-leg (monopodal) stance detection.

    ``method="lift"`` finds the stance from the leg lift index -- one foot
    higher than the other -- and is the default.  ``"knee_threshold"`` is the
    original rule, knee flexion above ``threshold_deg``, kept so v1 sessions
    reproduce; it cannot see a lift with the knee extended, and splits a kick
    where the knee straightens.
    """

    method: str = "lift"
    """``"lift"`` or ``"knee_threshold"``."""
    lift_min_peak: float = 0.25
    """How far a foot must rise for a stance to count, in thigh lengths (about
    10 cm): a physical definition of "clearly off the floor".  The single-leg
    phases here peak at 0.9-1.6; stepping and weight shifts stay below 0.1."""
    lift_rel_threshold: float = 0.15
    """Onset and offset, as a fraction of the event's own peak lift -- the same
    rule the trunk detector uses on its velocity envelope."""
    lift_min_duration_s: float = 0.5
    threshold_deg: float = 60.0
    """``knee_threshold`` only: flexion above which the stance counts as monopodal."""
    min_duration_s: float = 0.4
    """``knee_threshold`` only."""
    merge_gap_s: float = 0.2
    """``knee_threshold`` only: sub-threshold dips shorter than this do not split an event."""


@dataclass
class SegmentDetectorParams:
    """Tunable parameters for :func:`detect_yaw_turns`.

    The Tai Chi form is carried by large chest yaw rotations: the chest turns
    one way, then the other.  Each turn, from one turning point of the yaw to
    the next, starts and ends with the chest momentarily still, so it is a
    discrete movement in the sense the smoothness measures assume.  (The
    earlier unit, cut at the neutral crossings, started and ended at peak
    turning speed.)
    """

    yaw_cutoff_hz: float = 0.5
    """Smoothing applied before the turns are located.  Low, because only the
    carrier oscillation defines a turn, not the detail riding on it."""
    min_lobe_deg: float = 10.0
    """How far the chest must turn from neutral for a swing to count as one.
    The real turns here reach 40-70 deg, so this mainly rejects the chest
    hovering near neutral during the still passages of the form."""
    min_lobe_separation_s: float = 1.5
    """Minimum spacing between successive turning points, as for the trunk detector."""
    min_excursion_deg: float = 10.0
    min_duration_s: float = 1.0
    max_duration_s: float = 15.0
    """A turn longer than this spans a pause rather than a movement: the turns
    here last 2.5-13 s, and both recordings stand still for ~20 s at the start
    and end and hold the single-leg passage for ~35 s in the middle."""


@dataclass
class StabilizationParams:
    """Tunable parameters for :func:`find_stabilization`.

    Shared by both event families: the settling window is found the same way
    whether it follows a trunk rotation or a single-leg stance.

    Quietness is measured on the bias-corrected angular speed as
    ``z = (mean - baseline) / spread``, where the baseline is the recording's
    20th percentile and the spread the distance from there to its median.  So
    z = 0 is as quiet as the quietest fifth of the recording and z = 1 as busy
    as its median moment, whatever the participant's overall level or the
    sensors' bias.  (Before the gyroscope bias was removed the baseline was
    mostly bias -- about 21 deg/s of it -- and the ratio to it could not tell a
    settled window from an active one.)
    """

    min_duration_s: float = 3.0
    """Stabilization window length.  The pipeline used 2 s; measured against
    +/-0.25 s boundary jitter, the sway metrics' reliability rose from
    ICC 0.87 at 2 s to 0.96 at 3 s (0.99 at 4-5 s, but a longer window starts
    running into the next movement)."""
    horizon_s: float = 10.0
    latency_weight: float = 0.65
    """Penalty per second of delay after the event, in units of z.  0.65 is
    the trade-off the original ratio score made, expressed on the corrected
    signal: it reproduces 19 of 20 of the earlier automatic placements to within
    0.25 s."""
    baseline_percentile: float = 20.0
    scale_percentile: float = 50.0
    low_confidence_z: float = 1.0
    """A window busier than the recording's median moment is flagged: whoever
    it follows, the participant had not settled."""
    omega_cutoff_hz: float = 4.0


# ---------------------------------------------------------------------------
# Detected events and diagnostics
# ---------------------------------------------------------------------------


@dataclass
class TrunkEvent:
    """One detected trunk-rotation event, in sample indices."""

    start: int
    end: int
    peak: int
    excursion_deg: float
    peak_rate_dps: float
    score: float

    def duration_s(self, fs: float) -> float:
        return (self.end - self.start) / fs


@dataclass
class TrunkDiagnostics:
    """Signals the trunk detector thresholded on, for plotting."""

    yaw_deg: np.ndarray
    envelope_dps: np.ndarray
    floor_dps: float
    peak_height_dps: float


@dataclass
class StabilizationResult:
    """One stabilization window plus its self-assessed confidence."""

    start: int
    end: int
    quiet_ratio: float
    """Window mean angular speed divided by the recording's quiet baseline."""
    quiet_z: float
    """``(window mean - baseline) / spread``: 0 = the recording's quietest fifth, 1 = its median."""
    status: str
    """``"ok"``, ``"low-confidence"`` or ``"out-of-range"``."""

    @property
    def low_confidence(self) -> bool:
        return self.status != "ok"


@dataclass
class StabilizationDiagnostics:
    """Signals the stabilization search thresholded on, for plotting."""

    combined_omega_dps: np.ndarray
    baseline_dps: float
    scale_dps: float
    """Median minus baseline: the spread quietness is measured in."""

    def quiet_z(self, mean_dps: float) -> float:
        return float((mean_dps - self.baseline_dps) / max(self.scale_dps, 1e-9))


@dataclass
class SegmentDiagnostics:
    """Signals the turn detector thresholded on, for plotting."""

    chest_yaw_deg: np.ndarray
    pelvis_yaw_deg: np.ndarray
    smoothed_yaw_deg: np.ndarray
    min_lobe_deg: float


@dataclass
class LiftDiagnostics:
    """The signal the single-leg detector thresholded on, for plotting."""

    lift_index: np.ndarray
    """Left ankle height minus right, in thigh lengths (positive: left foot up)."""
    floor: float
    """The double-support level of |lift index|: no boundary is placed below it."""
    min_peak: float


@dataclass
class Turn:
    """One single-direction turn of the chest, in sample indices."""

    start: int
    end: int
    direction: int
    """+1 when the yaw increases through the turn, -1 when it decreases."""


# ---------------------------------------------------------------------------
# Trunk rotation
# ---------------------------------------------------------------------------


def trunk_yaw_envelope(
    kin: Kinematics, fs: float, params: TrunkDetectorParams | None = None
) -> TrunkDiagnostics:
    """Return the trunk yaw angle and the velocity envelope used to segment it.

    The envelope is the low-pass filtered magnitude of the yaw rate.  Because it
    is a derivative it is insensitive to the slow yaw drift that accumulates in
    a 6-axis IMU solution, unlike the amplitude-based score used by the original
    detector.
    """
    params = params or TrunkDetectorParams()

    yaw = lowpass_signal(
        highpass_detrend(kin.trunk_rel_euler_deg[:, 2], fs, cutoff_hz=0.05),
        fs,
        cutoff_hz=params.yaw_cutoff_hz,
    )
    rate = np.abs(np.gradient(yaw, 1.0 / fs))
    envelope = lowpass_signal(rate, fs, cutoff_hz=params.envelope_cutoff_hz)

    return TrunkDiagnostics(
        yaw_deg=yaw,
        envelope_dps=envelope,
        floor_dps=float(np.percentile(envelope, params.floor_percentile)),
        peak_height_dps=float(np.percentile(envelope, params.peak_percentile)),
    )


def trunk_rotation_candidates(
    kin: Kinematics, fs: float, params: TrunkDetectorParams | None = None
) -> tuple[list[TrunkEvent], TrunkDiagnostics]:
    """Every rotation the envelope supports, before the ``n_events`` cut.

    Each peak in the yaw-velocity envelope is a candidate rotation.  From the
    peak the search walks outward until the envelope falls below a threshold set
    to a fraction of that peak's own height — so a vigorous rotation and a gentle
    one are both bracketed at the same relative point — but never below a
    recording-wide floor, which stops the walk running away through a quiet
    stretch.  The resulting windows have whatever duration the movement had.
    Returned in time order.
    """
    params = params or TrunkDetectorParams()
    diagnostics = trunk_yaw_envelope(kin, fs, params)
    yaw, envelope = diagnostics.yaw_deg, diagnostics.envelope_dps

    peaks, _ = find_peaks(
        envelope,
        height=diagnostics.peak_height_dps,
        distance=max(1, int(params.peak_separation_s * fs)),
    )

    half_span = int(0.5 * params.max_duration_s * fs)
    candidates: list[TrunkEvent] = []
    for peak in peaks:
        threshold = max(params.rel_threshold * envelope[peak], diagnostics.floor_dps)

        start = int(peak)
        while start > 0 and envelope[start] > threshold and peak - start < half_span:
            start -= 1
        end = int(peak)
        last = len(envelope) - 1
        while end < last and envelope[end] > threshold and end - peak < half_span:
            end += 1

        if (end - start) / fs < params.min_duration_s:
            continue
        excursion = float(np.ptp(yaw[start:end]))
        if excursion < params.min_excursion_deg:
            continue

        candidates.append(
            TrunkEvent(
                start=start,
                end=end,
                peak=int(peak),
                excursion_deg=excursion,
                peak_rate_dps=float(envelope[peak]),
                score=excursion * float(np.log1p(envelope[peak])),
            )
        )
    return candidates, diagnostics


def _strongest_disjoint(items: list, score, n: int, spans) -> list:
    """The ``n`` highest-scoring items whose spans do not overlap."""
    selected: list = []
    for item in sorted(items, key=lambda item: -score(item)):
        if all(all(a_end < b_start or a_start > b_end
                   for (a_start, a_end), (b_start, b_end) in zip(spans(item), spans(kept)))
               for kept in selected):
            selected.append(item)
        if len(selected) == n:
            break
    return selected


def detect_trunk_rotation_events(
    kin: Kinematics, fs: float, params: TrunkDetectorParams | None = None
) -> tuple[list[TrunkEvent], TrunkDiagnostics]:
    """The ``n_events`` strongest non-overlapping rotations of one recording, in time order."""
    params = params or TrunkDetectorParams()
    candidates, diagnostics = trunk_rotation_candidates(kin, fs, params)
    selected = _strongest_disjoint(candidates, lambda event: event.score, params.n_events,
                                   lambda event: [(event.start, event.end)])
    selected.sort(key=lambda event: event.start)
    return selected, diagnostics


def recompute_trunk_peak(
    kin: Kinematics, fs: float, start: int, end: int, diagnostics: TrunkDiagnostics | None = None
) -> int:
    """Locate the yaw peak inside a window, matching the original definition.

    Mirrors the peak in :func:`analysis.detection_v1.detect_novice_trunk_rotation_events`
    so a manually adjusted window still reports a peak consistent with the rest
    of the pipeline.
    """
    yaw = diagnostics.yaw_deg if diagnostics is not None else trunk_yaw_envelope(kin, fs).yaw_deg
    segment = yaw[start:end]
    if len(segment) == 0:
        return int(start)
    return int(start + int(np.argmax(np.abs(segment - np.median(segment)))))


# ---------------------------------------------------------------------------
# Monopodal stance
# ---------------------------------------------------------------------------


def find_knee_flexion_windows(
    knee_angle_deg: np.ndarray,
    fs: float,
    threshold_deg: float = 60.0,
    min_duration_s: float = 0.4,
    merge_gap_s: float = 0.2,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Return windows where the knee flexion magnitude exceeds a threshold."""
    filtered = lowpass_signal(knee_angle_deg, fs, cutoff_hz=6.0)
    flexion_mag = np.abs(filtered)
    above = flexion_mag >= threshold_deg

    runs = extract_contiguous_runs(above)
    min_len = max(1, int(round(min_duration_s * fs)))
    gap_len = max(1, int(round(merge_gap_s * fs)))

    windows: list[tuple[int, int]] = []
    for start, end in runs:
        if end - start < min_len:
            continue
        if windows and start - windows[-1][1] <= gap_len:
            windows[-1] = (windows[-1][0], end)
        else:
            windows.append((start, end))

    return windows, flexion_mag


LIFT_CUTOFF_HZ = 2.0
"""Smoothing of the lift index: a lift lasts seconds, and above ~2 Hz the index
only carries soft-tissue and orientation-filter noise."""


def lift_signal(kin: Kinematics, fs: float, params: KneeDetectorParams | None = None) -> LiftDiagnostics:
    """The lift index the single-leg detector works on, with its floor.

    The floor is the recording's median |lift index|: most of any Tai Chi
    recording is spent with both feet down, so the median is the double-support
    level (0.03 thigh lengths here), and no boundary is walked below it.
    """
    params = params or KneeDetectorParams()
    lift = lowpass_signal(leg_lift_index(kin), fs, cutoff_hz=LIFT_CUTOFF_HZ)
    return LiftDiagnostics(lift_index=lift, floor=float(np.median(np.abs(lift))),
                           min_peak=float(params.lift_min_peak))


def find_single_leg_phases(
    kin: Kinematics, fs: float, params: KneeDetectorParams | None = None,
    diagnostics: LiftDiagnostics | None = None,
) -> list[tuple[int, int, str]]:
    """Single-leg support phases as ``(start, end, lifted_leg)``, in time order.

    Every peak of |lift index| above ``lift_min_peak`` is a lift.  From the peak
    the boundaries walk outward to where the lift falls below
    ``lift_rel_threshold`` of the peak's own height, never below the
    double-support floor, so each phase runs from lift-off to touch-down
    whatever the knee does in between.  The sign of the index at the peak says
    which leg is up.
    """
    params = params or KneeDetectorParams()
    diagnostics = diagnostics or lift_signal(kin, fs, params)
    lift = diagnostics.lift_index
    magnitude = np.abs(lift)
    peaks, _ = find_peaks(magnitude, height=params.lift_min_peak, distance=max(1, int(2.0 * fs)))

    phases: list[tuple[int, int, str]] = []
    for peak in peaks:
        threshold = max(params.lift_rel_threshold * magnitude[peak], diagnostics.floor)
        start = end = int(peak)
        while start > 0 and magnitude[start] > threshold:
            start -= 1
        while end < len(magnitude) - 1 and magnitude[end] > threshold:
            end += 1
        if (end - start) / fs < params.lift_min_duration_s:
            continue
        leg = "Left" if lift[peak] > 0 else "Right"
        if phases and start <= phases[-1][1] and phases[-1][2] == leg:
            phases[-1] = (phases[-1][0], max(end, phases[-1][1]), leg)  # two humps of one lift
            continue
        phases.append((start, end, leg))
    return phases


def peak_lift_index(lift: np.ndarray, start: int, end: int, leg: str) -> int:
    """Sample of the highest lift of ``leg``'s foot inside a window."""
    signed = lift[start:end] if leg == "Left" else -lift[start:end]
    return start + int(np.argmax(signed)) if len(signed) else start


# ---------------------------------------------------------------------------
# Sequence segments
# ---------------------------------------------------------------------------


def chest_yaw_signals(
    kin: Kinematics, fs: float, params: SegmentDetectorParams | None = None
) -> SegmentDiagnostics:
    """Return the chest and pelvis yaw traces the segment detector works on.

    Both are high-pass detrended the same way every other yaw channel in the
    pipeline is, because a 6-axis solution has no absolute heading reference.
    The smoothed copy is what the turns are located on: only the carrier
    oscillation defines a segment, not the detail riding on it.  The metrics are
    computed on the unsmoothed traces.
    """
    params = params or SegmentDetectorParams()

    chest = highpass_detrend(kin.eulers_deg["chestbone"][:, 2], fs, cutoff_hz=0.05)
    pelvis = highpass_detrend(kin.eulers_deg["lumbar"][:, 2], fs, cutoff_hz=0.05)

    return SegmentDiagnostics(
        chest_yaw_deg=chest,
        pelvis_yaw_deg=pelvis,
        smoothed_yaw_deg=lowpass_signal(chest, fs, cutoff_hz=params.yaw_cutoff_hz),
        min_lobe_deg=float(params.min_lobe_deg),
    )


def _turning_points(
    signal: np.ndarray, fs: float, params: SegmentDetectorParams
) -> list[tuple[int, int]]:
    """Alternating extrema of the yaw oscillation, as ``(index, direction)``.

    Keying on the extrema rather than on the neutral crossings is what makes the
    segmentation stable: a turn reaching 40-70 deg is unambiguous, whereas the
    crossings themselves are whatever the signal does while it is near zero, and
    debouncing them by amplitude silently welds the neighbours of any rejected
    wobble into one long lobe.
    """
    distance = max(1, int(params.min_lobe_separation_s * fs))
    positive, _ = find_peaks(signal, height=params.min_lobe_deg, distance=distance)
    negative, _ = find_peaks(-signal, height=params.min_lobe_deg, distance=distance)

    extrema = sorted(
        [(int(i), 1) for i in positive] + [(int(i), -1) for i in negative]
    )

    # A single turn can show two humps; keep only its largest, so that the list
    # strictly alternates in direction and every adjacent pair brackets one
    # neutral crossing.
    alternating: list[tuple[int, int]] = []
    for index, direction in extrema:
        if alternating and alternating[-1][1] == direction:
            if abs(signal[index]) > abs(signal[alternating[-1][0]]):
                alternating[-1] = (index, direction)
        else:
            alternating.append((index, direction))
    return alternating


def yaw_turning_points(
    kin: Kinematics, fs: float, params: SegmentDetectorParams | None = None
) -> tuple[list[tuple[int, int]], SegmentDiagnostics]:
    """The chest yaw's turning points as ``(index, direction)``, and the signals behind them."""
    params = params or SegmentDetectorParams()
    diagnostics = chest_yaw_signals(kin, fs, params)
    return _turning_points(diagnostics.smoothed_yaw_deg, fs, params), diagnostics


def turn_between(first: tuple[int, int], second: tuple[int, int], chest_yaw: np.ndarray, fs: float,
                 params: SegmentDetectorParams) -> Turn | None:
    """The turn from one turning point to another, or None if it cannot be one part of the form.

    Too short, too small or too long a span is dropped rather than reported
    with numbers that do not mean anything -- the duration cap in particular
    stops the turns either side of a pause being read as one slow turn.
    """
    (start, _), (end, direction) = first, second
    if not params.min_duration_s <= (end - start) / fs <= params.max_duration_s:
        return None
    if float(np.ptp(chest_yaw[start:end])) < params.min_excursion_deg:
        return None
    return Turn(start=int(start), end=int(end), direction=int(direction))


def detect_yaw_turns(
    kin: Kinematics, fs: float, params: SegmentDetectorParams | None = None
) -> tuple[list[Turn], SegmentDiagnostics]:
    """Cut the recording into single-direction turns of the chest.

    Each turn runs from one turning point of the (smoothed) chest yaw to the
    next, so it starts and ends with the chest momentarily still and turns one
    way throughout.  Consecutive turns abut.
    """
    params = params or SegmentDetectorParams()
    points, diagnostics = yaw_turning_points(kin, fs, params)
    turns = [turn_between(a, b, diagnostics.chest_yaw_deg, fs, params) for a, b in zip(points[:-1], points[1:])]
    return [turn for turn in turns if turn is not None], diagnostics


# ---------------------------------------------------------------------------
# Stabilization
# ---------------------------------------------------------------------------


def combined_omega(
    kin: Kinematics, fs: float, params: StabilizationParams | None = None
) -> StabilizationDiagnostics:
    """Return the lumbar+chest angular-velocity signal and its quiet baseline.

    This is the quantity the stabilization search minimises, so plotting it with
    the baseline makes the choice of window inspectable.
    """
    params = params or StabilizationParams()
    combined = lowpass_signal(
        kin.omega_mag["lumbar"], fs, cutoff_hz=params.omega_cutoff_hz
    ) + lowpass_signal(kin.omega_mag["chestbone"], fs, cutoff_hz=params.omega_cutoff_hz)
    baseline, middle = np.percentile(combined, [params.baseline_percentile, params.scale_percentile])
    return StabilizationDiagnostics(
        combined_omega_dps=combined,
        baseline_dps=float(baseline),
        scale_dps=float(middle - baseline),
    )


def find_stabilization(
    kin: Kinematics,
    fs: float,
    after_end: int,
    params: StabilizationParams | None = None,
    diagnostics: StabilizationDiagnostics | None = None,
) -> StabilizationResult:
    """Find the quiet window that follows an event, and say how quiet it is.

    Every candidate window in the horizon is scored as ``z + latency_weight *
    latency``, with z the window's quietness relative to the rest of *their own
    recording* (:class:`StabilizationParams`) rather than to a local percentile
    that always admits something; the latency term encodes that stabilization
    is what follows the event, which stops the search drifting to a quieter
    moment many seconds later.

    A plain global threshold was tried first and rejected: it removed the
    degenerate windows but pushed latencies out to 9-10 s, because in continuous
    Tai Chi there is never a genuinely still 2 s.  Nor is there, it turns out,
    after a trunk rotation: once the gyroscope bias is removed, 10 of the 12
    windows after the current trunk events are busier than the recording's
    median moment.  They are post-rotation windows, and flagged as such.
    """
    params = params or StabilizationParams()
    diagnostics = diagnostics or combined_omega(kin, fs, params)
    combined = diagnostics.combined_omega_dps
    baseline = max(diagnostics.baseline_dps, 1e-9)

    need = max(1, int(params.min_duration_s * fs))
    search_start = int(after_end)
    search_end = min(len(combined), search_start + int(params.horizon_s * fs))

    if search_end - search_start < need:
        start = min(search_start, max(0, len(combined) - need))
        end = min(start + need, len(combined))
        window = combined[start:end]
        mean = float(np.mean(window)) if len(window) else float("nan")
        return StabilizationResult(start, end, mean / baseline, diagnostics.quiet_z(mean), "out-of-range")

    rolling = pd.Series(combined[search_start:search_end]).rolling(need).mean().to_numpy()
    latency_s = (np.arange(len(rolling)) - need + 1) / fs
    z = (rolling - diagnostics.baseline_dps) / max(diagnostics.scale_dps, 1e-9)
    score = np.where(np.isfinite(rolling), z + params.latency_weight * latency_s, np.inf)

    best = int(np.argmin(score))
    start = search_start + best - need + 1
    end = search_start + best + 1
    quiet_z = float(z[best])
    status = "ok" if quiet_z <= params.low_confidence_z else "low-confidence"
    return StabilizationResult(start, end, float(rolling[best] / baseline), quiet_z, status)


def window_quietness(diagnostics: StabilizationDiagnostics, start: int, end: int) -> tuple[float, float]:
    """``(quiet_ratio, quiet_z)`` of an arbitrary window, e.g. one placed by hand."""
    mean = float(np.mean(diagnostics.combined_omega_dps[start:end])) if end > start else float("nan")
    return mean / max(diagnostics.baseline_dps, 1e-9), diagnostics.quiet_z(mean)


# ---------------------------------------------------------------------------
# Pairing novice and trained events
# ---------------------------------------------------------------------------


def pair_trunk_events(
    novice_kin: Kinematics,
    trained_kin: Kinematics,
    novice_fs: float,
    trained_fs: float,
    alignment: Alignment,
    trunk_params: TrunkDetectorParams | None = None,
    stab_params: StabilizationParams | None = None,
) -> tuple[dict[str, list[tuple[int, int, int, int, float, float]]], list[TrunkEvent], list[TrunkEvent]]:
    """Detect trunk rotations in both recordings and pair those that are the same movement.

    Every candidate rotation of each recording is carried onto the other's
    clock by the whole-recording alignment.  A novice and a trained candidate
    pair when each is the other's best match by intersection over union, at
    0.5 or more; the ``n_events`` pairs whose weaker partner is strongest are
    kept.  Pairing by order, which this replaces, compared a different movement
    in all six pairs of the working session: the top six rotations of one
    recording are not the top six of the other.

    Each window entry is ``(event_start, event_end, stab_start, stab_end,
    quiet_ratio, quiet_z)``.
    """
    trunk_params = trunk_params or TrunkDetectorParams()
    stab_params = stab_params or StabilizationParams()

    novice_all, _ = trunk_rotation_candidates(novice_kin, novice_fs, trunk_params)
    trained_all, _ = trunk_rotation_candidates(trained_kin, trained_fs, trunk_params)
    matches = match_windows(
        alignment,
        [(e.start / novice_fs, e.end / novice_fs) for e in novice_all],
        [(e.start / trained_fs, e.end / trained_fs) for e in trained_all],
    )
    pairs = [(novice_all[i], trained_all[j]) for i, j in matches]
    pairs = _strongest_disjoint(pairs, lambda pair: min(pair[0].score, pair[1].score), trunk_params.n_events,
                                lambda pair: [(pair[0].start, pair[0].end), (pair[1].start, pair[1].end)])
    pairs.sort(key=lambda pair: pair[0].start)

    windows: dict[str, list[tuple[int, int, int, int, float, float]]] = {"Novice": [], "Trained": []}
    for label, kin, fs, side in (("Novice", novice_kin, novice_fs, 0), ("Trained", trained_kin, trained_fs, 1)):
        omega = combined_omega(kin, fs, stab_params)
        for pair in pairs:
            event = pair[side]
            stabilization = find_stabilization(kin, fs, event.end, stab_params, diagnostics=omega)
            windows[label].append(
                (event.start, event.end, stabilization.start, stabilization.end,
                 round(stabilization.quiet_ratio, 3), round(stabilization.quiet_z, 3))
            )
    return windows, [pair[0] for pair in pairs], [pair[1] for pair in pairs]


# ---------------------------------------------------------------------------
# Storing parameter sets
# ---------------------------------------------------------------------------


PARAM_CLASSES = {
    "trunk": TrunkDetectorParams,
    "knee": KneeDetectorParams,
    "smooth": SegmentDetectorParams,
    "stab": StabilizationParams,
}


def params_to_dict(
    trunk: TrunkDetectorParams,
    stabilization: StabilizationParams,
    knee: KneeDetectorParams | None = None,
    segment: SegmentDetectorParams | None = None,
) -> dict[str, float]:
    """Flatten the parameter sets for storage in the session file."""
    merged: dict[str, float] = {}
    for prefix, params in (("trunk", trunk), ("knee", knee or KneeDetectorParams()),
                           ("smooth", segment or SegmentDetectorParams()),
                           ("stab", stabilization)):
        merged.update({f"{prefix}_{k}": v for k, v in asdict(params).items()})
    return merged


def params_from_dict(
    values: dict[str, float]
) -> tuple[TrunkDetectorParams, StabilizationParams, KneeDetectorParams, SegmentDetectorParams]:
    """Rebuild the parameter sets from a flattened dict, ignoring unknown keys."""
    built = {}
    for prefix, cls in PARAM_CLASSES.items():
        fields = set(cls.__dataclass_fields__)
        supplied = {
            key[len(prefix) + 1:]: value
            for key, value in values.items()
            if key.startswith(f"{prefix}_")
        }
        built[prefix] = cls(**{k: v for k, v in supplied.items() if k in fields})
    return built["trunk"], built["stab"], built["knee"], built["smooth"]
