#!/usr/bin/env python3
"""Motion-driven detection of Tai Chi balance events.

The original detectors in :mod:`tai_chi_trunk_and_knee` score a fixed-width 8 s
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
* :func:`find_stabilization` scores candidate quiet windows by how quiet they
  are *relative to the whole recording* and how soon they follow the event, and
  reports a quiet ratio so a window that is not actually quiet can be flagged
  rather than silently used.

Both return the diagnostic signals they thresholded on, so a user interface can
draw the envelope and the threshold next to the detected band and show *why* a
boundary landed where it did.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from tai_chi_trunk_and_knee import Kinematics, highpass_detrend, lowpass_signal


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
    """Tunable parameters for monopodal-stance detection.

    Mirrors the arguments of ``find_knee_flexion_windows`` so the thresholds
    that define a single-leg stance can be adjusted from the interface rather
    than being fixed in code.
    """

    threshold_deg: float = 60.0
    """Knee flexion magnitude above which the stance counts as monopodal."""
    min_duration_s: float = 0.4
    merge_gap_s: float = 0.2
    """Sub-threshold dips shorter than this do not split one event into two."""


@dataclass
class SegmentDetectorParams:
    """Tunable parameters for :func:`detect_yaw_cycle_segments`.

    The Tai Chi form is carried by large chest yaw rotations: the chest turns
    one way, back through neutral, then the other way.  Cutting the recording at
    those neutral crossings gives comparable parts of the sequence without
    hand-picking events, which is what the smoothness family is scored over.
    """

    half_cycles: bool = False
    """``False`` = one segment per full back-and-forth, ``True`` = one per
    single-direction excursion."""
    yaw_cutoff_hz: float = 0.5
    """Smoothing applied before the turns are located.  Low, because only the
    carrier oscillation defines a segment, not the detail riding on it."""
    min_lobe_deg: float = 10.0
    """How far the chest must turn from neutral for a swing to count as one.
    The real turns here reach 40-70 deg, so this mainly rejects the chest
    hovering near neutral during the still passages of the form."""
    min_lobe_separation_s: float = 1.5
    """Minimum spacing between successive turns, as for the trunk detector."""
    min_excursion_deg: float = 10.0
    min_duration_s: float = 2.0
    max_duration_s: float = 20.0
    """Segments longer than this span a pause rather than a movement.  Both
    recordings stand still for ~18 s at the start, ~35 s at the end and ~40 s in
    the middle; without this cap the turns either side of a pause are joined
    into one 30 s "cycle" that is mostly not moving."""


@dataclass
class StabilizationParams:
    """Tunable parameters for :func:`find_stabilization`.

    Shared by both event families: the settling window is found the same way
    whether it follows a trunk rotation or a single-leg stance.
    """

    min_duration_s: float = 3.0
    """Stabilization window length.  The pipeline used 2 s; measured against
    +/-0.25 s boundary jitter, the sway metrics' reliability rises from
    ICC 0.87 at 2 s to 0.96 at 3 s (0.99 at 4-5 s, but a longer window starts
    running into the next movement)."""
    horizon_s: float = 10.0
    latency_weight: float = 0.35
    """Penalty per second of delay after the event, in quiet-ratio units."""
    baseline_percentile: float = 20.0
    low_confidence_ratio: float = 1.6
    omega_cutoff_hz: float = 4.0


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
    """Window mean angular velocity divided by the recording quiet baseline."""
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


@dataclass
class SegmentDiagnostics:
    """Signals the segment detector thresholded on, for plotting."""

    chest_yaw_deg: np.ndarray
    pelvis_yaw_deg: np.ndarray
    smoothed_yaw_deg: np.ndarray
    min_lobe_deg: float


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


def detect_trunk_rotation_events(
    kin: Kinematics, fs: float, params: TrunkDetectorParams | None = None
) -> tuple[list[TrunkEvent], TrunkDiagnostics]:
    """Detect trunk-rotation events with boundaries taken from the movement.

    Each peak in the yaw-velocity envelope is a candidate rotation.  From the
    peak the search walks outward until the envelope falls below a threshold set
    to a fraction of that peak's own height — so a vigorous rotation and a gentle
    one are both bracketed at the same relative point — but never below a
    recording-wide floor, which stops the walk running away through a quiet
    stretch.  The resulting windows have whatever duration the movement had.
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

    candidates.sort(key=lambda event: -event.score)
    selected: list[TrunkEvent] = []
    for candidate in candidates:
        if all(candidate.end < kept.start or candidate.start > kept.end for kept in selected):
            selected.append(candidate)
        if len(selected) == params.n_events:
            break

    selected.sort(key=lambda event: event.start)
    return selected, diagnostics


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


def _neutral_between(signal: np.ndarray, first: int, second: int) -> int:
    """Where the yaw passes neutral between two opposing turns."""
    span = signal[first:second + 1]
    crossings = np.where(np.diff(np.sign(span)) != 0)[0]
    if len(crossings):
        return first + int(crossings[0]) + 1
    # No sign change: the oscillation never quite reached neutral.  The closest
    # approach is the boundary, which keeps the detector working whatever the
    # high-pass leaves behind.
    return first + int(np.argmin(np.abs(span)))


def detect_yaw_cycle_segments(
    kin: Kinematics, fs: float, params: SegmentDetectorParams | None = None
) -> tuple[list[tuple[int, int]], SegmentDiagnostics]:
    """Cut the recording into parts at the neutral crossings of the chest yaw.

    Boundaries are placed where the yaw passes neutral between two opposing
    turns, so the interval between consecutive boundaries is exactly one
    single-direction excursion: a full back-and-forth spans two of them, a half
    cycle one.  Segments too short, too small or too long to be one part of the
    form are dropped rather than reported with numbers that do not mean
    anything -- in particular the duration cap is what stops the turns either
    side of a pause in the form being joined into one very long "cycle".
    """
    params = params or SegmentDetectorParams()
    diagnostics = chest_yaw_signals(kin, fs, params)
    smoothed = diagnostics.smoothed_yaw_deg

    turns = _turning_points(smoothed, fs, params)
    boundaries = [
        _neutral_between(smoothed, first, second)
        for (first, _), (second, _) in zip(turns[:-1], turns[1:])
    ]

    step = 1 if params.half_cycles else 2
    spans = [
        (boundaries[i], boundaries[i + step])
        for i in range(0, len(boundaries) - step, step)
    ]

    chest = diagnostics.chest_yaw_deg
    segments = [
        (start, end)
        for start, end in spans
        if params.min_duration_s <= (end - start) / fs <= params.max_duration_s
        and float(np.ptp(chest[start:end])) >= params.min_excursion_deg
    ]
    return segments, diagnostics


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
    return StabilizationDiagnostics(
        combined_omega_dps=combined,
        baseline_dps=float(np.percentile(combined, params.baseline_percentile)),
    )


def find_stabilization(
    kin: Kinematics,
    fs: float,
    after_end: int,
    params: StabilizationParams | None = None,
    diagnostics: StabilizationDiagnostics | None = None,
) -> StabilizationResult:
    """Find the quiet window that follows an event, and say how quiet it is.

    Every candidate window in the horizon is scored as ``mean omega / baseline +
    latency_weight * latency``.  The first term asks whether the participant is
    quiet compared to the rest of *their own recording* rather than compared to a
    local percentile that always admits something; the second encodes that
    stabilization is what follows the event, which stops the search drifting to a
    quieter moment many seconds later.

    A plain global threshold was tried first and rejected: it removed the
    degenerate windows but pushed latencies out to 9-10 s, because in continuous
    Tai Chi there is never a genuinely still 2 s.
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
        ratio = float(np.mean(window) / baseline) if len(window) else float("nan")
        return StabilizationResult(start, end, ratio, "out-of-range")

    rolling = pd.Series(combined[search_start:search_end]).rolling(need).mean().to_numpy()
    latency_s = (np.arange(len(rolling)) - need + 1) / fs
    score = np.where(
        np.isfinite(rolling),
        rolling / baseline + params.latency_weight * latency_s,
        np.inf,
    )

    best = int(np.argmin(score))
    start = search_start + best - need + 1
    end = search_start + best + 1
    ratio = float(rolling[best] / baseline)
    status = "ok" if ratio <= params.low_confidence_ratio else "low-confidence"
    return StabilizationResult(start, end, ratio, status)


def recompute_trunk_peak(
    kin: Kinematics, fs: float, start: int, end: int, diagnostics: TrunkDiagnostics | None = None
) -> int:
    """Locate the yaw peak inside a window, matching the original definition.

    Mirrors ``detect_novice_trunk_rotation_events`` line 504 so a manually
    adjusted window still reports a peak consistent with the rest of the
    pipeline.
    """
    yaw = diagnostics.yaw_deg if diagnostics is not None else trunk_yaw_envelope(kin, fs).yaw_deg
    segment = yaw[start:end]
    if len(segment) == 0:
        return int(start)
    return int(start + int(np.argmax(np.abs(segment - np.median(segment)))))


# ----------------------------------------------------------------------------
# Fast DTW pairing
#
# signature_matrix() in the pipeline low-pass filters the full 89 k-sample
# signal for each of 7 channels on every candidate window, so find_trained_match
# spends minutes per event re-filtering identical data.  Filtering once up front
# and slicing afterwards is numerically identical and reduces the cost to the
# DTW itself.
# ----------------------------------------------------------------------------


SIGNATURE_CUTOFF_HZ = 6.0
"""Cutoff used by ``resample_filtered_window`` for a 20 Hz target rate."""


@dataclass
class SignatureChannels:
    """Pre-filtered channels for the DTW signature, in source-rate samples."""

    time_s: np.ndarray
    channels: list[np.ndarray] = field(default_factory=list)


def precompute_signature_channels(kin: Kinematics, fs: float) -> SignatureChannels:
    """Filter the seven DTW signature channels once, at the source rate.

    The channel list and their order match ``signature_matrix`` exactly.
    """
    raw = [
        highpass_detrend(kin.trunk_rel_euler_deg[:, 2], fs, cutoff_hz=0.05),
        kin.eulers_deg["lumbar"][:, 0],
        kin.omega_mag["chestbone"],
        kin.left_knee_deg[:, 0],
        kin.right_knee_deg[:, 0],
        kin.omega_mag["lhand"],
        kin.omega_mag["rhand"],
    ]
    return SignatureChannels(
        time_s=kin.t,
        channels=[lowpass_signal(channel, fs, cutoff_hz=SIGNATURE_CUTOFF_HZ) for channel in raw],
    )


def signature_matrix(
    precomputed: SignatureChannels, fs: float, start: int, end: int, target_fs: float = 20.0
) -> np.ndarray:
    """Resample the pre-filtered channels over a window and z-score them."""
    from scipy.stats import zscore

    time = precomputed.time_s
    n_target = max(2, int(round((end - start) / fs * target_fs)))
    target_time = time[start] + np.arange(n_target) / target_fs
    matrix = np.column_stack(
        [np.interp(target_time, time, channel) for channel in precomputed.channels]
    )
    return np.nan_to_num(zscore(matrix, axis=0, nan_policy="omit"))


def find_trained_match(
    novice: SignatureChannels,
    trained: SignatureChannels,
    fs: float,
    novice_start: int,
    novice_end: int,
    min_start: int = 0,
    target_fs: float = 20.0,
    ignore_s: float = 15.0,
    step_s: float = 1.0,
) -> tuple[int, int, float]:
    """Locate the trained window best matching a novice window, by banded DTW.

    Same search and same distance as the pipeline's ``find_trained_match``, but
    operating on pre-filtered channels so the low-pass filtering happens once
    rather than once per candidate window.
    """
    from tai_chi_trunk_and_knee import dtw_distance

    template = signature_matrix(novice, fs, novice_start, novice_end, target_fs)
    window = novice_end - novice_start
    ignore = int(ignore_s * fs)
    start_search = max(ignore, min_start)
    step = max(1, int(step_s * fs))
    band = int(1.5 * target_fs)

    stop = len(trained.time_s) - window - ignore
    if start_search >= stop:
        # No admissible window remains -- the caller's monotonic constraint has
        # run past the end of the recording.  Report it rather than returning a
        # silently meaningless window at an infinite distance.
        return start_search, min(start_search + window, len(trained.time_s) - 1), float("nan")

    best = (start_search, start_search + window, np.inf)
    for start in range(start_search, stop, step):
        end = start + window
        distance = dtw_distance(template, signature_matrix(trained, fs, start, end, target_fs), band=band)
        if distance < best[2]:
            best = (start, end, distance)
    return best


def pair_trunk_events(
    novice_kin: Kinematics,
    trained_kin: Kinematics,
    novice_fs: float,
    trained_fs: float,
    trunk_params: TrunkDetectorParams | None = None,
    stab_params: StabilizationParams | None = None,
) -> tuple[dict[str, list[tuple[int, int, int, int, float]]], list[TrunkEvent], list[TrunkEvent]]:
    """Detect trunk events in both recordings and pair them by order.

    Both participants perform the same form, so the k-th rotation in one
    recording corresponds to the k-th in the other.  Pairing by order is used
    rather than the DTW search because velocity-segmented events are only a few
    seconds long, and a short window is not distinctive enough for the DTW to
    match reliably -- a single mismatch then propagates, because the search is
    constrained to move forward monotonically.  Any mispairing that does occur
    is visible and correctable in the editor.

    Each entry is ``(event_start, event_end, stab_start, stab_end, quiet_ratio)``.
    """
    trunk_params = trunk_params or TrunkDetectorParams()
    stab_params = stab_params or StabilizationParams()

    novice_events, _ = detect_trunk_rotation_events(novice_kin, novice_fs, trunk_params)
    trained_events, _ = detect_trunk_rotation_events(trained_kin, trained_fs, trunk_params)

    novice_omega = combined_omega(novice_kin, novice_fs, stab_params)
    trained_omega = combined_omega(trained_kin, trained_fs, stab_params)

    windows: dict[str, list[tuple[int, int, int, int, float]]] = {"Novice": [], "Trained": []}
    for events, kin, fs, omega, label in (
        (novice_events, novice_kin, novice_fs, novice_omega, "Novice"),
        (trained_events, trained_kin, trained_fs, trained_omega, "Trained"),
    ):
        for event in events:
            stabilization = find_stabilization(kin, fs, event.end, stab_params, diagnostics=omega)
            windows[label].append(
                (event.start, event.end, stabilization.start, stabilization.end,
                 round(stabilization.quiet_ratio, 3))
            )

    # Only complete pairs are usable, since every trunk metric is reported for
    # both trials of the same event.
    paired = min(len(windows["Novice"]), len(windows["Trained"]))
    windows["Novice"] = windows["Novice"][:paired]
    windows["Trained"] = windows["Trained"][:paired]
    return windows, novice_events[:paired], trained_events[:paired]


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
