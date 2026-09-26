"""Sanity checks on a session's windows, run before every recalculation.

Errors block recalculation: a window out of order, outside the recording, or
too short for the metric code to handle; a pair whose two windows are not the
same movement of the form; a stance labelled with the leg that stayed down.
Warnings are shown in the editor and leave the call to the user: a
stabilization window the participant had not settled in, a pair whose windows
cover the same movement but with quite different boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analysis import alignment, detection
from analysis.config import TRIALS
from analysis.kinematics import LoadedTrial
from analysis.sessions import FAMILIES, FAMILY_KEY
from analysis.signals import LAG_MIN_OVERLAP, LAG_SEARCH_S


MIN_EVENT_S = 0.2
"""Hard floor: below ~24 samples (0.065 s) lowpass_signal silently returns the
signal unfiltered.  Kept permissive because the editor must be able to load
whatever a detector produced."""

SHORT_EVENT_WARN_S = 1.0
"""Below this the smoothness metrics are computed over very few cycles."""

MIN_STAB_S = 0.5
"""count_corrective_peaks calls find_peaks(distance=int(0.3*fs))."""

LAG_WINDOW_MIN_S = LAG_SEARCH_S / (1.0 - LAG_MIN_OVERLAP)
"""The shortest lag window in which the +/-1 s search keeps enough overlap at its edges."""

MIN_PAIR_OVERLAP = 0.5
"""A pair must overlap, once on one clock, by at least this share of its shorter window."""

MIN_PAIR_IOU = 0.5
"""Below this intersection over union the pair is the same movement cut differently."""


@dataclass
class Issue:
    event_id: str
    trial: str
    severity: str  # "error" or "warning"
    message: str


def validate_window(event_id: str, trial: str, window: dict, fs: float, n_samples: int,
                    lag_pad_s: float | None = None, low_confidence_z: float = 1.0) -> list[Issue]:
    issues: list[Issue] = []
    start, end = window["event_start"], window["event_end"]
    # Sequence turns have no stabilization band; everything about one is
    # skipped rather than defaulted, so a missing band cannot read as a broken one.
    has_stab = "stab_start" in window and "stab_end" in window
    stab_start, stab_end = window.get("stab_start", 0), window.get("stab_end", 0)

    def error(message: str) -> None:
        issues.append(Issue(event_id, trial, "error", message))

    def warn(message: str) -> None:
        issues.append(Issue(event_id, trial, "warning", message))

    spans = {"event": (start, end)}
    if has_stab:
        spans["stabilization"] = (stab_start, stab_end)
    for name, (lo, hi) in spans.items():
        if not (0 <= lo < hi <= n_samples - 1):
            error(f"{name} window [{lo}, {hi}] is out of order or outside the recording")

    duration_s = (end - start) / fs
    if end > start and duration_s < MIN_EVENT_S:
        error(f"event window is {duration_s:.2f} s, shorter than the {MIN_EVENT_S:.2f} s minimum")
    if has_stab and stab_end > stab_start and (stab_end - stab_start) / fs < MIN_STAB_S:
        error(
            f"stabilization window is {(stab_end - stab_start) / fs:.2f} s, "
            f"shorter than the {MIN_STAB_S:.1f} s minimum"
        )

    lag_window_s = duration_s + 2.0 * (lag_pad_s or 0.0)
    if end > start and duration_s < SHORT_EVENT_WARN_S:
        warn(f"event is only {duration_s:.2f} s; smoothness metrics span very few cycles")
    elif end > start and lag_window_s < LAG_WINDOW_MIN_S:
        warn(
            f"the coordination-lag window is {lag_window_s:.2f} s; the +/-{LAG_SEARCH_S:.0f} s search "
            f"needs at least {LAG_WINDOW_MIN_S:.0f} s and may return no lag"
        )
    if has_stab:
        if stab_start < end:
            warn("stabilization overlaps the event")
        z = window.get("stab_quiet_z")
        if z is not None and z > low_confidence_z:
            warn(f"stabilization is busier than the recording's median moment (quiet z {z:.2f}) - "
                 "the participant may not have settled")
    return issues


SEGMENT_OVERLAP_WARN_S = 0.5
"""Turns tile the form, so neighbours should meet.  A gap or an overlap larger
than this is usually a boundary dragged past its neighbour."""


def check_segment_continuity(events: list[dict], trials: dict[str, LoadedTrial]) -> list[Issue]:
    """Flag turns that overlap their neighbour.

    Unlike the other two families these windows are meant to abut: each ends
    at the turning point where the next begins.  A real pause in the form
    leaves a legitimate gap, so this warns rather than errors.
    """
    issues: list[Issue] = []
    for label in (label for label in TRIALS if label in trials):
        fs = trials[label].fs
        spans = [
            (event["event_id"], event["windows"][label])
            for event in events
            if label in event.get("windows", {})
        ]
        spans.sort(key=lambda item: item[1]["event_start"])
        for (_, first), (event_id, second) in zip(spans[:-1], spans[1:]):
            overlap_s = (first["event_end"] - second["event_start"]) / fs
            if overlap_s > SEGMENT_OVERLAP_WARN_S:
                issues.append(Issue(event_id, label, "warning",
                                    f"overlaps the previous turn by {overlap_s:.2f} s"))
    return issues


def pair_overlap(event: dict, trials: dict[str, LoadedTrial]) -> tuple[float, float] | None:
    """``(overlap / shorter, IoU)`` of an event's two windows on one clock, or None if unpaired."""
    windows = event.get("windows", {})
    aligned = alignment.for_trials(trials)
    if aligned is None or not all(label in windows for label in TRIALS):
        return None
    novice, trained = windows["Novice"], windows["Trained"]
    fs_n, fs_t = trials["Novice"].fs, trials["Trained"].fs
    return alignment.mapped_overlap(
        aligned,
        (novice["event_start"] / fs_n, novice["event_end"] / fs_n),
        (trained["event_start"] / fs_t, trained["event_end"] / fs_t),
    )


def check_pairing(events: list[dict], trials: dict[str, LoadedTrial], lenient: bool = False) -> list[Issue]:
    """Flag pairs whose two windows are not the same movement of the form.

    Each novice window is carried onto the trained clock by the whole-recording
    alignment.  Overlapping the trained window by less than half of the shorter
    one means a different movement: an error, unless the event is marked
    ``pair_confirmed`` (a deliberate exception) or ``lenient`` is set, as for v1
    sessions whose DTW pairs predate the alignment.  Same movement but an IoU
    below one half means the two windows are cut quite differently: a warning.
    """
    issues: list[Issue] = []
    aligned = alignment.for_trials(trials)
    for event in events:
        result = pair_overlap(event, trials)
        if result is None:
            continue
        shared, iou = result
        novice_start = event["windows"]["Novice"]["event_start"] / trials["Novice"].fs
        expected = float(aligned.to_trained(novice_start))
        if shared < MIN_PAIR_OVERLAP:
            confirmed = bool(event.get("pair_confirmed"))
            severity = "warning" if (confirmed or lenient) else "error"
            issues.append(Issue(
                event["event_id"], "both", severity,
                f"the two windows are not the same movement: they overlap by {shared:.0%} once aligned "
                f"(the novice window starts where the trained recording is at {expected:.1f} s)"
                + (" - pairing confirmed by hand" if confirmed else ""),
            ))
        elif iou < MIN_PAIR_IOU:
            issues.append(Issue(
                event["event_id"], "both", "warning",
                f"same movement, but the windows are cut differently (IoU {iou:.2f} once aligned)",
            ))
    return issues


_lift_cache: dict[str, np.ndarray] = {}


def lift_index_of(loaded: LoadedTrial) -> np.ndarray:
    """The recording's lift index, computed once per file: validation runs on every edit."""
    key = loaded.recording.sha256()
    if key not in _lift_cache:
        _lift_cache[key] = detection.lift_signal(loaded.kin, loaded.fs).lift_index
    return _lift_cache[key]


def check_leg_side(events: list[dict], trials: dict[str, LoadedTrial]) -> list[Issue]:
    """Flag a stance whose labelled lifted leg is the lower foot in the window."""
    issues: list[Issue] = []
    lift = {label: lift_index_of(loaded) for label, loaded in trials.items()}
    for event in events:
        leg = event.get("lifted_leg")
        for label, window in event.get("windows", {}).items():
            if label not in lift or leg not in ("Left", "Right"):
                continue
            mean = float(np.mean(lift[label][window["event_start"]:window["event_end"]]))
            higher = "Left" if mean > 0 else "Right"
            if higher != leg:
                issues.append(Issue(event["event_id"], label, "error",
                                    f"labelled with the {leg.lower()} leg lifted, but the {higher.lower()} foot "
                                    f"is the higher one in this window (lift index {mean:+.2f}): swap the leg"))
    return issues


def validate_session(session: dict, trials: dict[str, LoadedTrial]) -> list[Issue]:
    issues: list[Issue] = []
    detector = session.get("detector") or {}
    _, stab_params, _, _ = detection.params_from_dict(detector)
    lenient = str(detector.get("name", "")).startswith("v1")
    for family in FAMILIES:
        enabled = [e for e in session.get(FAMILY_KEY[family], []) if e.get("enabled", True)]
        for event in enabled:
            for label, window in event.get("windows", {}).items():
                if label not in trials:
                    issues.append(Issue(event["event_id"], label, "error",
                                        f"has a window for the {label.lower()} recording, which is not selected"))
                    continue
                loaded = trials[label]
                issues += validate_window(
                    event["event_id"], label, window, loaded.fs, loaded.n_samples,
                    lag_pad_s=session.get("lag_pad_s"), low_confidence_z=stab_params.low_confidence_z,
                )
        issues += check_pairing(enabled, trials, lenient=lenient)
        if family == "knee":
            issues += check_leg_side(enabled, trials)
        if family == "smooth":
            issues += check_segment_continuity(enabled, trials)
    return issues
