"""Sanity checks on a session's windows, run before every recalculation.

Errors block recalculation: a window out of order, outside the recording, or
too short for the metric code to handle.  Warnings are shown in the editor and
leave the call to the user: a window short enough for the coordination lag to
saturate, a stabilization window the participant had not settled in, a pair
that looks mismatched.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analysis.config import TRIALS
from analysis.editor.sessions import FAMILIES, FAMILY_KEY
from analysis.kinematics import LoadedTrial


MIN_EVENT_S = 0.2
"""Hard floor: below ~24 samples (0.065 s) lowpass_signal silently returns the
signal unfiltered.  Kept permissive because find_knee_flexion_windows accepts
events from 0.4 s, and the editor must be able to load what the pipeline
produced."""

SHORT_EVENT_WARN_S = 1.0
"""Below this the smoothness metrics are computed over very few cycles."""

MIN_STAB_S = 0.5
"""count_corrective_peaks calls find_peaks(distance=int(0.3*fs))."""

LAG_SATURATION_WARN_S = 4.0
"""cross_correlation_lag searches +/-2 s; a shorter window saturates the lag."""

LOW_CONFIDENCE_RATIO = 1.6


@dataclass
class Issue:
    event_id: str
    trial: str
    severity: str  # "error" or "warning"
    message: str


def validate_window(event_id: str, trial: str, window: dict, fs: float, n_samples: int) -> list[Issue]:
    issues: list[Issue] = []
    start, end = window["event_start"], window["event_end"]
    # Sequence segments have no stabilization band; everything about one is
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

    if end > start and duration_s < SHORT_EVENT_WARN_S:
        warn(f"event is only {duration_s:.2f} s; smoothness metrics span very few cycles")
    elif end > start and duration_s < LAG_SATURATION_WARN_S:
        warn(
            f"event is {duration_s:.2f} s; the coordination lag searches +/-2 s "
            "and may saturate on a window this short"
        )
    if has_stab:
        if stab_start <= end:
            warn("stabilization overlaps the event")
        ratio = window.get("stab_quiet_ratio")
        if ratio is not None and ratio > LOW_CONFIDENCE_RATIO:
            warn(f"stabilization is {ratio:.2f}x the quiet baseline - the participant may not have settled")
    return issues


SEGMENT_OVERLAP_WARN_S = 0.5
"""Sequence segments tile the form, so neighbours should meet.  A gap or an
overlap larger than this is usually a boundary dragged past its neighbour."""


def check_segment_continuity(events: list[dict], trials: dict[str, LoadedTrial]) -> list[Issue]:
    """Flag sequence segments that overlap or leave a hole against their neighbour.

    Unlike the other two families these windows are meant to abut: each ends
    where the chest passes neutral and the next begins.  A real pause in the
    form leaves a legitimate gap, so this warns rather than errors.
    """
    issues: list[Issue] = []
    for label in TRIALS:
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
                                    f"overlaps the previous segment by {overlap_s:.2f} s"))
    return issues


PAIR_OFFSET_TOLERANCE_S = 8.0
"""How far a pair's novice-to-trained offset may sit from the typical one."""


def check_pairing(events: list[dict], trials: dict[str, LoadedTrial]) -> list[Issue]:
    """Flag pairs whose two windows sit at an atypical offset from each other.

    Both participants perform the same form, so across a family the novice and
    trained windows should be separated by roughly a constant offset -- whatever
    the difference in when each recording started and how fast each moves.  A
    pair far from that typical offset is usually a mispairing rather than a real
    difference, since pairing is by order and one missing event shifts the rest.
    Comparing against the median offset rather than against zero keeps this
    robust to the recordings simply not starting together.
    """
    offsets = {}
    for event in events:
        windows = event.get("windows", {})
        if not all(label in windows for label in TRIALS):
            continue
        offsets[event["event_id"]] = (
            windows["Novice"]["event_start"] / trials["Novice"].fs
            - windows["Trained"]["event_start"] / trials["Trained"].fs
        )
    if len(offsets) < 3:
        return []  # too few pairs for a typical offset to mean anything

    typical = float(np.median(list(offsets.values())))
    return [
        Issue(event_id, "both", "warning",
              f"novice and trained windows are {offset:+.1f} s apart, against a typical "
              f"{typical:+.1f} s for this set - check they are the same movement")
        for event_id, offset in offsets.items()
        if abs(offset - typical) > PAIR_OFFSET_TOLERANCE_S
    ]


def validate_session(session: dict, trials: dict[str, LoadedTrial]) -> list[Issue]:
    issues: list[Issue] = []
    for family in FAMILIES:
        enabled = [e for e in session.get(FAMILY_KEY[family], []) if e.get("enabled", True)]
        for event in enabled:
            for label, window in event.get("windows", {}).items():
                loaded = trials[label]
                issues += validate_window(
                    event["event_id"], label, window, loaded.fs, loaded.n_samples
                )
        issues += check_pairing(enabled, trials)
        if family == "smooth":
            issues += check_segment_continuity(enabled, trials)
    return issues
