"""The editor's session file: the curated event windows and how they were made.

A session holds every event of the three families -- trunk rotation, monopodal
stance and sequence segments -- with one window per selected recording, the
detector settings that produced them, the metric options in force, and which
recording files (by name and SHA-256) filled each role.  It lives in the active
analysis folder (:func:`analysis.recordings.analysis_dir`), so every selection
of recordings has its own.  Either role may be empty: with one recording
selected, every event simply has one window.  Sample indices
are authoritative; seconds are written alongside for readability and ignored on
load, so there is no rounding ambiguity anywhere.

This module seeds a session from the detectors, edits it (add, delete to a
trash, restore), migrates older files, and keeps named copies alongside the
working one.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from analysis import alignment, config, detection, detection_v1, recordings
from analysis.config import TRIALS
from analysis.detection import find_knee_flexion_windows
from analysis.detection_v1 import find_stabilization as find_stabilization_v1
from analysis.kinematics import LoadedTrial
from analysis.signals import idx_to_sec, lowpass_signal, sec_to_idx

SESSION_FILENAME = "event_editor_session.json"
SCHEMA_VERSION = 2
"""1: every family paired by order, stances as ``flexed_leg``.  2: pairs from
the whole-recording alignment, stances as ``lifted_leg``, sequence turns."""
KNEE_THRESHOLD_DEG = 60.0


# ---------------------------------------------------------------------------
# Event families
# ---------------------------------------------------------------------------


FAMILY_KEY = {"trunk": "trunk_events", "knee": "knee_events", "smooth": "smooth_events"}
FAMILY_PREFIX = {"trunk": "trunk", "knee": "knee", "smooth": "smooth"}
FAMILIES = tuple(FAMILY_KEY)
STABILIZED_FAMILIES = ("trunk", "knee")
"""Families whose events carry a stabilization window.  Sequence segments do
not: they tile the movement, and there is nothing to settle from."""


FAMILY_DESCRIPTION = {
    "trunk": lambda event: "trunk rotation",
    "knee": lambda event: f"{event.get('lifted_leg', event.get('flexed_leg', '?'))} leg up",
    "smooth": lambda event: "turn",
}


# ---------------------------------------------------------------------------
# Metric options recorded in new sessions
# ---------------------------------------------------------------------------


LAG_PAD_S = 2.0
"""Seconds added each side of an event for the coordination-lag window only."""


KNEE_LAG_WINDOW_S = (2.0, 3.0)
"""Seconds before/after the highest lift for the coordination-lag window of a
single-leg stance.  (It was centred on peak knee flexion, which for a kick is
the start of the movement, not its middle.)  5 s leaves the +/-1 s search of
:func:`analysis.signals.pearson_lag` at least 60 % overlap."""


# ---------------------------------------------------------------------------
# Reading and writing
# ---------------------------------------------------------------------------


def session_path() -> Path:
    """The working session of the active selection."""
    return recordings.analysis_dir() / SESSION_FILENAME


def roles(trials: dict[str, LoadedTrial]) -> list[str]:
    """The roles that have a recording loaded, in role order."""
    return [label for label in TRIALS if label in trials]


def load_session(path: Path | None = None) -> dict | None:
    path = path or session_path()
    if not path.exists():
        return None
    with path.open() as handle:
        return json.load(handle)


def save_session(session: dict, path: Path | None = None) -> None:
    """Write the session atomically so an interrupted save cannot corrupt it."""
    path = path or session_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    session = dict(session)
    session["modified_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    # Re-read the flags on every write rather than only when the session was
    # seeded.  They describe the code that produced the numbers, not the
    # curation, so a stored copy from an older run would misreport how the
    # current CSVs were computed.
    session["ignore_high_pass_filter"] = _ignore_high_pass_flag()
    session["metrics_version"] = config.METRICS_VERSION
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        json.dump(session, handle, indent=2)
    os.replace(tmp, path)


def _ignore_high_pass_flag() -> bool:
    return bool(config.IGNORE_HIGH_PASS_FILTER)


def refresh_seconds(session: dict, trials: dict[str, LoadedTrial]) -> dict:
    """Recompute the derived ``*_s`` display fields from the authoritative indices."""
    for family in FAMILIES:
        for event in session.get(FAMILY_KEY[family], []):
            for label, window in event.get("windows", {}).items():
                if label not in trials:
                    continue
                fs = trials[label].fs
                # Sequence segments carry no stabilization band, so only the
                # keys actually present are mirrored into seconds.
                for key in ("event_start", "event_end", "stab_start", "stab_end"):
                    if key in window:
                        window[f"{key}_s"] = round(idx_to_sec(window[key], fs), 4)
    return session


# ---------------------------------------------------------------------------
# Seeding from the detectors
# ---------------------------------------------------------------------------


def _window_dict(
    start: int, end: int, stab_start: int, stab_end: int, fs: float, **extra
) -> dict:
    return {
        "event_start": int(start),
        "event_end": int(end),
        "stab_start": int(stab_start),
        "stab_end": int(stab_end),
        "event_start_s": round(idx_to_sec(start, fs), 4),
        "event_end_s": round(idx_to_sec(end, fs), 4),
        "stab_start_s": round(idx_to_sec(stab_start, fs), 4),
        "stab_end_s": round(idx_to_sec(stab_end, fs), 4),
        "source": {"event": "auto", "stab": "auto"},
        **extra,
    }


def _movement_window_dict(start: int, end: int, fs: float, **extra) -> dict:
    """A window with no stabilization band.

    Sequence segments have nothing to settle from — they tile the movement
    itself — so they carry only the movement boundaries.  Everything that reads
    a window must therefore use ``.get`` for the stabilization keys.
    """
    return {
        "event_start": int(start),
        "event_end": int(end),
        "event_start_s": round(idx_to_sec(start, fs), 4),
        "event_end_s": round(idx_to_sec(end, fs), 4),
        "source": {"event": "auto"},
        **extra,
    }


def seed_session(
    trials: dict[str, LoadedTrial],
    trunk_params: detection.TrunkDetectorParams | None = None,
    stab_params: detection.StabilizationParams | None = None,
    knee_params: detection.KneeDetectorParams | None = None,
    segment_params: detection.SegmentDetectorParams | None = None,
) -> dict:
    """Build a fresh session by running the detectors on the selected recordings."""
    trunk_params = trunk_params or detection.TrunkDetectorParams()
    stab_params = stab_params or detection.StabilizationParams()
    knee_params = knee_params or detection.KneeDetectorParams()
    segment_params = segment_params or detection.SegmentDetectorParams()

    return {
        **_header(trials),
        "detector": detection.params_to_dict(trunk_params, stab_params, knee_params, segment_params),
        "lag_pad_s": LAG_PAD_S,
        "knee_lag_window_s": list(KNEE_LAG_WINDOW_S),
        "trunk_events": seed_trunk_events(trials, trunk_params, stab_params),
        "knee_events": seed_knee_events(trials, stab_params, knee_params),
        "smooth_events": seed_smooth_events(trials, segment_params),
    }


def _header(trials: dict[str, LoadedTrial]) -> dict:
    """What every new session starts with: when, from which recordings, at which rates."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": now,
        "modified_utc": now,
        "recordings": {label: trials[label].recording.identity() for label in roles(trials)},
        "fs": {label: trials[label].fs for label in roles(trials)},
        "n_samples": {label: trials[label].n_samples for label in roles(trials)},
        "ignore_high_pass_filter": _ignore_high_pass_flag(),
        "metrics_version": config.METRICS_VERSION,
    }


def _quietness(window: tuple) -> dict:
    """The stored confidence of an automatically placed stabilization window."""
    return {"stab_quiet_ratio": window[4], "stab_quiet_z": window[5]} if len(window) >= 6 else {}


def _trunk_events(trials: dict[str, LoadedTrial], windows: dict[str, list[tuple]]) -> list[dict]:
    """Numbered trunk events from per-role window lists of equal length."""
    labels = [label for label in roles(trials) if label in windows]
    count = min((len(windows[label]) for label in labels), default=0)
    events = []
    for i in range(count):
        per_role = {}
        for label in labels:
            window = windows[label][i]
            per_role[label] = _window_dict(*window[:4], fs=trials[label].fs, **_quietness(window))
        events.append({"event_id": f"trunk-{i + 1:02d}", "enabled": True, "windows": per_role})
    return events


def detect_trunk_windows(
    trials: dict[str, LoadedTrial],
    trunk_params: detection.TrunkDetectorParams,
    stab_params: detection.StabilizationParams,
) -> dict[str, list[tuple[int, int, int, int, float, float]]]:
    """Detect trunk events, paired through the alignment when both recordings are selected.

    Each entry is ``(event_start, event_end, stab_start, stab_end, quiet_ratio, quiet_z)``.
    """
    if len(roles(trials)) == 2:
        novice, trained = trials["Novice"], trials["Trained"]
        windows, _, _ = detection.pair_trunk_events(
            novice.kin, trained.kin, novice.fs, trained.fs, alignment.for_trials(trials),
            trunk_params, stab_params,
        )
        return windows

    (label,) = roles(trials)
    loaded = trials[label]
    events, _ = detection.detect_trunk_rotation_events(loaded.kin, loaded.fs, trunk_params)
    omega = detection.combined_omega(loaded.kin, loaded.fs, stab_params)
    windows = []
    for event in events:
        stab = detection.find_stabilization(loaded.kin, loaded.fs, event.end, stab_params, diagnostics=omega)
        windows.append((event.start, event.end, stab.start, stab.end,
                        round(stab.quiet_ratio, 3), round(stab.quiet_z, 3)))
    return {label: windows}


def seed_trunk_events(
    trials: dict[str, LoadedTrial],
    trunk_params: detection.TrunkDetectorParams | None = None,
    stab_params: detection.StabilizationParams | None = None,
) -> list[dict]:
    return _trunk_events(trials, detect_trunk_windows(
        trials, trunk_params or detection.TrunkDetectorParams(), stab_params or detection.StabilizationParams()
    ))


def detect_stance_windows(
    loaded: LoadedTrial, knee_params: detection.KneeDetectorParams,
    stab_params: detection.StabilizationParams | None = None,
    omega: detection.StabilizationDiagnostics | None = None,
    v1: bool = False,
) -> list[tuple[str, dict]]:
    """Single-leg stances of one recording as ``(lifted_leg, window)``, in time order.

    With the ``lift`` method the window is the support phase and settling is
    searched for from touch-down.  With ``knee_threshold`` -- the original rule
    -- it is where the knee is flexed past the threshold, and the search starts
    at peak flexion, with the v1 search when ``v1`` is set (as
    :func:`seed_session_v1` needs).
    """
    kin, fs = loaded.kin, loaded.fs
    found: list[tuple[str, int, int, int]] = []
    if knee_params.method == "knee_threshold":
        for side in ("Left", "Right"):
            angle = kin.left_knee_deg[:, 0] if side == "Left" else kin.right_knee_deg[:, 0]
            knee_abs = np.abs(lowpass_signal(angle, fs, cutoff_hz=6.0))
            windows, _ = find_knee_flexion_windows(
                angle, fs, threshold_deg=knee_params.threshold_deg,
                min_duration_s=knee_params.min_duration_s, merge_gap_s=knee_params.merge_gap_s,
            )
            found += [(side, start, end, start + int(np.argmax(knee_abs[start:end]))) for start, end in windows]
    else:
        found = [(side, start, end, end)
                 for start, end, side in detection.find_single_leg_phases(kin, fs, knee_params)]
    found.sort(key=lambda item: item[1])

    out = []
    for side, start, end, search_from in found:
        if v1:
            stab_start, stab_end = find_stabilization_v1(kin, fs, search_from)
            out.append((side, _window_dict(start, end, stab_start, stab_end, fs=fs)))
            continue
        stab = detection.find_stabilization(kin, fs, search_from, stab_params, diagnostics=omega)
        out.append((side, _window_dict(start, end, stab.start, stab.end, fs=fs,
                                       stab_quiet_ratio=round(stab.quiet_ratio, 3),
                                       stab_quiet_z=round(stab.quiet_z, 3))))
    return out


def _span_s(window: dict, fs: float) -> tuple[float, float]:
    return window["event_start"] / fs, window["event_end"] / fs


def seed_knee_events(
    trials: dict[str, LoadedTrial],
    stab_params: detection.StabilizationParams | None = None,
    knee_params: detection.KneeDetectorParams | None = None,
    v1: bool = False,
) -> list[dict]:
    """Detect single-leg stances and pair them across the two recordings.

    Pairing is done **within each leg**, through the whole-recording alignment:
    a novice and a trained stance of the same leg pair when each is the other's
    best match once carried onto the same clock.  A stance with no partner
    becomes a single-sided event: it keeps a window for the recording it was
    found in and contributes nothing to the other.
    """
    stab_params = stab_params or detection.StabilizationParams()
    knee_params = knee_params or detection.KneeDetectorParams()
    labels = roles(trials)
    omega = {label: None if v1 else detection.combined_omega(trials[label].kin, trials[label].fs, stab_params)
             for label in labels}
    found = {label: detect_stance_windows(trials[label], knee_params, stab_params, omega[label], v1=v1)
             for label in labels}
    aligned = alignment.for_trials(trials)

    events: list[dict] = []
    for side in ("Left", "Right"):
        per_trial = {label: [w for leg, w in found[label] if leg == side] for label in labels}
        matched: list[tuple[int, int]] = []
        if aligned is not None:
            matched = alignment.match_windows(
                aligned,
                [_span_s(w, trials["Novice"].fs) for w in per_trial["Novice"]],
                [_span_s(w, trials["Trained"].fs) for w in per_trial["Trained"]],
            )
        used = {label: set() for label in labels}
        groups: list[dict[str, dict]] = []
        for i, j in matched:
            groups.append({"Novice": per_trial["Novice"][i], "Trained": per_trial["Trained"][j]})
            used["Novice"].add(i)
            used["Trained"].add(j)
        for label in labels:
            groups += [{label: w} for k, w in enumerate(per_trial[label]) if k not in used[label]]
        for windows in groups:
            events.append({
                "event_id": "",
                "lifted_leg": side,
                "stance_leg": "Right" if side == "Left" else "Left",
                "enabled": True,
                "windows": windows,
            })

    _sort_and_number(events, "knee")
    return events


def seed_smooth_events(
    trials: dict[str, LoadedTrial],
    segment_params: detection.SegmentDetectorParams | None = None,
) -> list[dict]:
    """Cut both recordings into single-direction turns and pair the same turn.

    The turning points of each recording's chest yaw are matched through the
    alignment (:func:`analysis.alignment.match_landmarks`), and a turn is made
    between every two consecutive matched points, in both recordings at once.
    A turning point one recording has and the other lacks -- an extra wiggle --
    therefore stays inside a turn instead of shifting every later pair, which
    is what pairing full cycles by order did: equal counts, 13 and 13, hid
    pairs that were half a cycle apart from the first one on.
    """
    segment_params = segment_params or detection.SegmentDetectorParams()
    labels = roles(trials)
    points = {}
    chest_yaw = {}
    for label in labels:
        points[label], diagnostics = detection.yaw_turning_points(trials[label].kin, trials[label].fs, segment_params)
        chest_yaw[label] = diagnostics.chest_yaw_deg

    events: list[dict] = []
    aligned = alignment.for_trials(trials)
    if aligned is not None:
        fs_n, fs_t = trials["Novice"].fs, trials["Trained"].fs
        matched = alignment.match_landmarks(
            aligned,
            [(index / fs_n, direction) for index, direction in points["Novice"]],
            [(index / fs_t, direction) for index, direction in points["Trained"]],
        )
        for (a_n, a_t), (b_n, b_t) in zip(matched[:-1], matched[1:]):
            direction = points["Novice"][b_n][1]
            if points["Novice"][a_n][1] == direction:
                continue  # consecutive matches turning the same way: not one turn
            windows = {}
            for label, a, b in (("Novice", a_n, b_n), ("Trained", a_t, b_t)):
                turn = detection.turn_between(points[label][a], points[label][b], chest_yaw[label],
                                              trials[label].fs, segment_params)
                if turn is not None:
                    windows[label] = _movement_window_dict(turn.start, turn.end, fs=trials[label].fs)
            if windows:
                events.append({"event_id": "", "enabled": True, "direction": int(direction), "windows": windows})
    else:
        (label,) = labels
        for first, second in zip(points[label][:-1], points[label][1:]):
            turn = detection.turn_between(first, second, chest_yaw[label], trials[label].fs, segment_params)
            if turn is not None:
                events.append({"event_id": "", "enabled": True, "direction": turn.direction,
                               "windows": {label: _movement_window_dict(turn.start, turn.end,
                                                                        fs=trials[label].fs)}})

    _sort_and_number(events, "smooth")
    return events


def _earliest_start(event: dict) -> int:
    """Sort key: the earliest window start across whichever trials are present."""
    starts = [w["event_start"] for w in event["windows"].values()]
    return min(starts) if starts else 0


def _sort_and_number(events: list[dict], family: str) -> None:
    events.sort(key=_earliest_start)
    prefix = FAMILY_PREFIX[family]
    for i, event in enumerate(events, 1):
        event["event_id"] = f"{prefix}-{i:02d}"


# ---------------------------------------------------------------------------
# Adding, deleting and restoring events
#
# Deleting moves an event to `trash` rather than dropping it.  The session file
# is written through on every change and has no undo of its own, so a removed
# event would otherwise only be recoverable from git.
# ---------------------------------------------------------------------------


DEFAULT_TRUNK_EVENT_S = 3.0
DEFAULT_KNEE_EVENT_S = 2.0
DEFAULT_SMOOTH_EVENT_S = 10.0
DEFAULT_STAB_GAP_S = 0.1
"""Gap between the end of a new event and the start of its stabilization window."""


def events_of(session: dict, family: str) -> list[dict]:
    return session.setdefault(FAMILY_KEY[family], [])


def trash_of(session: dict) -> list[dict]:
    return session.setdefault("trash", [])


def next_event_id(session: dict, family: str) -> str:
    """Lowest unused id for a family, counting trashed events so ids stay unique."""
    prefix = FAMILY_PREFIX[family]
    used = {e["event_id"] for e in events_of(session, family)}
    used |= {t["event"]["event_id"] for t in trash_of(session) if t["family"] == family}
    numbers = {int(i.rsplit("-", 1)[1]) for i in used if i.rsplit("-", 1)[-1].isdigit()}
    n = 1
    while n in numbers:
        n += 1
    return f"{prefix}-{n:02d}"


def sort_events(session: dict, family: str, trials: dict[str, LoadedTrial]) -> None:
    """Keep each family in chronological order, as the seeders produce them."""
    events_of(session, family).sort(key=_earliest_start)


def _default_windows(centre_idx: int, length_s: float, fs: float, n_samples: int,
                     stab_len_s: float) -> tuple[int, int, int, int]:
    half = int(round(0.5 * length_s * fs))
    start = int(np.clip(centre_idx - half, 0, n_samples - 2))
    end = int(np.clip(centre_idx + half, start + 1, n_samples - 1))
    stab_start = int(np.clip(end + int(DEFAULT_STAB_GAP_S * fs), 0, n_samples - 2))
    stab_end = int(np.clip(stab_start + int(stab_len_s * fs), stab_start + 1, n_samples - 1))
    return start, end, stab_start, stab_end


def add_trunk_event(session: dict, trials: dict[str, LoadedTrial],
                    centre_s: dict[str, float], stab_len_s: float = 3.0) -> dict:
    """Create a trunk-rotation event with a window in each loaded recording.

    ``centre_s`` gives the time to centre on per trial, so the new event lands
    where the user is looking in each graph rather than at a shared clock time
    the two recordings do not share.
    """
    windows = {}
    for label in roles(trials):
        loaded = trials[label]
        centre = sec_to_idx(centre_s.get(label, loaded.duration_s / 2), loaded.fs, loaded.n_samples)
        bounds = _default_windows(centre, DEFAULT_TRUNK_EVENT_S, loaded.fs, loaded.n_samples, stab_len_s)
        windows[label] = _window_dict(*bounds, fs=loaded.fs)
        windows[label]["source"] = {"event": "manual", "stab": "manual"}

    event = {"event_id": next_event_id(session, "trunk"), "enabled": True, "windows": windows}
    events_of(session, "trunk").append(event)
    sort_events(session, "trunk", trials)
    return event


def add_knee_event(session: dict, trials: dict[str, LoadedTrial], trial: str, lifted_leg: str,
                   centre_s: dict[str, float], stab_len_s: float = 3.0) -> dict:
    """Create a single-leg-stance event with ``lifted_leg`` off the floor.

    ``trial`` is ``"Both"`` for a paired event, or one recording's name for a
    single-sided one -- used when a movement genuinely appears in only one
    recording.  ``centre_s`` gives the time to centre on per trial.
    """
    labels = roles(trials) if trial == "Both" else (trial,)
    windows = {}
    for label in labels:
        loaded = trials[label]
        centre = sec_to_idx(centre_s.get(label, loaded.duration_s / 2), loaded.fs, loaded.n_samples)
        bounds = _default_windows(centre, DEFAULT_KNEE_EVENT_S, loaded.fs, loaded.n_samples, stab_len_s)
        windows[label] = _window_dict(*bounds, fs=loaded.fs)
        windows[label]["source"] = {"event": "manual", "stab": "manual"}

    event = {
        "event_id": next_event_id(session, "knee"),
        "lifted_leg": lifted_leg,
        "stance_leg": "Right" if lifted_leg == "Left" else "Left",
        "enabled": True,
        "windows": windows,
    }
    events_of(session, "knee").append(event)
    sort_events(session, "knee", trials)
    return event


def add_smooth_event(session: dict, trials: dict[str, LoadedTrial], trial: str,
                     centre_s: dict[str, float]) -> dict:
    """Create a sequence segment, with no stabilization window.

    ``trial`` is ``"Both"`` for a paired part, or one recording's name where the
    detector found a part in only one of them.
    """
    labels = roles(trials) if trial == "Both" else (trial,)
    windows = {}
    for label in labels:
        loaded = trials[label]
        centre = sec_to_idx(centre_s.get(label, loaded.duration_s / 2), loaded.fs, loaded.n_samples)
        half = int(round(0.5 * DEFAULT_SMOOTH_EVENT_S * loaded.fs))
        start = int(np.clip(centre - half, 0, loaded.n_samples - 2))
        end = int(np.clip(centre + half, start + 1, loaded.n_samples - 1))
        windows[label] = _movement_window_dict(start, end, fs=loaded.fs)
        windows[label]["source"] = {"event": "manual"}

    event = {"event_id": next_event_id(session, "smooth"), "enabled": True, "windows": windows}
    events_of(session, "smooth").append(event)
    sort_events(session, "smooth", trials)
    return event


def delete_event(session: dict, family: str, event_id: str) -> dict | None:
    """Move an event to the trash.  Returns the event, or None if not found."""
    events = events_of(session, family)
    for i, event in enumerate(events):
        if event["event_id"] == event_id:
            removed = events.pop(i)
            trash_of(session).append({
                "family": family,
                "deleted_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "event": removed,
            })
            return removed
    return None


def restore_event(session: dict, trash_index: int, trials: dict[str, LoadedTrial]) -> dict | None:
    """Move a trashed event back into its family."""
    trash = trash_of(session)
    if not 0 <= trash_index < len(trash):
        return None
    entry = trash.pop(trash_index)
    family = entry["family"]
    event = entry["event"]
    # An id freed since deletion may have been reused, so re-issue if it clashes.
    if any(e["event_id"] == event["event_id"] for e in events_of(session, family)):
        event["event_id"] = next_event_id(session, family)
    events_of(session, family).append(event)
    sort_events(session, family, trials)
    return event


def describe_trashed(entry: dict, trials: dict[str, LoadedTrial]) -> str:
    """One-line label for the restore list."""
    event = entry["event"]
    windows = event.get("windows", {})
    label = next((label for label in TRIALS if label in windows and label in trials), None)
    if label is None:
        return event.get("event_id", "?")
    window, fs = windows[label], trials[label].fs
    kind = FAMILY_DESCRIPTION.get(entry["family"], lambda e: "?")(event)
    span = f"{window['event_start']/fs:.1f}-{window['event_end']/fs:.1f}s"
    sides = "+".join(sorted(windows))
    return f"{event['event_id']} - {kind}  {label[:1]} {span}  [{sides}]"


# ---------------------------------------------------------------------------
# Migrating older session files
# ---------------------------------------------------------------------------


def _is_legacy_knee_event(event: dict) -> bool:
    """Pre-pairing shape: one trial per event, boundaries at the top level."""
    return "windows" not in event and "trial" in event


def migrate_knee_events(events: list[dict]) -> tuple[list[dict], int]:
    """Convert unpaired knee events into the paired shape, preserving edits.

    Legacy events carried a single ``trial`` and their boundaries at the top
    level.  They are grouped by leg, ordered in time within each recording, and
    zipped together — the same rule the seeder uses — so hand-adjusted
    boundaries and their provenance survive the change.  A recording with more
    events of a leg than the other yields single-sided pairs.
    """
    legacy = [e for e in events if _is_legacy_knee_event(e)]
    if not legacy:
        return events, 0

    already_paired = [e for e in events if not _is_legacy_knee_event(e)]
    migrated: list[dict] = []
    for side in ("Left", "Right"):
        per_trial = {
            label: sorted(
                (e for e in legacy if e["flexed_leg"] == side and e["trial"] == label),
                key=lambda e: e["event_start"],
            )
            for label in TRIALS
        }
        for i in range(max(len(per_trial[label]) for label in TRIALS)):
            windows = {}
            for label in TRIALS:
                if i < len(per_trial[label]):
                    source = dict(per_trial[label][i])
                    for key in ("event_id", "trial", "flexed_leg", "stance_leg", "enabled"):
                        source.pop(key, None)
                    windows[label] = source
            if not windows:
                continue
            first = next(e for label in TRIALS for e in per_trial[label][i:i + 1])
            migrated.append({
                "event_id": "",
                "flexed_leg": side,
                "stance_leg": first.get("stance_leg", "Right" if side == "Left" else "Left"),
                "enabled": any(
                    per_trial[label][i].get("enabled", True)
                    for label in TRIALS if i < len(per_trial[label])
                ),
                "windows": windows,
            })

    combined = already_paired + migrated
    _sort_and_number(combined, "knee")
    return combined, len(legacy)


def migrate_session(
    session: dict, trials: dict[str, LoadedTrial] | None = None
) -> tuple[dict, list[str]]:
    """Bring a session file up to the current schema.  Returns it plus a log.

    ``trials`` is needed only to seed families added since the file was written;
    without it such a family is left empty and can be filled by re-detecting.
    """
    notes: list[str] = []

    if "recordings" not in session and trials is not None:
        session["recordings"] = {label: trials[label].recording.identity() for label in roles(trials)}
        notes.append("recorded which recording files the session belongs to")

    if "smooth_events" not in session and trials is not None:
        params = detection.params_from_dict(session.get("detector", {}) or {})[3]
        session["smooth_events"] = seed_smooth_events(trials, params)
        session.setdefault("detector", {}).update(
            {f"smooth_{k}": v for k, v in asdict(params).items()}
        )
        notes.append(f"detected {len(session['smooth_events'])} sequence segments")

    knee, converted = migrate_knee_events(session.get("knee_events", []))
    if converted:
        session["knee_events"] = knee
        notes.append(f"paired {converted} unpaired monopodal-stance events into {len(knee)} events")

    # Trashed events must migrate too, or restoring one would reintroduce the
    # old shape.  A trashed event has no partner to pair with, so it becomes
    # single-sided and can be re-paired by hand after restoring.
    trashed = 0
    for entry in trash_of(session):
        event = entry.get("event", {})
        if entry.get("family") == "knee" and _is_legacy_knee_event(event):
            label = event["trial"]
            window = {k: v for k, v in event.items()
                      if k not in ("event_id", "trial", "flexed_leg", "stance_leg", "enabled")}
            entry["event"] = {
                "event_id": event["event_id"],
                "flexed_leg": event["flexed_leg"],
                "stance_leg": event.get("stance_leg", "Right" if event["flexed_leg"] == "Left" else "Left"),
                "enabled": event.get("enabled", True),
                "windows": {label: window},
            }
            trashed += 1
    if trashed:
        notes.append(f"converted {trashed} deleted monopodal-stance events")

    if session.get("schema_version", 1) < 2 and trials is not None:
        notes += _migrate_to_schema_2(session, trials)

    if notes:
        session["schema_version"] = SCHEMA_VERSION
    return session, notes


STAB_KEYS_CHANGED = ("stab_latency_weight", "stab_low_confidence_ratio")
"""Detector keys whose meaning changed with the bias-corrected quietness score:
an old value would be read in the wrong units, so it is dropped for the default."""


def _all_auto(events: list[dict]) -> bool:
    return all(source == "auto"
               for event in events for window in event.get("windows", {}).values()
               for source in (window.get("source") or {}).values())


def _migrate_to_schema_2(session: dict, trials: dict[str, LoadedTrial]) -> list[str]:
    """Schema 1 -> 2: aligned pairing, stances by the lifted leg, sequence turns.

    The session as it was is parked first, in ``sessions/pre-schema-2-<name>.json``.
    Families whose windows were all placed automatically -- nothing curated to
    lose -- are detected afresh with the aligned pairing; a family with any
    hand-placed window is kept and left to validation, which flags a pair that
    is not the same movement.  A stance whose lift index says the other leg was
    up in every recording it has a window in is relabelled, and logged.
    """
    notes: list[str] = []
    stem = sanitise_session_name(session.get("session_name") or "working") or "working"
    # Written as it was, not through save_session, which would stamp the copy
    # with the current metrics version and modification time.
    parked = sessions_dir() / f"pre-schema-2-{stem}.json"
    parked.parent.mkdir(parents=True, exist_ok=True)
    parked.write_text(json.dumps(session, indent=2))
    notes.append(f"parked the schema-1 session as sessions/pre-schema-2-{stem}.json")

    for event in session.get("knee_events", []) + [e["event"] for e in trash_of(session) if e.get("family") == "knee"]:
        if "flexed_leg" in event:
            event["lifted_leg"] = event.pop("flexed_leg")

    detector = session.setdefault("detector", {})
    for key in STAB_KEYS_CHANGED + ("smooth_half_cycles",):
        detector.pop(key, None)
    detector.setdefault("knee_method", "knee_threshold" if str(detector.get("name", "")).startswith("v1") else "lift")
    trunk_params, stab_params, knee_params, segment_params = detection.params_from_dict(detector)

    lift = {label: detection.lift_signal(loaded.kin, loaded.fs).lift_index for label, loaded in trials.items()}
    for event in session.get("knee_events", []):
        votes = []
        for label, window in event.get("windows", {}).items():
            if label in lift:
                mean = float(np.mean(lift[label][window["event_start"]:window["event_end"]]))
                votes.append("Left" if mean > 0 else "Right")
        if votes and len(set(votes)) == 1 and votes[0] != event.get("lifted_leg"):
            notes.append(f"{event['event_id']}: relabelled {event.get('lifted_leg')} -> {votes[0]} leg lifted "
                         f"(the {votes[0].lower()} foot is the higher one in every window)")
            event["lifted_leg"] = votes[0]
            event["stance_leg"] = "Right" if votes[0] == "Left" else "Left"

    if detector.get("name") != "v1":
        if _all_auto(session.get("trunk_events", [])):
            session["trunk_events"] = seed_trunk_events(trials, trunk_params, stab_params)
            notes.append(f"re-detected {len(session['trunk_events'])} trunk rotations, paired through the alignment")
        if _all_auto(session.get("smooth_events", [])):
            session["smooth_events"] = seed_smooth_events(trials, segment_params)
            notes.append(f"re-cut the sequence into {len(session['smooth_events'])} single-direction turns")
    detector.update(detection.params_to_dict(trunk_params, stab_params, knee_params, segment_params))

    omega = {label: detection.combined_omega(loaded.kin, loaded.fs, stab_params) for label, loaded in trials.items()}
    for family in STABILIZED_FAMILIES:
        for event in session.get(FAMILY_KEY[family], []):
            for label, window in event.get("windows", {}).items():
                if label in omega and "stab_start" in window:
                    ratio, z = detection.window_quietness(omega[label], window["stab_start"], window["stab_end"])
                    window["stab_quiet_ratio"], window["stab_quiet_z"] = round(ratio, 3), round(z, 3)
    notes.append("recomputed the quietness of every stabilization window on bias-corrected angular speed")
    return notes


# ---------------------------------------------------------------------------
# Named sessions
#
# The working session is saved continuously to session_path(), so nothing is ever
# lost to a crash.  Named sessions are copies of it, kept alongside, so several
# curations of the same recordings -- a conservative one and a permissive one,
# say -- can be held and compared rather than overwriting each other.
# ---------------------------------------------------------------------------


def sessions_dir() -> Path:
    """Named copies of the active selection's working session."""
    return recordings.analysis_dir() / "sessions"


PREVIOUS_SLOT = "_previous"
"""Where the working session is parked before a load, as a one-step undo."""


def sanitise_session_name(name: str) -> str:
    """Reduce a typed name to a safe file stem.

    The name reaches this from a text box, so anything that could escape the
    sessions directory or collide with the filesystem is stripped rather than
    escaped -- there is no reason to support such names.
    """
    cleaned = "".join(c if (c.isalnum() or c in "-_ ") else "-" for c in (name or "").strip())
    cleaned = "-".join(cleaned.split())
    return cleaned[:64].strip("-")


def session_path_for(name: str) -> Path:
    stem = sanitise_session_name(name)
    if not stem:
        raise ValueError("A session name must contain at least one letter or digit.")
    return sessions_dir() / f"{stem}.json"


def list_sessions() -> list[str]:
    """Saved session names, newest first, with the undo slot last."""
    if not sessions_dir().exists():
        return []
    paths = sorted(sessions_dir().glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    names = [p.stem for p in paths]
    ordinary = [n for n in names if not n.startswith("_")]
    special = [n for n in names if n.startswith("_")]
    return ordinary + special


def save_session_as(session: dict, name: str) -> Path:
    """Write the session to a named file and record the name on it."""
    path = session_path_for(name)
    session = dict(session)
    session["session_name"] = path.stem
    save_session(session, path)
    # Keep the working copy in step so the name shown in the interface persists.
    save_session(session)
    return path


def load_session_named(name: str, trials: dict[str, LoadedTrial] | None = None) -> dict:
    """Load a named session and make it the working one.

    The current working session is parked in the undo slot first, so loading
    can never silently discard unsaved curation.
    """
    path = session_path_for(name)
    if not path.exists():
        raise FileNotFoundError(f"No saved session called {path.stem!r}.")

    # Read the target before parking: loading the undo slot itself would
    # otherwise overwrite it with the current session and read back what it
    # had just destroyed.
    with path.open() as handle:
        loaded = json.load(handle)

    park_working_session()
    loaded["session_name"] = path.stem
    loaded, _ = migrate_session(loaded, trials)
    if trials is not None:
        loaded = refresh_seconds(loaded, trials)
    save_session(loaded)
    return loaded


def park_working_session() -> bool:
    """Copy the working session into the undo slot.  Returns whether it did.

    Called before anything that replaces the whole session.  It parks
    unconditionally, including when reloading the session already open: that is
    how unsaved edits get discarded, and it is exactly the case where losing
    them silently would hurt most.
    """
    current = load_session()
    if current is None:
        return False
    parked = dict(current)
    parked["session_name"] = PREVIOUS_SLOT
    save_session(parked, sessions_dir() / f"{PREVIOUS_SLOT}.json")
    return True


NEW_SESSION_SOURCES = {
    "v2": "v2 detectors (boundaries from the motion, pairs from the alignment)",
    "v1": "v1 detectors (fixed 8 s windows, DTW pairing)",
}
"""Where :func:`start_new_session` can take its windows from."""


def start_new_session(
    trials: dict[str, LoadedTrial],
    trunk_params: detection.TrunkDetectorParams | None = None,
    stab_params: detection.StabilizationParams | None = None,
    knee_params: detection.KneeDetectorParams | None = None,
    segment_params: detection.SegmentDetectorParams | None = None,
    source: str = "v2",
) -> dict:
    """Discard the working session and detect all families afresh.

    The current session is parked first, so a fresh start is one ``Load`` away
    from being undone.  Unlike ``Re-detect``, which replaces one family and
    leaves the others alone, this resets everything: every family, the deleted
    list and the session name.

    ``source`` is one of :data:`NEW_SESSION_SOURCES`; the parameter sets apply
    to the v2 detectors only.
    """
    if source not in NEW_SESSION_SOURCES:
        raise ValueError(f"Unknown session source {source!r}; expected one of {sorted(NEW_SESSION_SOURCES)}.")
    park_working_session()
    if source == "v1":
        session = seed_session_v1(trials)
    else:
        session = seed_session(trials, trunk_params, stab_params, knee_params, segment_params)
    session["trash"] = []
    session.pop("session_name", None)
    session = refresh_seconds(session, trials)
    save_session(session)
    return session


def delete_named_session(name: str) -> None:
    path = session_path_for(name)
    if path.exists():
        path.unlink()


# ---------------------------------------------------------------------------
# Seeding with the v1 methods, to reproduce earlier results
# ---------------------------------------------------------------------------


def seed_session_v1(trials: dict[str, LoadedTrial]) -> dict:
    """Build a session from the v1 detectors, with the original metric definitions.

    Trunk events are the fixed 8 s windows, matched in the trained recording by
    DTW (:func:`analysis.detection_v1.pair_trunk_events_v1`).  Single-leg
    stances use the original knee flexion > 60 deg windows, each followed by the
    v1 stabilization search from peak flexion.  ``lag_pad_s`` and
    ``knee_lag_window_s`` are left unset, as they were.  The v1 detectors run on
    the raw gyroscope magnitude they were designed on, so the windows are the
    original ones; the metrics computed on them are the current definitions
    (``config.METRICS_VERSION``) -- the original numbers are reproduced by the
    git tag ``metrics-v1``.  Sequence turns had no v1 method and are detected as
    in v2.

    With one recording selected there is nothing to match, so its fixed 8 s
    windows are used as they are.
    """
    if len(roles(trials)) == 2:
        novice, trained = trials["Novice"], trials["Trained"]
        windows, _ = detection_v1.pair_trunk_events_v1(novice.kin, trained.kin, novice.fs, trained.fs)
    else:
        (label,) = roles(trials)
        loaded = trials[label]
        events, _ = detection_v1.detect_novice_trunk_rotation_events(loaded.kin, loaded.fs, 6)
        windows = {label: [
            (start, end, *detection_v1.find_stabilization(loaded.kin, loaded.fs, end), peak)
            for start, end, peak in events
        ]}
    trunk_events = _trunk_events(trials, {label: [w[:4] for w in rows] for label, rows in windows.items()})

    knee_params = detection.KneeDetectorParams(method="knee_threshold")
    segment_params = detection.SegmentDetectorParams()
    detector = {"name": "v1"}
    detector.update({f"knee_{k}": v for k, v in asdict(knee_params).items()})
    detector.update({f"smooth_{k}": v for k, v in asdict(segment_params).items()})

    return {
        **_header(trials),
        "detector": detector,
        "lag_pad_s": None,
        "knee_lag_window_s": None,
        "trunk_events": trunk_events,
        "knee_events": seed_knee_events(trials, knee_params=knee_params, v1=True),
        "smooth_events": seed_smooth_events(trials, segment_params),
    }
