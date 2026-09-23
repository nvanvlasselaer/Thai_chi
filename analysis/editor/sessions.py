"""The editor's session file: the curated event windows and how they were made.

A session holds every event of the three families -- trunk rotation, monopodal
stance and sequence segments -- with one window per recording, the detector
settings that produced them, and the metric options in force.  Sample indices
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
import pandas as pd

from analysis import config, detection
from analysis.config import TRIALS
from analysis.detection import find_knee_flexion_windows
from analysis.detection_v1 import find_stabilization as find_stabilization_v1
from analysis.kinematics import LoadedTrial
from analysis.signals import idx_to_sec, lowpass_signal, sec_to_idx

SESSION_PATH = config.OUTPUT_DIR / "event_editor_session.json"
SCHEMA_VERSION = 1
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
    "knee": lambda event: f"{event.get('flexed_leg', '?')} knee",
    "smooth": lambda event: "sequence part",
}


# ---------------------------------------------------------------------------
# Metric options recorded in new sessions
# ---------------------------------------------------------------------------


LAG_PAD_S = 2.0
"""Seconds added each side of an event for the coordination-lag window only."""


KNEE_LAG_WINDOW_S = (2.0, 3.0)
"""Seconds before/after peak knee flexion for the coordination-lag window.

The pipeline default is (1.0, 2.0), a 3 s window, but ``cross_correlation_lag``
searches +/-2 s, so at the extreme lag only 1 s of the two signals overlaps and
the overlap-normalised correlation is dominated by that short tail: 9 of 11
events rail at the bound.  Widening to 5 s keeps 60 % overlap at the extreme and
drops the mean |lag| from 1.72 s to 0.19 s.  Going wider still starts absorbing
neighbouring movements and the lag drifts up again, so 5 s is the useful point.
"""


# ---------------------------------------------------------------------------
# Reading and writing
# ---------------------------------------------------------------------------


def load_session(path: Path = SESSION_PATH) -> dict | None:
    if not path.exists():
        return None
    with path.open() as handle:
        return json.load(handle)


def save_session(session: dict, path: Path = SESSION_PATH) -> None:
    """Write the session atomically so an interrupted save cannot corrupt it."""
    session = dict(session)
    session["modified_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    # Re-read the flag on every write rather than only when the session was
    # seeded.  It describes the code that produced the numbers, not the
    # curation, so a stored copy from an older run would misreport how the
    # current CSVs were computed.
    session["ignore_high_pass_filter"] = _ignore_high_pass_flag()
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
    """Build a fresh session by running the detectors on both recordings."""
    trunk_params = trunk_params or detection.TrunkDetectorParams()
    stab_params = stab_params or detection.StabilizationParams()
    knee_params = knee_params or detection.KneeDetectorParams()
    segment_params = segment_params or detection.SegmentDetectorParams()

    novice, trained = trials["Novice"], trials["Trained"]
    windows, _ = detect_trunk_windows(trials, trunk_params, stab_params)

    trunk_events = []
    for i, (novice_window, trained_window) in enumerate(zip(windows["Novice"], windows["Trained"]), 1):
        trunk_events.append(
            {
                "event_id": f"trunk-{i:02d}",
                "enabled": True,
                "windows": {
                    "Novice": _window_dict(*novice_window[:4], fs=novice.fs, stab_quiet_ratio=novice_window[4]),
                    "Trained": _window_dict(*trained_window[:4], fs=trained.fs, stab_quiet_ratio=trained_window[4]),
                },
            }
        )

    knee_events = seed_knee_events(trials, stab_params, knee_params)
    smooth_events = seed_smooth_events(trials, segment_params)

    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "modified_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fs": {label: trial.fs for label, trial in trials.items()},
        "n_samples": {label: trial.n_samples for label, trial in trials.items()},
        "ignore_high_pass_filter": _ignore_high_pass_flag(),
        "detector": detection.params_to_dict(trunk_params, stab_params, knee_params, segment_params),
        "lag_pad_s": LAG_PAD_S,
        "knee_lag_window_s": list(KNEE_LAG_WINDOW_S),
        "trunk_events": trunk_events,
        "knee_events": knee_events,
        "smooth_events": smooth_events,
    }


def detect_trunk_windows(
    trials: dict[str, LoadedTrial],
    trunk_params: detection.TrunkDetectorParams,
    stab_params: detection.StabilizationParams,
) -> tuple[dict[str, list[tuple[int, int, int, int, float]]], list[float]]:
    """Detect trunk events in both recordings and pair them by order.

    Each entry is ``(event_start, event_end, stab_start, stab_end, quiet_ratio)``.
    """
    novice, trained = trials["Novice"], trials["Trained"]
    windows, _, _ = detection.pair_trunk_events(
        novice.kin, trained.kin, novice.fs, trained.fs, trunk_params, stab_params
    )
    return windows, []


def detect_knee_windows(
    loaded: LoadedTrial, side: str, knee_params: detection.KneeDetectorParams,
    stab_params: detection.StabilizationParams, omega: detection.StabilizationDiagnostics,
) -> list[dict]:
    """Knee-flexion events for one leg of one recording, in time order."""
    angle = loaded.kin.left_knee_deg[:, 0] if side == "Left" else loaded.kin.right_knee_deg[:, 0]
    knee_abs = np.abs(lowpass_signal(angle, loaded.fs, cutoff_hz=6.0))
    windows, _ = find_knee_flexion_windows(
        angle, loaded.fs,
        threshold_deg=knee_params.threshold_deg,
        min_duration_s=knee_params.min_duration_s,
        merge_gap_s=knee_params.merge_gap_s,
    )
    out = []
    for start, end in windows:
        peak_idx = start + int(np.argmax(knee_abs[start:end]))
        stab = detection.find_stabilization(loaded.kin, loaded.fs, peak_idx, stab_params, diagnostics=omega)
        out.append(_window_dict(start, end, stab.start, stab.end, fs=loaded.fs,
                                stab_quiet_ratio=round(stab.quiet_ratio, 3)))
    return out


def seed_knee_events(
    trials: dict[str, LoadedTrial],
    stab_params: detection.StabilizationParams | None = None,
    knee_params: detection.KneeDetectorParams | None = None,
) -> list[dict]:
    """Detect monopodal-stance events and pair them across the two recordings.

    Pairing is done **within each leg**, by time order: the k-th left-knee event
    in one recording corresponds to the k-th in the other, because both
    participants perform the same form.  Pairing within a leg rather than across
    all events guarantees that a pair always compares like with like, which
    matters because ``stance_leg`` drives the asymmetry metrics.

    Where one recording has more events of a leg than the other, the surplus
    becomes a single-sided event: it keeps a window for the recording it was
    found in and contributes nothing to the other.
    """
    stab_params = stab_params or detection.StabilizationParams()
    knee_params = knee_params or detection.KneeDetectorParams()
    omega = {label: detection.combined_omega(trials[label].kin, trials[label].fs, stab_params)
             for label in TRIALS}

    events: list[dict] = []
    for side in ("Left", "Right"):
        per_trial = {
            label: detect_knee_windows(trials[label], side, knee_params, stab_params, omega[label])
            for label in TRIALS
        }
        for i in range(max(len(per_trial[label]) for label in TRIALS)):
            windows = {label: per_trial[label][i] for label in TRIALS if i < len(per_trial[label])}
            events.append({
                "event_id": "",
                "flexed_leg": side,
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
    """Cut both recordings into parts of the sequence and pair them by order.

    Unlike the other two families these segments are not picked out of the
    recording — they tile the moving passages of it — so pairing by order is
    less fragile here than elsewhere: the detector finds the same number of
    parts in both recordings when both perform the same form, and a mismatch is
    itself worth seeing.  A surplus part becomes a single-sided event, as for
    monopodal stances.
    """
    segment_params = segment_params or detection.SegmentDetectorParams()

    per_trial = {
        label: detection.detect_yaw_cycle_segments(trials[label].kin, trials[label].fs, segment_params)[0]
        for label in TRIALS
    }

    events: list[dict] = []
    for i in range(max(len(per_trial[label]) for label in TRIALS)):
        windows = {}
        for label in TRIALS:
            if i >= len(per_trial[label]):
                continue
            start, end = per_trial[label][i]
            windows[label] = _movement_window_dict(start, end, fs=trials[label].fs)
        events.append({"event_id": "", "enabled": True, "windows": windows})

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
    """Create a trunk-rotation event with a window in each recording.

    ``centre_s`` gives the time to centre on per trial, so the new event lands
    where the user is looking in each graph rather than at a shared clock time
    the two recordings do not share.
    """
    windows = {}
    for label in TRIALS:
        loaded = trials[label]
        centre = sec_to_idx(centre_s.get(label, loaded.duration_s / 2), loaded.fs, loaded.n_samples)
        bounds = _default_windows(centre, DEFAULT_TRUNK_EVENT_S, loaded.fs, loaded.n_samples, stab_len_s)
        windows[label] = _window_dict(*bounds, fs=loaded.fs)
        windows[label]["source"] = {"event": "manual", "stab": "manual"}

    event = {"event_id": next_event_id(session, "trunk"), "enabled": True, "windows": windows}
    events_of(session, "trunk").append(event)
    sort_events(session, "trunk", trials)
    return event


def add_knee_event(session: dict, trials: dict[str, LoadedTrial], trial: str, flexed_leg: str,
                   centre_s: dict[str, float], stab_len_s: float = 3.0) -> dict:
    """Create a monopodal-stance event.

    ``trial`` is ``"Both"`` for a paired event, or one recording's name for a
    single-sided one -- used when a movement genuinely appears in only one
    recording.  ``centre_s`` gives the time to centre on per trial.
    """
    labels = TRIALS if trial == "Both" else (trial,)
    windows = {}
    for label in labels:
        loaded = trials[label]
        centre = sec_to_idx(centre_s.get(label, loaded.duration_s / 2), loaded.fs, loaded.n_samples)
        bounds = _default_windows(centre, DEFAULT_KNEE_EVENT_S, loaded.fs, loaded.n_samples, stab_len_s)
        windows[label] = _window_dict(*bounds, fs=loaded.fs)
        windows[label]["source"] = {"event": "manual", "stab": "manual"}

    event = {
        "event_id": next_event_id(session, "knee"),
        "flexed_leg": flexed_leg,
        "stance_leg": "Right" if flexed_leg == "Left" else "Left",
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
    labels = TRIALS if trial == "Both" else (trial,)
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
    label = "Novice" if "Novice" in windows else next(iter(windows), None)
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

    if notes:
        session["schema_version"] = SCHEMA_VERSION
    return session, notes


# ---------------------------------------------------------------------------
# Named sessions
#
# The working session is saved continuously to SESSION_PATH, so nothing is ever
# lost to a crash.  Named sessions are copies of it, kept alongside, so several
# curations of the same recordings -- a conservative one and a permissive one,
# say -- can be held and compared rather than overwriting each other.
# ---------------------------------------------------------------------------


SESSIONS_DIR = config.OUTPUT_DIR / "sessions"
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
    return SESSIONS_DIR / f"{stem}.json"


def list_sessions() -> list[str]:
    """Saved session names, newest first, with the undo slot last."""
    if not SESSIONS_DIR.exists():
        return []
    paths = sorted(SESSIONS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    names = [p.stem for p in paths]
    ordinary = [n for n in names if not n.startswith("_")]
    special = [n for n in names if n.startswith("_")]
    return ordinary + special


def save_session_as(session: dict, name: str) -> Path:
    """Write the session to a named file and record the name on it."""
    path = session_path_for(name)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    session = dict(session)
    session["session_name"] = path.stem
    save_session(session, path)
    # Keep the working copy in step so the name shown in the interface persists.
    save_session(session, SESSION_PATH)
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
    save_session(loaded, SESSION_PATH)
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
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    parked = dict(current)
    parked["session_name"] = PREVIOUS_SLOT
    save_session(parked, SESSIONS_DIR / f"{PREVIOUS_SLOT}.json")
    return True


def start_new_session(
    trials: dict[str, LoadedTrial],
    trunk_params: detection.TrunkDetectorParams | None = None,
    stab_params: detection.StabilizationParams | None = None,
    knee_params: detection.KneeDetectorParams | None = None,
    segment_params: detection.SegmentDetectorParams | None = None,
) -> dict:
    """Discard the working session and detect all families afresh.

    The current session is parked first, so a fresh start is one ``Load`` away
    from being undone.  Unlike ``Re-detect``, which replaces one family and
    leaves the others alone, this resets everything: every family, the deleted
    list and the session name.
    """
    park_working_session()
    session = seed_session(trials, trunk_params, stab_params, knee_params, segment_params)
    session["trash"] = []
    session.pop("session_name", None)
    session = refresh_seconds(session, trials)
    save_session(session, SESSION_PATH)
    return session


def delete_named_session(name: str) -> None:
    path = session_path_for(name)
    if path.exists():
        path.unlink()


# ---------------------------------------------------------------------------
# Seeding from the committed v1 outputs, for comparison
# ---------------------------------------------------------------------------


def seed_session_from_v1_csvs(trials: dict[str, LoadedTrial]) -> dict:
    """Rebuild a session from whatever is currently in the output window CSVs.

    Recalculating overwrites those CSVs, so after the first recalculation this
    reads back the most recent result rather than the original automatic one.
    The untouched automatic baseline lives in git:
    ``git show HEAD:outputs/trunk_rotation_event_windows.csv``.

    ``lag_pad_s`` and ``knee_lag_window_s`` are left unset here, reproducing the
    original metric definitions.  A session seeded this way will therefore report
    different coordination lags from one seeded by the detectors, which set both.
    """
    trunk_csv = config.OUTPUT_DIR / "trunk_rotation_event_windows.csv"
    knee_csv = config.OUTPUT_DIR / "monopodal_stance_event_windows.csv"
    if not trunk_csv.exists() or not knee_csv.exists():
        raise FileNotFoundError("Committed v1 window CSVs not found in outputs/")

    trunk_table = pd.read_csv(trunk_csv)
    trunk_events = []
    for index in sorted(trunk_table.event_index.unique()):
        windows = {}
        for label in TRIALS:
            rows = trunk_table[(trunk_table.trial == label) & (trunk_table.event_index == index)]
            if len(rows) == 0:
                break
            row = rows.iloc[0]
            fs, n = trials[label].fs, trials[label].n_samples
            windows[label] = _window_dict(
                sec_to_idx(row.event_start_s, fs, n),
                sec_to_idx(row.event_end_s, fs, n),
                sec_to_idx(row.stabilization_start_s, fs, n),
                sec_to_idx(row.stabilization_end_s, fs, n),
                fs=fs,
            )
        if len(windows) == len(TRIALS):
            trunk_events.append(
                {"event_id": f"trunk-{int(index):02d}", "enabled": True, "windows": windows}
            )

    knee_table = pd.read_csv(knee_csv)
    knee_events = []
    for i, row in enumerate(knee_table.itertuples(), 1):
        loaded = trials[row.trial]
        fs, n = loaded.fs, loaded.n_samples
        # Newer CSVs carry the curated stabilization window; older ones do not,
        # because the pipeline derived it inside compute_knee_balance_metrics
        # from the peak.  Prefer the recorded window so a reseed does not
        # silently discard hand-placed stabilization boundaries.
        recorded_start = getattr(row, "stabilization_start_s", None)
        recorded_end = getattr(row, "stabilization_end_s", None)
        if recorded_start is not None and recorded_end is not None and np.isfinite(recorded_start):
            stab_start = sec_to_idx(recorded_start, fs, n)
            stab_end = sec_to_idx(recorded_end, fs, n)
        else:
            peak_idx = sec_to_idx(row.peak_time_s, fs, n)
            stab_start, stab_end = find_stabilization_v1(loaded.kin, fs, peak_idx)
        knee_events.append(
            {
                "event_id": f"knee-{i:02d}",
                "trial": row.trial,
                "flexed_leg": row.flexed_leg,
                "stance_leg": row.stance_leg,
                "enabled": True,
                **_window_dict(
                    sec_to_idx(row.window_start_s, fs, n),
                    sec_to_idx(row.window_end_s, fs, n),
                    stab_start,
                    stab_end,
                    fs=fs,
                ),
            }
        )

    # The v1 CSV is one row per trial, so build the flat events above and let the
    # migration pair them, rather than repeating the pairing rule here.
    knee_events, _ = migrate_knee_events(knee_events)

    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "modified_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fs": {label: trial.fs for label, trial in trials.items()},
        "n_samples": {label: trial.n_samples for label, trial in trials.items()},
        "ignore_high_pass_filter": _ignore_high_pass_flag(),
        "detector": {"name": "v1-from-committed-csv"},
        "lag_pad_s": None,
        "knee_lag_window_s": None,
        "trunk_events": trunk_events,
        "knee_events": knee_events,
    }
