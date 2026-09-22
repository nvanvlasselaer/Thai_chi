#!/usr/bin/env python3
"""Data layer for the Tai Chi event-window editor.

Three responsibilities, none of which involve Dash so that they can be exercised
from a plain script:

* **Loading** — reconstitute a full :class:`Kinematics` without re-running the
  Madgwick filter.  ``madgwick_imu`` is a pure-Python per-sample loop and is the
  only slow part of the pipeline, but its output already sits in
  ``outputs/orientation_*.npz``.  Parsing a raw CSV takes ~1.6 s, loading the
  quaternions ~0.2 s and rederiving every joint angle from them ~0.35 s, so the
  whole editor starts in about 4 s.
* **Session state** — load, seed and save the curated windows.  Sample indices
  are authoritative; seconds are written alongside for readability and ignored
  on load, so there is no rounding ambiguity anywhere.
* **Recalculation** — feed the curated windows through the pipeline's own metric
  functions, unmodified, and write the same output files ``main()`` writes.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import event_detection as ed
from tai_chi_trunk_and_knee import (
    Kinematics,
    OUTPUT_DIR,
    NOVICE_CSV,
    TRAINED_CSV,
    TrialData,
    compute_asymmetry_metrics,
    compute_kinematics,
    compute_knee_balance_metrics,
    compute_trunk_rotation_balance_metrics,
    find_knee_flexion_windows,
    lowpass_signal,
    make_knee_flexion_overview_figure,
    make_knee_traceability_figure,
    make_trunk_traceability_figure,
    parse_IMU_csv,
    quat_inverse,
    quat_multiply,
    quat_to_euler_deg,
    save_orientation_npz,
    summarize_knee_flexion_event,
)

SESSION_PATH = OUTPUT_DIR / "event_editor_session.json"
SCHEMA_VERSION = 1
TRIALS = ("Novice", "Trained")
CSV_FOR_TRIAL = {"Novice": NOVICE_CSV, "Trained": TRAINED_CSV}
KNEE_THRESHOLD_DEG = 60.0

# Joint angles are relative orientations between a proximal and a distal sensor.
# The pairs and their order mirror compute_kinematics() exactly.
JOINT_SENSOR_PAIRS = {
    "left_hip_deg": ("lumbar", "lthigh"),
    "right_hip_deg": ("lumbar", "rthigh"),
    "left_knee_deg": ("lthigh", "ltibia"),
    "right_knee_deg": ("rthigh", "rtibia"),
    "left_ankle_deg": ("ltibia", "lfoot"),
    "right_ankle_deg": ("rtibia", "rfoot"),
    "left_shoulder_deg": ("chestbone", "lhumerus"),
    "right_shoulder_deg": ("chestbone", "rhumerus"),
    "left_elbow_deg": ("lhumerus", "lulna"),
    "right_elbow_deg": ("rhumerus", "rulna"),
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@dataclass
class LoadedTrial:
    """Everything the editor needs for one recording."""

    label: str
    kin: Kinematics
    trial: TrialData
    fs: float
    n_samples: int

    @property
    def duration_s(self) -> float:
        return float(self.kin.t[-1])


def orientation_path(label: str) -> Path:
    return OUTPUT_DIR / f"orientation_{label.lower()}.npz"


def relative_euler(quats: dict[str, np.ndarray], proximal: str, distal: str) -> np.ndarray:
    """Euler angles of the distal segment expressed in the proximal frame."""
    return quat_to_euler_deg(quat_multiply(quat_inverse(quats[proximal]), quats[distal]))


def load_trial(label: str, recompute_orientation: bool = False) -> LoadedTrial:
    """Load one recording, rebuilding kinematics from the cached quaternions.

    The npz is trusted as the Madgwick output.  If the orientation code itself
    changes, pass ``recompute_orientation=True`` to run the real
    ``compute_kinematics`` and rewrite the npz.
    """
    csv_path = CSV_FOR_TRIAL[label]
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw recording not found: {csv_path}\n"
            "data/ is git-ignored, so a fresh clone has no IMU CSVs."
        )

    trial = parse_IMU_csv(csv_path, label)
    n = min(len(df) for df in trial.data.values())

    if recompute_orientation:
        kin = compute_kinematics(trial)
        save_orientation_npz(label, kin)
        return LoadedTrial(label=label, kin=kin, trial=trial, fs=trial.fs, n_samples=len(kin.t))

    npz_path = orientation_path(label)
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Orientation cache not found: {npz_path}\n"
            "Run the pipeline once, or start the editor with --recompute-orientation."
        )

    with np.load(npz_path) as data:
        time_s = data["time_s"]
        quats = {key[: -len("_q_wxyz")]: data[key][:n] for key in data.files if key.endswith("_q_wxyz")}

    if len(time_s) != n:
        raise ValueError(
            f"{npz_path.name} holds {len(time_s)} samples but {csv_path.name} parses to {n}. "
            "The orientation cache is out of sync with the recording; "
            "restart with --recompute-orientation."
        )

    eulers = {sensor: quat_to_euler_deg(q) for sensor, q in quats.items()}
    omega = {
        sensor: np.linalg.norm(
            trial.data[sensor][["gyro_x_dps", "gyro_y_dps", "gyro_z_dps"]].to_numpy()[:n], axis=1
        )
        for sensor in quats
    }
    joints = {
        field: relative_euler(quats, proximal, distal)
        for field, (proximal, distal) in JOINT_SENSOR_PAIRS.items()
    }

    kin = Kinematics(
        t=time_s,
        quaternions=quats,
        eulers_deg=eulers,
        omega_mag=omega,
        trunk_rel_euler_deg=relative_euler(quats, "lumbar", "chestbone"),
        **joints,
    )
    return LoadedTrial(label=label, kin=kin, trial=trial, fs=trial.fs, n_samples=n)


def load_all_trials(recompute_orientation: bool = False) -> dict[str, LoadedTrial]:
    return {label: load_trial(label, recompute_orientation) for label in TRIALS}


# ---------------------------------------------------------------------------
# Index/time conversion
#
# The pipeline converts an index to seconds as idx / fs, so the exact inverse is
# round(t * fs).  Truncating instead shifts the window by one sample, which
# changes lumbar_ap_acc_variance_g2 by ~0.8 % -- small enough to miss, large
# enough to matter.  Every conversion in the editor goes through these two.
# ---------------------------------------------------------------------------


def sec_to_idx(seconds: float, fs: float, n_samples: int) -> int:
    return int(np.clip(round(float(seconds) * fs), 0, n_samples - 1))


def idx_to_sec(index: int, fs: float) -> float:
    return float(index) / fs


# ---------------------------------------------------------------------------
# Session state
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


def seed_session(
    trials: dict[str, LoadedTrial],
    trunk_params: ed.TrunkDetectorParams | None = None,
    stab_params: ed.StabilizationParams | None = None,
    knee_params: ed.KneeDetectorParams | None = None,
) -> dict:
    """Build a fresh session by running the detectors on both recordings."""
    trunk_params = trunk_params or ed.TrunkDetectorParams()
    stab_params = stab_params or ed.StabilizationParams()
    knee_params = knee_params or ed.KneeDetectorParams()

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

    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "modified_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fs": {label: trial.fs for label, trial in trials.items()},
        "n_samples": {label: trial.n_samples for label, trial in trials.items()},
        "ignore_high_pass_filter": _ignore_high_pass_flag(),
        "detector": ed.params_to_dict(trunk_params, stab_params, knee_params),
        "lag_pad_s": LAG_PAD_S,
        "knee_lag_window_s": list(KNEE_LAG_WINDOW_S),
        "trunk_events": trunk_events,
        "knee_events": knee_events,
    }


def _ignore_high_pass_flag() -> bool:
    import tai_chi_trunk_and_knee as pipeline

    return bool(pipeline.Ignore_high_pass_filter)


def detect_trunk_windows(
    trials: dict[str, LoadedTrial],
    trunk_params: ed.TrunkDetectorParams,
    stab_params: ed.StabilizationParams,
) -> tuple[dict[str, list[tuple[int, int, int, int, float]]], list[float]]:
    """Detect trunk events in both recordings and pair them by order.

    Each entry is ``(event_start, event_end, stab_start, stab_end, quiet_ratio)``.
    """
    novice, trained = trials["Novice"], trials["Trained"]
    windows, _, _ = ed.pair_trunk_events(
        novice.kin, trained.kin, novice.fs, trained.fs, trunk_params, stab_params
    )
    return windows, []


def detect_knee_windows(
    loaded: LoadedTrial, side: str, knee_params: ed.KneeDetectorParams,
    stab_params: ed.StabilizationParams, omega: ed.StabilizationDiagnostics,
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
        stab = ed.find_stabilization(loaded.kin, loaded.fs, peak_idx, stab_params, diagnostics=omega)
        out.append(_window_dict(start, end, stab.start, stab.end, fs=loaded.fs,
                                stab_quiet_ratio=round(stab.quiet_ratio, 3)))
    return out


def seed_knee_events(
    trials: dict[str, LoadedTrial],
    stab_params: ed.StabilizationParams | None = None,
    knee_params: ed.KneeDetectorParams | None = None,
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
    stab_params = stab_params or ed.StabilizationParams()
    knee_params = knee_params or ed.KneeDetectorParams()
    omega = {label: ed.combined_omega(trials[label].kin, trials[label].fs, stab_params)
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


def _earliest_start(event: dict) -> int:
    """Sort key: the earliest window start across whichever trials are present."""
    starts = [w["event_start"] for w in event["windows"].values()]
    return min(starts) if starts else 0


def _sort_and_number(events: list[dict], family: str) -> None:
    events.sort(key=_earliest_start)
    prefix = "trunk" if family == "trunk" else "knee"
    for i, event in enumerate(events, 1):
        event["event_id"] = f"{prefix}-{i:02d}"


# ---------------------------------------------------------------------------
# Adding, deleting and restoring events
#
# Deleting moves an event to `trash` rather than dropping it.  The session file
# is written through on every change and has no undo of its own, so a removed
# event would otherwise only be recoverable from git.
# ---------------------------------------------------------------------------


FAMILY_KEY = {"trunk": "trunk_events", "knee": "knee_events"}
DEFAULT_TRUNK_EVENT_S = 3.0
DEFAULT_KNEE_EVENT_S = 2.0
DEFAULT_STAB_GAP_S = 0.1
"""Gap between the end of a new event and the start of its stabilization window."""


def events_of(session: dict, family: str) -> list[dict]:
    return session.setdefault(FAMILY_KEY[family], [])


def trash_of(session: dict) -> list[dict]:
    return session.setdefault("trash", [])


def next_event_id(session: dict, family: str) -> str:
    """Lowest unused id for a family, counting trashed events so ids stay unique."""
    prefix = "trunk" if family == "trunk" else "knee"
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
    kind = "trunk rotation" if entry["family"] == "trunk" else f"{event.get('flexed_leg','?')} knee"
    span = f"{window['event_start']/fs:.1f}-{window['event_end']/fs:.1f}s"
    sides = "+".join(sorted(windows))
    return f"{event['event_id']} - {kind}  {label[:1]} {span}  [{sides}]"


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


def migrate_session(session: dict) -> tuple[dict, list[str]]:
    """Bring a session file up to the current schema.  Returns it plus a log."""
    notes: list[str] = []

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


SESSIONS_DIR = OUTPUT_DIR / "sessions"
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
    loaded, _ = migrate_session(loaded)
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
    trunk_params: ed.TrunkDetectorParams | None = None,
    stab_params: ed.StabilizationParams | None = None,
    knee_params: ed.KneeDetectorParams | None = None,
) -> dict:
    """Discard the working session and detect both families afresh.

    The current session is parked first, so a fresh start is one ``Load`` away
    from being undone.  Unlike ``Re-detect``, which replaces one family and
    leaves the other alone, this resets everything: both families, the deleted
    list and the session name.
    """
    park_working_session()
    session = seed_session(trials, trunk_params, stab_params, knee_params)
    session["trash"] = []
    session.pop("session_name", None)
    session = refresh_seconds(session, trials)
    save_session(session, SESSION_PATH)
    return session


def delete_named_session(name: str) -> None:
    path = session_path_for(name)
    if path.exists():
        path.unlink()


def load_session(path: Path = SESSION_PATH) -> dict | None:
    if not path.exists():
        return None
    with path.open() as handle:
        return json.load(handle)


def save_session(session: dict, path: Path = SESSION_PATH) -> None:
    """Write the session atomically so an interrupted save cannot corrupt it."""
    session = dict(session)
    session["modified_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        json.dump(session, handle, indent=2)
    os.replace(tmp, path)


def refresh_seconds(session: dict, trials: dict[str, LoadedTrial]) -> dict:
    """Recompute the derived ``*_s`` display fields from the authoritative indices."""
    for event in session.get("trunk_events", []):
        for label, window in event["windows"].items():
            fs = trials[label].fs
            for key in ("event_start", "event_end", "stab_start", "stab_end"):
                window[f"{key}_s"] = round(idx_to_sec(window[key], fs), 4)
    for event in session.get("knee_events", []):
        for label, window in event.get("windows", {}).items():
            fs = trials[label].fs
            for key in ("event_start", "event_end", "stab_start", "stab_end"):
                window[f"{key}_s"] = round(idx_to_sec(window[key], fs), 4)
    return session


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


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
    stab_start, stab_end = window["stab_start"], window["stab_end"]

    def error(message: str) -> None:
        issues.append(Issue(event_id, trial, "error", message))

    def warn(message: str) -> None:
        issues.append(Issue(event_id, trial, "warning", message))

    for name, (lo, hi) in {"event": (start, end), "stabilization": (stab_start, stab_end)}.items():
        if not (0 <= lo < hi <= n_samples - 1):
            error(f"{name} window [{lo}, {hi}] is out of order or outside the recording")

    duration_s = (end - start) / fs
    if end > start and duration_s < MIN_EVENT_S:
        error(f"event window is {duration_s:.2f} s, shorter than the {MIN_EVENT_S:.2f} s minimum")
    if stab_end > stab_start and (stab_end - stab_start) / fs < MIN_STAB_S:
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
    if stab_start <= end:
        warn("stabilization overlaps the event")
    ratio = window.get("stab_quiet_ratio")
    if ratio is not None and ratio > LOW_CONFIDENCE_RATIO:
        warn(f"stabilization is {ratio:.2f}x the quiet baseline - the participant may not have settled")
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
    for event in session.get("trunk_events", []):
        if not event.get("enabled", True):
            continue
        for label, window in event["windows"].items():
            loaded = trials[label]
            issues += validate_window(event["event_id"], label, window, loaded.fs, loaded.n_samples)
    for event in session.get("knee_events", []):
        if not event.get("enabled", True):
            continue
        for label, window in event["windows"].items():
            loaded = trials[label]
            issues += validate_window(event["event_id"], label, window, loaded.fs, loaded.n_samples)

    for key in ("trunk_events", "knee_events"):
        enabled = [e for e in session.get(key, []) if e.get("enabled", True)]
        issues += check_pairing(enabled, trials)
    return issues


# ---------------------------------------------------------------------------
# Recalculation
# ---------------------------------------------------------------------------


@dataclass
class RecomputeResult:
    trunk_metrics: pd.DataFrame
    knee_metrics: pd.DataFrame
    asymmetry: pd.DataFrame
    written: list[str]
    log: list[str]


LAG_BOUND_S = 2.0
"""cross_correlation_lag's max_lag_s; a result at the bound is not a measurement."""

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


def flag_saturated_lags(metrics: pd.DataFrame, columns: list[str]) -> list[str]:
    notes = []
    for column in columns:
        if column not in metrics:
            continue
        saturated = metrics[np.abs(metrics[column]) >= 0.99 * LAG_BOUND_S]
        for _, row in saturated.iterrows():
            notes.append(
                f"{row['trial']}: {column} = {row[column]:+.3f} s is at the +/-{LAG_BOUND_S} s "
                "search bound, so the cross-correlation did not find a peak"
            )
    return notes


def recompute(session: dict, trials: dict[str, LoadedTrial], write: bool = True) -> RecomputeResult:
    """Run the curated windows through the pipeline's own metric functions.

    Every metric call here is the same function ``main()`` calls, with the same
    arguments, so curated numbers are directly comparable to automatic ones.
    """
    log: list[str] = []
    errors = [i for i in validate_session(session, trials) if i.severity == "error"]
    if errors:
        raise ValueError(
            "Cannot recalculate while windows are invalid:\n"
            + "\n".join(f"  {i.event_id} ({i.trial}): {i.message}" for i in errors)
        )

    # Velocity-segmented events are often shorter than the +/-2 s lag search, so
    # the cross-correlation gets a padded window.  Sessions seeded from the v1
    # windows leave this None and reproduce the original numbers exactly.
    lag_pad_s = session.get("lag_pad_s")
    _, _, session_knee_params = ed.params_from_dict(session.get("detector", {}) or {})
    knee_threshold = session_knee_params.threshold_deg
    knee_lag = session.get("knee_lag_window_s")
    knee_lag_kwargs = (
        {"lag_pre_s": float(knee_lag[0]), "lag_post_s": float(knee_lag[1])} if knee_lag else {}
    )

    trunk_rows: list[dict] = []
    trunk_windows: dict[str, list[tuple]] = {"Novice": [], "Trained": []}

    enabled_trunk = [e for e in session.get("trunk_events", []) if e.get("enabled", True)]
    for event in enabled_trunk:
        for label in TRIALS:
            loaded = trials[label]
            window = event["windows"][label]
            start, end = window["event_start"], window["event_end"]
            stab_start, stab_end = window["stab_start"], window["stab_end"]

            peak = ed.recompute_trunk_peak(loaded.kin, loaded.fs, start, end) if label == "Novice" else np.nan
            trunk_windows[label].append((start, end, stab_start, stab_end, peak))

            # Metric rows alternate Novice/Trained per event, as main() emits them.
            trunk_rows.append(
                compute_trunk_rotation_balance_metrics(
                    label, loaded.kin, loaded.trial, start, end, stab_start, stab_end,
                    lag_pad_s=lag_pad_s,
                )
            )

    # The window CSV is grouped by trial, not interleaved -- also as main() emits it.
    trunk_event_rows: list[dict] = []
    for label, windows in trunk_windows.items():
        fs = trials[label].fs
        for index, (start, end, stab_start, stab_end, peak) in enumerate(windows, 1):
            trunk_event_rows.append(
                {
                    "trial": label,
                    "event_index": index,
                    "event_start_s": idx_to_sec(start, fs),
                    "event_end_s": idx_to_sec(end, fs),
                    "event_peak_s": idx_to_sec(peak, fs) if label == "Novice" else np.nan,
                    "stabilization_start_s": idx_to_sec(stab_start, fs),
                    "stabilization_end_s": idx_to_sec(stab_end, fs),
                }
            )

    trunk_metrics = pd.DataFrame(trunk_rows)
    log.append(f"Trunk rotation: {len(enabled_trunk)} events x 2 trials = {len(trunk_rows)} metric rows")

    knee_rows: list[dict] = []
    knee_event_rows: list[dict] = []
    knee_events: dict[str, list[dict]] = {"Novice": [], "Trained": []}
    stab_overrides: dict[str, list[tuple[int, int]]] = {"Novice": [], "Trained": []}

    enabled_knee = [e for e in session.get("knee_events", []) if e.get("enabled", True)]
    for index, event in enumerate(enabled_knee, 1):
        side = event["flexed_leg"]
        for label in TRIALS:
            window = event["windows"].get(label)
            if window is None:
                continue  # single-sided event: this recording contributes nothing
            loaded = trials[label]
            start, end = window["event_start"], window["event_end"]
            stab_start, stab_end = window["stab_start"], window["stab_end"]

            summary = summarize_knee_flexion_event(
                label, loaded.kin, loaded.fs, side, start, end, threshold_deg=knee_threshold
            )
            summary["stabilization_start_s"] = idx_to_sec(stab_start, loaded.fs)
            summary["stabilization_end_s"] = idx_to_sec(stab_end, loaded.fs)
            knee_events[label].append(summary)
            stab_overrides[label].append((stab_start, stab_end))

            knee_rows.append(
                compute_knee_balance_metrics(
                    label, loaded.kin, loaded.trial, side, start, end,
                    stab_start=stab_start, stab_end=stab_end, **knee_lag_kwargs,
                )
            )
            knee_event_rows.append(
                {
                    "trial": label,
                    "event_index": index,
                    "flexed_leg": side,
                    "stance_leg": summary["stance_leg"],
                    "window_start_s": summary["window_start_s"],
                    "window_end_s": summary["window_end_s"],
                    "peak_time_s": summary["peak_time_s"],
                    "peak_abs_knee_flexion_deg": summary["peak_abs_knee_flexion_deg"],
                    "stabilization_start_s": summary["stabilization_start_s"],
                    "stabilization_end_s": summary["stabilization_end_s"],
                }
            )

    # Window CSVs are grouped by trial and metric CSVs interleave the pair, which
    # is the convention the trunk family already follows.
    knee_event_rows.sort(key=lambda r: (TRIALS.index(r["trial"]), r["event_index"]))

    knee_metrics = pd.DataFrame(knee_rows)
    asymmetry = compute_asymmetry_metrics(knee_metrics)
    log.append(f"Monopodal stance: {len(enabled_knee)} events")

    for note in flag_saturated_lags(trunk_metrics, ["trunk_pelvis_lag_s"]):
        log.append("WARNING  " + note)
    for note in flag_saturated_lags(knee_metrics, ["trunk_pelvis_lag_s", "trunk_pelvis_pitch_lag_s"]):
        log.append("WARNING  " + note)

    written: list[str] = []
    if write:
        OUTPUT_DIR.mkdir(exist_ok=True)

        def write_csv(frame: pd.DataFrame, name: str) -> None:
            if len(frame) == 0:
                log.append(f"skipped {name} (no rows)")
                return
            frame.to_csv(OUTPUT_DIR / name, index=False)
            written.append(name)

        write_csv(pd.DataFrame(trunk_event_rows), "trunk_rotation_event_windows.csv")
        write_csv(trunk_metrics, "trunk_rotation_balance_metrics.csv")
        write_csv(pd.DataFrame(knee_event_rows), "monopodal_stance_event_windows.csv")
        write_csv(knee_metrics, "monopodal_stance_balance_metrics.csv")
        write_csv(asymmetry, "monopodal_stance_asymmetry_metrics.csv")

        novice, trained = trials["Novice"], trials["Trained"]
        if len(trunk_metrics) > 0:
            make_trunk_traceability_figure(
                novice.kin, trained.kin, trunk_metrics, trunk_windows, novice.fs
            )
            written.append("trunk_traceability_figure.png")
        if len(knee_metrics) > 0:
            make_knee_flexion_overview_figure(novice.kin, trained.kin, knee_events, novice.fs)
            written.append("monopodal_stance_overview_figure.png")
            make_knee_traceability_figure(
                novice.kin, trained.kin, knee_metrics, knee_events, novice.fs,
                stab_overrides=stab_overrides,
            )
            written.append("monopodal_stance_traceability_figure.png")

        save_session(refresh_seconds(session, trials))
        written.append(SESSION_PATH.name)

    return RecomputeResult(
        trunk_metrics=trunk_metrics,
        knee_metrics=knee_metrics,
        asymmetry=asymmetry,
        written=written,
        log=log,
    )


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
    trunk_csv = OUTPUT_DIR / "trunk_rotation_event_windows.csv"
    knee_csv = OUTPUT_DIR / "monopodal_stance_event_windows.csv"
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
    stab_params = ed.StabilizationParams()
    omega_cache = {
        label: ed.combined_omega(trials[label].kin, trials[label].fs, stab_params) for label in TRIALS
    }
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
            from tai_chi_trunk_and_knee import find_stabilization as find_stabilization_v1

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
