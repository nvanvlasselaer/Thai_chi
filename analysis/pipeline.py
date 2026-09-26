#!/usr/bin/env python3
"""The analysis pipeline, from raw recordings to metrics, in four stages.

The stages work on a *selection*: a recording from ``data/`` for the novice
role, one for the trained role, or just one of the two
(:mod:`analysis.recordings`).

    1. preprocess   per recording: parse it, run the orientation filter, and write
                    what does not depend on event windows -- the orientation
                    cache, the 50 Hz kinematic CSV, the sensor inventory and the
                    orientation validation figure -- to outputs/recordings/<id>/.
                    Done once per recording and reused by every selection.
    2. load         rebuild the selected recordings' kinematics from their caches
    3. session      open the event windows every metric is computed on: the
                    selection's saved session, or a new one from the detectors
    4. recalculate  run the windows through the metrics; write the CSVs and
                    figures to outputs/analyses/<selection>/

The dashboard (``python3 app.py``) runs these stages from its Pipeline page.
They also run headless::

    python3 -m analysis.pipeline [--novice FILE] [--trained FILE] [--list]
                                 [--force-orientation] [--new-session {v2,v1}]

Without ``--novice``/``--trained`` the last selection is used.  The saved
session is reused unless ``--new-session`` is given, so a headless run
reproduces exactly what the curated windows produce.  With no session saved
yet, one is detected with ``config.DETECTOR``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

if __package__ in (None, ""):
    # Run as a script rather than with -m: make the ``analysis`` package importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from analysis import config, recordings, sessions
from analysis.data_io import (
    TrialData,
    make_sensor_inventory,
    parse_IMU_csv,
    save_kinematic_variables,
    save_orientation_npz,
)
from analysis.figures import make_orientation_validation_figure
from analysis.kinematics import Kinematics, LoadedTrial, compute_kinematics, load_all_trials, load_trial, sensor_signals
from analysis.orientation import madgwick_imu
from analysis.recompute import RecomputeResult, read_provenance, recompute, session_digest
from analysis.recordings import Recording, Selection

Progress = Callable[..., None]
"""``progress(message, fraction=None)``: what is happening, and where known the
fraction of the current stage that is done."""


def _silent(message: str, fraction: float | None = None) -> None:
    pass


# ---------------------------------------------------------------------------
# Stage 1: preprocessing, per recording
# ---------------------------------------------------------------------------


def run_orientation_filter(
    trials: list[TrialData], progress: Progress | None = None, workers: int | None = None
) -> dict[str, dict[str, np.ndarray]]:
    """Madgwick output per recording and sensor, computed in parallel.

    The filter is a pure-Python per-sample loop and by far the slowest step of
    the pipeline.  Its runs -- one per sensor per recording -- are independent,
    so they are spread over ``workers`` processes (default: one per CPU; ``1``
    runs them one after another in this process).  The result is the same
    either way.
    """
    report = progress or _silent
    jobs = [
        (trial.label, sensor, acc, gyro, trial.fs)
        for trial in trials
        for sensor, (acc, gyro) in sensor_signals(trial).items()
    ]
    raw: dict[str, dict[str, np.ndarray]] = {trial.label: {} for trial in trials}
    started = time.monotonic()

    def done(count: int) -> None:
        report(f"Orientation filter: {count} of {len(jobs)} sensors "
               f"({time.monotonic() - started:.0f} s)", count / len(jobs))

    workers = min(workers or os.cpu_count() or 1, len(jobs))
    if workers <= 1:
        for count, (label, sensor, acc, gyro, fs) in enumerate(jobs, 1):
            raw[label][sensor] = madgwick_imu(acc, gyro, fs)
            done(count)
        return raw

    # Spawn rather than fork: the dashboard calls this from a worker thread,
    # and forking a process that has threads is unsafe.
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = {
            pool.submit(madgwick_imu, acc, gyro, fs): (label, sensor)
            for label, sensor, acc, gyro, fs in jobs
        }
        for count, future in enumerate(as_completed(futures), 1):
            label, sensor = futures[future]
            raw[label][sensor] = future.result()
            done(count)
    return raw


def preprocess(
    targets: list[Recording], progress: Progress | None = None, workers: int | None = None,
    force_orientation: bool = False,
) -> None:
    """Stage 1 for each recording in ``targets``: parse, filter, write its output folder.

    The orientation filters of all of them run in one parallel pool.  A
    recording whose orientation cache is current and only its 50 Hz export is
    out of date is re-exported from the cache instead, which takes seconds and
    leaves ``orientation.npz`` untouched -- unless ``force_orientation``.
    """
    report = progress or _silent
    if not force_orientation:
        exports = [recording for recording in targets if recording.orientation_current()]
        for recording in exports:
            export_kinematics(recording, report)
        targets = [recording for recording in targets if recording not in exports]
    parsed = []
    for recording in targets:
        if not recording.path.exists():
            raise FileNotFoundError(f"Raw recording not found: {recording.path}")
        report(f"Parsing {recording.name}")
        # Labelled by file name, not role: this stage belongs to the recording
        # and is shared by every selection it appears in.
        parsed.append((recording, parse_IMU_csv(recording.path, recording.name)))

    raw = run_orientation_filter([trial for _, trial in parsed], report, workers)

    for recording, trial in parsed:
        report(f"Deriving joint angles and the 50 Hz export: {recording.name}")
        recording.output_dir.mkdir(parents=True, exist_ok=True)
        kin = compute_kinematics(trial, raw_quats=raw[trial.label])
        save_orientation_npz(recording.orientation_path, kin)
        save_kinematic_variables(recording.kinematics_path, kin, trial.fs)
        make_sensor_inventory([trial]).to_csv(recording.inventory_path, index=False)
        make_orientation_validation_figure(kin, trial.fs, recording.name, recording.validation_figure_path)
        _write_recording_manifest(recording, trial, kin)


def export_kinematics(recording: Recording, progress: Progress | None = None) -> None:
    """Rebuild a recording's 50 Hz export from its orientation cache, without filtering again."""
    report = progress or _silent
    report(f"Rewriting the 50 Hz export of {recording.name} in format {config.EXPORT_VERSION} "
           "(the orientation cache is current)")
    loaded = load_trial(recording, recording.name)
    save_kinematic_variables(recording.kinematics_path, loaded.kin, loaded.fs)
    manifest = recording.manifest() or {}
    manifest["export_version"] = config.EXPORT_VERSION
    manifest["exported_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    recording.manifest_path.write_text(json.dumps(manifest, indent=2))


def _write_recording_manifest(recording: Recording, trial: TrialData, kin: Kinematics) -> None:
    """``recording.json``: which file, byte for byte, the folder was computed from."""
    now = datetime.now(timezone.utc)
    recording.manifest_path.write_text(json.dumps({
        "source": {"file": recording.name, "size": recording.size, "mtime_ns": recording.mtime_ns,
                   "sha256": recording.sha256()},
        "fs_hz": trial.fs,
        "n_samples": len(kin.t),
        "duration_s": round(float(kin.t[-1]), 3),
        "sensors": trial.sensors,
        "export_version": config.EXPORT_VERSION,
        "processed_utc": now.isoformat(timespec="seconds"),
        "processed_local": now.astimezone().strftime("%d %b %Y %H:%M"),
    }, indent=2))


def pending_preprocessing(selection: Selection | None = None) -> list[Recording]:
    """The selected recordings whose stage-1 outputs are missing or out of date."""
    selection = selection or recordings.active()
    return [recording for recording in selection.recordings().values()
            if recording is not None and recording.cache_state()[0] != "done"]


# ---------------------------------------------------------------------------
# Stages 2-4: loading, the session, recalculation
# ---------------------------------------------------------------------------


def load(progress: Progress | None = None, selection: Selection | None = None) -> dict[str, LoadedTrial]:
    """Stage 2: rebuild the selected recordings' kinematics from their orientation caches."""
    report = progress or _silent
    selection = selection or recordings.active()
    report("Loading " + " and ".join(f"the {role.lower()} recording ({name})"
                                     for role, name in selection.names().items()))
    trials = load_all_trials(selection)
    for label, loaded in trials.items():
        for warning in loaded.kin.calibration.warnings():
            report(f"Warning, {label.lower()}: {warning}")
    return trials


def open_session(
    trials: dict[str, LoadedTrial], source: str | None = None, progress: Progress | None = None
) -> dict:
    """Stage 3: the saved session, or -- given ``source``, or with none saved -- a new one.

    A saved session is migrated to the current schema, and written back if that
    changed it.  A new one takes its windows from ``source`` (one of
    :data:`analysis.sessions.NEW_SESSION_SOURCES`, by default
    ``config.DETECTOR``); the session it replaces is parked in ``_previous``.
    """
    report = progress or _silent
    saved = None if source else sessions.load_session()
    if saved is None:
        source = source or config.DETECTOR
        report(f"Detecting events with the {sessions.NEW_SESSION_SOURCES[source]}")
        return sessions.start_new_session(trials, source=source)

    session, notes = sessions.migrate_session(saved, trials)
    for note in notes:
        report(f"Session migrated: {note}")
    if notes:
        sessions.save_session(sessions.refresh_seconds(session, trials))
    return session


def recalculate(
    session: dict, trials: dict[str, LoadedTrial], progress: Progress | None = None
) -> RecomputeResult:
    """Stage 4: run the session's windows through every metric and write the outputs."""
    report = progress or _silent
    report("Recalculating the metrics and figures")
    result = recompute(session, trials, write=True)
    for line in result.log:
        report(line)
    return result


def run(
    selection: Selection | None = None,
    force_orientation: bool = False,
    new_session: str | None = None,
    progress: Progress | None = None,
    workers: int | None = None,
) -> RecomputeResult:
    """Every stage in turn, for ``selection`` (the active one by default).

    Stage 1 only runs for recordings whose outputs are missing or out of date,
    unless ``force_orientation`` is set.
    """
    selection = recordings.activate(selection or recordings.active())
    targets = ([recording for recording in selection.recordings().values() if recording]
               if force_orientation else pending_preprocessing(selection))
    if targets:
        preprocess(targets, progress, workers, force_orientation=force_orientation)
    trials = load(progress, selection)
    session = open_session(trials, new_session, progress)
    return recalculate(session, trials, progress)


# ---------------------------------------------------------------------------
# Status, for the dashboard
# ---------------------------------------------------------------------------


METRIC_FILES = (
    "trunk_rotation_balance_metrics.csv",
    "monopodal_stance_balance_metrics.csv",
    "sequence_smoothness_metrics.csv",
)


@dataclass
class StageStatus:
    state: str
    """``"done"``, ``"stale"`` (present but out of date) or ``"missing"``."""
    detail: str


def _when(path_or_iso) -> str:
    if isinstance(path_or_iso, Path):
        moment = datetime.fromtimestamp(path_or_iso.stat().st_mtime)
    else:
        moment = datetime.fromisoformat(path_or_iso).astimezone()
    return moment.strftime("%d %b %H:%M")


def status(selection: Selection | None = None) -> dict[str, StageStatus]:
    """What is on disk for each stage of ``selection`` (the active one by default).

    Stage 2 (loading) is state held by whoever loaded the recordings, so it is
    not reported here.
    """
    selection = selection or recordings.active()
    result: dict[str, StageStatus] = {}
    chosen = selection.recordings()

    problem = selection.problem()
    if problem:
        result["recordings"] = StageStatus("missing", problem[0].upper() + problem[1:])
    else:
        result["recordings"] = StageStatus("done", " · ".join(
            f"{role}: {recording.name} ({recording.size / 1e6:.0f} MB)" for role, recording in chosen.items()
        ))

    states = {role: recording.cache_state() if recording else ("missing", "not found")
              for role, recording in chosen.items()}
    if not states:
        result["preprocess"] = StageStatus("missing", "no recording selected")
    else:
        found = {state for state, _ in states.values()}
        worst = "missing" if "missing" in found else "stale" if "stale" in found else "done"
        result["preprocess"] = StageStatus(worst, " · ".join(f"{role}: {why}" for role, (_, why) in states.items()))

    session = None if problem else sessions.load_session()
    if session is None:
        result["session"] = StageStatus(
            "missing", "no session for this selection yet: one is detected when the recordings are loaded")
    else:
        name = session.get("session_name") or "working session (unnamed)"
        detector = (session.get("detector") or {}).get("name", "v2")
        counts = (f"{len(session.get('trunk_events', []))} trunk rotations · "
                  f"{len(session.get('knee_events', []))} single-leg stances · "
                  f"{len(session.get('smooth_events', []))} sequence turns")
        result["session"] = StageStatus("done", f"{name}: {counts} (detector {detector}, "
                                                f"last edited {_when(sessions.session_path())})")

    provenance = None if problem else read_provenance()
    have_metrics = all((selection.output_dir / name).exists() for name in METRIC_FILES)
    if session is None or not have_metrics:
        result["metrics"] = StageStatus("missing", "not calculated yet")
    elif provenance is None:
        result["metrics"] = StageStatus(
            "stale", "calculated before provenance was recorded, so they cannot be matched to "
                     "the current windows: recalculate to be sure"
        )
    elif provenance.get("session_digest") != session_digest(session):
        result["metrics"] = StageStatus(
            "stale", f"the windows, settings or recordings have changed since the last recalculation "
                     f"({_when(provenance['computed_utc'])})"
        )
    else:
        result["metrics"] = StageStatus("done", f"up to date: recalculated {_when(provenance['computed_utc'])}")
    return result


# ---------------------------------------------------------------------------
# Headless entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the whole analysis without the dashboard.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--novice", metavar="FILE",
                        help="recording for the novice role, as a path relative to data/")
    parser.add_argument("--trained", metavar="FILE",
                        help="recording for the trained role. Give only one of the two to analyse a "
                             "single recording; give neither to reuse the last selection")
    parser.add_argument("--list", action="store_true", help="list the recordings in data/ and exit")
    parser.add_argument("--force-orientation", action="store_true",
                        help="re-run the orientation filter even if its cache is up to date")
    parser.add_argument("--new-session", choices=sorted(sessions.NEW_SESSION_SOURCES),
                        help="replace the saved session with a new one from this source "
                             "(the old one is parked in the analysis's sessions/_previous.json)")
    parser.add_argument("--workers", type=int, default=None,
                        help="processes for the orientation filter (default: one per CPU)")
    args = parser.parse_args()

    if args.list:
        for recording in recordings.scan():
            state = recording.cache_state()[1] if recording.usable else recording.problem
            print(f"{recording.name:60} {recording.size / 1e6:6.0f} MB  {state}")
        return

    if args.novice or args.trained:
        selection = Selection(novice=args.novice, trained=args.trained)
    else:
        selection = recordings.active()
    problem = selection.problem()
    if problem:
        parser.error(f"cannot use {selection.describe()}: {problem}")

    started = time.monotonic()

    def progress(message: str, fraction: float | None = None) -> None:
        print(f"[{time.monotonic() - started:5.1f} s] {message}", flush=True)

    progress(f"Selection: {selection.describe()}")
    result = run(selection, args.force_orientation, args.new_session, progress, args.workers)
    print(f"\nWrote to {selection.output_dir.relative_to(config.ROOT)}/:\n  " + "\n  ".join(result.written))


if __name__ == "__main__":
    main()
