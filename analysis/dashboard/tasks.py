"""The dashboard's long-running jobs, each a sequence of :mod:`analysis.pipeline` stages.

Every function here starts a job on the shared :data:`~analysis.dashboard.state.jobs`
runner and returns at once -- ``False`` if another job is still running.  When a
job changes what the browser shows, it bumps the matching revision on the
workspace, and the browser reloads that part on its next poll.
"""

from __future__ import annotations

from typing import Callable

from analysis import pipeline, recordings
from analysis.dashboard.editor_page import build_display
from analysis.dashboard.state import jobs, workspace
from analysis.kinematics import LoadedTrial
from analysis.recordings import Selection

Progress = Callable[..., None]


def _load(progress: Progress) -> dict[str, LoadedTrial]:
    trials = pipeline.load(progress)
    # Opening the session here means one always exists once the recordings are
    # loaded: a saved one is migrated, and with none saved one is detected.
    pipeline.open_session(trials, progress=progress)
    progress("Preparing the event editor's plots")
    displays = {label: build_display(loaded) for label, loaded in trials.items()}
    workspace.set_trials(trials, displays)
    return trials


def _loaded(progress: Progress) -> dict[str, LoadedTrial]:
    return workspace.trials if workspace.ready else _load(progress)


def load_recordings() -> bool:
    return jobs.start("Load recordings", _load)


def use_recordings(novice: str | None, trained: str | None) -> bool:
    """Switch to another selection of recordings, and load it if it is preprocessed."""
    selection = Selection(novice=novice or None, trained=trained or None)

    def task(progress: Progress) -> None:
        recordings.activate(selection)
        workspace.clear()
        progress(f"Selected {selection.describe()}; outputs go to outputs/analyses/{selection.id}/")
        pending = pipeline.pending_preprocessing(selection)
        if pending:
            progress("Not preprocessed yet: " + ", ".join(r.name for r in pending)
                     + ". Press Run pipeline to process and load them.")
            return
        _load(progress)

    return jobs.start("Select recordings", task)


def preprocess() -> bool:
    """Stage 1 again for every selected recording, then reload them."""

    def task(progress: Progress) -> None:
        targets = [r for r in recordings.active().recordings().values() if r]
        pipeline.preprocess(targets, progress)
        _load(progress)  # the caches have been rewritten, so reload from them

    return jobs.start("Preprocess recordings", task)


def new_session(source: str) -> bool:
    def task(progress: Progress) -> None:
        pipeline.open_session(_loaded(progress), source, progress)
        workspace.bump("session")

    return jobs.start("Detect events", task)


def recalculate() -> bool:
    def task(progress: Progress) -> None:
        trials = _loaded(progress)
        pipeline.recalculate(pipeline.open_session(trials, progress=progress), trials, progress)
        workspace.bump("results")

    return jobs.start("Recalculate metrics", task)


def run_pipeline() -> bool:
    """Every stage that is missing or out of date, in order, for the active selection."""

    def task(progress: Progress) -> None:
        pending = pipeline.pending_preprocessing()
        if pending:
            pipeline.preprocess(pending, progress)
            trials = _load(progress)
        else:
            trials = _loaded(progress)

        session = pipeline.open_session(trials, progress=progress)
        if pipeline.status()["metrics"].state == "done":
            progress("The metrics are already up to date with the session.")
            return
        pipeline.recalculate(session, trials, progress)
        workspace.bump("results")

    return jobs.start("Run pipeline", task)
