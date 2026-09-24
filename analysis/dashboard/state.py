"""Server-side state shared by every callback: the loaded recordings and the running job.

Dash callbacks are functions of what the browser sends, but two things cannot
live in the browser: the loaded kinematics (hundreds of MB) and a pipeline run
that outlasts any request.  Both are held here, in the server process, behind
locks.  The browser polls :func:`JobRunner.snapshot` and the revision counters
of :class:`Workspace`, and reloads whatever a finished job has changed.
"""

from __future__ import annotations

import threading
import time
import traceback
from datetime import datetime
from typing import Callable

from analysis.kinematics import LoadedTrial

WORK_LOCK = threading.Lock()
"""Held while anything writes outputs/ or draws a figure: matplotlib is not
thread-safe, and two writers would interleave their CSVs."""


class Workspace:
    """The recordings currently loaded, plus revision counters the browser watches.

    ``data`` changes when the recordings are (re)loaded, ``session`` when a job
    replaces the session file, and ``results`` when the outputs are rewritten.
    Edits made in the event editor are saved by the editor itself and bump
    nothing, since the browser that made them already has them.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.trials: dict[str, LoadedTrial] | None = None
        self.displays: dict[str, dict] | None = None
        self._revision = {"data": 0, "session": 0, "results": 0}

    @property
    def ready(self) -> bool:
        return self.trials is not None

    def set_trials(self, trials: dict[str, LoadedTrial], displays: dict[str, dict]) -> None:
        with self._lock:
            self.trials, self.displays = trials, displays
            self._revision["data"] += 1
            self._revision["session"] += 1

    def clear(self) -> None:
        """Forget the loaded recordings, as when a different selection is made."""
        with self._lock:
            self.trials, self.displays = None, None
            for kind in self._revision:
                self._revision[kind] += 1

    def bump(self, *kinds: str) -> None:
        with self._lock:
            for kind in kinds:
                self._revision[kind] += 1

    def revision(self) -> dict[str, int]:
        with self._lock:
            return dict(self._revision)


class JobRunner:
    """Runs one long task at a time on a background thread, and reports on it.

    A task is a function taking a ``progress(message, fraction=None)`` callback
    -- the same protocol :mod:`analysis.pipeline` uses -- so the pipeline stages
    can be passed their progress reporting directly.
    """

    MAX_LOG = 400

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = self._idle()

    @staticmethod
    def _idle() -> dict:
        return {"name": None, "running": False, "message": "", "fraction": None,
                "log": [], "error": None, "started": None, "elapsed_s": 0.0, "finished": False}

    def start(self, name: str, task: Callable[[Callable[..., None]], None]) -> bool:
        """Start ``task`` unless a job is already running.  Returns whether it started."""
        with self._lock:
            if self._state["running"]:
                return False
            self._state = {**self._idle(), "name": name, "running": True, "started": time.monotonic()}
        threading.Thread(target=self._run, args=(task,), name=f"job: {name}", daemon=True).start()
        return True

    def _run(self, task: Callable[[Callable[..., None]], None]) -> None:
        try:
            with WORK_LOCK:
                task(self.progress)
            self.progress("Done.", 1.0)
        except Exception as error:  # reported in the interface rather than lost on a thread
            detail = traceback.format_exc(limit=6)
            with self._lock:
                self._state["error"] = f"{type(error).__name__}: {error}"
                self._state["log"].append(self._stamp(detail.rstrip()))
        finally:
            with self._lock:
                self._state["running"] = False
                self._state["finished"] = True
                self._state["elapsed_s"] = time.monotonic() - self._state["started"]

    @staticmethod
    def _stamp(message: str) -> str:
        return f"{datetime.now():%H:%M:%S}  {message}"

    def progress(self, message: str, fraction: float | None = None) -> None:
        with self._lock:
            self._state["message"] = message
            # A step that reports no fraction has an unknown one; keeping the
            # previous step's would show "100%" while the next step runs.
            self._state["fraction"] = None if fraction is None else max(0.0, min(1.0, float(fraction)))
            log = self._state["log"]
            # Repeated progress lines of one step (the filter reports per sensor)
            # replace each other rather than filling the log.
            if log and message.split(":")[0] == log[-1][10:].split(":")[0] and fraction is not None:
                log[-1] = self._stamp(message)
            else:
                log.append(self._stamp(message))
            del log[:-self.MAX_LOG]

    def snapshot(self) -> dict:
        with self._lock:
            state = dict(self._state, log=list(self._state["log"]))
        if state["running"]:
            state["elapsed_s"] = time.monotonic() - state["started"]
        return state

    @property
    def running(self) -> bool:
        with self._lock:
            return self._state["running"]


workspace = Workspace()
jobs = JobRunner()
