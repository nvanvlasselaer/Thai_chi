"""Which recordings exist, which are selected, and where their outputs go.

Every CSV under ``data/`` is a candidate recording.  A *selection* assigns one
to each role -- novice and trained -- or to only one of them.  Outputs are kept
apart so that every file can be traced back to the recordings behind it:

    outputs/recordings/<recording>/     what depends on one recording alone: the
                                        orientation cache, the 50 Hz export, the
                                        sensor inventory, the orientation figure,
                                        and recording.json naming the source file
    outputs/analyses/<selection>/       what depends on the selection: the session,
                                        the metrics and figures, and analysis.json

A recording's folder is named after its file, made safe for any file system,
and a selection's after its recordings with their roles, for example
``novice-IMU_Trial_1_RC_Novice__trained-IMU_Trial_3_RC_Trained``.  Because file
names follow no fixed scheme, the exact source name, size and SHA-256 are also
written into those JSON files.

One selection is *active* at a time.  It is remembered in
``outputs/selection.json``, and every module that writes analysis outputs asks
:func:`analysis_dir` where to put them.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from analysis import config

SELECTION_PATH = config.OUTPUT_DIR / "selection.json"
MAX_ID_LENGTH = 80


# ---------------------------------------------------------------------------
# The recordings in data/
# ---------------------------------------------------------------------------


def _safe_id(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-._")
    return cleaned or "recording"


def recording_id(name: str) -> str:
    """A folder name for a recording, from its path relative to ``data/``.

    Kept readable -- the file's own name with unsafe characters replaced -- so
    the folder says which file it came from.  Where that loses information
    (characters replaced, or a long name shortened), a short hash of the exact
    name is appended, so two differently named files never share a folder.
    """
    stem = name[: -len(".csv")] if name.lower().endswith(".csv") else name
    readable = _safe_id(stem.replace("/", "__"))
    if readable == stem and len(readable) <= MAX_ID_LENGTH:
        return readable
    digest = hashlib.sha1(name.encode()).hexdigest()[:6]
    return f"{readable[:MAX_ID_LENGTH]}-{digest}"


def _delsys_problem(path: Path) -> str | None:
    """Why a CSV is not a Delsys Trigno export the parser can read, or None."""
    try:
        with path.open(newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.reader(handle)
            rows = [next(reader, []) for _ in range(9)]
    except OSError as error:
        return f"cannot be read ({error.strerror})"
    channels = [cell.strip().lower() for cell in rows[5]]
    if not any("acc x time" in cell for cell in channels):
        return "not a Delsys Trigno export (no ACC X time column in row 6)"
    if not any("hz" in cell.lower() for cell in rows[6]):
        return "not a Delsys Trigno export (no sampling rate in row 7)"
    if not any(cell.strip() for cell in rows[3]):
        return "not a Delsys Trigno export (no sensor names in row 4)"
    return None


_problem_cache: dict[tuple[str, int, int], str | None] = {}


@dataclass(frozen=True)
class Recording:
    """One CSV in ``data/``."""

    name: str
    """Path relative to ``data/``, the name shown everywhere."""
    path: Path
    size: int
    mtime_ns: int
    problem: str | None
    """Why it cannot be used, or None if it is a readable Delsys export."""

    @property
    def id(self) -> str:
        return recording_id(self.name)

    @property
    def usable(self) -> bool:
        return self.problem is None

    @property
    def output_dir(self) -> Path:
        return config.RECORDINGS_DIR / self.id

    orientation_path = property(lambda self: self.output_dir / "orientation.npz")
    kinematics_path = property(lambda self: self.output_dir / "kinematic_variables_50hz.csv")
    inventory_path = property(lambda self: self.output_dir / "sensor_inventory.csv")
    validation_figure_path = property(lambda self: self.output_dir / "orientation_validation.png")
    sensor_check_path = property(lambda self: self.output_dir / "sensor_check.csv")
    manifest_path = property(lambda self: self.output_dir / "recording.json")

    def outputs(self) -> list[Path]:
        return [self.orientation_path, self.kinematics_path, self.inventory_path,
                self.validation_figure_path, self.sensor_check_path, self.manifest_path]

    def sha256(self) -> str:
        return _sha256(self.path, self.size, self.mtime_ns)

    def manifest(self) -> dict | None:
        """What ``recording.json`` says the outputs were computed from, if it exists."""
        try:
            return json.loads(self.manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def cache_state(self) -> tuple[str, str]:
        """``("done" | "stale" | "missing", why)`` for this recording's outputs.

        Stale means the source file is no longer the one they were computed
        from.  The size and modification time are compared first; only when the
        time differs -- as it does after copying the file -- is the content
        hashed, so an unchanged copy is not reported as changed.
        """
        if not self.orientation_path.exists():
            return "missing", "not processed yet"
        manifest = self.manifest()
        if manifest is None:
            return "stale", "processed before recording.json was written"
        source = manifest.get("source", {})
        if source.get("size") != self.size or (
            source.get("mtime_ns") != self.mtime_ns and source.get("sha256") != self.sha256()
        ):
            return "stale", "the file has changed since it was processed"
        absent = [path.name for path in self.outputs() if not path.exists()]
        if absent:
            return "stale", f"missing {', '.join(absent)}"
        if self.export_stale():
            return "stale", (f"the 50 Hz export is format {manifest.get('export_version', 1)}, the current one "
                             f"is {config.EXPORT_VERSION} (rebuilt from the orientation cache, no filtering)")
        return "done", f"processed {manifest.get('processed_local', '')}".strip()

    def orientation_current(self) -> bool:
        """Whether the orientation cache itself is up to date, whatever the export."""
        manifest = self.manifest()
        if not self.orientation_path.exists() or manifest is None:
            return False
        source = manifest.get("source", {})
        return source.get("size") == self.size and (
            source.get("mtime_ns") == self.mtime_ns or source.get("sha256") == self.sha256()
        )

    def export_stale(self) -> bool:
        """The 50 Hz export was written in an older format than ``config.EXPORT_VERSION``."""
        manifest = self.manifest() or {}
        return int(manifest.get("export_version", 1)) < config.EXPORT_VERSION

    def identity(self) -> dict:
        """How this recording is named in session, analysis and provenance files."""
        return {"file": self.name, "id": self.id, "size": self.size, "sha256": self.sha256()}


_sha_cache: dict[tuple[str, int, int], str] = {}


def _sha256(path: Path, size: int, mtime_ns: int) -> str:
    key = (str(path), size, mtime_ns)
    if key not in _sha_cache:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        _sha_cache[key] = digest.hexdigest()
    return _sha_cache[key]


def scan() -> list[Recording]:
    """Every CSV under ``data/``, sorted by name, each checked for readability."""
    if not config.DATA_DIR.exists():
        return []
    found = []
    for path in sorted(config.DATA_DIR.rglob("*.csv")) + sorted(config.DATA_DIR.rglob("*.CSV")):
        if any(part.startswith(".") for part in path.relative_to(config.DATA_DIR).parts):
            continue
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        if key not in _problem_cache:
            _problem_cache[key] = _delsys_problem(path)
        found.append(Recording(
            name=path.relative_to(config.DATA_DIR).as_posix(), path=path, size=stat.st_size,
            mtime_ns=stat.st_mtime_ns, problem=_problem_cache[key],
        ))
    return sorted({r.name: r for r in found}.values(), key=lambda r: r.name.lower())


def find(name: str | None) -> Recording | None:
    if not name:
        return None
    return next((recording for recording in scan() if recording.name == name), None)


# ---------------------------------------------------------------------------
# The selection, and where its outputs go
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Selection:
    """Which recording fills each role.  One of the two may be empty."""

    novice: str | None = None
    trained: str | None = None

    def names(self) -> dict[str, str]:
        """``{role: file name}`` for the roles that are filled, in role order."""
        chosen = {"Novice": self.novice, "Trained": self.trained}
        return {role: chosen[role] for role in config.TRIALS if chosen[role]}

    def recordings(self) -> dict[str, Recording | None]:
        """``{role: Recording}``; None where the named file is no longer in ``data/``."""
        catalogue = {recording.name: recording for recording in scan()}
        return {role: catalogue.get(name) for role, name in self.names().items()}

    def problem(self) -> str | None:
        """Why this selection cannot be used, or None."""
        names = self.names()
        if not names:
            return "no recording selected"
        if len(set(names.values())) < len(names):
            return "the same recording is selected for both roles"
        for role, recording in self.recordings().items():
            if recording is None:
                return f"{names[role]} is no longer in data/"
            if not recording.usable:
                return f"{recording.name}: {recording.problem}"
        return None

    @property
    def id(self) -> str:
        return "__".join(f"{role.lower()}-{recording_id(name)}" for role, name in self.names().items())

    @property
    def output_dir(self) -> Path:
        return config.ANALYSES_DIR / self.id

    def describe(self) -> str:
        names = self.names()
        return " · ".join(f"{role}: {name}" for role, name in names.items()) or "nothing selected"


def selected_recording(prefer: str = "Novice") -> Recording | None:
    """One recording of the active selection, for tools that show a single one."""
    chosen = {role: recording for role, recording in active().recordings().items() if recording}
    return chosen.get(prefer) or next(iter(chosen.values()), None)


def default_selection() -> Selection:
    """The configured recordings, as far as they are present in ``data/``."""
    present = {recording.name for recording in scan() if recording.usable}
    chosen = {role: name for role, name in config.DEFAULT_RECORDINGS.items() if name in present}
    return Selection(novice=chosen.get("Novice"), trained=chosen.get("Trained"))


_active: Selection | None = None


def active() -> Selection:
    """The selection outputs are currently written for: the last one chosen, else the default."""
    global _active
    if _active is None:
        try:
            saved = json.loads(SELECTION_PATH.read_text())
            _active = Selection(novice=saved.get("Novice"), trained=saved.get("Trained"))
        except (OSError, json.JSONDecodeError):
            _active = default_selection()
    return _active


def activate(selection: Selection) -> Selection:
    """Make ``selection`` the active one, remember it, and record it in its folder."""
    global _active
    problem = selection.problem()
    if problem:
        raise ValueError(f"Cannot use this selection: {problem}.")
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    SELECTION_PATH.write_text(json.dumps(selection.names(), indent=2))
    _active = selection
    write_analysis_manifest(selection)
    return selection


def analysis_dir() -> Path:
    """Where the active selection's session, metrics and figures go."""
    return active().output_dir


def write_analysis_manifest(selection: Selection) -> None:
    """``analysis.json``: which recordings, byte for byte, this folder was made from."""
    selection.output_dir.mkdir(parents=True, exist_ok=True)
    path = selection.output_dir / "analysis.json"
    recordings = {role: recording.identity() for role, recording in selection.recordings().items() if recording}
    try:
        previous = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        previous = {}
    if previous.get("recordings") == recordings:
        return
    path.write_text(json.dumps({
        "analysis": selection.id,
        "recordings": recordings,
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2))
