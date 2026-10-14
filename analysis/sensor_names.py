"""Which sensor in a recording is which: the names the analysis knows its sensors by.

The analysis refers to its sensors by fixed names -- the keys of
``config.SENSOR_MAP``: ``chestbone``, ``lumbar``, ``lthigh``, ... -- and hangs
everything on them: the mounting each is assumed to have
(:func:`analysis.orientation.mounting_matrix`), the joints between them
(:data:`analysis.kinematics.JOINT_SENSOR_PAIRS`) and the stick figure.  A Delsys
Trigno export labels each sensor as it was set up in Trigno Discover, a free
text and the sensor's serial number: ``L_humerus (84483)``.  This module maps
the one onto the other, per recording.

* :func:`suggest` proposes a name for each label: the one it spells (as the
  current recordings' labels do); else one it reads as (``Left Upper Arm``,
  ``R_Shank``, ``Sternum``); else the name the same serial number has in another
  recording.  Only a spelled name is used without being checked.
* A mapping someone has checked is saved as ``sensor_names.json`` in the
  recording's output folder (:func:`save`), and stage 1 parses the recording
  with it.  A label can also be marked *not used* -- a spare or an EMG sensor.
* :func:`problems` says what keeps a mapping from being used: a name two
  sensors have, a label not decided on, or no name at all.  A name no sensor
  has is not one of them: :func:`absent` lists those, and the analysis runs
  without them (:mod:`analysis.coverage`, ``docs/missing_sensors.md``).
* A limb sensor's *placement* -- on the side of the segment, its front or its
  back (:data:`analysis.orientation.PLACEMENTS`) -- is saved with the names;
  where none is saved it is the one the first recordings had
  (:data:`analysis.orientation.DEFAULT_PLACEMENT`).

Nothing here reads more of a recording than its eight header rows.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from analysis import config
from analysis.data_io import read_header
from analysis.orientation import DEFAULT_PLACEMENT, PLACEMENTS

NAMES = tuple(config.SENSOR_MAP)
"""Every name the analysis uses, in the order it lists its sensors.  A recording may lack some of them."""

UNUSED = None
"""What a label maps to when its sensor is left out of the analysis."""

SPELLED, READ_AS, SERIAL = "spelled", "read as", "serial"
"""How a suggestion was found, strongest first."""


# ---------------------------------------------------------------------------
# The labels in a recording
# ---------------------------------------------------------------------------


def read_labels(path: Path) -> list[str]:
    """The sensor labels in a Delsys export's fourth row, in file order, e.g. ``L_humerus (84483)``."""
    rows, _, _ = read_header(path)
    return [cell.strip() for cell in rows[3] if cell.strip()]


def split_label(label: str) -> tuple[str, str | None]:
    """``(name, serial number)`` of a label: ``"L_humerus (84483)"`` -> ``("L_humerus", "84483")``.

    A label that is only a number is taken for the serial number.
    """
    match = re.match(r"^(.*?)\s*\((\d+)\)\s*$", label)
    if match:
        return match.group(1).strip(), match.group(2)
    if label.strip().isdigit():
        return "", label.strip()
    return label.strip(), None


def spelled(name: str) -> str | None:
    """The analysis name ``name`` spells, ignoring case, spaces, dashes and underscores."""
    key = re.sub(r"[\s_\-.]+", "", name.lower())
    return key if key in NAMES else None


# The words a label may use for each segment, and whether the segment has a side.
SEGMENT_WORDS = {
    "chestbone": ({"chestbone", "chest", "sternum", "thorax", "breastbone"}, False),
    "lumbar": ({"lumbar", "pelvis", "sacrum", "sacral", "lowerback", "lowback", "lumbosacral", "waist"}, False),
    "thigh": ({"thigh", "femur", "upperleg"}, True),
    "tibia": ({"tibia", "shank", "shin", "lowerleg", "calf"}, True),
    "foot": ({"foot", "feet", "instep"}, True),
    "humerus": ({"humerus", "upperarm"}, True),
    "ulna": ({"ulna", "radius", "forearm", "lowerarm", "wrist"}, True),
    "hand": ({"hand", "palm"}, True),
}
SIDE_WORDS = {"l": {"l", "left", "lt", "lft"}, "r": {"r", "right", "rt", "rgt"}}


def read_as(name: str) -> str | None:
    """The analysis name a label's words amount to, or None: ``Left Upper Arm`` -> ``lhumerus``.

    A segment with a side needs the side in the label -- ``Thigh`` alone is
    none of the two.  Words are split at anything but letters and at a
    lower-to-upper case change, so ``RightShank``, ``R_shank`` and ``shank R``
    all read as ``rtibia``.
    """
    words = re.split(r"[^a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", name).lower())
    words = [word for word in words if word]
    side = next((key for key, options in SIDE_WORDS.items() if any(word in options for word in words)), None)
    rest = "".join(word for word in words if not any(word in options for options in SIDE_WORDS.values()))
    if side is None:  # a side written together with the segment: "lthigh", "rightfoot"
        for key, options in SIDE_WORDS.items():
            for prefix in sorted(options, key=len, reverse=True):
                if rest.startswith(prefix) and any(rest[len(prefix):] in words_ for words_, _ in SEGMENT_WORDS.values()):
                    side, rest = key, rest[len(prefix):]
                    break
            if side:
                break
    for stem, (vocabulary, sided) in SEGMENT_WORDS.items():
        if rest in vocabulary:
            if not sided:
                return stem
            return f"{side}{stem}" if side else None
    return None


# ---------------------------------------------------------------------------
# Suggesting and checking a mapping
# ---------------------------------------------------------------------------


def suggest(labels: list[str], history: dict[str, dict[str, list[str]]] | None = None
            ) -> dict[str, tuple[str | None, str, str]]:
    """``{label: (name or None, how, why)}``: a proposed name for every label.

    ``how`` is :data:`SPELLED`, :data:`READ_AS`, :data:`SERIAL` or ``""`` (no
    proposal); ``why`` says it in words.  ``history`` maps serial numbers to
    the names they have in other recordings (``{serial: {name: [files]}}``),
    and is used only for labels that name nothing.  Where two labels would get
    the same name, the stronger reason keeps it, and a tie gives it to neither.
    """
    history = history or {}
    proposals: dict[str, tuple[str | None, str, str]] = {}
    for label in labels:
        name, serial = split_label(label)
        if spelled(name):
            proposals[label] = (spelled(name), SPELLED, "the label spells it")
        elif read_as(name):
            proposals[label] = (read_as(name), READ_AS, f"“{name}” reads as {read_as(name)}")
        elif serial and len(history.get(serial, {})) == 1:
            (known, files), = history[serial].items()
            where = files[0] if len(files) == 1 else f"{files[0]} and {len(files) - 1} other" + "s" * (len(files) > 2)
            proposals[label] = (known, SERIAL, f"sensor {serial} is {known} in {where}")
        elif serial and len(history.get(serial, {})) > 1:
            names = ", ".join(sorted(history[serial]))
            proposals[label] = (None, "", f"sensor {serial} has had several names ({names})")
        else:
            proposals[label] = (None, "", "no name found in the label")

    strength = {SPELLED: 3, READ_AS: 2, SERIAL: 1}
    for name in NAMES:
        claims = [label for label, (proposed, _, _) in proposals.items() if proposed == name]
        if len(claims) < 2:
            continue
        best = max(strength[proposals[label][1]] for label in claims)
        winners = [label for label in claims if strength[proposals[label][1]] == best]
        for label in claims:
            if len(winners) > 1 or label not in winners:
                others = ", ".join(other for other in claims if other != label)
                proposals[label] = (None, "", f"would be {name}, as would {others}")
    return proposals


def automatic(labels: list[str]) -> dict[str, str]:
    """The names the labels spell: what is used without anyone checking it."""
    return {label: spelled(split_label(label)[0]) for label in labels if spelled(split_label(label)[0])}


def problems(names: dict[str, str | None], labels: list[str]) -> list[str]:
    """What keeps ``names`` (``{label: name or None for not used}``) from being used, in words.

    A name no sensor has is not a problem: the analysis runs without that
    sensor (:func:`absent`).
    """
    found = []
    undecided = [label for label in labels if label not in names]
    if undecided:
        found.append("no name chosen for " + ", ".join(undecided))
    for name in NAMES:
        holders = [label for label in labels if names.get(label) == name]
        if len(holders) > 1:
            found.append(f"{name} is given to {len(holders)} sensors: " + ", ".join(holders))
    if labels and not undecided and not any(names.get(label) in NAMES for label in labels):
        found.append("no sensor has one of the analysis's names")
    return found


def absent(names: dict[str, str | None], labels: list[str]) -> list[str]:
    """The analysis names no sensor has, in the order it lists them: what the analysis will run without."""
    given = {names.get(label) for label in labels}
    return [name for name in NAMES if name not in given]


def used(names: dict[str, str | None]) -> dict[str, str]:
    """``{label: name}`` for the labels that map to one of the analysis names."""
    return {label: name for label, name in names.items() if name in NAMES}


def placements(names: dict[str, str | None], chosen: dict[str, str] | None = None) -> dict[str, str]:
    """``{name: placement}`` for the limb sensors among ``names``: the one chosen for its label, else the default."""
    chosen = chosen or {}
    return {name: chosen.get(label) or DEFAULT_PLACEMENT[name]
            for label, name in used(names).items() if name in DEFAULT_PLACEMENT}


# ---------------------------------------------------------------------------
# The saved mapping
# ---------------------------------------------------------------------------


def load(path: Path) -> dict[str, str | None] | None:
    """The saved ``{label: name or None}``, or None if nothing has been saved."""
    try:
        saved = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    sensors = saved.get("sensors")
    if not isinstance(sensors, dict):
        return None
    return {label: (name if name in NAMES else UNUSED) for label, name in sensors.items()}


def load_placements(path: Path) -> dict[str, str]:
    """The saved ``{label: placement}``; empty if none is saved, as in files written before placements were."""
    try:
        saved = json.loads(path.read_text()).get("placements")
    except (OSError, json.JSONDecodeError, AttributeError):
        return {}
    if not isinstance(saved, dict):
        return {}
    return {label: placement for label, placement in saved.items() if placement in PLACEMENTS}


def save(path: Path, names: dict[str, str | None], labels: list[str], source: dict,
         chosen: dict[str, str] | None = None) -> None:
    """Write ``sensor_names.json``: the checked names, in the file's order, and the file they belong to.

    ``chosen`` is ``{label: placement}``; every limb sensor's placement is
    written, the default where none was chosen.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    placed = placements(names, chosen)
    path.write_text(json.dumps({
        "recording": source,
        "saved_utc": now.isoformat(timespec="seconds"),
        "saved_local": now.astimezone().strftime("%d %b %Y %H:%M"),
        "sensors": {label: names.get(label) for label in labels if label in names},
        "placements": {label: placed[names[label]] for label in labels if names.get(label) in placed},
    }, indent=2))
