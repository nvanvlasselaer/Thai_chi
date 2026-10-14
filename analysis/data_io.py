"""Reading the Delsys recordings, and reading and writing the derived data files.

* **Raw input** -- the Delsys Trigno Discover CSV export, parsed into a
  :class:`TrialData` holding one DataFrame per sensor.
* **Orientation cache** -- ``orientation.npz`` in each recording's output folder
  (:mod:`analysis.recordings`) holds the body-frame quaternions.  The Madgwick filter that produces them is the only
  slow step of the pipeline, so everything else rebuilds its kinematics from
  this file (:func:`analysis.kinematics.load_trial`).
* **Exports** -- the per-sensor inventory and the 50 Hz kinematic-variable CSV.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from analysis import config
from analysis.signals import highpass_detrend, resample_filtered_full

if TYPE_CHECKING:
    from analysis.kinematics import Kinematics

ACC_COLUMNS = ["acc_x_g", "acc_y_g", "acc_z_g"]
GYRO_COLUMNS = ["gyro_x_dps", "gyro_y_dps", "gyro_z_dps"]


# ---------------------------------------------------------------------------
# Raw recordings
# ---------------------------------------------------------------------------


@dataclass
class TrialData:
    label: str
    path: Path
    duration_meta_s: float
    fs: float
    sensors: list[str]
    data: dict[str, pd.DataFrame]
    recorded_as: dict[str, str] = field(default_factory=dict)
    """Each sensor's label in the file, by the name the analysis gives it."""
    placements: dict[str, str] = field(default_factory=dict)
    """Where each limb sensor sits on its segment, by name; one not in it sits where
    :data:`analysis.orientation.DEFAULT_PLACEMENT` says."""


HEADER_ROWS = 8
"""Rows of metadata above the samples in a Delsys export."""
SEPARATORS = (",", ";", "\t")


def read_header(path: Path) -> tuple[list[list[str]], str, str]:
    """The header rows of a Delsys export split into cells, and its ``(separator, decimal mark)``.

    Trigno Discover writes numbers the way the computer's language settings
    do: ``0.0027, 0.1245`` on an English one, ``0,0027; 0,1245`` on a Dutch or
    French one.  The separator is the one the channel-name row (row 6) is split
    by, since its cells hold none of them; the decimal mark is a comma only
    where the separator is not, and the duration or sampling rate is written
    with one.
    """
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        lines = [handle.readline() for _ in range(HEADER_ROWS)]
    separator = max(SEPARATORS, key=lines[5].count)
    reader = csv.reader(lines, delimiter=separator)
    rows = [next(reader, []) for _ in range(HEADER_ROWS)]
    decimal = "," if separator != "," and re.search(r"\d,\d", lines[2] + lines[6]) else "."
    return rows, separator, decimal


def header_number(text: str, decimal: str) -> float:
    """A number written in a header cell, ``"370,3704 Hz"`` or ``"370.3704 Hz"`` -> 370.3704."""
    return float(re.sub(r"[^\d.\-]", "", text.replace(decimal, ".")))


def canonical_sensor(name: str) -> str:
    stem = re.sub(r"\s*\(\d+\)\s*$", "", name.strip())
    return stem.lower().replace("_", "")


def parse_IMU_csv(path: Path, label: str, names: dict[str, str | None] | None = None,
                  placements: dict[str, str] | None = None) -> TrialData:
    """Parse a Delsys Trigno Discover CSV export.

    Eight metadata rows, then one 12-column block per sensor (ACC X/Y/Z and
    GYRO X/Y/Z, each with its own time column), with commas or semicolons
    between the values (:func:`read_header`).  Sensor order differs between
    recordings, so blocks are matched by name rather than position.

    ``names`` maps each sensor label in the file (``L_humerus (84483)``) to the
    name the analysis uses (:mod:`analysis.sensor_names`); a sensor it maps to
    None, or leaves out, is not read.  Without it every label is reduced to
    its lower-case letters, which is what the labels of the first recordings
    spell.  ``placements`` (``{name: placement}``) says where the limb sensors
    sit, which the mounting of each depends on; it is kept with the data.
    """
    rows, separator, decimal = read_header(path)

    duration_meta_s = header_number(rows[2][1], decimal)
    sensor_row = rows[3]
    channel_row = rows[5]
    fs_row = rows[6]

    sensor_starts = [i for i, value in enumerate(sensor_row) if value.strip()]
    labels = [sensor_row[i].strip() for i in sensor_starts]
    if names is None:
        blocks = [(index, canonical_sensor(text), text) for index, text in enumerate(labels)]
    else:
        blocks = [(index, names[text], text) for index, text in enumerate(labels) if names.get(text)]
    n_sensor_cols = len(labels) * 12

    block_fs = []
    for start in sensor_starts:
        block_fs.append(header_number(fs_row[start + 1], decimal))
    fs = float(np.median(block_fs))

    raw = pd.read_csv(path, skiprows=HEADER_ROWS, header=None, sep=separator, decimal=decimal,
                      usecols=range(n_sensor_cols), low_memory=False)
    raw = raw.apply(pd.to_numeric, errors="coerce")

    data: dict[str, pd.DataFrame] = {}
    for block_index, sensor, _ in blocks:
        start = block_index * 12
        cols = raw.iloc[:, start : start + 12].copy()
        names = channel_row[start : start + 12]
        channel_map = {}
        for idx, name in enumerate(names):
            cleaned = name.strip().lower()
            if "acc x time" in cleaned:
                channel_map["time"] = cols.iloc[:, idx]
            elif cleaned == "acc x (g)":
                channel_map["acc_x_g"] = cols.iloc[:, idx]
            elif cleaned == "acc y (g)":
                channel_map["acc_y_g"] = cols.iloc[:, idx]
            elif cleaned == "acc z (g)":
                channel_map["acc_z_g"] = cols.iloc[:, idx]
            elif cleaned == "gyro x (deg/s)":
                channel_map["gyro_x_dps"] = cols.iloc[:, idx]
            elif cleaned == "gyro y (deg/s)":
                channel_map["gyro_y_dps"] = cols.iloc[:, idx]
            elif cleaned == "gyro z (deg/s)":
                channel_map["gyro_z_dps"] = cols.iloc[:, idx]

        sensor_df = pd.DataFrame(channel_map).dropna(how="all")
        sensor_df = sensor_df.dropna(subset=["time"])
        data[sensor] = sensor_df.reset_index(drop=True)

    return TrialData(label=label, path=path, duration_meta_s=duration_meta_s, fs=fs,
                     sensors=[sensor for _, sensor, _ in blocks], data=data,
                     recorded_as={sensor: text for _, sensor, text in blocks},
                     placements={sensor: placement for sensor, placement in (placements or {}).items()
                                 if sensor in data})


def make_sensor_inventory(trials: list[TrialData]) -> pd.DataFrame:
    rows = []
    required = ACC_COLUMNS + GYRO_COLUMNS
    for trial in trials:
        for sensor in sorted(trial.sensors):
            df = trial.data[sensor]
            time = df["time"].to_numpy()
            diffs = np.diff(time)
            rows.append(
                {
                    "recording": trial.label,
                    "sensor": sensor,
                    "recorded_as": trial.recorded_as.get(sensor, sensor),
                    "body_segment": config.SENSOR_MAP.get(sensor, "Unmapped"),
                    "samples": len(df),
                    "duration_s": round(float(time[-1] - time[0]), 3),
                    "fs_hz": round(trial.fs, 4),
                    "median_dt_s": round(float(np.nanmedian(diffs)), 6),
                    "max_dt_s": round(float(np.nanmax(diffs)), 6),
                    "missing_channel_values": int(df[required].isna().sum().sum()),
                    "complete_accel_gyro": all(col in df.columns for col in required),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Orientation cache
# ---------------------------------------------------------------------------


def save_orientation_npz(path: Path, kin: Kinematics) -> None:
    payload = {"time_s": kin.t}
    for sensor, q in kin.quaternions.items():
        payload[f"{sensor}_q_wxyz"] = q
    np.savez_compressed(path, **payload)


def load_orientation_npz(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Time stamps and per-sensor wxyz quaternions from an orientation cache."""
    with np.load(path) as data:
        time_s = data["time_s"]
        quats = {key[: -len("_q_wxyz")]: data[key] for key in data.files if key.endswith("_q_wxyz")}
    return time_s, quats


# ---------------------------------------------------------------------------
# 50 Hz kinematic-variable export
# ---------------------------------------------------------------------------


EXPORTED_JOINTS = (
    "left_knee", "right_knee", "left_ankle", "right_ankle", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_hip", "right_hip",
)
"""Joint angles in the export, in column order.  Each contributes x, y and z."""

def save_kinematic_variables(path: Path, kin: Kinematics, fs: float, target_fs: float = 50.0) -> None:
    """Write the trunk angles, angular speeds and every joint angle at 50 Hz.

    Each channel is anti-alias filtered and resampled.  Yaw and the axial (z)
    component of every joint angle are high-pass detrended first to remove the
    heading drift of a 6-axis estimate; tilt and joint flexion (x) and y
    components are gravity-referenced and exported as they are, and the
    angular speeds have the gyroscope bias removed.  Axes: x = mediolateral,
    y = anteroposterior, z = vertical (:func:`analysis.orientation.mounting_matrix`).

    A channel whose sensor the recording lacks is left empty (NaN), so every
    export has the same columns.
    """

    def detrended(signal: np.ndarray) -> np.ndarray:
        return highpass_detrend(signal, fs, cutoff_hz=0.05)

    empty = np.full(len(kin.t), np.nan)
    eulers = {sensor: kin.eulers_deg.get(sensor, np.full((len(kin.t), 3), np.nan)) for sensor in ("lumbar", "chestbone")}
    omega = {sensor: kin.omega_mag.get(sensor, empty) for sensor in ("lumbar", "chestbone")}
    channels = {
        "trunk_pelvis_x_deg": kin.trunk_rel_euler_deg[:, 0],
        "trunk_pelvis_y_deg": kin.trunk_rel_euler_deg[:, 1],
        "trunk_pelvis_z_deg": detrended(kin.trunk_rel_euler_deg[:, 2]),
        "lumbar_x_deg": eulers["lumbar"][:, 0],
        "lumbar_y_deg": eulers["lumbar"][:, 1],
        "lumbar_z_deg": detrended(eulers["lumbar"][:, 2]),
        "chest_z_deg": detrended(eulers["chestbone"][:, 2]),
        "lumbar_omega_dps": omega["lumbar"],
        "chest_omega_dps": omega["chestbone"],
    }
    for joint in EXPORTED_JOINTS:
        angles = getattr(kin, f"{joint}_deg")
        for axis, name in enumerate("xyz"):
            channels[f"{joint}_{name}_est_deg"] = detrended(angles[:, axis]) if name == "z" else angles[:, axis]

    resampled = {name: resample_filtered_full(kin.t, signal, fs, target_fs) for name, signal in channels.items()}
    time_s = resampled["trunk_pelvis_x_deg"][0]
    df = pd.DataFrame({"time_s": time_s, **{name: values for name, (_, values) in resampled.items()}})
    df.to_csv(path, index=False)
