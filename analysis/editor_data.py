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
) -> dict:
    """Build a fresh session by running the detectors on both recordings."""
    trunk_params = trunk_params or ed.TrunkDetectorParams()
    stab_params = stab_params or ed.StabilizationParams()

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

    knee_events = seed_knee_events(trials, stab_params)

    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "modified_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fs": {label: trial.fs for label, trial in trials.items()},
        "n_samples": {label: trial.n_samples for label, trial in trials.items()},
        "ignore_high_pass_filter": _ignore_high_pass_flag(),
        "detector": ed.params_to_dict(trunk_params, stab_params),
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


def seed_knee_events(
    trials: dict[str, LoadedTrial], stab_params: ed.StabilizationParams | None = None
) -> list[dict]:
    """Detect monopodal-stance events by knee-flexion threshold, per leg per trial."""
    stab_params = stab_params or ed.StabilizationParams()
    events: list[dict] = []
    counter = 1
    for label in TRIALS:
        loaded = trials[label]
        omega = ed.combined_omega(loaded.kin, loaded.fs, stab_params)
        for side, angle in [
            ("Left", loaded.kin.left_knee_deg[:, 0]),
            ("Right", loaded.kin.right_knee_deg[:, 0]),
        ]:
            windows, _ = find_knee_flexion_windows(
                angle, loaded.fs, threshold_deg=KNEE_THRESHOLD_DEG, min_duration_s=0.4, merge_gap_s=0.2
            )
            for start, end in windows:
                knee_abs = np.abs(lowpass_signal(angle, loaded.fs, cutoff_hz=6.0))
                peak_idx = start + int(np.argmax(knee_abs[start:end]))
                stab = ed.find_stabilization(
                    loaded.kin, loaded.fs, peak_idx, stab_params, diagnostics=omega
                )
                events.append(
                    {
                        "event_id": f"knee-{counter:02d}",
                        "trial": label,
                        "flexed_leg": side,
                        "stance_leg": "Right" if side == "Left" else "Left",
                        "enabled": True,
                        **_window_dict(start, end, stab.start, stab.end, fs=loaded.fs,
                                       stab_quiet_ratio=round(stab.quiet_ratio, 3)),
                    }
                )
                counter += 1
    events.sort(key=lambda e: (e["trial"], e["event_start"]))
    for i, event in enumerate(events, 1):
        event["event_id"] = f"knee-{i:02d}"
    return events


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
        fs = trials[event["trial"]].fs
        for key in ("event_start", "event_end", "stab_start", "stab_end"):
            event[f"{key}_s"] = round(idx_to_sec(event[key], fs), 4)
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
        loaded = trials[event["trial"]]
        issues += validate_window(event["event_id"], event["trial"], event, loaded.fs, loaded.n_samples)
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
    for event in enabled_knee:
        label = event["trial"]
        loaded = trials[label]
        start, end = event["event_start"], event["event_end"]
        stab_start, stab_end = event["stab_start"], event["stab_end"]
        side = event["flexed_leg"]

        summary = summarize_knee_flexion_event(
            label, loaded.kin, loaded.fs, side, start, end, threshold_deg=KNEE_THRESHOLD_DEG
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
                "flexed_leg": side,
                "stance_leg": summary["stance_leg"],
                "window_start_s": summary["window_start_s"],
                "window_end_s": summary["window_end_s"],
                "peak_time_s": summary["peak_time_s"],
                "peak_abs_knee_flexion_deg": summary["peak_abs_knee_flexion_deg"],
            }
        )

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
    """Rebuild a session from the committed automatic windows.

    Used to reproduce the pre-existing results exactly, which is how the editor
    is verified: seeding from these and recalculating without edits must leave
    the output CSVs byte-identical.
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
        # The v1 CSV carries no stabilization columns because the pipeline
        # derives them inside compute_knee_balance_metrics from the peak.
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
