"""Recalculate every metric from the curated windows, and write the outputs.

Overwrites the event-window CSVs, the metric CSVs and the traceability figures
in the active analysis folder, and saves the session they were computed from
alongside them.  Every table has a ``recording`` column naming the source file
of each row, and every figure names its source files underneath, so a file
copied out of its folder still says where it came from.

Each recalculation also writes ``metrics_provenance.json``, recording a digest
of the windows and recordings it used, so the dashboard can tell whether the
metrics on disk still match the session or something has changed since.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import config, detection, recordings, smoothness_metrics
from analysis.balance_metrics import (
    compute_asymmetry_metrics,
    compute_knee_balance_metrics,
    compute_trunk_rotation_balance_metrics,
    summarize_knee_flexion_event,
)
from analysis.config import TRIALS
from analysis.figures import (
    make_knee_flexion_overview_figure,
    make_knee_traceability_figure,
    make_sequence_smoothness_figure,
    make_trunk_traceability_figure,
)
from analysis.kinematics import LoadedTrial
from analysis.sessions import FAMILIES, FAMILY_KEY, SESSION_FILENAME, refresh_seconds, roles, save_session
from analysis.signals import idx_to_sec
from analysis.validation import validate_session

PROVENANCE_FILENAME = "metrics_provenance.json"


@dataclass
class RecomputeResult:
    trunk_metrics: pd.DataFrame
    knee_metrics: pd.DataFrame
    asymmetry: pd.DataFrame
    smooth_metrics: pd.DataFrame
    variability: pd.DataFrame
    written: list[str]
    log: list[str]


LAG_BOUND_S = 2.0
"""cross_correlation_lag's max_lag_s; a result at the bound is not a measurement."""


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

    Works with one recording selected as well as two: every family is computed
    for whichever roles are loaded.  The outputs go to the active analysis
    folder (:func:`analysis.recordings.analysis_dir`).
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
    _, _, session_knee_params, _ = detection.params_from_dict(session.get("detector", {}) or {})
    knee_threshold = session_knee_params.threshold_deg
    knee_lag = session.get("knee_lag_window_s")
    knee_lag_kwargs = (
        {"lag_pre_s": float(knee_lag[0]), "lag_post_s": float(knee_lag[1])} if knee_lag else {}
    )

    labels = roles(trials)
    # The trunk-rotation peak is reported for the first selected recording --
    # the novice when both are -- which is the recording v1 matched the other to.
    reference = labels[0]

    trunk_rows: list[dict] = []
    trunk_windows: dict[str, list[tuple]] = {label: [] for label in labels}

    enabled_trunk = [e for e in session.get("trunk_events", []) if e.get("enabled", True)]
    for event in enabled_trunk:
        for label in labels:
            window = event["windows"].get(label)
            if window is None:
                continue
            loaded = trials[label]
            start, end = window["event_start"], window["event_end"]
            stab_start, stab_end = window["stab_start"], window["stab_end"]

            peak = detection.recompute_trunk_peak(loaded.kin, loaded.fs, start, end) if label == reference else np.nan
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
                    "event_peak_s": idx_to_sec(peak, fs) if label == reference else np.nan,
                    "stabilization_start_s": idx_to_sec(stab_start, fs),
                    "stabilization_end_s": idx_to_sec(stab_end, fs),
                }
            )

    trunk_metrics = pd.DataFrame(trunk_rows)
    log.append(f"Trunk rotation: {len(enabled_trunk)} events x {len(labels)} recording(s) = {len(trunk_rows)} metric rows")

    knee_rows: list[dict] = []
    knee_event_rows: list[dict] = []
    knee_events: dict[str, list[dict]] = {label: [] for label in labels}
    stab_overrides: dict[str, list[tuple[int, int]]] = {label: [] for label in labels}

    enabled_knee = [e for e in session.get("knee_events", []) if e.get("enabled", True)]
    for index, event in enumerate(enabled_knee, 1):
        side = event["flexed_leg"]
        for label in labels:
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

    smooth_rows: list[dict] = []
    waveform_segments: dict[str, list[np.ndarray]] = {label: [] for label in labels}
    enabled_smooth = [e for e in session.get("smooth_events", []) if e.get("enabled", True)]
    # The detrended yaw traces are the same for every segment of a recording, so
    # they are filtered once here rather than inside each of ~26 metric calls.
    detrended = {label: smoothness_metrics.detrended_yaw(trials[label].kin, trials[label].fs)
                 for label in labels}
    for event in enabled_smooth:
        for label in labels:
            window = event["windows"].get(label)
            if window is None:
                continue  # single-sided segment
            loaded = trials[label]
            start, end = window["event_start"], window["event_end"]
            smooth_rows.append(
                smoothness_metrics.compute_sequence_smoothness_metrics(
                    label, loaded.kin, loaded.fs, event["event_id"], start, end,
                    lag_pad_s=lag_pad_s, yaw=detrended[label],
                )
            )
            waveform_segments[label].append(detrended[label][0][start:end])

    smooth_metrics = pd.DataFrame(smooth_rows)
    waveforms = {label: smoothness_metrics.waveform_consistency(segments)
                 for label, segments in waveform_segments.items()}
    variability = (
        smoothness_metrics.summarize_sequence_variability(smooth_metrics, waveforms)
        if len(smooth_metrics) else pd.DataFrame()
    )
    log.append(f"Sequence smoothness: {len(enabled_smooth)} segments")

    for note in flag_saturated_lags(smooth_metrics, ["chest_pelvis_lag_s"]):
        log.append("WARNING  " + note)
    for note in flag_saturated_lags(trunk_metrics, ["trunk_pelvis_lag_s"]):
        log.append("WARNING  " + note)
    for note in flag_saturated_lags(knee_metrics, ["trunk_pelvis_lag_s", "trunk_pelvis_pitch_lag_s"]):
        log.append("WARNING  " + note)

    sources = {label: trials[label].recording.name for label in labels}
    trunk_metrics, knee_metrics, asymmetry, smooth_metrics, variability = (
        with_recording(frame, sources) for frame in (trunk_metrics, knee_metrics, asymmetry, smooth_metrics, variability)
    )

    written: list[str] = []
    if write:
        folder = recordings.analysis_dir()
        folder.mkdir(parents=True, exist_ok=True)

        def write_csv(frame: pd.DataFrame, name: str) -> None:
            if len(frame) == 0:
                log.append(f"skipped {name} (no rows)")
                return
            frame.to_csv(folder / name, index=False)
            written.append(name)

        write_csv(with_recording(pd.DataFrame(trunk_event_rows), sources), "trunk_rotation_event_windows.csv")
        write_csv(trunk_metrics, "trunk_rotation_balance_metrics.csv")
        write_csv(with_recording(pd.DataFrame(knee_event_rows), sources), "monopodal_stance_event_windows.csv")
        write_csv(knee_metrics, "monopodal_stance_balance_metrics.csv")
        write_csv(asymmetry, "monopodal_stance_asymmetry_metrics.csv")
        write_csv(smooth_metrics, "sequence_smoothness_metrics.csv")
        write_csv(variability, "sequence_variability_summary.csv")

        kins = {label: trials[label].kin for label in labels}
        fs_of = {label: trials[label].fs for label in labels}
        if len(trunk_metrics) > 0:
            make_trunk_traceability_figure(kins, fs_of, trunk_metrics, trunk_windows, sources, folder)
            written.append("trunk_traceability_figure.png")
        if len(knee_metrics) > 0:
            make_knee_flexion_overview_figure(kins, fs_of, knee_events, sources, folder)
            written.append("monopodal_stance_overview_figure.png")
            make_knee_traceability_figure(kins, fs_of, knee_metrics, knee_events, sources, folder,
                                          stab_overrides=stab_overrides)
            written.append("monopodal_stance_traceability_figure.png")
        if len(smooth_metrics) > 0:
            make_sequence_smoothness_figure(detrended, fs_of, smooth_metrics, waveforms, sources, folder)
            written.append("sequence_smoothness_figure.png")

        save_session(refresh_seconds(session, trials))
        written.append(SESSION_FILENAME)
        write_provenance(session, written)

    return RecomputeResult(
        trunk_metrics=trunk_metrics,
        knee_metrics=knee_metrics,
        asymmetry=asymmetry,
        smooth_metrics=smooth_metrics,
        variability=variability,
        written=written,
        log=log,
    )


def with_recording(frame: pd.DataFrame, sources: dict[str, str]) -> pd.DataFrame:
    """``frame`` with a ``recording`` column, the source file of each row, after ``trial``."""
    if len(frame) == 0 or "trial" not in frame or "recording" in frame:
        return frame
    frame = frame.copy()
    frame.insert(frame.columns.get_loc("trial") + 1, "recording", frame["trial"].map(sources))
    return frame


# ---------------------------------------------------------------------------
# Provenance: which windows the metrics on disk were computed from
# ---------------------------------------------------------------------------


WINDOW_KEYS = ("event_start", "event_end", "stab_start", "stab_end")


def session_digest(session: dict) -> str:
    """Fingerprint of everything in a session that changes a metric.

    The window indices, which events are enabled, the metric options, the knee
    threshold, the high-pass switch in force, and the content of the selected
    recording files -- but not names, timestamps, the trash or the display-only
    seconds, so renaming or deleting-then-restoring does not mark the metrics out
    of date.
    """
    events = {
        family: [
            {
                "event_id": event.get("event_id"),
                "enabled": event.get("enabled", True),
                "flexed_leg": event.get("flexed_leg"),
                "windows": {
                    label: {key: window[key] for key in WINDOW_KEYS if key in window}
                    for label, window in sorted(event.get("windows", {}).items())
                },
            }
            for event in session.get(FAMILY_KEY[family], [])
        ]
        for family in FAMILIES
    }
    relevant = {
        "events": events,
        "lag_pad_s": session.get("lag_pad_s"),
        "knee_lag_window_s": session.get("knee_lag_window_s"),
        "knee_threshold_deg": detection.params_from_dict(session.get("detector", {}) or {})[2].threshold_deg,
        "ignore_high_pass_filter": bool(config.IGNORE_HIGH_PASS_FILTER),
        "recordings": {role: recording.sha256() if recording else None
                       for role, recording in recordings.active().recordings().items()},
    }
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:16]


def provenance_path() -> Path:
    return recordings.analysis_dir() / PROVENANCE_FILENAME


def write_provenance(session: dict, written: list[str]) -> None:
    record = {
        "computed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "analysis": recordings.active().id,
        "recordings": {role: recording.identity()
                       for role, recording in recordings.active().recordings().items() if recording},
        "session_name": session.get("session_name"),
        "session_digest": session_digest(session),
        "ignore_high_pass_filter": bool(config.IGNORE_HIGH_PASS_FILTER),
        "files": written,
    }
    provenance_path().write_text(json.dumps(record, indent=2))


def read_provenance() -> dict | None:
    if not provenance_path().exists():
        return None
    try:
        return json.loads(provenance_path().read_text())
    except json.JSONDecodeError:
        return None
