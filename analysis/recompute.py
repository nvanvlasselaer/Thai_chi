"""Recalculate every metric from the curated windows, and write the outputs.

Overwrites the event-window CSVs, the metric CSVs and the traceability figures
in the active analysis folder, and saves the session they were computed from
alongside them.  Every table has ``recording`` and ``metrics_version`` columns
naming the source file of each row and the metric definitions it was computed
with, and every figure names its source files underneath, so a file copied out
of its folder still says where it came from.

Each recalculation also writes ``metrics_provenance.json``, recording a digest
of the windows and recordings it used, the gyroscope bias it removed and the
alignment the pairs were checked against, so the dashboard can tell whether
the metrics on disk still match the session or something has changed since.

When a selected recording lacks a sensor, ``sensor_coverage.csv`` says, per
recording and metric, whether it was measured, measured with a stand-in or not
calculated, and where windows were carried over from the other recording
(:mod:`analysis.coverage`).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import alignment, config, coverage, detection, recordings, smoothness_metrics
from analysis.balance_metrics import (
    balance_signals,
    compute_asymmetry_metrics,
    compute_single_leg_metrics,
    compute_trunk_rotation_balance_metrics,
    summarize_single_leg_event,
)
from analysis.comparison import paired_comparison
from analysis.config import TRIALS
from analysis.figures import (
    make_centre_of_mass_figure,
    make_knee_flexion_overview_figure,
    make_knee_traceability_figure,
    make_sequence_smoothness_figure,
    make_trunk_traceability_figure,
)
from analysis.kinematics import LoadedTrial
from analysis.sessions import CARRIED_OVER, FAMILIES, FAMILY_KEY, SESSION_FILENAME, refresh_seconds, roles, save_session
from analysis.signals import idx_to_sec
from analysis.validation import validate_session

PROVENANCE_FILENAME = "metrics_provenance.json"
COVERAGE_FILENAME = "sensor_coverage.csv"

RETIRED_OUTPUTS = ("sequence_variability_summary.csv",)
"""Files earlier metric versions wrote that nothing writes any more; removed on
recalculation so a stale copy cannot be read as current."""

OUTPUTS = (
    "trunk_rotation_event_windows.csv", "trunk_rotation_balance_metrics.csv",
    "monopodal_stance_event_windows.csv", "monopodal_stance_balance_metrics.csv",
    "monopodal_stance_asymmetry_metrics.csv", "sequence_smoothness_metrics.csv", "sequence_summary.csv",
    "paired_comparison.csv", "recording_alignment.csv", COVERAGE_FILENAME,
    "trunk_traceability_figure.png", "monopodal_stance_overview_figure.png",
    "monopodal_stance_traceability_figure.png", "sequence_smoothness_figure.png", "centre_of_mass_figure.png",
)
"""Every table and figure a recalculation may write.  One it does not write this time -- a family
with no events, a figure with nothing to draw -- is removed, so an earlier copy cannot pass for
current."""

DEFAULT_KNEE_LAG_WINDOW_S = (1.0, 2.0)
"""The single-leg lag window of sessions that leave it unset (v1)."""


@dataclass
class RecomputeResult:
    trunk_metrics: pd.DataFrame
    knee_metrics: pd.DataFrame
    asymmetry: pd.DataFrame
    smooth_metrics: pd.DataFrame
    summary: pd.DataFrame
    paired: pd.DataFrame
    written: list[str]
    log: list[str]


LAG_COLUMNS = {
    "trunk": ["trunk_pelvis_lag_s"],
    "knee": ["trunk_pelvis_lag_s", "trunk_pelvis_frontal_lag_s"],
    "smooth": ["chest_pelvis_lag_s"],
}


def flag_missing_lags(metrics: pd.DataFrame, columns: list[str], unmeasured: dict[str, set] | None = None) -> list[str]:
    """Rows whose lag is NaN: the correlation peaked on the edge of the search.

    ``unmeasured`` gives per recording the columns its sensors cannot provide;
    a lag that is NaN for that reason is in ``sensor_coverage.csv`` instead.
    """
    notes = []
    for column in columns:
        if column not in metrics:
            continue
        missing = metrics[metrics[column].isna()]
        for _, row in missing.iterrows():
            if column in (unmeasured or {}).get(row["trial"], set()):
                continue
            notes.append(
                f"{row['trial']} {row.get('event_id', '')}: {column} has no interior peak within the "
                "+/-1 s search, so no lag is reported"
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

    # Velocity-segmented events are often shorter than the lag search, so the
    # lag gets a padded window.  v1 sessions leave this None.
    lag_pad_s = session.get("lag_pad_s")
    _, stab_params, _, _ = detection.params_from_dict(session.get("detector", {}) or {})
    knee_lag = tuple(session.get("knee_lag_window_s") or DEFAULT_KNEE_LAG_WINDOW_S)

    labels = roles(trials)
    # The trunk-rotation peak is reported for the first selected recording --
    # the novice when both are -- which is the recording v1 matched the other to.
    reference = labels[0]
    signals = {label: balance_signals(trials[label].kin, trials[label].trial, trials[label].fs, stab_params)
               for label in labels}
    for label in labels:
        for warning in trials[label].kin.calibration.warnings():
            log.append(f"WARNING  {label}: {warning}")
        for line in coverage.summary(trials[label].kin.sensors):
            log.append(f"note  {label}: {line}")
    carried = carried_over(session, labels)
    for label, families in carried.items():
        for family, note in families.items():
            log.append(f"note  {label}: {coverage.FAMILY_NAMES[family]} -- {note}; check them in the event editor")

    # --- trunk rotation ----------------------------------------------------
    trunk_rows: list[dict] = []
    trunk_windows: dict[str, list[tuple]] = {label: [] for label in labels}
    trunk_event_rows: list[dict] = []
    enabled_trunk = [e for e in session.get("trunk_events", []) if e.get("enabled", True)]
    for event in enabled_trunk:
        for label in labels:
            window = event["windows"].get(label)
            if window is None:
                continue
            loaded = trials[label]
            start, end = window["event_start"], window["event_end"]
            stab_start, stab_end = window["stab_start"], window["stab_end"]
            peak = (detection.recompute_trunk_peak(loaded.kin, loaded.fs, start, end)
                    if label == reference and loaded.kin.has("lumbar", "chestbone") else np.nan)
            trunk_windows[label].append((start, end, stab_start, stab_end, peak))
            trunk_event_rows.append({
                "trial": label,
                "event_id": event["event_id"],
                "event_start_s": idx_to_sec(start, loaded.fs),
                "event_end_s": idx_to_sec(end, loaded.fs),
                "event_peak_s": idx_to_sec(peak, loaded.fs) if label == reference else np.nan,
                "stabilization_start_s": idx_to_sec(stab_start, loaded.fs),
                "stabilization_end_s": idx_to_sec(stab_end, loaded.fs),
            })
            trunk_rows.append(compute_trunk_rotation_balance_metrics(
                label, loaded.fs, event["event_id"], start, end, stab_start, stab_end, signals[label],
                lag_pad_s=lag_pad_s,
            ))
    trunk_event_rows.sort(key=lambda r: TRIALS.index(r["trial"]))
    trunk_metrics = pd.DataFrame(trunk_rows)
    log.append(f"Trunk rotation: {len(enabled_trunk)} events x {len(labels)} recording(s) = {len(trunk_rows)} metric rows")

    # --- single-leg stance -------------------------------------------------
    knee_rows: list[dict] = []
    knee_event_rows: list[dict] = []
    knee_events: dict[str, list[dict]] = {label: [] for label in labels}
    enabled_knee = [e for e in session.get("knee_events", []) if e.get("enabled", True)]
    for event in enabled_knee:
        leg = event.get("lifted_leg", event.get("flexed_leg"))
        for label in labels:
            window = event["windows"].get(label)
            if window is None:
                continue  # single-sided event: this recording contributes nothing
            loaded = trials[label]
            start, end = window["event_start"], window["event_end"]
            stab_start, stab_end = window["stab_start"], window["stab_end"]
            summary = summarize_single_leg_event(label, loaded.kin, loaded.fs, event["event_id"], leg,
                                                 start, end, signals[label]["lift"])
            summary["stabilization_start_s"] = idx_to_sec(stab_start, loaded.fs)
            summary["stabilization_end_s"] = idx_to_sec(stab_end, loaded.fs)
            knee_events[label].append(summary)
            knee_event_rows.append(dict(summary))
            knee_rows.append(compute_single_leg_metrics(
                label, loaded.kin, loaded.fs, event["event_id"], leg, start, end, stab_start, stab_end,
                signals[label], lag_window_s=knee_lag,
            ))
    knee_event_rows.sort(key=lambda r: (TRIALS.index(r["trial"]), r["window_start_s"]))
    knee_metrics = pd.DataFrame(knee_rows)
    asymmetry = compute_asymmetry_metrics(knee_metrics, [
        {**e, "lifted_leg": e.get("lifted_leg", e.get("flexed_leg"))} for e in enabled_knee
    ])
    log.append(f"Single-leg stance: {len(enabled_knee)} events")

    # --- turns ---------------------------------------------------------------
    smooth_rows: list[dict] = []
    progress: dict[str, list[np.ndarray]] = {label: [] for label in labels}
    enabled_smooth = [e for e in session.get("smooth_events", []) if e.get("enabled", True)]
    for event in enabled_smooth:
        for label in labels:
            window = event["windows"].get(label)
            if window is None:
                continue  # single-sided turn
            loaded = trials[label]
            start, end = window["event_start"], window["event_end"]
            smooth_rows.append(smoothness_metrics.compute_turn_metrics(
                label, loaded.fs, event["event_id"], start, end, signals[label],
                direction=event.get("direction"), lag_pad_s=lag_pad_s,
            ))
            progress[label].append(smoothness_metrics.turn_progress(signals[label], loaded.fs, start, end))
    smooth_metrics = pd.DataFrame(smooth_rows)
    corridors = {label: smoothness_metrics.turn_shape_corridor(progress[label]) for label in labels}
    summary = smoothness_metrics.sequence_summary(smooth_metrics) if len(smooth_metrics) else pd.DataFrame()
    log.append(f"Sequence turns: {len(enabled_smooth)} turns")

    paired = paired_comparison({"trunk": trunk_metrics, "knee": knee_metrics, "smooth": smooth_metrics})

    for family, frame in (("smooth", smooth_metrics), ("trunk", trunk_metrics), ("knee", knee_metrics)):
        unmeasured = {label: coverage.unmeasured(trials[label].kin.sensors, family) for label in labels}
        for note in flag_missing_lags(frame, LAG_COLUMNS[family], unmeasured):
            log.append("note  " + note)

    sources = {label: trials[label].recording.name for label in labels}
    trunk_metrics, knee_metrics, asymmetry, smooth_metrics, summary = (
        with_recording(frame, sources) for frame in (trunk_metrics, knee_metrics, asymmetry, smooth_metrics, summary)
    )
    paired = with_version(paired)
    aligned = alignment.for_trials(trials)

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
        write_csv(summary, "sequence_summary.csv")
        write_csv(paired, "paired_comparison.csv")
        if aligned is not None:
            write_csv(alignment_table(aligned, trials), "recording_alignment.csv")
        for name in RETIRED_OUTPUTS:
            if (folder / name).exists():
                (folder / name).unlink()
                log.append(f"removed {name}, which the current metric version no longer writes")
        table = coverage.coverage_table({label: set(trials[label].kin.sensors) for label in labels}, sources,
                                        carried)
        if len(table):
            table.to_csv(folder / COVERAGE_FILENAME, index=False)
            written.append(COVERAGE_FILENAME)
        elif (folder / COVERAGE_FILENAME).exists():
            (folder / COVERAGE_FILENAME).unlink()
            log.append(f"removed {COVERAGE_FILENAME}: every selected recording now has all its sensors")

        kins = {label: trials[label].kin for label in labels}
        fs_of = {label: trials[label].fs for label in labels}
        if len(trunk_metrics) > 0:
            make_trunk_traceability_figure(kins, fs_of, trunk_metrics, trunk_windows, sources, folder,
                                           lag_pad_s=lag_pad_s)
            written.append("trunk_traceability_figure.png")
        if len(knee_metrics) > 0:
            make_knee_flexion_overview_figure(kins, fs_of, knee_events, signals, sources, folder)
            written.append("monopodal_stance_overview_figure.png")
            make_knee_traceability_figure(kins, fs_of, knee_metrics, knee_events, signals, sources, folder)
            written.append("monopodal_stance_traceability_figure.png")
        if len(smooth_metrics) > 0:
            make_sequence_smoothness_figure(signals, fs_of, smooth_metrics, corridors, sources, folder)
            written.append("sequence_smoothness_figure.png")
        has_com = any(signals[label]["com_height"] is not None or signals[label]["weight_index"] is not None
                      for label in labels)
        if len(trunk_metrics) + len(knee_metrics) + len(smooth_metrics) > 0 and has_com:
            spans = {label: {
                "trunk": [(start / fs_of[label], end / fs_of[label]) for start, end, *_ in trunk_windows[label]],
                "knee": [(ev["window_start_s"], ev["window_end_s"]) for ev in knee_events[label]],
                "smooth": [(row["turn_start_s"], row["turn_end_s"]) for _, row in smooth_metrics.iterrows()
                           if row["trial"] == label],
            } for label in labels}
            make_centre_of_mass_figure(signals, fs_of, {"trunk": trunk_metrics, "knee": knee_metrics,
                                                        "smooth": smooth_metrics}, spans, sources, folder)
            written.append("centre_of_mass_figure.png")

        for name in OUTPUTS:
            if name not in written and (folder / name).exists():
                (folder / name).unlink()
                log.append(f"removed {name} from an earlier recalculation: nothing to put in it this time")

        save_session(refresh_seconds(session, trials))
        written.append(SESSION_FILENAME)
        write_provenance(session, written, trials)

    return RecomputeResult(
        trunk_metrics=trunk_metrics,
        knee_metrics=knee_metrics,
        asymmetry=asymmetry,
        smooth_metrics=smooth_metrics,
        summary=summary,
        paired=paired,
        written=written,
        log=log,
    )


def carried_over(session: dict, labels: list[str]) -> dict[str, dict[str, str]]:
    """Per recording and family, how many enabled windows were carried over from the other recording.

    A window keeps that mark until its movement boundaries are edited by hand
    (:data:`analysis.sessions.CARRIED_OVER`).
    """
    out: dict[str, dict[str, str]] = {}
    for family in FAMILIES:
        for label in labels:
            windows = [event["windows"][label] for event in session.get(FAMILY_KEY[family], [])
                       if event.get("enabled", True) and label in event.get("windows", {})]
            count = sum((window.get("source") or {}).get("event") == CARRIED_OVER for window in windows)
            if count:
                other = next((window.get("carried_from") for window in windows if window.get("carried_from")), "")
                out.setdefault(label, {})[family] = (
                    f"{count} of {len(windows)} windows carried over from the {other.lower() or 'other'} recording "
                    "through the alignment")
    return out


def with_version(frame: pd.DataFrame) -> pd.DataFrame:
    """``frame`` with a leading ``metrics_version`` column, if it has none."""
    if len(frame) == 0 or "metrics_version" in frame:
        return frame
    frame = frame.copy()
    frame.insert(0, "metrics_version", config.METRICS_VERSION)
    return frame


def with_recording(frame: pd.DataFrame, sources: dict[str, str]) -> pd.DataFrame:
    """``frame`` with ``recording`` (the source file of each row) and ``metrics_version`` after ``trial``."""
    if len(frame) == 0 or "trial" not in frame or "recording" in frame:
        return frame
    frame = frame.copy()
    position = frame.columns.get_loc("trial") + 1
    frame.insert(position, "recording", frame["trial"].map(sources))
    frame.insert(position + 1, "metrics_version", config.METRICS_VERSION)
    return frame


def alignment_table(aligned: alignment.Alignment, trials: dict[str, LoadedTrial]) -> pd.DataFrame:
    """The novice-to-trained clock map at 1 Hz, for tracing any pair by hand."""
    novice_s = np.arange(0.0, trials["Novice"].duration_s, 1.0)
    trained_s = aligned.to_trained(novice_s)
    return pd.DataFrame({
        "metrics_version": config.METRICS_VERSION,
        "novice_time_s": novice_s,
        "trained_time_s": np.round(trained_s, 3),
        "novice_minus_trained_s": np.round(novice_s - trained_s, 3),
    })


# ---------------------------------------------------------------------------
# Provenance: which windows the metrics on disk were computed from
# ---------------------------------------------------------------------------


WINDOW_KEYS = ("event_start", "event_end", "stab_start", "stab_end")


def session_digest(session: dict) -> str:
    """Fingerprint of everything in a session that changes a metric.

    The window indices, which events are enabled and which leg each stance
    lifts, the metric options, the high-pass switch and the metric definitions
    in force, and the content of the selected recording files -- but not names,
    timestamps, the trash, a hand confirmation of a pair or the display-only
    seconds, so renaming or deleting-then-restoring does not mark the metrics
    out of date.  The alignment is not in it either: it only suggests and
    checks pairs, and the pairs themselves are in the windows.
    """
    events = {
        family: [
            {
                "event_id": event.get("event_id"),
                "enabled": event.get("enabled", True),
                "lifted_leg": event.get("lifted_leg", event.get("flexed_leg")),
                "direction": event.get("direction"),
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
        "stabilization": asdict(detection.params_from_dict(session.get("detector", {}) or {})[1]),
        "ignore_high_pass_filter": bool(config.IGNORE_HIGH_PASS_FILTER),
        "metrics_version": config.METRICS_VERSION,
        "recordings": {role: recording.sha256() if recording else None
                       for role, recording in recordings.active().recordings().items()},
    }
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:16]


def provenance_path() -> Path:
    return recordings.analysis_dir() / PROVENANCE_FILENAME


def calibration_record(loaded: LoadedTrial) -> dict:
    """What was removed from a recording's gyroscope, and where it was read."""
    calibration, fs = loaded.kin.calibration, loaded.fs
    return {
        "quiet_spans_s": [[round(start / fs, 2), round(end / fs, 2)] for start, end in calibration.quiet_spans],
        "gyro_bias_dps": {sensor: [round(float(v), 3) for v in bias]
                          for sensor, bias in sorted(calibration.gyro_bias_dps.items())},
        "bias_start_end_difference_dps": {sensor: round(value, 3)
                                          for sensor, value in sorted(calibration.bias_drift_dps.items())},
    }


def write_provenance(session: dict, written: list[str], trials: dict[str, LoadedTrial] | None = None) -> None:
    record = {
        "computed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "analysis": recordings.active().id,
        "recordings": {role: recording.identity()
                       for role, recording in recordings.active().recordings().items() if recording},
        "session_name": session.get("session_name"),
        "session_digest": session_digest(session),
        "metrics_version": config.METRICS_VERSION,
        "ignore_high_pass_filter": bool(config.IGNORE_HIGH_PASS_FILTER),
        "files": written,
    }
    if trials:
        record["calibration"] = {label: calibration_record(loaded) for label, loaded in trials.items()}
        aligned = alignment.for_trials(trials)
        if aligned is not None:
            record["alignment"] = {
                "digest": aligned.digest(),
                "median_novice_minus_trained_s": round(aligned.median_offset_s, 3),
                "mean_step_cost": round(aligned.cost, 4),
                "params": asdict(aligned.params),
            }
    provenance_path().write_text(json.dumps(record, indent=2))


def read_provenance() -> dict | None:
    if not provenance_path().exists():
        return None
    try:
        return json.loads(provenance_path().read_text())
    except json.JSONDecodeError:
        return None
