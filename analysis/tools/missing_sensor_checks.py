#!/usr/bin/env python3
"""Re-run the simulations the missing-sensor rules rest on (docs/missing_sensors.md).

    python3 analysis/tools/missing_sensor_checks.py [--novice FILE] [--trained FILE]

Takes two recordings with all 14 sensors -- by default the June pair -- and
the windows of their saved session (detected afresh if there is none).  Each
sensor is then removed in turn, the recordings rebuilt as stage 1 would have
built them without it, and every metric recalculated on the same windows and
compared with the real values:

* every column :data:`analysis.coverage.METRICS` marks *not calculated* must
  come out empty, every *stand-in* column within the tolerance, and every
  *measured* column is reported against it too;
* the tolerance is the "rule of ten": a tenth of the metric's interquartile
  range across the windows, not to be exceeded in 95 % of them, in any family.

It also checks every combination of missing hands, forearms, upper arms and
feet against the centre-of-mass rules (:func:`analysis.coverage.com_coverage`),
the stabilization search on one trunk sensor, the single-leg detector with the
thigh standing in for a shank, the alignment on fewer channels, and windows
carried over from the other recording.  Writes nothing.  Exits with status 1
if a rule no longer holds.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from analysis import alignment, com, coverage, detection, kinematics, recordings, sessions, validation
from analysis.data_io import GYRO_COLUMNS, TrialData
from analysis.kinematics import LoadedTrial
from analysis.kinematics_check import dedrifted_rotations
from analysis.orientation import get_nominal_sensor_quaternion, mean_quaternion, quat_inverse, quat_multiply
from analysis.recompute import recompute

FAMILIES = ("trunk", "knee", "smooth")
PLACEMENT_SHARE = 0.90
"""A placement fallback on a recording's own signals must put at least this share of the boundaries
within :data:`PLACEMENT_S` of where the full method does."""
PLACEMENT_S = 0.25
"""The precision of a manual drag, and the jitter of the boundary-robustness check."""

failures: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"   [{'ok' if ok else 'FAIL'}] {what}")
    if not ok:
        failures.append(what)


# ---------------------------------------------------------------------------
# A recording without some of its sensors
# ---------------------------------------------------------------------------


def rereferenced(quats: dict[str, np.ndarray], window: tuple[int, int],
                 placements: dict[str, str] | None = None) -> dict[str, np.ndarray]:
    """Body quaternions re-referenced to another neutral-pose window.

    ``q_body = q_nom q_off^-1 q_raw q_nom^-1``, with ``q_off`` the mean raw
    orientation over the neutral window, so with ``r = q_nom^-1 q_body q_nom``
    a new window gives ``q_nom m^-1 r q_nom^-1``, ``m`` the mean of ``r`` over it.
    ``placements`` are the recording's (:attr:`analysis.data_io.TrialData.placements`).
    """
    out = {}
    for sensor, q in quats.items():
        nominal = get_nominal_sensor_quaternion(sensor, (placements or {}).get(sensor))
        r = quat_multiply(quat_multiply(quat_inverse(nominal), q), nominal)
        m = mean_quaternion(r[window[0]:window[1]])
        out[sensor] = quat_multiply(quat_multiply(quat_multiply(nominal, quat_inverse(m)), r), quat_inverse(nominal))
    return out


def without(loaded: LoadedTrial, drop: tuple[str, ...]) -> LoadedTrial:
    """``loaded`` as stage 1 would have built it with the sensors ``drop`` missing.

    Each sensor's orientation filter runs on its own, so the others' are as
    they were, unless the neutral pose moves (no lumbar sensor): then every
    orientation is re-referenced to the window the fallback finds.
    """
    trial = loaded.trial
    keep = [sensor for sensor in trial.sensors if sensor not in drop]
    reduced = TrialData(label=trial.label, path=trial.path, duration_meta_s=trial.duration_meta_s, fs=trial.fs,
                        sensors=keep, data={sensor: trial.data[sensor] for sensor in keep},
                        recorded_as={sensor: trial.recorded_as[sensor] for sensor in keep if sensor in trial.recorded_as},
                        placements={sensor: trial.placements[sensor] for sensor in keep if sensor in trial.placements})
    n = len(loaded.kin.t)
    quats = {sensor: loaded.kin.quaternions[sensor] for sensor in keep}
    neutral = kinematics.find_neutral_pose_window(reduced)
    if neutral != kinematics.find_neutral_pose_window(trial):
        quats = rereferenced(quats, neutral, trial.placements)
    gyros = {sensor: reduced.data[sensor][GYRO_COLUMNS].to_numpy()[:n] for sensor in keep}
    kin = kinematics.kinematics_from_quaternions(loaded.kin.t, quats, gyros, kinematics.calibrate(reduced, n))
    return LoadedTrial(label=loaded.label, kin=kin, trial=reduced, fs=loaded.fs, n_samples=loaded.n_samples,
                       recording=loaded.recording)


def clear_caches() -> None:
    """Simulated recordings share their files' digests with the real ones: drop what is cached by digest."""
    alignment._cache.clear()
    validation._lift_cache.clear()


# ---------------------------------------------------------------------------
# Tolerances and comparisons
# ---------------------------------------------------------------------------


def frames(result) -> dict[str, pd.DataFrame]:
    return {"trunk": result.trunk_metrics, "knee": result.knee_metrics, "smooth": result.smooth_metrics}


def tolerances(result) -> dict[str, dict[str, float]]:
    """Per family and metric column, a tenth of its interquartile range across the windows."""
    out: dict[str, dict[str, float]] = {}
    for family, frame in frames(result).items():
        out[family] = {}
        for column in frame.columns:
            if frame[column].dtype.kind == "f" or frame[column].dtype.kind == "i":
                values = frame[column].astype(float).dropna().to_numpy()
                if len(values) > 3:
                    q25, q75 = np.percentile(values, [25, 75])
                    out[family][column] = 0.1 * float(q75 - q25)
    return out


def ratio(errors: np.ndarray, tolerance: float) -> float:
    """The 95th percentile of the errors over the tolerance; 0 for no error at all."""
    errors = errors[np.isfinite(errors)]
    if not len(errors):
        return float("nan")
    p95 = float(np.percentile(errors, 95))
    if p95 == 0.0:
        return 0.0
    return p95 / tolerance if tolerance > 0 else float("inf")


def keyed(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.set_index(["trial", "event_id"])


# ---------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------


def single_sensors(trials, session, base, tol) -> None:
    print("\n1. Each sensor removed from both recordings, the metrics recalculated on the same windows")
    print("   ratio = 95th percentile of |change| / tolerance, worst over the family's columns")
    print(f"   {'sensor':10s} {'not calculated':>15s} {'stand-in (worst)':>18s} {'measured (worst)':>18s}  worst measured column")
    for sensor in coverage.ALL:
        clear_caches()
        reduced = {label: without(loaded, (sensor,)) for label, loaded in trials.items()}
        result = recompute(session, reduced, write=False)
        present = set(coverage.ALL) - {sensor}
        empty_ok, stand_in, measured = True, [], []
        for family in FAMILIES:
            before, after = keyed(frames(base)[family]), keyed(frames(result)[family])
            for requirement in coverage.METRICS[family]:
                for column in requirement.columns:
                    if column not in after:
                        continue
                    for (label, event_id), row in after.iterrows():
                        leg = row.get("lifted_leg") if family == "knee" else None
                        unmeasured = column in coverage.unmeasured(present, family, leg)
                        if unmeasured and np.isfinite(row[column]):
                            empty_ok = False
                    lost = coverage.unmeasured(present, family)
                    status = coverage.metric_status(present, family, requirement)[0]
                    if column in lost and status == "not calculated":
                        continue
                    common = before.index.intersection(after.index)
                    rows = [key for key in common if column not in coverage.unmeasured(
                        present, family, after.loc[key].get("lifted_leg") if family == "knee" else None)]
                    if not rows or column not in tol[family]:
                        continue
                    errors = np.abs(after.loc[rows, column].astype(float).to_numpy()
                                    - before.loc[rows, column].astype(float).to_numpy())
                    entry = (ratio(errors, tol[family][column]), f"{family}:{column}")
                    (stand_in if status == "stand-in" else measured).append(entry)
        worst_stand_in = max(stand_in, default=(0.0, "-"))
        worst_measured = max((m for m in measured if np.isfinite(m[0])), default=(0.0, "-"))
        print(f"   {sensor:10s} {'all empty' if empty_ok else 'NOT EMPTY':>15s} "
              f"{worst_stand_in[0]:>18.2f} {worst_measured[0]:>18.2f}  {worst_measured[1]}")
        check(empty_ok, f"{sensor}: every column the coverage table marks not calculated is empty")
        check(worst_stand_in[0] <= 1.0, f"{sensor}: every stand-in within the tolerance")
        check(worst_measured[0] <= 1.0, f"{sensor}: every measured column within the tolerance")


def centre_of_mass_combinations(trials, session, tol) -> None:
    print("\n2. Centre of mass: every combination of missing hands, forearms, upper arms and feet")
    arm = ("humerus", "ulna", "hand")
    chains = [("l", m) for r in range(1, 4) for m in itertools.combinations(arm, r)]
    chains += [("r", m) for r in range(1, 4) for m in itertools.combinations(arm, r)]
    chains += [("l", ("foot",)), ("r", ("foot",))]

    def stand_ins(missing: set[str]) -> dict[str, str]:
        """The rules' stand-ins, also where the rules refuse them; a whole missing arm hangs along the trunk."""
        rules = coverage.com_coverage(set(coverage.ALL) - missing)
        if rules.stand_ins or rules.computed:
            return rules.stand_ins
        return {sensor: ("chestbone" if sensor[1:] in arm else f"{sensor[0]}tibia") for sensor in missing}

    errors: dict[tuple[str, ...], dict[tuple[str, str], list[float]]] = {}
    additivity = 0.0
    for label, loaded in trials.items():
        kin, fs = loaded.kin, loaded.fs
        G = com.levelled_rotations(kin, dedrifted_rotations(kin, fs))
        full = com.centre_of_mass_from(kin, fs, G)
        deltas = {}
        for side, segments in chains:
            missing = {f"{side}{segment}" for segment in segments}
            G2 = dict(G)
            G2.update({segment: G[donor] for segment, donor in stand_ins(missing).items()})
            centre = com.centre_of_mass_from(kin, fs, G2)
            deltas[(side, segments)] = (centre.height - full.height, centre.weight_index - full.weight_index,
                                        centre.neutral_height - full.neutral_height)
        # The superposition below is exact: check it on one combination of three chains.
        missing = {"lhand", "rulna", "lfoot"}
        G2 = dict(G)
        G2.update({segment: G[donor] for segment, donor in stand_ins(missing).items()})
        direct = com.centre_of_mass_from(kin, fs, G2).weight_index
        summed = full.weight_index + sum(deltas[key][1] for key in (("l", ("hand",)), ("r", ("ulna",)), ("l", ("foot",))))
        additivity = max(additivity, float(np.max(np.abs(direct - summed))))

        windows = [(family, slice(w["event_start"], w["event_end"]))
                   for family in FAMILIES for event in session.get(sessions.FAMILY_KEY[family], [])
                   if event.get("enabled", True) for lab, w in event["windows"].items() if lab == label]
        options = {side: [()] + [segments for s, segments in chains if s == side and segments != ("foot",)]
                   for side in "lr"}
        base_signals = {"com_height": full.height, "com_neutral_height": full.neutral_height,
                        "weight_index": full.weight_index, "weight_valid": full.weight_valid}
        for left in options["l"]:
            for right in options["r"]:
                for feet in itertools.product((False, True), repeat=2):
                    keys = ([("l", left)] if left else []) + ([("r", right)] if right else []) + \
                           [(side, ("foot",)) for side, on in zip("lr", feet) if on]
                    if not keys:
                        continue
                    combo = tuple(sorted({f"{side}{segment}" for side, segments in keys for segment in segments}))
                    test = {"com_height": full.height + sum(deltas[k][0] for k in keys),
                            "weight_index": full.weight_index + sum(deltas[k][1] for k in keys),
                            "com_neutral_height": full.neutral_height + sum(deltas[k][2] for k in keys),
                            "weight_valid": full.weight_valid}
                    for family, window in windows:
                        prefix = "support_" if family == "knee" else ""
                        a = com.com_metrics(base_signals, window, prefix, weight_transfer=family != "knee")
                        b = com.com_metrics(test, window, prefix, weight_transfer=family != "knee")
                        for column in a:
                            if np.isfinite(a[column]) and np.isfinite(b[column]):
                                errors.setdefault(combo, {}).setdefault((family, column), []).append(abs(b[column] - a[column]))

    print(f"   superposition of the arm and foot chains exact to {additivity:.1e} (weight index)")
    groups = {coverage.COM_WEIGHT_TRANSFER: ("com_weight_transfer",),
              coverage.COM_HEIGHTS: ("com_height_range_pct_height", "com_lowering_pct_height")}
    for group, columns in groups.items():
        allowed, refused = [], []
        for combo, per in errors.items():
            worst = max(ratio(np.array(d), tol[family][column]) for (family, column), d in per.items()
                        if column.replace("support_", "") in columns)
            rules = coverage.com_coverage(set(coverage.ALL) - set(combo))
            (allowed if rules.valid(group) else refused).append((worst, combo))
        worst_allowed = max(allowed, default=(0.0, ()))
        print(f"   {group}: the rules allow {len(allowed)} of {len(errors)} combinations, worst "
              f"{worst_allowed[0]:.2f} ({'+'.join(worst_allowed[1]) or '-'}); of the {len(refused)} refused, "
              f"{sum(r > 1 for r, _ in refused)} exceed the tolerance")
        check(worst_allowed[0] <= 1.0, f"centre of mass, {group}: every combination the rules allow is within the tolerance")
    for combo in (("lhand",), ("rhand",), ("lhand", "rhand"), ("lfoot",), ("rfoot",), ("lfoot", "rfoot"),
                  ("lulna",), ("rulna",), ("lhumerus",), ("rhumerus",)):
        per = errors[combo]
        cells = []
        for columns in (("com_weight_transfer",), ("com_height_range_pct_height",), ("com_lowering_pct_height",)):
            cells.append(max(ratio(np.array(d), tol[family][column]) for (family, column), d in per.items()
                             if column.replace("support_", "") in columns))
        print(f"   {'+'.join(combo):12s} weight transfer {cells[0]:5.2f}   height range {cells[1]:5.2f}   lowering {cells[2]:5.2f}")


def trunk_stand_ins(trials, tol) -> None:
    print("\n3. The rigid-trunk and leg stand-ins the rules refuse")
    rows = {}
    for label, loaded in trials.items():
        kin, fs = loaded.kin, loaded.fs
        G = com.levelled_rotations(kin, dedrifted_rotations(kin, fs))
        full = com.centre_of_mass_from(kin, fs, G)
        signals = lambda c: {"com_height": c.height, "com_neutral_height": c.neutral_height,
                             "weight_index": c.weight_index, "weight_valid": c.weight_valid}
        for name, (segment, donor) in {"chest -> lumbar": ("chestbone", "lumbar"),
                                       "lumbar -> chest": ("lumbar", "chestbone"),
                                       "thigh -> shank": ("lthigh", "ltibia"),
                                       "shank -> thigh": ("ltibia", "lthigh")}.items():
            G2 = dict(G)
            G2[segment] = G[donor]
            test = com.centre_of_mass_from(kin, fs, G2)
            for family in FAMILIES:
                for event in loaded_events(family):
                    window = event["windows"].get(label)
                    if window is None:
                        continue
                    prefix = "support_" if family == "knee" else ""
                    span = slice(window["event_start"], window["event_end"])
                    a = com.com_metrics(signals(full), span, prefix, weight_transfer=family != "knee")
                    b = com.com_metrics(signals(test), span, prefix, weight_transfer=family != "knee")
                    for column in a:
                        if np.isfinite(a[column]) and np.isfinite(b[column]):
                            rows.setdefault(name, {}).setdefault((family, column), []).append(abs(b[column] - a[column]))
    for name, per in rows.items():
        worst = max(ratio(np.array(d), tol[family][column]) for (family, column), d in per.items())
        print(f"   {name:16s} worst {worst:6.2f} x the tolerance")
        check(worst > 1.0, f"{name}: refused, and indeed beyond the tolerance")


def stabilization(trials, session, tol, stab_params) -> None:
    print(f"\n4. Stabilization windows placed on one trunk sensor (automatic placement after each event's end)")
    for sensor, other in (("lumbar", "chestbone"), ("chestbone", "lumbar")):
        offsets, latency = [], []
        for label, loaded in trials.items():
            full = detection.combined_omega(loaded.kin, loaded.fs, stab_params)
            alone = detection.combined_omega(without(loaded, (other,)).kin, loaded.fs, stab_params)
            for family in ("trunk", "knee"):
                for event in loaded_events(family):
                    window = event["windows"].get(label)
                    if window is None:
                        continue
                    a = detection.find_stabilization(loaded.kin, loaded.fs, window["event_end"], stab_params, diagnostics=full)
                    b = detection.find_stabilization(loaded.kin, loaded.fs, window["event_end"], stab_params, diagnostics=alone)
                    offsets.append(abs(b.start - a.start) / loaded.fs)
                    if family == "knee":
                        latency.append(abs(b.start - a.start) / loaded.fs)
        offsets = np.array(offsets)
        share = float(np.mean(offsets <= PLACEMENT_S))
        print(f"   {sensor} alone: {share:.0%} of windows within {PLACEMENT_S} s of the two-sensor placement, "
              f"median {np.median(offsets):.2f} s, max {offsets.max():.1f} s; time to stabilization "
              f"{ratio(np.array(latency), tol['knee']['time_to_stabilization_s']):.1f} x its tolerance (not reported)")
        check(share >= PLACEMENT_SHARE, f"stabilization on the {sensor} alone places {PLACEMENT_SHARE:.0%} of the windows "
                                        f"within {PLACEMENT_S} s")


def lift_stand_in(trials) -> None:
    print("\n5. Single-leg stances found with the thigh standing in for a missing shank")
    for shank in ("ltibia", "rtibia"):
        for label, loaded in trials.items():
            phases = detection.find_single_leg_phases(loaded.kin, loaded.fs)
            reduced = without(loaded, (shank,))
            found = detection.find_single_leg_phases(reduced.kin, loaded.fs)
            offsets = []
            for start, end, leg in phases:
                best = min(found, key=lambda p: abs(p[0] - start) + abs(p[1] - end)) if found else None
                offsets += ([abs(best[0] - start) / loaded.fs, abs(best[1] - end) / loaded.fs]
                            if best is not None and best[2] == leg else [np.inf, np.inf])
            share = float(np.mean(np.array(offsets) <= PLACEMENT_S)) if offsets else float("nan")
            print(f"   no {shank}, {label.lower()}: {len(phases)} stances -> {len(found)}, "
                  f"{share:.0%} of boundaries within {PLACEMENT_S} s")
            check(share >= PLACEMENT_SHARE and abs(len(found) - len(phases)) <= 1,
                  f"no {shank} ({label.lower()}): the same stances, boundaries within {PLACEMENT_S} s")


def reduced_alignments(trials) -> None:
    print("\n6. The whole-recording alignment on fewer channels, against the full one")
    full = alignment.align(trials["Novice"], trials["Trained"])
    grid = np.arange(30.0, trials["Novice"].duration_s - 30.0, 0.5)
    pairs = curated_pairs(trials)
    for drop in (("chestbone",), ("lumbar",), ("lthigh",), ("rtibia",), ("chestbone", "lthigh")):
        reduced = {"Novice": trials["Novice"], "Trained": without(trials["Trained"], drop)}
        channels = alignment.channels_for(reduced["Novice"].kin.sensors, reduced["Trained"].kin.sensors)
        # The novice is complete, so its channels are the trained one's: the map isolates the channel choice.
        a = alignment.align(trials["Novice"], trials["Trained"], channels=channels)
        d = np.abs(a.to_trained(grid) - full.to_trained(grid))
        kept = sum(alignment.mapped_overlap(a, n, t)[0] >= validation.MIN_PAIR_OVERLAP for _, n, t in pairs)
        print(f"   no {'+'.join(drop):18s} channels {', '.join(channels):46s} map: median {np.median(d):.2f} s, "
              f"p95 {np.percentile(d, 95):.2f} s, max {d.max():.2f} s; pairs kept {kept}/{len(pairs)}")
        check(kept == len(pairs), f"alignment without {'+'.join(drop)} keeps every curated pair the same movement")


def carried_over(trials) -> None:
    print("\n7. Windows carried over from the novice, against the trained recording's own curated windows")
    pairs = curated_pairs(trials)
    for family, drop in (("trunk", ("chestbone",)), ("smooth", ("chestbone",)), ("knee", ("lthigh",))):
        clear_caches()
        reduced = {"Novice": trials["Novice"], "Trained": without(trials["Trained"], drop)}
        aligned = alignment.for_trials(reduced)
        offsets, ious = [], []
        for fam, novice, trained in pairs:
            if fam != family:
                continue
            fs_n, fs_t = trials["Novice"].fs, trials["Trained"].fs
            start, end = sessions.carry_span(round(novice[0] * fs_n), round(novice[1] * fs_n), reduced, "Novice",
                                             "Trained", aligned)
            carried = (start / fs_t, end / fs_t)
            offsets += [abs(carried[0] - trained[0]), abs(carried[1] - trained[1])]
            ious.append(alignment.overlap(carried, trained)[1])
        offsets, ious = np.array(offsets), np.array(ious)
        print(f"   {family:6s} (no {'+'.join(drop)}): n={len(ious)}, IoU median {np.median(ious):.2f} (min {ious.min():.2f}), "
              f"boundaries median {np.median(offsets):.2f} s apart, {np.mean(offsets <= 0.5):.0%} within 0.5 s")
        check(ious.min() >= validation.MIN_PAIR_IOU,
              f"carried-over {family} windows cover the same movement as the recording's own (IoU >= {validation.MIN_PAIR_IOU})")


# ---------------------------------------------------------------------------
# Helpers over the session
# ---------------------------------------------------------------------------


_session: dict = {}


def loaded_events(family: str) -> list[dict]:
    return [event for event in _session.get(sessions.FAMILY_KEY[family], []) if event.get("enabled", True)]


def curated_pairs(trials) -> list[tuple[str, tuple[float, float], tuple[float, float]]]:
    fs_n, fs_t = trials["Novice"].fs, trials["Trained"].fs
    return [(family, (e["windows"]["Novice"]["event_start"] / fs_n, e["windows"]["Novice"]["event_end"] / fs_n),
             (e["windows"]["Trained"]["event_start"] / fs_t, e["windows"]["Trained"]["event_end"] / fs_t))
            for family in FAMILIES for e in loaded_events(family)
            if "Novice" in e["windows"] and "Trained" in e["windows"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--novice", default="IMU_Trial_1_RC_Novice.csv")
    parser.add_argument("--trained", default="IMU_Trial_3_RC_Trained.csv")
    args = parser.parse_args()

    selection = recordings.Selection(novice=args.novice, trained=args.trained)
    problem = selection.problem()
    if problem:
        sys.exit(f"Cannot use {selection.describe()}: {problem}.")
    trials = {label: kinematics.load_trial(recording, label) for label, recording in selection.recordings().items()}
    incomplete = {label: coverage.missing(loaded.kin.sensors) for label, loaded in trials.items()
                  if coverage.missing(loaded.kin.sensors)}
    if incomplete:
        sys.exit(f"Both recordings need all 14 sensors to be compared with: {incomplete}.")
    saved = sessions.load_session(selection.output_dir / sessions.SESSION_FILENAME)
    session = saved if saved is not None else sessions.seed_session(trials)
    _session.update(session)
    _, stab_params, _, _ = detection.params_from_dict(session.get("detector") or {})
    print(f"{selection.describe()}: {'the saved session' if saved else 'freshly detected windows'}, "
          + ", ".join(f"{len(loaded_events(f))} {coverage.FAMILY_NAMES[f]}" for f in FAMILIES))

    base = recompute(session, trials, write=False)
    tol = tolerances(base)
    single_sensors(trials, session, base, tol)
    centre_of_mass_combinations(trials, session, tol)
    trunk_stand_ins(trials, tol)
    stabilization(trials, session, tol, stab_params)
    lift_stand_in(trials)
    reduced_alignments(trials)
    carried_over(trials)

    print(f"\n{len(failures)} rule{'s' * (len(failures) != 1)} no longer hold" + (":" if failures else "."))
    for failure in failures:
        print(f"   {failure}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
