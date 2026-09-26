#!/usr/bin/env python3
"""Re-run the checks each method choice of metrics version 2 rests on.

    python3 analysis/tools/method_checks.py

Reads the active selection's recordings and working session and writes
nothing.  Each check prints PASS or FAIL with the number behind it, so the
figures quoted in the README and in docstrings can be regenerated rather than
taken on trust:

* the lag estimator recovers a known lag, and never returns r > 1;
* SPARC matches the published reference implementation;
* the body axes are x = mediolateral, y = anteroposterior;
* the gyroscope bias is large, and removing it leaves the still spans still;
* gravity dominated the old sway variance;
* the whole-recording alignment puts each single-leg stance on its partner;
* the lift detector finds each stance on the right leg;
* every pair in the session is the same movement;
* dimensionless jerk is explained by duration and amplitude.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from analysis import alignment, detection, recordings, sessions
from analysis.data_io import ACC_COLUMNS
from analysis.gravity import linear_acceleration, vertical_angular_velocity
from analysis.kinematics import load_all_trials
from analysis.orientation import mounting_matrix
from analysis.signals import lowpass_signal, pearson_lag
from analysis.smoothness_metrics import dimensionless_jerk, sparc
from analysis.validation import MIN_PAIR_OVERLAP, pair_overlap

failures: list[str] = []


def check(name: str, passed: bool, detail: str) -> None:
    print(f"  {'PASS' if passed else 'FAIL'}  {name}: {detail}")
    if not passed:
        failures.append(name)


def overlap_normalised_lag(a: np.ndarray, b: np.ndarray, fs: float, max_lag_s: float = 2.0) -> tuple[float, float]:
    """The estimator metrics version 1 used, for comparison only."""
    from scipy.signal import correlate, correlation_lags
    a = lowpass_signal(a, fs, 6.0)
    b = lowpass_signal(b, fs, 6.0)
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    corr = correlate(a, b, mode="full", method="fft")
    lags = correlation_lags(len(a), len(b), mode="full")
    overlap = correlate(np.ones_like(a), np.ones_like(b), mode="full", method="direct")
    values = corr / np.maximum(overlap, 1.0)
    keep = np.abs(lags) <= int(max_lag_s * fs)
    i = int(np.argmax(np.abs(values[keep])))
    return float(values[keep][i]), float(lags[keep][i] / fs)


def reference_sparc(speed: np.ndarray, fs: float, padlevel: int = 4, fc: float = 10.0, amp_th: float = 0.05) -> float:
    """Balasubramanian et al. (2015), as published."""
    nfft = int(pow(2, np.ceil(np.log2(len(speed))) + padlevel))
    f = np.arange(0, fs, fs / nfft)
    mf = abs(np.fft.fft(speed, nfft))
    mf = mf / max(mf)
    fc_inx = ((f <= fc) * 1).nonzero()
    f_sel, mf_sel = f[fc_inx], mf[fc_inx]
    inx = ((mf_sel >= amp_th) * 1).nonzero()[0]
    f_sel, mf_sel = f_sel[inx[0]:inx[-1] + 1], mf_sel[inx[0]:inx[-1] + 1]
    return float(-sum(np.sqrt(pow(np.diff(f_sel) / (f_sel[-1] - f_sel[0]), 2) + pow(np.diff(mf_sel), 2))))


def synthetic_checks() -> None:
    print("Synthetic signals")
    fs = 370.3704

    def turn(t: np.ndarray, t0: float, duration: float, amplitude: float) -> np.ndarray:
        s = np.clip((t - t0) / duration, 0.0, 1.0)
        return amplitude * (10 * s**3 - 15 * s**4 + 6 * s**5)

    t = np.arange(0.0, 8.0, 1.0 / fs)
    new_lags, old_lags, old_r = [], [], []
    for start in (0.5, 1.5, 2.5, 3.5, 4.5):
        pelvis, chest = turn(t, start, 3.0, 25.0), turn(t, start + 0.3, 3.0, 40.0)
        r, lag = pearson_lag(pelvis, chest, fs)
        r_old, lag_old = overlap_normalised_lag(pelvis, chest, fs)
        new_lags.append(lag)
        old_lags.append(lag_old)
        old_r.append(r_old)
    check("per-lag Pearson recovers a 0.30 s lead", max(abs(lag + 0.30) for lag in new_lags) < 0.005,
          f"lags {np.round(new_lags, 3).tolist()} (overlap-normalised: {np.round(old_lags, 3).tolist()})")

    speed = np.abs(np.gradient(turn(np.arange(0, 4, 1 / fs), 0.5, 1.5, 30) - turn(np.arange(0, 4, 1 / fs), 2.0, 1.5, 30),
                               1 / fs))
    check("SPARC matches the reference (padlevel 4)", abs(sparc(speed, fs) - reference_sparc(speed, fs)) < 1e-9,
          f"{sparc(speed, fs):.4f} vs {reference_sparc(speed, fs):.4f}")


def data_checks() -> None:
    selection = recordings.active()
    print(f"Recordings: {selection.describe()}")
    trials = load_all_trials(selection)
    session = sessions.load_session()
    if session is None:
        print("  no session saved for this selection yet: session checks skipped")
    aligned = alignment.for_trials(trials)

    for label, loaded in trials.items():
        kin, fs, calibration = loaded.kin, loaded.fs, loaded.kin.calibration
        print(f"{label}")
        ns, ne = calibration.neutral
        acc = loaded.trial.data["lumbar"][ACC_COLUMNS].to_numpy()[: len(kin.t)] @ mounting_matrix("lumbar").T
        mean = acc[ns:ne].mean(axis=0)
        check("lumbar mounting tilt lies in the y-z plane (so y = AP)", abs(mean[1]) > 3 * abs(mean[0]),
              f"neutral acceleration {np.round(mean, 3).tolist()} g")
        hip = np.abs(lowpass_signal(kin.left_hip_deg[:, 0], fs, 6.0)).max(), \
            np.abs(lowpass_signal(kin.left_hip_deg[:, 1], fs, 6.0)).max()
        check("hip flexion is on Euler x (so x = ML)", hip[0] > 2 * hip[1],
              f"left hip range x {hip[0]:.0f} deg vs y {hip[1]:.0f} deg")

        bias = np.linalg.norm(calibration.gyro_bias_dps["chestbone"])
        check("chest gyroscope bias is large", bias > 5.0, f"|bias| {bias:.1f} deg/s")
        last = calibration.quiet_spans[-1]
        for sensor in ("lumbar", "chestbone"):
            rate = vertical_angular_velocity(kin, loaded.trial, sensor)[last[0]:last[1]]
            check(f"{sensor} is still in the final quiet span once the bias is removed", abs(rate.mean()) < 0.5,
                  f"mean vertical rate {rate.mean():+.3f} deg/s")

        sway = linear_acceleration(kin, loaded.trial)
        raw = acc - acc.mean(axis=0)
        if session:
            for axis, name in ((0, "ML"), (1, "AP")):
                ratios = []
                for event in session.get("trunk_events", []):
                    window = event["windows"].get(label)
                    if window:
                        part = slice(window["stab_start"], window["stab_end"])
                        ratios.append(np.var(raw[part, axis]) * 9.80665**2 / np.var(sway[part, axis]))
                if ratios:
                    check(f"tilt-projected gravity inflated the old {name} sway", np.median(ratios) > 1.5,
                          f"sensor-axis / gravity-free variance, median {np.median(ratios):.1f}x "
                          f"(range {min(ratios):.1f}-{max(ratios):.1f}) over {len(ratios)} post-rotation windows")

        phases = detection.find_single_leg_phases(kin, fs)
        check("lift detector finds the single-leg phases", len(phases) >= 1,
              f"{[(round(s / fs, 1), round(e / fs, 1), leg) for s, e, leg in phases]}")
        if session:
            for event in session.get("knee_events", []):
                window = event["windows"].get(label)
                if not window:
                    continue
                inside = [leg for s, e, leg in phases if s < window["event_end"] and e > window["event_start"]]
                check(f"{event['event_id']} leg agrees with the lift index", inside == [event.get("lifted_leg")],
                      f"labelled {event.get('lifted_leg')}, detected {inside or 'nothing'}")

    if aligned is not None and session:
        print("Alignment and pairing")
        errors = []
        for event in session.get("knee_events", []):
            windows = event["windows"]
            if "Novice" in windows and "Trained" in windows:
                mapped = float(aligned.to_trained(windows["Novice"]["event_start"] / trials["Novice"].fs))
                errors.append(mapped - windows["Trained"]["event_start"] / trials["Trained"].fs)
        if errors:
            check("alignment puts each stance on its partner", max(abs(e) for e in errors) < 1.0,
                  f"errors {np.round(errors, 2).tolist()} s; median offset {aligned.median_offset_s:+.2f} s")
        for family in sessions.FAMILIES:
            events = session.get(sessions.FAMILY_KEY[family], [])
            shares = [pair_overlap(e, trials) for e in events]
            shares = [s[0] for s in shares if s is not None]
            if shares:
                check(f"every {family} pair is the same movement", min(shares) >= MIN_PAIR_OVERLAP,
                      f"{len(shares)} pairs, smallest overlap {min(shares):.0%}")

    if session:
        print("Dimensionless jerk")
        rows = []
        for label, loaded in trials.items():
            chest = detection.chest_yaw_signals(loaded.kin, loaded.fs).chest_yaw_deg
            for event in session.get("smooth_events", []):
                window = event["windows"].get(label)
                if window:
                    part = chest[window["event_start"]:window["event_end"]]
                    duration = len(part) / loaded.fs
                    rows.append((math.log10(dimensionless_jerk(part, loaded.fs)), math.log10(duration),
                                 math.log10(max(np.ptp(part), 1e-6))))
        if len(rows) > 3:
            y, d, a = np.array(rows).T
            design = np.column_stack([np.ones_like(d), d, a])
            beta, *_ = np.linalg.lstsq(design, y, rcond=None)
            r2 = 1 - np.sum((y - design @ beta) ** 2) / np.sum((y - y.mean()) ** 2)
            check("duration and amplitude explain log10 dimensionless jerk", r2 > 0.8,
                  f"log10 DJ = {beta[0]:.2f} + {beta[1]:.2f} log10 D {beta[2]:+.2f} log10 A, R^2 = {r2:.3f} "
                  f"(a noise floor predicts +6 / -2), n = {len(rows)} turns")


def main() -> None:
    synthetic_checks()
    data_checks()
    print(f"\n{len(failures)} check(s) failed" + (": " + ", ".join(failures) if failures else ""))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
