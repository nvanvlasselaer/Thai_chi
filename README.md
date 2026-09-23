# Tai Chi balance analysis from IMU recordings

Kinematic comparison of a novice and a trained Tai Chi practitioner from full-body inertial
measurement unit (IMU) recordings. The analysis estimates segment orientations, derives trunk and
lower-limb joint angles, segments three families of balance-relevant windows, and computes balance
and movement-variability metrics on them.

Three families are analysed:

- **Trunk rotation / weight shift** — axial rotation of the trunk relative to the pelvis, and the
  postural settling that follows it.
- **Monopodal stance** — single-leg support, identified by unilateral knee flexion above 60°.
- **Sequence segments** — the moving passages of the whole form, cut at the neutral crossings of the
  chest yaw into one back-and-forth turn each, and scored for smoothness, chest–pelvis coordination
  and consistency from one part to the next. Unlike the first two these are not picked out of the
  recording but tile it, and they have no stabilization window.

---

## Data

The recordings are **not** included in this repository (`data/` is git-ignored; the two CSVs are
~150 MB each). The analysis expects:

| File | Participant |
| --- | --- |
| `data/IMU_Trial_1_RC_Novice.csv` | Novice |
| `data/IMU_Trial_3_RC_Trained.csv` | Trained |

**Format.** Delsys Trigno Discover 2.0.1.3 CSV export: eight metadata rows, then one 12-column block
per sensor (ACC X/Y/Z and GYRO X/Y/Z, each with its own time column). Sampling rate **370.3704 Hz**
on every channel; ~240 s per recording. Sensor order differs between the two files, so the parser
matches by name rather than position.

**Sensors (14).** Sternum (`chestbone`) and lower back (`lumbar`); left/right thigh, tibia and foot;
left/right humerus, ulna and hand. Placement is documented in [docs/notes.txt](docs/notes.txt).

**Important limitation.** The sensors are 6-axis (accelerometer + gyroscope, **no magnetometer**).
Rotation about the gravity vector — yaw — is therefore not directly observable and is subject to
drift. Tilt-based quantities (roll, pitch) and relative joint angles (knee, ankle, trunk-pelvis) are
unaffected. Absolute yaw should be interpreted with this in mind.

---

## Requirements

Python 3.13 with:

| package | tested version |
| --- | --- |
| numpy | 2.3.5 |
| scipy | 1.17.0 |
| pandas | 2.3.3 |
| matplotlib | 3.10.8 |
| dash | 2.14.2 |
| plotly | 6.5.0 |
| flask | (via dash) |

`ffmpeg` is additionally required by `analysis/tools/animate_kinematics.py`.

> **Note on dash/plotly.** dash 2.14.2 bundles plotly.js 2.23.2, which cannot decode the base64
> typed arrays that plotly ≥ 6 emits for numpy input. Any figure code in this repository must
> convert arrays with `.tolist()` before passing them to a trace, or the plot renders blank with no
> error. `analysis/editor/app.py` routes everything through a `_series()` helper for this reason.

---

## Running the analysis

### Full pipeline

```bash
python3 analysis/pipeline.py              # or: python3 -m analysis.pipeline
```

Parses both recordings, estimates orientations with a 6-axis Madgwick filter, detects the
trunk-rotation and monopodal-stance events, computes their metrics and writes everything under
`outputs/`. Takes roughly a minute.

Behaviour is controlled by the switches in `analysis/config.py`, which also holds the data and
output paths:

| constant | default | meaning |
| --- | --- | --- |
| `DETECTOR` | `"v2"` | event-segmentation method — see [Event detection](#event-detection) |
| `IGNORE_HIGH_PASS_FILTER` | `False` | when true, `highpass_detrend` is a pass-through and the 0.05 Hz yaw de-drifting is disabled. At the default the high-pass is applied, which is what the committed results were produced with. It changes every yaw-derived metric, so the value in force is written into the session file on each save and shown in the editor header. |

The sampling rate is read from each file's header (370.3704 Hz on every channel).

### Interactive event editor

```bash
python3 analysis/editor/app.py            # or: python3 -m analysis.editor.app
                                          # serves at http://127.0.0.1:8051
```

Automatic segmentation is a starting point, not an answer — see
[Event detection](#event-detection). The editor plots the kinematics on a zoomable timeline and lets
each event's four boundaries (movement start/end, stabilization start/end) be dragged into place,
then recalculates every metric through the pipeline's own functions.

**[docs/user_manual.md](docs/user_manual.md)** covers the dashboard in detail: what each plot shows, how to
adjust the windows, sensible ranges for the detector settings, and how to read the results.

| flag | effect |
| --- | --- |
| `--port N` | serve on a different port (default 8051; `plot_imu.py` uses 8050) |
| `--reseed auto` | discard the saved session and re-run the detectors |
| `--reseed v1` | reseed from the committed window CSVs, reproducing the original automatic result |
| `--recompute-orientation` | re-run the Madgwick filter instead of reusing `outputs/orientation_*.npz` |

Startup takes about 4 s: the orientation quaternions are read from `outputs/orientation_*.npz` and
all joint angles are rederived from them, rather than re-running the orientation filter.

Curated boundaries are stored in `outputs/event_editor_session.json`, which records per-boundary
provenance (`auto` or `manual`), the detector settings used, and the metric options in force. It is
the audit trail for every number in the regenerated CSVs and is intended to be committed.

**Pressing "Recalculate metrics" overwrites the output CSVs and figures in place.** The previous
values remain available through git.

### Supporting tools

Standalone viewers in `analysis/tools/`. None is part of the pipeline, and all find `data/` and
`outputs/` through `analysis/config.py`, so they run from any directory.

| command | purpose |
| --- | --- |
| `python3 analysis/tools/plot_kinematics.py` | plot the 50 Hz joint-angle series |
| `python3 analysis/tools/plot_imu.py` | Dash dashboard for browsing raw accelerometer/gyroscope channels |
| `python3 analysis/tools/animate_kinematics.py [npz] [out.mp4] [--start S --end E --fps N]` | render a 3D stick-figure animation (requires `ffmpeg`) |
| `python3 analysis/tools/plot_frame.py [npz] [--axis-scale S]` | plot one skeleton frame with local axis triads, to check sensor alignment |
| `python3 analysis/tools/yt_download.py` | download the video at the URL in the script from YouTube (requires `yt-dlp`) |

---

## Event detection

Both event families are segmented automatically, and both are editable afterwards.

### Trunk rotation (`DETECTOR = "v2"`)

A trunk rotation begins when the trunk starts turning and ends when it stops, which is a statement
about angular *velocity*. Peaks in the low-pass-filtered yaw-rate envelope are located, and from each
peak the boundaries are walked outward to where the envelope falls below a fraction of that peak's
own height — never below a recording-wide floor, which prevents the walk running away through a quiet
stretch. Event duration is therefore measured rather than assumed.

Because it operates on a derivative, this segmentation is insensitive to the slow yaw drift inherent
to 6-axis orientation estimates.

On the present recordings it yields events of **2.1–7.8 s** (mean 3.7 s). The earlier method
(`DETECTOR = "v1"`, retained so previously published results remain reproducible) scored a
**fixed 8 s** sliding window, so every event came out exactly 8.000 s long — roughly twice the
duration of the actual movement, with the remainder averaging over near-stationary data.

Novice and trained events are paired **by temporal order**, on the premise that both participants
perform the same form. Dynamic-time-warping matching is implemented but not used by default: events
of 2–4 s are not distinctive enough for it to match reliably, and because the search is constrained
to advance monotonically, a single mismatch propagates to every later event. Pairings should be
checked in the editor.

### Stabilization

The settling window that follows each event is chosen by minimising
`mean(ω_lumbar + ω_chest) / quiet_baseline + λ · latency` over a search horizon, where the baseline is
a recording-wide percentile. The latency term encodes that stabilization is what *follows* the event
and stops the search drifting to a quieter moment several seconds later.

Each window carries a **quiet ratio** — its mean angular velocity divided by the recording's quiet
baseline. Values above 1.6 are flagged in the editor as low-confidence: they indicate the participant
had not actually settled, which on these recordings is sometimes genuinely the case rather than a
detection failure.

### Monopodal stance

Windows where the low-pass-filtered knee flexion magnitude exceeds 60° for at least 0.4 s, merging
gaps below 0.2 s.

### Tunable parameters

The detector thresholds are exposed as live controls in the editor rather than fixed in code.
Changing one re-runs detection and redraws the bands immediately, and the plot shows the velocity
envelope and the threshold the detector actually used, so every boundary can be checked against the
signal that produced it.

---

## Metrics

Written to `outputs/trunk_rotation_balance_metrics.csv` and
`outputs/monopodal_stance_balance_metrics.csv`.

| metric | window | meaning |
| --- | --- | --- |
| `trunk_pelvis_peak_cross_correlation`, `trunk_pelvis_lag_s` | event | trunk–pelvis yaw coordination and its time offset |
| `trunk_pelvis_pitch_lag_s` | event | same in the sagittal plane (monopodal stance only) |
| `weight_shift_dimensionless_jerk`, `..._log10_...` | event | movement smoothness; see [docs/notes.txt](docs/notes.txt) for why log₁₀ is the inferential form |
| `lumbar_ap_acc_variance_g2`, `lumbar_ml_acc_variance_g2` | stabilization | anteroposterior / mediolateral postural sway |
| `lumbar_orientation_variability_deg` | stabilization | pooled SD of lumbar orientation |
| `corrective_peak_rate_hz` | stabilization | rate of corrective angular-velocity bursts |
| `lumbar_rms_angular_velocity_dps` | stabilization | RMS lumbar angular velocity |
| `peak_knee_flexion_deg`, `time_to_stabilization_s` | event | monopodal stance only |

### Sequence smoothness

Written per segment to `outputs/sequence_smoothness_metrics.csv`, and summarised per participant to
`outputs/sequence_variability_summary.csv`. The summary file is the movement-variability result: it
reports the spread of each per-segment metric across the parts of the form.

| metric | meaning |
| --- | --- |
| `chest_yaw_log10_dimensionless_jerk` | smoothness of the turn, same definition as the weight-shift jerk above; lower is smoother |
| `chest_yaw_sparc` | spectral arc length of the yaw speed profile (Balasubramanian et al. 2015); negative, less negative is smoother |
| `chest_yaw_submovement_rate_hz` | separate speed peaks per second — one continuous turn has one peak, a hesitant one has several |
| `chest_pelvis_lag_s`, `chest_pelvis_peak_cross_correlation` | chest–pelvis coordination and its time offset, over the segment |
| `chest_pelvis_gain` | pelvis yaw range ÷ chest yaw range; 1 = the trunk turns as a unit |
| `relative_yaw_range_deg`, `relative_yaw_rms_deg` | how much axial twist opens up inside the segment |
| `duration_cv`, `chest_yaw_excursion_cv`, `*_sd` | *(summary)* consistency of the parts in length, size and movement quality |
| `waveform_mean_sd_deg`, `waveform_variance_ratio` | *(summary)* spread of the time-normalised, sign-aligned corridor; the variance ratio is the dimensionless Kadaba form, lower being more repeatable |

Jerk and SPARC measure different aspects of smoothness and need not agree — on these recordings they
point in opposite directions. Report both rather than picking the flattering one.

`monopodal_stance_asymmetry_metrics.csv` reports absolute left-versus-right differences per
participant, averaged over events.

### Metric reliability

Because event boundaries can be placed by hand, each metric was tested by **jittering every boundary
by ±0.25 s** — about the precision of a manual adjustment — and measuring how much it moved.
Reliability is reported as an ICC: between-event signal divided by signal plus measurement noise.

| metric | ICC | note |
| --- | --- | --- |
| `corrective_peak_rate_hz` | 0.97 | replaces the count below |
| `lumbar_rms_angular_velocity_dps` | 0.98 | threshold-free alternative |
| `lumbar_ml_acc_variance_g2` | 0.97 | |
| `lumbar_ap_acc_variance_g2` | 0.96 | |
| `corrective_lumbar_angular_velocity_peak_count` | 0.41 | **superseded — retained only for comparability** |

Two design consequences follow, both empirically determined:

- **Corrective activity is reported as a rate against a recording-wide threshold.** The original
  count derived its threshold from each window's own distribution, so a window twice as busy was
  scored against a bar more than twice as high; it measured shape rather than magnitude (ICC 0.41).
  The columns `corrective_lumbar_angular_velocity_peak_count`,
  `largest_corrective_lumbar_angular_velocity_dps` and `corrective_peak_count` are kept so earlier
  results can still be reproduced, but should not be used for new inference.
- **The stabilization window is 3 s** rather than 2 s, which raises mediolateral sway reliability
  from ICC 0.91 to 0.97. Longer windows score marginally higher still but begin to overlap the
  following movement.

Detrending or tapering the sway variances was tested and **rejected**: it lowered reliability
(mediolateral ICC 0.91 → 0.86), because the trend within a stabilization window is postural signal
rather than artefact.

### Coordination-lag windows

`cross_correlation_lag` searches ±2 s. A window shorter than roughly 4 s cannot support that search —
at the extreme lag the two signals barely overlap, the overlap-normalised correlation is computed
from a short tail, and the estimate pins to the bound rather than finding a peak. Any result at
±1.998 s is a failure marker, not a measurement, and the editor flags it.

The lag window is therefore decoupled from the event window:

- trunk rotation — the event window padded by 2 s on each side (`lag_pad_s`)
- monopodal stance — 2 s before to 3 s after peak flexion (`knee_lag_window_s`)

Both are recorded in the session file. Sessions seeded with `--reseed v1` leave them unset and
reproduce the original behaviour exactly.

---

## Outputs

| file | contents |
| --- | --- |
| `sensor_inventory.csv` | per-sensor sample counts, duration, sampling rate, missing values |
| `orientation_{novice,trained}.npz` | full-rate quaternions (wxyz) for all 14 sensors, plus `time_s` |
| `kinematic_variables_{novice,trained}_50hz.csv` | all joint angles resampled to 50 Hz (40 columns) |
| `trunk_rotation_event_windows.csv` | event and stabilization boundaries, per trial |
| `trunk_rotation_balance_metrics.csv` | trunk-rotation metrics |
| `monopodal_stance_event_windows.csv` | knee-flexion event boundaries |
| `monopodal_stance_balance_metrics.csv` | monopodal-stance metrics |
| `monopodal_stance_asymmetry_metrics.csv` | left/right differences per participant |
| `event_editor_session.json` | curated boundaries and their provenance |
| `trunk_traceability_figure.png` | trunk-rotation time series with event and stabilization bands, plus metric comparison |
| `monopodal_stance_overview_figure.png` | knee flexion and trunk response |
| `monopodal_stance_traceability_figure.png` | monopodal-stance equivalent of the above |
| `orientation_validation.png` | lumbar and sternum orientation, for sanity-checking the filter |

Recalculating from the editor rewrites the event-window CSVs, the metric CSVs and the three
traceability figures. The orientation `.npz`, the 50 Hz kinematic CSVs, `sensor_inventory.csv` and
`orientation_validation.png` do not depend on event windows and are left untouched.

---

## Repository layout

`analysis/` is a Python package whose modules follow the processing chain, one stage each. Every
module can be imported from a script or notebook (`from analysis.kinematics import load_trial`);
the entry points can also be run directly as files.

```
analysis/
  config.py               paths, recordings and the analysis-wide switches
  signals.py              filtering, resampling, cross-correlation lag, index/time conversion
  orientation.py          quaternion algebra, Madgwick filter, sensor-mounting alignment
  data_io.py              Delsys CSV parsing, sensor inventory, orientation cache, 50 Hz export
  kinematics.py           segment and joint angles; fast loading from the orientation cache
  detection.py            v2 event detection (trunk, stance, segments, stabilization), pairing
  detection_v1.py         original fixed-window detector and DTW matching (DETECTOR = "v1")
  smoothness_metrics.py   smoothness, coordination and consistency per sequence segment
  balance_metrics.py      trunk-rotation and monopodal-stance metrics
  figures.py              all output figures
  pipeline.py             batch entry point — runs everything and writes outputs/
  editor/
    app.py                interactive boundary editor (Dash) — entry point
    sessions.py           the session file: seeding, editing, named copies, migration
    validation.py         window checks that gate recalculation
    recompute.py          curated windows → metrics, CSVs and figures
  tools/                  standalone viewers (see Supporting tools)
data/                     recordings (not in the repository)
outputs/                  generated results
docs/
  user_manual.md          guide to using the event editor and interpreting its output
  notes.txt               sensor placement, calibration notes, metric rationale
  Implementation_strategy.md  planned nonlinear (Lyapunov) analysis — not yet implemented
  IMU sensor placement.pdf, Tai Chi Balance Analysis.{docx,pdf}, Technical Assignment.docx
  literature/             Literature.md, BibLaTeX.txt and the PDFs in pdf/
```

Each module imports only modules listed above it (type annotations aside), so any stage can be
used without the ones after it.

---

## Known limitations

- **No magnetometer.** Absolute yaw is unobservable and drifts; only tilt and relative joint angles
  are reliable in that axis. Mitigation options — functional calibration, anatomical alignment, joint
  constraints — are discussed in [docs/notes.txt](docs/notes.txt).
- **Two participants, one trial each.** Differences between the novice and trained recordings are
  descriptive. No statistical inference about training effects is supported by this sample.
- **Novice/trained event pairing is by order** and should be verified visually.
- **Sequence segments cover the moving passages only**, about 60 % of each recording. The still
  spans at the start, middle and end of the form produce no segment, by design — a "cycle" spanning
  a pause would be mostly not moving.
- **Segment lengths are nominal.** No anthropometric scaling or functional joint-axis calibration is
  applied, so joint angles carry soft-tissue and mounting error.
- [docs/Implementation_strategy.md](docs/Implementation_strategy.md) describes a largest-Lyapunov-exponent analysis that is planned but not
  implemented in code.

---

Copyright © 2026 Nicolas Van Vlasselaer

All rights reserved.

This repository is made publicly available for transparency,
scientific reproducibility, and peer review.

No permission is granted to use, copy, modify, distribute,
sublicense, publish, or commercialize this software or any
derivative works without prior written permission from the
copyright holder.
