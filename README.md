# Tai Chi balance analysis from IMU recordings

Kinematic comparison of a novice and a trained Tai Chi practitioner from full-body inertial
measurement unit (IMU) recordings. The analysis estimates segment orientations, derives trunk and
lower-limb joint angles, segments three families of balance-relevant windows, and computes balance
and movement-variability metrics on them.

Three families are analysed:

- **Trunk rotation** — axial rotation of the trunk relative to the pelvis, and the window that
  follows it.
- **Single-leg (monopodal) stance** — the support phase from lift-off to touch-down, found from leg
  geometry (one foot clearly higher than the other), and the settling after touch-down.
- **Sequence turns** — the whole form cut into single-direction turns of the chest, from one turning
  point of the yaw to the next, scored for smoothness and chest–pelvis coordination. Unlike the
  first two these are not picked out of the recording but tile its moving passages, and they have no
  stabilization window.

Every event of every family is paired with the same movement in the other recording through a
whole-recording alignment, and the novice–trained comparison is made pair by pair
(`paired_comparison.csv`).

> **Metrics version 2.** The metric definitions were revised after an external review
> ([Review.md](Review.md)), with every claim checked on the recordings (see
> [What changed in version 2](#what-changed-in-version-2)). The numbers of the June write-up
> (`docs/Tai Chi Balance Analysis.pdf`) are metrics version 1 and are reproduced by the git tag
> `metrics-v1`.

## Quick start

```bash
python3 app.py                            # opens the dashboard at http://127.0.0.1:8051
```

Everything is done from the dashboard. On the **Pipeline** page press **Run pipeline**: it computes
whatever is missing (about 25 s from scratch). On later starts the recordings load by themselves and
the stages already done show as done. Check on the **Kinematics check** page that the sensors sit
where the analysis assumes, check and correct the event windows in the **Event editor**, press
**Recalculate**, and read the tables and figures on the **Results** page. The two recordings must be in `data/` first — see
[Data](#data).

---

## Data

The recordings are **not** included in this repository (`data/` is git-ignored; a recording is
~150 MB). Put them in `data/`, in subfolders if you like and under any name: every CSV there is
offered on the Pipeline page, where you choose one for the **novice** role, one for the **trained**
role, or only one of the two. Files that are not a readable Delsys export are listed but cannot be
chosen. Until you choose, the two recordings of the original study are selected when present:

| File | Role |
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
> error. `analysis/dashboard/editor_page.py` routes everything through a `_series()` helper for this reason.

---

## Running the analysis

### The dashboard

```bash
python3 app.py [--port 8051] [--no-browser] [--debug]
```

Starts the server and opens a browser tab. When the orientation cache exists the recordings load in
the background within a few seconds of starting; until then the editor shows a placeholder. There
are four pages, each with its own address:

| page | address | what it is for |
| --- | --- | --- |
| **Pipeline** | `/pipeline` | status of every stage, and the buttons that run them |
| **Kinematics check** | `/check` | whether each sensor sits where the analysis assumes, and whether the movement looks like the video |
| **Event editor** | `/editor` | check and correct the windows every metric is computed on |
| **Results** | `/results` | the metric tables and figures in `outputs/`, and where they came from |

The Pipeline page shows one card per stage, each marked *done*, *out of date* or *not yet*, with
the button that re-runs it:

| stage | what it does | re-run when |
| --- | --- | --- |
| Recordings | choose a novice and a trained recording from the CSVs in `data/`, or leave one empty; **Use these recordings** switches to that selection | you want to analyse other recordings |
| 1 · Orientation and kinematics | per selected recording: parses it, runs the Madgwick filter on all its sensors in parallel, and writes its orientation cache, 50 Hz kinematic CSV, sensor inventory, orientation validation figure and sensor check. Done once per recording and reused by every selection it is in. The card says which sensors, if any, need a look, and **Check sensors and kinematics** opens the Kinematics check page | a recording changes (detected from its size and SHA-256) or the orientation code changes |
| 2 · Recordings loaded | rebuilds the kinematics from the caches (about 5 s) | after stage 1 (automatic) |
| 3 · Event windows | the selection's session, which every metric is computed on. **New session** detects one from the v2 detectors or the v1 detectors; the session it replaces is kept in its `sessions/_previous.json`. A session from an older schema is migrated when it is opened (the original is kept as `sessions/pre-schema-2-<name>.json`) | you want to start the curation over |
| 4 · Metrics and figures | runs the windows through every metric and writes the CSVs and figures to the selection's folder | the windows, settings or recordings changed — the card says so |

**Run pipeline** runs every stage that is missing or out of date, in order. Long work runs in the
background, with a progress bar, a log, and a status line in the header of every page.

Whether the metrics are up to date is judged from the selection's `metrics_provenance.json`, which
each recalculation writes: it records a digest of the windows, the metric options, the high-pass
setting and the recording files it used, and the card compares that digest with the current state.

With one recording selected, everything works the same with one row instead of two: every event has
one window, the editor shows one graph, and the figures and tables hold that recording alone.

**[docs/user_manual.md](docs/user_manual.md)** covers the event editor in detail: what each plot
shows, how to adjust the windows, sensible ranges for the detector settings, and how to read the
results.

Curated boundaries are stored in the selection's `event_editor_session.json`, which records the
recording files it belongs to, per-boundary provenance (`auto` or `manual`), the detector settings
used, and the metric options in force. It is the audit trail for every number in the regenerated
CSVs and is intended to be committed.

**Recalculating overwrites the output CSVs and figures in place.** The previous values remain
available through git.

### The kinematics check

Stage 1 ends with a check of every sensor (`analysis/kinematics_check.py`), written to the
recording's `sensor_check.csv` and shown on the Kinematics check page. A sensor is marked **check**
when:

- in the neutral pose it sits more than 45° from the orientation its mounting assumes (rotated or
  turned over on the segment);
- its accelerometer does not read 1 g ± 0.05 at rest;
- it moves while the participant stands still: more than 3 robust SDs (1.4826 × MAD) above the
  median sensor, in the stiller of the two quiet spans;
- samples are missing;
- a knee's flexion axis lies more than 45° from x (between 20° and 45° it is a note);
- a leg segment moves the way its joint cannot: a raised thigh pointing backward, or a bent knee
  swinging the shank forward, more than half the time. That is what a leg sensor turned around on its
  segment, or left and right swapped, produces, and the neutral pose cannot show either, because
  turning a sensor about the vertical leaves its tilt unchanged.

Whether each sensor is on the segment it is labelled with is judged by eye. The page's stick figure
can be scrubbed through the recording, or jumped to the moments that show a misplaced sensor: at the
highest left-foot lift the figure's left leg must be the one up. Beside it are the joint angles, with
left and right overlaid. Headings are de-drifted for display by taking out each sensor's slow turn
about the vertical (below 0.05 Hz), computed from the quaternion's twist, which does not jump the way
an Euler angle does. The arms are shown as shoulder elevation and elbow bend, the angles between long
axes (see [Known limitations](#known-limitations)). The same panels and table are available outside
the dashboard from `analysis/tools/plot_kinematics.py`.

On the two recordings, every leg segment always moves the way its joint allows, and the knees'
flexion axes lie 28–42° from x, turned opposite ways on the two legs. The trained participant's right
foot is marked: in the stiller quiet span it moves at 17 °/s RMS, 2.4 times the median sensor and 9
times the left foot.

### Without the dashboard

The same stages run headless — useful to regenerate every output from the recordings and the
committed session in one command:

```bash
python3 -m analysis.pipeline --list                       # the recordings in data/, and their state
python3 -m analysis.pipeline [--novice FILE] [--trained FILE]
                             [--force-orientation] [--new-session {v2,v1}] [--workers N]
```

`FILE` is a path relative to `data/`. Give one of `--novice`/`--trained` to analyse a single
recording; give neither to reuse the last selection (the one the dashboard has open). The saved
session is reused unless `--new-session` is given, so this reproduces exactly the outputs the curated
windows produced. The orientation filter only runs for recordings whose outputs are missing or out of
date, or for all selected ones with `--force-orientation`.

### Settings

Two switches in `analysis/config.py` change the results, and are shown on the Pipeline page:

| constant | default | meaning |
| --- | --- | --- |
| `DETECTOR` | `"v2"` | detectors a new session is taken from when none is chosen — see [Event detection](#event-detection) |
| `METRICS_VERSION` | `2` | the metric definitions in force; stamped into the session, every CSV and `metrics_provenance.json` |
| `EXPORT_VERSION` | `2` | format of the 50 Hz export; an older one is rebuilt from the orientation cache, without filtering |
| `IGNORE_HIGH_PASS_FILTER` | `False` | when true, `highpass_detrend` is a pass-through and the 0.05 Hz yaw de-drifting is disabled. At the default the high-pass is applied, which is what the committed results were produced with. It changes every yaw-derived metric, so the value in force is written into the session file on each save and shown in the editor header. |

`DEFAULT_RECORDINGS` sets what is selected before anything has been chosen; afterwards the choice
is remembered in `outputs/selection.json` (not tracked by git). The sampling rate is read from each
file's header.

### Supporting tools

Standalone viewers in `analysis/tools/`. None is part of the pipeline, and all find `data/` and
`outputs/` through `analysis/config.py`, so they run from any directory.

| command | purpose |
| --- | --- |
| `python3 analysis/tools/plot_kinematics.py [--role R] [--recording FILE] [--save PNG]` | print the sensor check and plot the joint-angle panels of the Kinematics check page, for one recording |
| `python3 analysis/tools/plot_imu.py` | separate Dash viewer for the raw accelerometer/gyroscope channels (port 8050) |
| `python3 analysis/tools/animate_kinematics.py [npz] [out.mp4] [--start S --end E --fps N]` | render a 3D stick-figure animation (requires `ffmpeg`) |
| `python3 analysis/tools/plot_frame.py [npz] [--axis-scale S]` | plot one skeleton frame with local axis triads, to check sensor alignment |
| `python3 analysis/tools/method_checks.py` | re-run the checks the method choices rest on, and print their numbers (see [Boundary robustness](#boundary-robustness)) |
| `python3 analysis/tools/boundary_robustness.py [--jitter 0.25] [--replicates 20] [--seed 0]` | jitter every event boundary and report each metric's ICC |
| `python3 analysis/tools/yt_download.py` | download the video at the URL in the script from YouTube (requires `yt-dlp`) |

---

## Event detection

Every family is detected automatically and editable afterwards. The detector thresholds are exposed
as live controls in the editor rather than fixed in code; changing one re-runs detection and redraws
the bands, and each plot shows the signal and threshold the detector actually used, so every
boundary can be checked against the signal that produced it.

### Axes

The body frame is **x = mediolateral** (the flexion/extension axis), **y = anteroposterior**,
**z = vertical**. Three independent checks on the recordings agree: at rest the lumbar sensor's
mounting tilt on the lordosis lies in the y–z plane (neutral acceleration 0.02, −0.30, 0.95 g), hip and
knee flexion appear on Euler x, and the lumbar tilt about y changes sign with the lifted leg. (The
code used to say x = forward; the AP and ML sway columns of metrics version 1 are therefore swapped.)
For the leg sensors the signs follow from what the hip and knee can do. In both recordings the hips
reach 86–111° of positive x rotation and the knees 85–116° of negative x rotation, which only flexion
can. Every raised thigh points to +y and every bent knee swings the shank to −y, so x points **right**
and y **forward** (the sensor check tests this per sensor). For the trunk and arm sensors the signs
are still unverified, and a functional-calibration trial would settle them, so only the spread and
range of signed trunk angles are interpreted.

### Gyroscope bias and the vertical reference

When a recording loads, the gyroscope bias of every sensor is taken as the median over the stillest
10 s near the start and near the end (the stillest by raw angular power summed over all sensors, which
a constant bias cannot mislead). It is large — about 6.6 °/s on the lumbar sensor and 15 °/s on the
chest — and is removed from every angular speed. With it left in, the 21 °/s "quiet baseline" of the
stabilization search was almost all bias, and which participant looked busier depended on the
direction of turning. The bias spans and values are written to `metrics_provenance.json`, and the log
warns when the start and end estimates disagree (a recording without genuine quiet standing).
The Madgwick filter itself is unchanged.

The vertical in each sensor's frame, taken from the neutral pose, gives tilt, turning rate and
gravity-free acceleration with no yaw in them (`analysis/gravity.py`), so none of these are affected by
the heading drift of a 6-axis estimate.

### Whole-recording alignment and pairing

Both participants perform the same form, so the two recordings are warped onto each other once, by
dynamic time warping on slow whole-body channels (chest yaw, trunk–pelvis yaw, both knees' flexion, at
5 Hz, open ends). Checked against the four single-leg stances, whose pairing is unambiguous, the map
puts every novice stance within 0.44 s of its trained partner. On these recordings the same moment of
the form occurs a near-constant **3.4 s later in the novice recording** throughout — both participants
followed the same instruction video — which is itself worth knowing: tempo was paced externally.

Every family is paired through this map. Pairing by order, which it replaces, compared a different
movement in all six trunk-rotation pairs of the curated session, and put the full-cycle sequence
segments half a cycle out of step from the first one on, although both recordings had 13. Validation
now blocks a pair whose two windows overlap by less than half of the shorter once aligned (unless
confirmed by hand with **Confirm pair**), and the editor outlines each partner where the map puts it.
The map is written to `recording_alignment.csv`.

### Trunk rotation

A trunk rotation begins when the trunk starts turning and ends when it stops, which is a statement
about angular *velocity*. Peaks in the low-pass-filtered yaw-rate envelope are located, and from each
peak the boundaries are walked outward to where the envelope falls below a fraction of that peak's
own height — never below a recording-wide floor. Event duration is therefore measured rather than
assumed (2–7 s here). Candidates of both recordings are paired when each is the other's best match by
intersection over union (≥ 0.5) once aligned, and the six pairs whose weaker partner is strongest are
kept. The original fixed 8 s windows with DTW template matching (v1) remain selectable under
**New session**.

### Stabilization

The window that follows each event is chosen by minimising `z + λ · latency` over a search horizon,
where `z = (mean ω − baseline) / spread` measures the bias-corrected lumbar + chest angular speed
against the recording itself (baseline: its 20th percentile, spread: from there to its median), and
λ = 0.65 per second is the trade-off the original ratio score made. A window with z > 1 — busier than
the recording's median moment — is flagged `unsettled`. **After the trunk rotations of this form nobody
settles**: 10 of the 12 post-rotation windows are flagged, so their sway describes the transition into
the next movement, not a recovery. The windows after a single-leg stance's touch-down mostly are quiet.

### Single-leg stance

A single-leg phase is where one foot is clearly higher than the other. The **lift index** is each
leg's vertical reach — the cosine of the thigh's tilt from vertical plus the shank's — for the right
leg minus the left, in thigh lengths: positive when the left foot is up. Peaks above 0.25 (about
10 cm) are lifts, and the boundaries walk out to 15 % of each peak, never below the double-support
level (the median |lift|). This finds the whole support phase whatever the knee does, and which leg
is up. On these recordings it finds exactly four per participant: a knee lift on each side, then a lift
with a kick on each side.

The earlier rule, knee flexion above 60°, is still selectable (it is what v1 sessions use). It split
each kick where the knee straightened, and when the right kick was re-entered by hand it was entered as
a left-leg event — its "11° peak flexion" was the stance knee. Validation now errors on a stance whose
labelled leg is the lower foot, the editor has **Swap leg**, and the migration to schema 2 relabelled
that event.

### Sequence turns

The form is carried by large chest yaw rotations. Each turn runs from one turning point of the
(0.5 Hz-smoothed) chest yaw to the next, so it starts and ends with the chest momentarily still and
turns one way throughout — a discrete movement, which is what SPARC assumes. (The earlier segments were
cut at neutral crossings, where the turning speed peaks.) The turning points of the two recordings are
matched through the alignment, and each turn is made between consecutive matched points in both
recordings at once, so an extra wiggle in one recording stays inside a turn instead of shifting every
later pair.

---

## Metrics

Metric CSVs have one row per event per recording, with `event_id` shared by the two members of a pair,
and `recording` and `metrics_version` columns.

### Trunk rotation — `trunk_rotation_balance_metrics.csv`

| metric | window | meaning |
| --- | --- | --- |
| `trunk_pelvis_lag_s`, `trunk_pelvis_yaw_r` | event, padded 2 s each side | pelvis-to-chest turning lag (negative: the pelvis leads) and the correlation behind it |
| `trunk_yaw_sparc` | event | smoothness of the trunk-on-pelvis turning speed; less negative is smoother |
| `trunk_yaw_excursion_deg`, `trunk_yaw_peak_rate_dps` | event | how far and how fast the trunk turned on the pelvis |
| `lumbar_ml_acc_rms_mps2`, `lumbar_ap_acc_rms_mps2` | post-rotation | mediolateral / anteroposterior sway: gravity-free lumbar acceleration in the pelvis-heading frame, RMS about the window mean |
| `lumbar_frontal_tilt_sd_deg`, `lumbar_sagittal_tilt_sd_deg` | post-rotation | sideways / forward-backward tilt spread of the pelvis, with no yaw in it |
| `lumbar_rms_angular_velocity_dps` | post-rotation | overall lumbar angular activity, bias-corrected; needs no threshold |
| `lumbar_corrective_peak_rate_hz` | post-rotation | angular-speed bursts per second above a recording-wide threshold — secondary: zero in most windows |
| `stabilization_quiet_z` | post-rotation | how quiet the window was (0 = the recording's quietest fifth, 1 = its median) |

### Single-leg stance — `monopodal_stance_balance_metrics.csv`

| metric | window | meaning |
| --- | --- | --- |
| `support_duration_s`, `peak_lift_index` | support | how long, and how high the foot went (thigh lengths) |
| `peak_knee_flexion_deg`, `knee_extension_while_lifted_deg`, `peak_hip_flexion_deg` | support | what the lifted leg did; the extension is large for a kick |
| `support_*` (`ml_acc_rms_mps2`, `ap_acc_rms_mps2`, `frontal_tilt_sd_deg`, `sagittal_tilt_sd_deg`, `rms_angular_velocity_dps`, `corrective_peak_rate_hz`) | support | balance on one foot: the sway, tilt and angular activity of the pelvis while the base of support is a single foot |
| `settle_*` (the same) | after touch-down | settling |
| `time_to_stabilization_s` | — | touch-down to the start of settling |
| `lumbar_frontal_tilt_at_peak_deg` | highest lift | sideways pelvis tilt at the highest lift |
| `trunk_pelvis_lag_s`, `trunk_pelvis_frontal_lag_s` (with `_r`) | 2 s before to 3 s after the highest lift | kept in the CSV only: the correlation behind them is weak during a stance |

`monopodal_stance_asymmetry_metrics.csv` compares **mirrored stances** — the k-th left lift with the
k-th right lift, or events sharing a `movement` label — as signed left-minus-right differences per
pair, plus the mean absolute difference. Pooling all left against all right events, as before,
compared a knee lift with a kick.

### Sequence turns — `sequence_smoothness_metrics.csv`, `sequence_summary.csv`

| metric | meaning |
| --- | --- |
| `turn_duration_s`, `chest_yaw_excursion_deg`, `chest_yaw_peak_rate_dps` | how long, how far and how fast the chest turned (from the integrated gyroscope turning rate) |
| `chest_yaw_sparc` | smoothness of the turning speed (SPARC, Balasubramanian et al. 2012/2015, padlevel 4); less negative is smoother |
| `chest_yaw_submovement_rate_hz` | separate speed peaks per second — one continuous turn has one |
| `chest_pelvis_lag_s`, `chest_pelvis_yaw_r` | pelvis-to-chest turning lag and its correlation |
| `chest_pelvis_gain` | pelvis turn ÷ chest turn; 1 = the trunk turns as a unit |
| `relative_yaw_range_deg`, `relative_yaw_sd_deg` | how much axial twist opens up inside the turn |
| `chest_yaw_log10_dimensionless_jerk` | secondary, for comparability only (see below) |

`sequence_summary.csv` gives per recording the number of turns and the median of each metric.
Medians, not spreads: every turn is a different movement of the form, performed once, so the spread of
a metric across turns describes the choreography rather than anyone's consistency. That is why the
across-segment CV, SD and variance ratio of metrics version 1 are gone.

### Novice versus trained — `paired_comparison.csv`

For every family and metric, over the events present in both recordings: the number of pairs, each
performer's median, the median trained-minus-novice difference with its quartiles, and the fraction of
pairs in which the trained value is the higher. It is descriptive — one participant per group, and
consecutive events of one performance are not independent samples — so no p-values are given. On
these recordings, for example, the trained performer's turns are smoother (higher SPARC) in 24 of 28
matched turns of equal median duration, and their mediolateral sway during single-leg support is lower
in all four stances.

### Why dimensionless jerk is secondary

For these slow, long movements the jerk integral is dominated by noise near the low-pass cutoff, whose
contribution grows like duration⁶/amplitude². Over the 56 turns, log₁₀ DJ = 4.90 + 5.69 log₁₀ D − 1.61
log₁₀ A (R² = 0.955) — the exponents a noise floor predicts — and 86–93 % of the jerk power lies above
1 Hz while the turns are ~0.1 Hz. SPARC is valid on gyroscope angular velocity where log dimensionless
jerk is error-prone (Melendez-Calderon et al. 2021), so it is the primary smoothness measure.

### Boundary robustness

Because boundaries can be placed by hand, `analysis/tools/boundary_robustness.py` jitters every
boundary of the working session by up to ±0.25 s (about the precision of a manual drag), 20 times, and
reports ICC(1,1) of each metric — the share of its variance that is between events rather than due to
where exactly the boundaries fell. This is robustness to curation, **not test–retest reliability**,
which needs a second session. On the current session:

| family | metrics | ICC |
| --- | --- | --- |
| trunk rotation | lag, excursion, ML and AP sway | 0.98–1.00 |
| trunk rotation | SPARC, frontal / sagittal tilt SD, lumbar ω RMS | 0.93–0.97 |
| single-leg stance | peak lift, peak knee flexion, support duration, support sway, tilt and ω RMS, settling sway | 0.96–1.00 |
| single-leg stance | `time_to_stabilization_s` | **0.51** — the gap between two boundaries: read with caution |
| turns | duration, excursion, peak rate, lag, twist range | 0.99–1.00 |
| turns | SPARC, gain, submovement rate | 0.90–0.97 |

`analysis/tools/method_checks.py` re-runs every check the method choices above rest on and prints the
numbers quoted in this README.

### What changed in version 2

The external review's claims were each checked on the recordings, which the reviewer did not have:

| issue | on these recordings | change |
| --- | --- | --- |
| Lag estimator z-scored the whole window, then divided by the overlap count | r > 1 in 7 of 12 trunk rows; a synthetic 0.30 s lag came out as 0.00–0.17 s | per-lag Pearson on the overlap, most positive peak, ±1 s, NaN at the bound |
| AP and ML sway labels swapped | x is the flexion axis (see [Axes](#axes)) | columns named by the correct axis |
| Sway variance in sensor axes | tilt-projected gravity inflated it 1.8–2.9× (ML) and 7.7–9.4× (AP, median) | gravity-free acceleration in the pelvis-heading frame, as RMS |
| Orientation variability pooled Euler x, y, z | dominated by continuing yaw (SD 6–16°; tilt 0.5–3.5°) | frontal and sagittal tilt SD, no yaw |
| Gyroscope bias in every angular speed | 6.6 °/s lumbar, 15 °/s chest; reversed the novice/trained ordering of lumbar ω | removed at load time |
| Pairing by order | all six trunk pairs and half the sequence pairs were different movements | whole-recording alignment |
| Stance = knee > 60° | the kicks were split; one hand-entered event had the wrong leg | leg lift index; leg-side validation |
| SPARC on full cycles cut at neutral, padlevel 2 | cut where speed peaks | per single-direction turn from the gyroscope, padlevel 4 |
| 0.05 Hz high-pass on yaw | shrank 8–14 s turns by 23–26 % | turn angles from the integrated gyroscope rate |
| Across-segment spread reported as "the variability answer" | the segments are different movements | per-recording medians; paired comparison |
| All joint angles high-passed in the 50 Hz export | a held knee flexion lost its level | only axial (z) channels high-passed |

Not taken up, because these recordings cannot support it: repetition variability and coordination
(coupling-angle) variability need the same movement performed many times; entropy of quiet standing
needs longer holds than the ~15–25 s here; Lyapunov exponents need hundreds of cycles
([docs/Implementation_strategy.md](docs/Implementation_strategy.md)). What the next recordings would
need is in [Recommendations_for_next_recordings.md](Recommendations_for_next_recordings.md).

---

## Outputs

Outputs are kept apart by what they depend on, and named after the recordings they came from:

```
outputs/
  recordings/<recording>/          one folder per recording, whichever selections use it
    recording.json                 the source file: name, size, SHA-256; sampling rate, samples, sensors
    orientation.npz                full-rate quaternions (wxyz) for every sensor, plus time_s
    kinematic_variables_50hz.csv   all joint angles resampled to 50 Hz (40 columns; only yaw and joint z high-passed)
    sensor_inventory.csv           per-sensor sample counts, duration, sampling rate, missing values
    orientation_validation.png     lumbar and sternum orientation, for sanity-checking the filter
    sensor_check.csv               per sensor: mounting tilt, |g| at rest, stillness, knee axis, leg direction, status
  analyses/<selection>/            one folder per selection of recordings
    analysis.json                  which recordings, by name, size and SHA-256
    event_editor_session.json      the curated windows and their provenance
    sessions/                      named copies of the session, and the _previous undo slot
    metrics_provenance.json        which windows, settings, recordings, gyroscope bias and alignment the metrics came from
    ... the CSVs and figures below
```

A recording's folder is its file name with anything unsafe for a file system replaced — with a short
hash appended when that changed the name, so two files never share a folder. A selection's folder
names its recordings with their roles, e.g. `novice-IMU_Trial_1_RC_Novice__trained-IMU_Trial_3_RC_Trained`
or `trained-IMU_Trial_3_RC_Trained`. Every table below has a `recording` column naming the source file
of each row, and every figure names its source files underneath, so a file copied out of its folder
still says where it came from.

| file (in the selection's folder) | contents |
| --- | --- |
| `paired_comparison.csv` | novice versus trained, pair by pair, for every family and metric |
| `trunk_rotation_event_windows.csv` | event and stabilization boundaries, per recording |
| `trunk_rotation_balance_metrics.csv` | trunk-rotation metrics |
| `monopodal_stance_event_windows.csv` | single-leg support windows: lifted leg, highest lift, settling |
| `monopodal_stance_balance_metrics.csv` | single-leg-stance metrics |
| `monopodal_stance_asymmetry_metrics.csv` | left-minus-right differences between mirrored stances |
| `sequence_smoothness_metrics.csv`, `sequence_summary.csv` | per-turn metrics, and their medians per recording |
| `recording_alignment.csv` | the novice-to-trained clock map at 1 Hz, for tracing any pair by hand |
| `trunk_traceability_figure.png` | trunk-rotation time series with the movement (rotation) and stabilization (post-rotation) bands, above the paired metrics grouped by the window they are computed on, each group on its band's colour |
| `monopodal_stance_overview_figure.png` | lift index, knee flexion and pelvis tilt |
| `monopodal_stance_traceability_figure.png` | the lift index with the movement (single-leg support) and stabilization (settling) bands, above the paired metrics grouped the same way |
| `sequence_smoothness_figure.png` | the turns, their typical time course, and paired metrics |

Recalculating (stage 4) rewrites the selection's CSVs, figures and `metrics_provenance.json`. Stage 1
writes the recording folders, which do not depend on event windows.

---

## Repository layout

`app.py` starts the dashboard. Everything else is in `analysis/`, a Python package whose modules
follow the processing chain, one stage each, with the dashboard on top. Every module can be imported
from a script or notebook (`from analysis.kinematics import load_trial`).

```
app.py                    starts the dashboard
analysis/
  config.py               paths, default recordings and the analysis-wide switches
  signals.py              filtering, resampling, cross-correlation lag, index/time conversion
  orientation.py          quaternion algebra, Madgwick filter, sensor-mounting alignment
  recordings.py           the recordings in data/, the selection, and where their outputs go
  data_io.py              Delsys CSV parsing, sensor inventory, orientation cache, 50 Hz export
  kinematics.py           segment and joint angles; gyroscope bias and vertical reference; fast loading
  gravity.py              tilt, turning rate, gravity-free sway and leg lift, from each segment's vertical
  skeleton.py             the stick figure: its segments, and a pose from their orientations
  kinematics_check.py     the sensor check, joint-angle panels and moments of the Kinematics check page
  alignment.py            whole-recording DTW between the two recordings; event and landmark matching
  detection.py            v2 event detection (trunk, stance, turns, stabilization), pairing via the alignment
  detection_v1.py         original fixed-window detector and DTW matching (v1)
  smoothness_metrics.py   smoothness and chest–pelvis coordination per turn
  balance_metrics.py      trunk-rotation and single-leg-stance metrics, mirrored asymmetry
  comparison.py           novice versus trained, paired event by event
  figures.py              all output figures
  sessions.py             the session file: seeding, editing, named copies, migration
  validation.py           window checks that gate recalculation
  recompute.py            windows → metrics, CSVs and figures, and their provenance
  pipeline.py             the four stages and their status; also runs headless
  dashboard/
    app.py                the Dash app: navigation, progress polling, the server
    pipeline_page.py      stage cards and the buttons that run them
    check_page.py         the kinematics check: sensor table, stick figure, joint angles
    editor_page.py        the event-window editor
    results_page.py       metric tables and figures
    tasks.py              background jobs, each a sequence of pipeline stages
    state.py              what the server holds: the loaded recordings and the running job
  tools/                  standalone viewers (see Supporting tools)
data/                     recordings, any names, subfolders allowed (not in the repository)
outputs/                  generated results: recordings/<recording>/ and analyses/<selection>/
docs/
  user_manual.md          guide to using the event editor and interpreting its output
  notes.txt               sensor placement, calibration notes, metric rationale
  Implementation_strategy.md  planned nonlinear (Lyapunov) analysis — not yet implemented
  IMU sensor placement.pdf, Tai Chi Balance Analysis.{docx,pdf}, Technical Assignment.docx
  literature/             Literature.md, BibLaTeX.txt and the PDFs in pdf/
```

Each module imports only modules listed above it (type annotations aside), so any stage can be
used without the ones after it — nothing outside `dashboard/` depends on Dash.

---

## Known limitations

- **No magnetometer.** Absolute yaw is unobservable and drifts; only tilt and relative joint angles
  are reliable in that axis. Mitigation options — functional calibration, anatomical alignment, joint
  constraints — are discussed in [docs/notes.txt](docs/notes.txt).
- **Two participants, one trial each.** Differences between the novice and trained recordings are
  descriptive. No statistical inference about training effects is supported by this sample.
- **Pairs rest on the whole-recording alignment**, which is only meaningful where the form is being
  performed; in the quiet standing at either end it maps arbitrarily (no event lies there).
- **Each movement is performed once.** Variability across repetitions of the same movement, and
  coordination (coupling-angle) variability, cannot be computed from these recordings; the spread
  across different events describes the choreography.
- **Post-rotation windows are not quiet** (see [Stabilization](#stabilization)): read the trunk
  family's sway as the transition into the next movement.
- **The trained participant's right-foot sensor moves while they stand still**: 17 °/s RMS in the
  stiller quiet span, against 2 °/s for the left foot. Its strap was loose or the foot moved, and its
  `sensor_check.csv` marks it. The novice's right foot is the stillest sensor of that recording. No
  stance-foot measure is reported.
- **The exported elbow angles are not usable, and the shoulder angles only roughly.** In Tai Chi's
  arm positions the elbow's Euler angles pass through the decomposition's singularity: its
  ab/adduction reaches ±77–88°, and its flexion, unwrapped, runs to −743° and −1060° in
  `kinematic_variables_50hz.csv`. No Euler ordering fixes this, because the upper-arm sensor, on soft
  tissue, misses much of the humerus's axial rotation. The shoulder's components mix as the chest and
  upper-arm headings drift apart (next point). No metric uses either. The Kinematics check shows the
  arms as shoulder elevation and elbow bend instead: the angle between the upper arm and the trunk's
  axis, and between the upper arm and the forearm.
- **The upper-body sensors' headings drift**, each by its own amount. Between the start and the end
  of the recordings they drift by up to 115°, against at most 44° for the legs, so two of them end up
  as much as 130° apart. Angles between arm segments drift with them. Re-running the orientation
  filter with the quiet-standing bias should cut this.
- **Sequence turns cover the moving passages only**. The still spans at the start and end, and the
  single-leg passage in the middle (where the chest barely turns), produce no turn, by design.
- **Segment lengths are nominal.** No anthropometric scaling or functional joint-axis calibration is
  applied, so joint angles carry soft-tissue and mounting error. The knees' flexion axes lie 28–42°
  from the assumed one, so part of the knee flexion shows up in the other two components.
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
