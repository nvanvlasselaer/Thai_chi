# Missing sensors: what the analysis does without them

A recording does not always have all 14 sensors. A sensor runs flat, drops out, or is left off. Of
the 18 recordings added in September 2026, 15 have only 13 sensors: one hand or the sternum is
missing (see [the recordings in data/](#the-recordings-in-data)). This file lists, for each sensor,
which metrics need it, what happens when it is missing, how that is mitigated, and why. The rules
live in `analysis/coverage.py`. `python3 analysis/tools/missing_sensor_checks.py` re-runs every
simulation quoted here.

## The principle

* **A missing sensor never stops the analysis.** Before, the Sensor names card refused to go on
  ("the analysis needs all 14"). Now a missing sensor is reported, and the analysis runs without it.
* **Every metric is one of three things:** *measured* exactly as with all sensors; *measured with a
  stand-in*, where a neighbouring sensor provably does the job (below); or *not calculated*, an empty
  cell (NaN). Every table keeps the same columns, so a missing value never shifts another.
* **One missing sensor does not cost the others their metrics.** The lumbar sway after a trunk
  rotation needs no chest sensor, so it is still calculated when the chest is missing, even though
  the rotation itself can no longer be detected in that recording (see
  [windows](#windows-stabilization-and-pairing-without-a-sensor)).
* **A recording with all 14 sensors gives exactly the outputs it gave before.** This was checked byte
  for byte on the June pair.
* **What a recording lacks is shown wherever it matters:** on the Sensor names card, in the
  recalculation log, in `sensor_coverage.csv` next to the metric tables (one row per recording and
  metric: measured, stand-in, or not calculated and why), on the Kinematics check page and in the
  event editor.

## How each decision was made

Each sensor was removed in turn from the two complete June recordings (novice and trained), and the
analysis was run again and compared with the real values over the 76 curated windows: 12 trunk
rotations, 8 single-leg stances and 56 turns.

**Stand-ins for a metric's value: the rule of ten.** A stand-in is used only if, in 95 % of windows,
it moves the metric by less than a tenth of that metric's spread (its interquartile range across the
windows), in every family the metric appears in. A tenth follows the "rule of ten" of metrology: an
error that small cannot reorder two windows unless they are near-ties. The rule was first set to
the ±0.25 s boundary-jitter noise of the boundary-robustness check. It was dropped because, for a
range measured over a long window (such as weight transfer over a turn), that noise is exactly zero.
The hand stand-in, which moves the centre-of-mass height by about half a millimetre, would have
failed it.

**Fallbacks for window placement and pairing.** Automatic windows are a proposal that is reviewed in
the event editor, and a hand-placed window gives the same metric whichever sensor placed it. So a
fallback that replaces a recording's own detector is judged by its boundaries. It must put at least
90 % of them within ±0.25 s of where the full method does; that is the precision of a manual drag,
and the jitter of the boundary-robustness check. A pairing fallback must keep every curated
novice–trained pair the same movement of the form. Windows *carried over* from the other recording
are different: they exist only where the recording has no signal of its own for that family. They
are judged as a pair is (the same movement, IoU ≥ 0.5) and marked in the editor until checked.

**What "measured" means.** Removing a sensor also shifts, a little, the quiet-standing spans the
gyroscope bias is read from, because they are found from all sensors together. So even a metric
that does not need the missing sensor can move. In the simulations no such metric moved by more
than 0.53 of its tolerance (the chest–pelvis correlation of the turns, without a hand); most moved
by nothing at all.

## Overview

| Missing | Not calculated (empty) | Stand-in or fallback |
| --- | --- | --- |
| **lumbar** (pelvis) | All sway, tilt and pelvic angular-speed measures (every stabilization window, the single-leg support); trunk–pelvis turning and lags; pelvis excursion, chest–pelvis gain and twist range; hip flexion; peak knee flexion; the centre of mass; `stabilization_quiet_z`, `time_to_stabilization_s` | Neutral pose found from all the other sensors; stabilization windows placed on the chest alone; trunk rotations carried over from the other recording; alignment on chest yaw and knees |
| **chestbone** (sternum) | Trunk–pelvis turning and lags (trunk rotation, single-leg); every chest-turn measure (excursion, speed, SPARC, submovements, jerk); chest–pelvis lag, gain and twist range; the centre of mass; `stabilization_quiet_z`, `time_to_stabilization_s` | Stabilization windows placed on the lumbar sensor alone; trunk rotations and turns carried over from the other recording; alignment on pelvis yaw and knees |
| **thigh** (left or right) | Peak lift and knee extension while lifted (every stance); that leg's peak knee and hip flexion; lumbar tilt at the highest lift; single-leg trunk–pelvis lags; the centre of mass | Single-leg stances carried over from the other recording; alignment without that knee |
| **shank** (left or right) | Peak lift and knee extension while lifted (every stance); that leg's peak knee flexion; lumbar tilt at the highest lift; single-leg trunk–pelvis lags; the centre of mass | Stances still detected, with the thigh's tilt standing in for the shank's in the lift index; alignment without that knee |
| **foot** (left or right) | Nothing | Centre of mass: the foot moves with the shank (ankle in its neutral angle) |
| **upper arm** (left or right) | Centre-of-mass height range and lowering | Weight transfer: the upper arm along the forearm (if it is the only arm segment missing besides hands) |
| **forearm** (left or right) | Centre-of-mass height range and lowering | Weight transfer: the forearm along the hand, or along the upper arm if the hand is missing too |
| **hand** (left or right) | Nothing | Centre of mass: the hand along the forearm |

The 50 Hz kinematic export (`kinematic_variables_50hz.csv`) leaves every channel empty whose sensors
are missing, and the Kinematics check page leaves out those traces.

## Per sensor

### Lumbar (pelvis)

The most important sensor. It is the reference every trunk measure is taken from.

* **Used for:** the neutral pose every orientation is referenced to; all sway, tilt and pelvic
  angular-speed measures; the pelvis's turning (trunk–pelvis yaw and lags, pelvis excursion,
  chest–pelvis gain and twist range); hip angles; the pelvis and lower trunk of the centre of mass;
  half of the stabilization quietness signal; the trunk–pelvis yaw channel of the alignment.
* **Not calculated without it:**
  * the whole trunk-rotation family except the window times;
  * of the single-leg stance: peak knee and hip flexion, the lumbar tilt at the highest lift, all
    `support_*` and `settle_*` measures, `time_to_stabilization_s`, `stabilization_quiet_z` and the
    lags;
  * of the turns: pelvis excursion, chest–pelvis lag, gain and twist range, and the centre of mass.
* **Still calculated:** single-leg support duration, peak lift, knee extension while lifted, and
  every chest-turn measure (excursion, peak speed, SPARC, submovements, jerk).
* **Stand-ins and fallbacks:**
  * **Neutral pose.** It is normally the stillest 2 s of the lumbar sensor after the sync movement.
    Without the lumbar sensor, the same search runs on all the other sensors together: the sync peak
    of their summed |acceleration − 1 g|, then their stillest 2 s. That lands on a different still
    moment (2–8 s away in the June recordings), where the arms are held a little differently. The
    lift index and every chest-turn measure stay well within tolerance (worst 0.05 of it), but
    peak knee flexion moves by up to 8° (6.2× the tolerance). That is why it is not calculated.
  * **Stabilization windows** are placed on the chest's angular speed alone (see
    [windows](#windows-stabilization-and-pairing-without-a-sensor)).
  * **Centre of mass:** the chest cannot stand in for the pelvis. Moving the hip line and the lower
    trunk with the chest is 3–36× the tolerance.

### Chestbone (sternum)

* **Used for:** the chest's turning (the turns of the form, trunk–pelvis twist); the thorax, head
  and shoulders of the centre of mass; half of the stabilization quietness signal; the chest yaw
  and trunk–pelvis yaw channels of the alignment.
* **Not calculated without it:**
  * of the trunk rotation: trunk–pelvis lag and correlation, rotation SPARC, excursion and peak
    speed, the centre of mass and `stabilization_quiet_z`;
  * of the single-leg stance: the centre of mass, `time_to_stabilization_s`,
    `stabilization_quiet_z` and the trunk–pelvis lags;
  * of the turns: every chest measure, the chest–pelvis lag, gain and twist range, and the centre
    of mass.
* **Still calculated:**
  * every lumbar sway, tilt and angular-speed measure, in the stabilization windows after trunk
    rotations and in the single-leg support and settling;
  * peak lift, knee and hip flexion;
  * pelvis excursion over each turn.
* **Stand-ins and fallbacks:**
  * **Trunk rotations and turns** cannot be detected without the chest (both are defined by its
    turning). With two recordings selected, they are carried over from the other recording.
  * **Stabilization windows** are placed on the lumbar sensor alone.
  * **Alignment:** pelvis yaw replaces chest yaw.
  * **Centre of mass:** the pelvis cannot stand in for the thorax. Turning the shoulders and arms
    with the pelvis instead of the chest moves the weight transfer over the turns by 1.8× the
    tolerance and the height range over the rotations by 3.6×.

### Thigh (left or right)

* **Used for:** the lift index (how much higher one ankle is than the other, from the tilt of both
  thighs and shanks), which finds the single-leg stances and measures peak lift; the knee and hip
  angles of that leg; the leg chain of the centre of mass; that knee's alignment channel.
* **Not calculated without it:**
  * peak lift, knee extension while lifted, the lumbar tilt at the highest lift and the single-leg
    lags, for every stance (they all need the measured lift index);
  * peak knee and hip flexion of stances that lift that leg;
  * the centre of mass.
* **Stand-ins and fallbacks:**
  * **Stances.** Taking the shank's tilt for the thigh's loses stances: a knee lift raises the
    thigh while the shank stays vertical, so 4 stances became 3 in both recordings. Stances are
    therefore carried over from the other recording.
  * **Lift index and centre of mass:** no stand-in. The shank for the thigh is 17–43× the
    tolerance.

### Shank (left or right)

* **Used for:** as the thigh, plus that leg's ankle angle.
* **Not calculated without it:**
  * peak lift, knee extension while lifted, the lumbar tilt at the highest lift and the single-leg
    lags, for every stance;
  * peak knee flexion of stances that lift that leg;
  * the centre of mass.
* **Stand-ins and fallbacks:**
  * **Stances are still detected in the recording itself.** In the lift index the thigh's tilt
    stands in for the shank's (placement only, not the peak-lift value). In the June recordings
    this finds the same stances with 100 % of boundaries within 0.25 s, plus one extra stance in
    one recording, which the editor shows. The value of peak lift with this stand-in is 18–19× the
    tolerance, so it is not reported.
  * **Centre of mass:** no stand-in (the thigh for the shank is 20–24× the tolerance).

### Foot (left or right)

* **Used for:** the foot segment of the centre of mass (1.45 % of body mass) and that ankle's angle.
  The step detection of the centre of mass does not use the feet, by design.
* **Not calculated without it:** nothing (the ankle angle is left empty in the export).
* **Stand-in:** the foot moves rigidly with the shank, with the ankle at its neutral-pose angle.
  Worst case 0.76 of the tolerance (height range over trunk rotations); both feet 0.56.

### Upper arm (left or right)

* **Used for:** the upper arm of the centre of mass (2.8 % of body mass), and shoulder and elbow
  angles (export and Kinematics check only).
* **Not calculated without it:** the centre-of-mass height range and lowering, in every family.
  Raising an arm raises the centre of mass, so a wrong arm elevation is a wrong height. The upper
  arm along the forearm is 10–17× the tolerance on the height range, and holding the arm hanging is
  no better.
* **Stand-in:** weight transfer is calculated with the upper arm along the forearm (0.80–0.86 of
  the tolerance). The weight index is a horizontal position divided by the ankle-to-ankle distance,
  and the arm's elevation barely enters it.

### Forearm (left or right)

* **Used for:** the forearm of the centre of mass (1.6 % of body mass) and the elbow angle.
* **Not calculated without it:** the centre-of-mass height range and lowering (forearm along the
  hand: 1.8–3.9× the tolerance; along the upper arm: 3–6×).
* **Stand-in:** weight transfer is calculated with the forearm along the hand (wrist straight), or
  along the upper arm if the hand is missing too (0.24–0.49 of the tolerance).

### Hand (left or right)

* **Used for:** the hand of the centre of mass (0.6 % of body mass), and one channel of the v1
  trunk detector's DTW matching.
* **Not calculated without it:** nothing.
* **Stand-ins:**
  * **Centre of mass:** the hand points along the forearm (wrist straight). Worst case 0.68 of the
    tolerance for one hand, 0.86 for both.
  * **v1 detector:** the hand's channel is left out of the DTW signature in both recordings.

### Several sensors missing

Each sensor's rules apply on their own: a metric is calculated only if none of its sensors is
missing without a stand-in. The centre-of-mass stand-ins were checked for every one of the 255
combinations of missing hands, forearms, upper arms and feet. This is exact, because the centre of
mass is linear in the segment positions and, with the legs measured, the ankles do not move: the
error of a combination is the sum of the per-arm and per-foot errors (checked to 10⁻¹³).

* **Height range and lowering** are calculated if the only missing centre-of-mass sensors are hands,
  or only feet: 6 combinations, worst 0.86 of the tolerance. A missing hand together with a missing
  foot is left out. Three of the nine such mixes exceed the tolerance on the height range (left foot
  with right hand 1.14×, with both hands 1.31×, all four 1.06×), and near the threshold the errors do
  not grow in a simple way, so no simpler rule would hold.
* **Weight transfer** is calculated if, besides any hands and feet, at most one forearm or upper arm
  is missing: 79 combinations, worst 0.90. With two or more forearms or upper arms missing, 120 of
  the 176 combinations exceed the tolerance, so none of them is used.
* If the lumbar, chest, a thigh or a shank is missing, the centre of mass is not calculated.

With neither trunk sensor, there is no stabilization signal. Stabilization windows then start at the
end of their event and every stabilization measure is empty; stances are still detected from the
legs.

## Windows, stabilization and pairing without a sensor

Event windows come from detectors that each need certain sensors.

| Windows | Detected from | Without its sensors |
| --- | --- | --- |
| Trunk rotations | trunk–pelvis yaw (lumbar, chestbone) | carried over from the other recording through the alignment; none in a single-recording analysis |
| Single-leg stances | lift index (both thighs and shanks) | missing shank: detected with the thigh standing in; missing thigh: carried over from the other recording |
| Turns | chest yaw (chestbone) | carried over from the other recording; none in a single-recording analysis |
| Stabilization | angular speed of lumbar + chest | the one trunk sensor present; with neither, the window starts at the event's end |
| Novice–trained pairs | whole-recording alignment on chest yaw, trunk–pelvis yaw and both knees | the channels both recordings have; pelvis yaw replaces chest yaw when either lacks the chest |
| Neutral pose (calibration) | the lumbar sensor's stillest 2 s after the sync movement | the same search on all other sensors together |

**Carried over from the other recording.** When one recording cannot detect a family, each of the
other recording's windows is carried onto its clock by the whole-recording alignment. Its
stabilization window is then searched for in the recording itself. In the June pair, compared with
the trained recording's own curated windows, carried-over windows overlap them by an IoU of 0.85
(trunk rotations), 0.91 (turns) and 0.89 (stances) at the median (at least 0.54). Boundaries are
0.21–0.34 s apart at the median, and 67–84 % are within 0.5 s. That is the same movement cut slightly differently, so
these windows are marked *carried over* in the editor's event list and reported in the
recalculation log. Check them against the recording's own signals before relying on their
movement-window metrics.

**Stabilization on one trunk sensor.** 95 % of the automatically placed windows land within 0.25 s
of where the two-sensor search puts them, with the lumbar sensor alone and with the chest alone
(the criterion is 90 %). The rest move up to 1.8 s (lumbar) or 0.5 s (chest). The quietness score of a window is then a different
quantity (one sensor's distribution instead of two), so the editor still uses it to flag an
unsettled window, but `stabilization_quiet_z` is not reported. `time_to_stabilization_s` *is* the
automatic placement (from touch-down to the settling window), and it moves by 10–13× its tolerance,
so it is not reported either.

**Alignment on fewer channels.** Every combination simulated kept all 38 curated pairs the same
movement:

| Channels (missing sensor) | Map vs the full alignment: median, 95th percentile, max |
| --- | --- |
| pelvis yaw + both knees (no chest) | 0.2 s, 0.6 s, 1.2 s |
| both knees only (no chest, without the pelvis yaw stand-in) | 0.2 s, 1.0 s, 2.4 s |
| chest yaw + both knees (no lumbar) | 0.0 s, 0.2 s, 0.6 s |
| chest yaw, trunk–pelvis yaw, one knee (no thigh or shank) | 0.0 s, 0.3–0.4 s, 1.4–1.5 s |
| pelvis yaw + one knee (no chest, no thigh or shank) | 0.2 s, 1.2 s, 4.0 s |

The full alignment itself puts each novice stance within 0.1–0.7 s of its trained partner.

## The validation numbers

Each ratio is the 95th percentile of the error, over the windows, divided by the tolerance (a tenth
of the interquartile range). It is the worst over the metric's families. ≤ 1 passes.

**Centre of mass, one stand-in at a time:**

| Stand-in | Weight transfer | Height range | Lowering | Used |
| --- | --- | --- | --- | --- |
| left hand → forearm | 0.04 | 0.31 | 0.44 | all three |
| right hand → forearm | 0.09 | 0.68 | 0.56 | all three |
| both hands → forearms | 0.11 | 0.86 | 0.85 | all three |
| left foot → shank | 0.02 | 0.76 | 0.19 | all three |
| right foot → shank | 0.02 | 0.72 | 0.24 | all three |
| both feet → shanks | 0.03 | 0.56 | 0.19 | all three |
| left forearm → hand / upper arm | 0.24 / 0.26 | 1.76 / 3.12 | 2.53 / 1.75 | weight transfer |
| right forearm → hand / upper arm | 0.49 / 0.24 | 3.87 / 6.02 | 3.24 / 1.97 | weight transfer |
| left upper arm → forearm | 0.86 | 9.82 | 5.91 | weight transfer |
| right upper arm → forearm | 0.80 | 16.73 | 6.68 | weight transfer |
| chest → lumbar (rigid trunk) | 1.76 | 3.63 | 1.37 | none |
| lumbar → chest (rigid trunk) | 2.94 | 35.5 | 8.57 | none |
| thigh → shank, shank → thigh | — | 20–52 (worst metric) | | none |

**Metrics without the lumbar sensor, with the neutral pose from the other sensors:** peak lift 0.04,
knee extension while lifted 0.14, chest excursion, speed, SPARC, submovements and jerk ≤ 0.03; peak
knee flexion 6.2 (not calculated).

## The recordings in data/

| Recordings | Missing | What is lost |
| --- | --- | --- |
| `IMU_Trial_1_RC_Novice`, `IMU_Trial_3_RC_Trained`, `IMUmama_testmoment1`, `IMUPOCpapa_eerstetestmoment` (3) | — | nothing |
| `IMUmama_testmoment1_2`, `_3`, `_4` | right hand | nothing (centre of mass with the hand along the forearm) |
| `IMUPOCmama_derdetestmoment` (3), `IMUPOCpapa-tweedetestmoment` (3), `IMUPOCpapa-derdetestmoment` (3) | left hand | nothing (centre of mass with the hand along the forearm) |
| `IMUPOCmama_tweedetestmoment` (3) | chestbone | trunk-rotation movement metrics, every chest-turn measure, chest–pelvis coordination, the centre of mass, `stabilization_quiet_z`, `time_to_stabilization_s`; trunk rotations and turns carried over from the other recording |

## Re-running the checks

`python3 analysis/tools/missing_sensor_checks.py` (about 75 s) removes each sensor in turn from two
complete recordings, rebuilds them as stage 1 would have, recalculates every metric on the same
windows, and checks every rule above:

* every column marked *not calculated* comes out empty, every stand-in stays within the tolerance,
  and so does every *measured* column;
* every combination the centre-of-mass rules allow stays within the tolerance (all 255 are tried);
* the refused stand-ins are indeed beyond it;
* the stabilization search on one trunk sensor, and the stance detector with the thigh standing in
  for a shank, place at least 90 % of the boundaries within 0.25 s;
* every reduced alignment keeps every curated pair the same movement;
* carried-over windows overlap the recording's own windows by an IoU of at least 0.5.

It prints the numbers quoted in this file, and exits with status 1 if any rule no longer holds.

## Limits

* The tolerances and every verdict come from two recordings of two participants (76 windows).
  Verdicts well inside or outside the tolerance (the hands and feet, the trunk sensors, the legs) will
  not change with more data. Those near it, such as the upper arm for weight transfer (0.80–0.86),
  might. The check tool takes any two complete recordings (`--novice`, `--trained`), so the verdicts
  can be re-checked on the new complete recordings once their windows are curated.
* A stand-in is validated against the analysis's own model. It shows that the stand-in does not
  change the numbers beyond the tolerance. It does not show that the numbers are right, which needs
  an independent reference such as the video-based body models planned for later.
* The v1 detectors are kept to reproduce the June results, which had every sensor. With a sensor
  missing they do only what their sensors allow: no v1 trunk rotations without the lumbar and chest
  sensors, and the DTW matching without the missing channels.
