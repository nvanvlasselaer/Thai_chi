# Dashboard — user manual

A guide to the dashboard the analysis is run from: running the pipeline, reading the event editor, adjusting event windows, tuning the detector, and interpreting what comes out.

See [README.md](../README.md) for installation.

---

## Getting started

Start the app from the repository root:

```bash
python3 app.py
```

A browser tab opens at http://127.0.0.1:8051. The header has four pages — **Pipeline**, **Kinematics check**, **Event editor** and **Results** — and on the right a status line for whatever is running in the background. Each page has its own address (`/pipeline`, `/check`, `/editor`, `/results`), so a refresh stays where you are.

### The Pipeline page

This is where the analysis starts. Each stage has a card saying whether it is **done**, **out of date** or **not yet** done, with the button that runs it again:

| card | what it does |
| --- | --- |
| Recordings | which recordings to analyse — see [Choosing recordings](#choosing-recordings) below |
| 1 · Orientation and kinematics | runs the orientation filter on every sensor of each selected recording and writes its orientation cache, 50 Hz joint angles, sensor inventory, orientation validation figure and sensor check. About 10 s per recording, done once: every selection that uses the recording reuses it, until the file itself changes. The card says which sensors, if any, need a look; **Check sensors and kinematics** opens the [Kinematics check](#the-kinematics-check-page) page |
| 2 · Recordings loaded | loads the selected recordings into the app, for the editor. A few seconds, and it happens by itself when the server starts once stage 1 has been run |
| 3 · Event windows | the session you edit in the event editor — one per selection. **New session** starts over from the source chosen beside it — the v2 detectors or the v1 detectors (fixed 8 s windows, to reproduce earlier windows) — and keeps the session it replaces in `_previous` (see [§4](#saving-and-loading-sessions)). A session saved by an older version is migrated when it is opened; the original is kept as `sessions/pre-schema-2-<name>.json` |
| 4 · Metrics and figures | recalculates everything from the session. The card says **out of date** as soon as a window, a setting or a recording file has changed since the last recalculation |

**Run pipeline** does whatever is missing or out of date, in order. On a fresh copy of the repository that is all of it, about 25 s; after that it usually only recalculates. While a job runs, a progress bar and a log appear under the cards and the buttons are greyed out until it finishes. A job keeps running if you switch page, and the page you are on picks up its result when it finishes: a new session appears in the editor without reloading.

### Choosing recordings

Every CSV in `data/` — subfolders included, under any name — appears in the two dropdowns of the Recordings card, one for the **novice** role and one for the **trained** role, each marked *processed*, *not processed yet* or *preprocess again*. A file that is not a readable Delsys export is listed but greyed out, with the reason.

Pick a recording for each role, or clear one (×) to analyse a single recording, and press **Use these recordings**. The line underneath says which folder the selection's outputs go to — `outputs/analyses/novice-<recording>__trained-<recording>/`, or `novice-<recording>` alone — and whether that analysis already exists. Switching is instant when the recordings have been processed before; otherwise the cards show what is missing and **Run pipeline** processes only the new recordings.

Each selection keeps its own session, named sessions, metrics and figures, so switching back later returns to the curation exactly as you left it. With one recording selected, the editor shows one graph and every event has one window; the tables and figures hold that recording alone.

Every output says where it came from: the folder is named after the recordings, `analysis.json` and each recording's `recording.json` record the exact file (name, size, SHA-256), every table has a `recording` column, and every figure names its files underneath.

### The Kinematics check page

Open it after stage 1, before placing any event: it shows whether each sensor sits where the analysis assumes, and whether the movement it reconstructs looks like the person in the video. Choose the recording at the top.

**The sensor table** is the recording's `sensor_check.csv`. Rows marked **check** are shaded, with the reason in *notes*:

| column | what it says | marked **check** when |
| --- | --- | --- |
| mounting tilt | how far the axis the mounting assumes is vertical sits from vertical in the neutral pose. The trunk sensors sit 13–18° off on the lordosis and the sternum, the feet ~40° on the slope of the instep | over 45°: the sensor is rotated or turned over on its segment |
| \|g\| at rest | the accelerometer's reading in the neutral pose | not 1 g ± 0.05 |
| gyro bias | what the gyroscope reads while nothing moves; it is subtracted from every angular speed | — |
| still | the sensor's angular speed in the stiller of the two quiet standing spans | more than 3 robust SDs above the median sensor: a loose strap, or that segment moved |
| knee axis off x | how far each knee's flexion axis lies from the one assumed | over 45° (20–45° is a note: part of the knee flexion shows up in the other components) |
| moves the allowed way | of the time a thigh is raised or a knee bent, how often the thigh points forward and the shank swings back | under 50 %: the sensor is turned around on its segment, or left and right are swapped — which the neutral pose cannot show |
| missing | missing samples | any |

Nothing on this page changes the analysis: a sensor marked **check** is for you to look at. If one really is misplaced, the metrics that use it cannot be trusted: the thigh and shank sensors for the single-leg stances, the chest and lumbar sensors for every family.

**The stick figure** shows the recording at the time on the slider: left segments blue, right red, the grey outline the neutral pose, and a head drawn on the chest's axis (there is no head sensor). Drag it to rotate; the **view** buttons switch between a diagonal view and straight front, side and top views, which show angles without perspective. **Jump to a moment** goes to the moments where a misplaced sensor shows most:

- *neutral pose*: the figure stands upright, arms hanging;
- *deepest left / right knee bend* and *highest left / right foot lift*: the figure's leg on that side, and only that one, bends or rises, and in the video it must be the person's leg on the same side. If it is their other leg, the left and right leg sensors are swapped;
- *largest trunk turn each way*: the chest turns on the pelvis.

Compare each with the video. **De-drift headings** (on by default) takes out each sensor's slow turn about the vertical: without a magnetometer every sensor's heading drifts on its own (the arm sensors by up to 115° over these recordings), and without the correction the figure slowly twists apart. It also takes out a slow turn of the whole body, so the figure always faces roughly forward. **Sensor axes** draws each sensor's x (red), y (green) and z (blue).

**The joint angles** beside it are grouped by joint, with left and right overlaid, the neutral pose shaded green and the quiet standing spans grey. Click anywhere in them to move the figure there. Every angle is measured from the neutral pose. For the legs, the panel titles say which sign is which: hip flexion positive, knee flexion negative, ankle dorsiflexion positive, and hip abduction positive (the right side mirrored so both read the same way). The arms are shown as **shoulder elevation** (the upper arm's angle from the trunk's axis) and **elbow bend** (the angle between upper arm and forearm), which ignore rotation about each segment's own axis. The elbow's Euler angles in the 50 Hz export pass through their singularity in Tai Chi's arm positions and are not usable.

What to look for: both knees flexing the same sign, never much past 0 the other way; hip flexion to ~90° when a leg is lifted; similar ranges on the two sides for a symmetric form; no slow drift in the trunk and pelvis angles; shoulder elevation near 0 while the arms hang.

The same table and panels are available outside the dashboard: `python3 analysis/tools/plot_kinematics.py [--role Trained] [--save file.png]`.

### The Results page

Shows the metric tables — pick one with the buttons above the table; every column sorts — and every figure, each opening full size when clicked. The line at the top says which session the numbers were computed from and whether they still match it.

Everything on this page is read back from `outputs/`, so it is exactly what someone opening that folder would see.

---

## 1. Why this tool exists

Every balance metric in this analysis is computed **over a time window**, not over the whole recording. 
There are two windows per event:

- the **movement window** — the trunk rotation itself, or the single-leg stance itself;
- the **stabilization window** — the settling period immediately afterwards, where you measure how well the person recovers their balance.

Get those windows wrong and the numbers are still produced, still look plausible, and are still wrong. If the movement window is twice as long as the movement, you are averaging over several seconds of someone standing still. 
If the stabilization window lands while the person is still turning, you are measuring the movement, not the recovery.

The software places both windows automatically, and gets most of them approximately right and some of them badly wrong. This editor is where you check its work and correct it.

---

## 2. The screen at a glance

```
 Tai Chi balance analysis   Pipeline  [Event editor]  Results      ● status
┌────────────────────┬───────────────────────────────────────────────────────┐
│ SESSION            │  fs 370.37 Hz | Novice 241 s / Trained 236 s | ...    │
│  [saved ▾] [Load]  │                                                       │
│  [name___] [Save]  ├───────────────────────────────────────────────────────┤
│  [New session]     │                                                       │
│ DETECTOR — <tab>   │  [ Trunk rotation ] [ Single-leg stance ]             │
│  settings for the  │  [ Sequence turns ]                                   │
│  open tab, plus    ├───────────────────────────────────────────────────────┤
│  stabilization     │                                                       │
│  [Re-detect <fam>] │                                                       │
│ EVENTS             │  warnings and errors for the current windows          │
│  id  what  window  ├───────────────────────────────────────────────────────┤
│  ▸ trunk-01 …      │                                                       │
│    trunk-02 …      │  NOVICE                                               │
│  [Toggle] [Delete] │   1  trunk-pelvis yaw        ▓▓▓▓▓▓    ░░░░           │
│  [Confirm] [Swap]  │   2  yaw speed envelope      ▓▓▓▓▓▓    ░░░░           │
│ ADD AN EVENT       │   3  lumbar + chest ang.vel. ▓▓▓▓▓▓    ░░░░           │
│  [trial▾] [leg▾]   │   4  leg lift                ▓▓▓▓▓▓    ░░░░           │
│  [+ Add event here]│   5  knee flexion            ▓▓▓▓▓▓    ░░░░           │
│                    │   6  chest + pelvis yaw      ▓▓▓▓▓▓    ░░░░           │
│                    │  TRAINED                                              │
│                    │   (same six plots, independent time axis)             │
│ DELETED EVENTS     ├───────────────────────────────────────────────────────┤
│  [select ▾]        │  Recalculated metrics (table)                         │
│  [Restore selected]│  Regenerated figures                                  │
│                    │                                                       │
│ BOUNDARIES (s)     │                                                       │
│  Novice   Trained  │                                                       │
│  ev start ev start │                                                       │
│  ev end   ev end   │                                                       │
│  stab st  stab st  │                                                       │
│  stab end stab end │                                                       │
│ [Recalculate]      │                                                       │
└────────────────────┴───────────────────────────────────────────────────────┘
```

**Three tabs**, labelled `Trunk rotation`, `Single-leg stance` and `Sequence turns`. Events in all three families are **paired**: one event holds a window in each recording, so selecting it shows the same movement in the novice and the trained participant side by side, and every metric is reported for both.

The first two tabs hold **events** — moments picked out of the recording, each followed by a settling period. The third holds **turns**, which are different in kind: they tile the moving passages of the form end to end rather than being picked out of it, and they have no stabilization window, so the green band and its two boundary boxes grey out on that tab. See [§10](#10-sequence-turns-the-whole-form).

**Pairs come from a whole-recording alignment**, never from the order of events: the two recordings are warped onto each other once (dynamic time warping on chest yaw, trunk–pelvis yaw and both knees), and an event of one recording pairs with the event of the other that covers the same stretch of the form. On these recordings the same moment of the form occurs about 3.4 s later in the novice recording throughout. Single-leg stances pair only with a stance of the same leg.

Where an event has no partner, it becomes **single-sided**: it keeps a window for the recording it was found in, and the other participant's boundary boxes grey out. Such an event contributes only to the trial it has a window for, and to no paired comparison.

**Two graphs.** Novice on top, Trained below. They pan and zoom **independently**, because the two recordings are not on the same clock. For the selected event, each graph also shows a **dashed outline where its partner lands** once carried onto this graph's clock: green when the two windows are the same movement, red when they are not.

**Coloured bands.** Amber = movement window. Green = stabilization window. Faint grey = other events.

---

## 3. Reading the six plots

All six rows share one time axis.

### Row 1 — trunk-pelvis yaw (deg)

Rotation of the chest **relative to the pelvis**, about the vertical axis. This is axial trunk twist: positive and negative correspond to turning one way and the other. It is a *relative* angle between two sensors.

This is the signal that defines a trunk-rotation event. A clean event looks like a ramp from one plateau to another, typically 40–70° on these recordings.

### Row 2 — yaw speed envelope (deg/s)

How fast that twist is changing, smoothed. **This is the row that determines the movement boundaries**, so it is the one to look at when judging whether a band is placed correctly.

A trunk rotation appears as a clear hump. The movement starts where the envelope rises off the baseline and ends where it settles back. The dotted horizontal line is the **onset/offset floor** — the level the detector walks out to. If a band looks too narrow or too wide, compare it against this line first; usually the floor is in the wrong place rather than the algorithm being confused.

### Row 3 — lumbar + chest angular speed (deg/s)

Three traces: the lumbar sensor, the chest sensor, and their sum in bold, all with the **gyroscope bias removed** (it was about 6.6 °/s on the lumbar sensor and 15 °/s on the chest — more than the quiet level itself). The sum is what the stabilization search minimises, so **this is the row for judging the green band**.

The dotted line is the **quiet baseline** — the recording's 20th percentile, roughly "how still this person gets when they are being still" — and the dashed line is its **median moment**. A good stabilization window sits in a trough near the dotted line; a green band above the dashed line is flagged `unsettled`.

### Row 4 — leg lift (thigh lengths)

**This is the row that defines a single-leg stance.** Positive means the left foot is higher than the right, negative the right foot higher, in units of thigh length (0.25 is about 10 cm). It comes from the tilt of both thighs and shanks against gravity, so it sees a lift whatever the knee does — a knee lift and a kick with the knee straight alike — and says which leg is up. The dashed lines are the **minimum lift** a stance must reach (±0.25 by default); the amber band of a stance should run from where the curve leaves zero to where it returns.

### Row 5 — knee flexion |x| (deg)

Absolute flexion angle of the left and right knee. The dashed line is the threshold of the **legacy knee-threshold method** (60°), which you can still select for stances; with the default lift method this row is context — it shows what the knee did during a lift, and whether a trunk rotation coincided with a deep knee bend.

### Row 6 — chest and pelvis yaw (deg)

Rotation of the chest and of the pelvis about the vertical, each de-drifted. Row 1 shows the *difference* between these two; this row shows them separately, so you can see whether a twist came from the chest turning, the pelvis staying, or both.

**This is the row that defines a sequence turn.** A turn runs from one turning point of the chest yaw to the next — a peak one way to a peak the other way — and the two dashed lines are the **minimum turn from neutral** a peak must reach to count (±10° by default).

Read the two traces together on the other tabs as well: where they run parallel the trunk is turning as a unit, and where the pale pelvis trace flattens while the chest keeps going, the person is twisting rather than turning.

---

## 4. Adjusting event windows

### Navigating

| action | how |
| --- | --- |
| pan along time | click and drag on any plot |
| zoom | scroll wheel |
| reset the view | double-click |

Zoom is preserved when you edit, so you can zoom into one event and work there.

### Moving a boundary

**Drag.** Grab an edge of the amber or green band and pull. The band spans all four plots, so you can grab it on whichever row you are using to judge the placement — the yaw-speed row for a movement edge, the angular-velocity row for a stabilization edge.

**Type.** The four boxes under *Boundaries (s)* take times in seconds, to two decimals. Use the spinner arrows for 0.01 s nudges. Typing and dragging stay in sync: drag a band and the numbers update; type a number and the band moves.

If you enter a start later than its end, the editor swaps them rather than creating an invalid window.

### Other event actions

- **Toggle on/off** — excludes an event from recalculation without deleting it. Use this when an
  event is spurious but you want a record that you saw it. Disabled events show the flag `off` and
  are drawn faintly.
- **Delete** — moves the event to the *Deleted events* list. It is not lost; see below.
- **Confirm pair** — accepts a pair the alignment says is not the same movement (a red outline, the
  flag `mismatch`, and an error that blocks recalculation). Use it only when you have checked the two
  windows are the same movement; the flag becomes `confirmed` and the error a warning. Press again to
  undo.
- **Swap leg** — single-leg stances only: exchanges the lifted and the stance leg. Validation errors
  when the labelled leg is the lower foot in the window, which is how a stance entered with the wrong
  leg shows up.

All changes save immediately to the selection's `event_editor_session.json` (in `outputs/analyses/<selection>/`). There is no separate save step,
and your work survives restarting the app.

### Adding an event the detector missed

Automatic detection keeps only the highest-scoring events, so a genuine movement can be left out —
particularly a gentle one, or one close in time to a larger movement.

1. **Zoom to where the event belongs.** The new event is placed at the centre of the currently
   visible time range, so frame the movement before adding it.
2. On the *Single-leg stance* tab, choose which **leg is lifted**, and whether the event is created in
   `Both` recordings (the default, giving a paired event) or in just one — use a single recording
   when a movement genuinely appears in only one of them. On the *Trunk rotation* tab these
   dropdowns are disabled, because a trunk event is always created in both.
3. Press **+ Add event here**.

You get a default window — 3 s for a trunk rotation, 2 s for a single-leg stance, with a
stabilization window of the current *stabilization length* immediately after — which you then drag
to fit, exactly as for a detected event. A manually added event is marked `edited` from the start,
since none of its boundaries came from the detector.

The two windows of a paired event are placed independently, each at the centre of its own graph's
view. That is deliberate: the recordings are not on the same clock, so a single shared time would put
one of them in the wrong place. **Frame the movement in both graphs before adding**, or you will have
to drag one of the two windows a long way; the partner outline shows where the alignment expects it.

### Saving and loading sessions

Your editing is written continuously to the **working session**, so nothing is lost to a crash or a
restart. A *named* session is a copy of it kept alongside, which lets you hold several curations of
the same recordings — a conservative segmentation and a permissive one, say — and move between them
instead of overwriting one with the other.

- **Save** — type a name and press it. The name is tidied into a filename (`tight windows` becomes
  `tight-windows`), the copy is written to the selection's `sessions/` folder, and that name becomes the one you
  are editing. Saving again under the same name overwrites that copy.
- **Load** — pick from the dropdown. The session replaces everything on screen: events, deleted
  events and detector settings.
- **New session from detection** — throws the current curation away and re-detects all three families
  from scratch, using the settings currently in the detector panels. (The **New session** button on
  the Pipeline page does the same from the default settings, and can also start from the v1
  detectors.) The deleted-events list is
  emptied and the session becomes unnamed again. Use it to start over, or to see what the detector
  makes of a new set of thresholds without a half-edited session in the way.

*New* differs from *Re-detect* in scope: Re-detect replaces one family and leaves the other and the
deleted list alone, whereas New resets everything.

**Loading and New both park what they replaced in `_previous`.** Load that to undo, including when
you deliberately reload the session you are already on to throw away unsaved edits. Only one step of
undo is kept, so the next load overwrites it.

Named sessions live in the selection's `sessions/` folder as readable JSON and are tracked by git, so a curation can
be committed as the provenance for whatever numbers you publish from it.

### Restoring a deleted event

Deleting moves an event to the **Deleted events** list rather than discarding it. Pick it from the
dropdown — entries are labelled with the event id, recording, leg and original time span — and press
**Restore selected**. It returns to the list in chronological order with its boundaries intact.

The list persists in the session file, so a deletion can still be undone after restarting the app.
An event keeps its original id on restore unless that id has since been reused, in which case it is
issued a new one.

---

## 5. The detector settings

These re-run automatic detection. **"Re-detect" replaces all events of the open family and discards any manual edits to them**, so tune the settings first and hand-correct afterwards, not the other way round.

The most useful thing about these controls is not the numbers themselves — it is that the plots draw the thresholds the detector actually used, so you can see *why* a boundary landed where it did.

The panel follows the tab, and its **Re-detect** button re-detects only its own family, so the other families' curated windows are never replaced by a click meant for this one; only that family's settings are recorded in the session. The stabilization settings sit below the trunk and stance panels, because the settling window is found the same way whichever movement precedes it.

Switching tabs does not reset anything — each panel keeps its values while hidden.

### Trunk rotation — movement boundaries

| setting | default | sensible range | what it does |
| --- | --- | --- | --- |
| **onset threshold** (fraction of peak) | 0.15 | 0.10 – 0.30 | Where the movement is considered to start and stop, as a fraction of that event's own peak speed. Higher = tighter windows. Measured on these recordings: 0.15 → mean 3.8 s, 0.25 → 3.0 s, 0.40 → 2.9 s, 0.60 → 1.9 s (by then it is clipping into the movement). |
| **velocity floor** (percentile) | 40 | 25 – 55 | A hard lower bound on that threshold, taken from the whole recording, so the search cannot wander off through a quiet stretch. Below 25 it has no effect here; at 70 it starts truncating real movement (mean 2.5 s), at 85 more so (1.9 s). |
| **min event duration** (s) | 1.5 | 1.0 – 2.5 | Rejects blips. Keep at or above 1.0 s: shorter windows make the smoothness and coordination metrics unreliable (§7). |
| **min yaw excursion** (deg) | 8 | 10 – 40 | Rejects small wobbles. The real rotations here are 45–70°, so 8° is permissive. Raising to 20–40° filters noise without losing genuine events; above 60° almost everything is rejected. |
| **number of events** | 6 | — | How many pairs to keep. Every candidate rotation of both recordings is considered; a novice and a trained candidate pair when each is the other's best match once aligned, and the pairs whose weaker partner scores highest (excursion × peak speed) are kept. |

The two thresholds interact: the effective bar is `max(onset threshold × peak speed, velocity floor)`, so whichever is higher wins. That is why lowering the floor below ~25 changes nothing — the relative threshold is already binding.

### Single-leg stance — movement boundaries

| setting | default | sensible range | what it does |
| --- | --- | --- | --- |
| **method** | leg lift | — | *Leg lift* finds a stance where one foot is clearly higher than the other (row 4). *Knee flexion threshold (v1)* is the original rule, the knee flexed past a threshold (row 5): it cannot see a lift with the knee straight and splits a kick where the knee extends. |
| **min lift** (thigh lengths) | 0.25 | 0.15 – 0.5 | How high a foot must rise for a stance to count — about 10 cm at the default. The stances here peak at 0.9–1.6; stepping and weight shifts stay below 0.1, so the result is insensitive to this over the whole range. |
| **onset threshold** (fraction of peak lift) | 0.15 | 0.10 – 0.30 | Lift-off and touch-down, as a fraction of that stance's own highest lift — the same rule as the trunk detector. Never below the double-support level (the median lift of the recording). |
| **min support duration** (s) | 0.5 | 0.3 – 1.0 | Rejects brief lifts. |
| knee flexion threshold, min event duration, merge gap | 60°, 0.4 s, 0.2 s | — | The knee-threshold method only. |

### Sequence turns — turn boundaries

| setting | default | sensible range | what it does |
| --- | --- | --- | --- |
| **min turn from neutral** (deg) | 10 | 5 – 20 | How far the chest must swing for a turning point to count. This decides how many turns you get, because the real turns reach 40–70° and everything below this is the chest hovering near neutral. |
| **min turning-point separation** (s) | 1.5 | 1.0 – 3.0 | Minimum spacing between successive turning points in the same direction. |
| **min turn excursion** (deg) | 10 | 10 – 30 | Rejects a turn whose yaw range is too small to score. |
| **min turn duration** (s) | 1.0 | 1.0 – 2.0 | Rejects blips. |
| **max turn duration** (s) | 15 | 13 – 20 | **The setting that keeps the pauses out.** Both recordings stand still for ~20 s at the start and end, and the chest barely turns during the ~35 s single-leg passage in the middle. Without this cap the turning points either side of a pause make one slow "turn" that is mostly not moving. |

### Stabilization window (applies to trunk and stance events)

Sequence turns have no stabilization window, so this panel is hidden on the *Sequence turns* tab.

The search scores every candidate window as `z + λ · latency`, where **z** is how busy the window is relative to the recording itself: `(mean angular speed − quiet baseline) / (median − quiet baseline)`, on row 3's bold trace. z = 0 is as quiet as the recording's quietest fifth, z = 1 as busy as its median moment.

| setting | default | sensible range | what it does |
| --- | --- | --- | --- |
| **latency penalty** (λ, per s) | 0.65 | 0.3 – 1.0 | Trades *quiet* against *soon*, in units of z per second of delay. 0.65 is the trade-off the original score made, expressed on the bias-corrected signal. λ near 0 picks the quietest window anywhere in the horizon — quiet, but arguably no longer a recovery from *that* event. |
| **search horizon** (s) | 10 | 8 – 15 | How far after the movement to look. For a stance the search starts at touch-down. |
| **length** (s) | 3.0 | 3.0 – 4.0 | How long the window is. Against ±0.25 s boundary jitter, sway measured over 2 s windows had an ICC of 0.87–0.91, over 3 s 0.96–0.97. Longer starts overlapping the next movement. |

---

## 6. Flags and warnings

### Flags in the event list

| flag | meaning |
| --- | --- |
| `edited` | at least one boundary of this event was set by hand, not by the detector |
| `mismatch` | the two windows are not the same movement once aligned (overlap column below 50 %) — **fix this one** |
| `confirmed` | a `mismatch` you accepted with **Confirm pair** |
| `unsettled` | the stabilization window is busier than the recording's median moment (z > 1) |
| `off` | disabled, excluded from recalculation |

The **overlap** column is how much of the shorter of the two windows the other covers once the novice window is carried onto the trained clock.

`unsettled` is worth deciding on deliberately. Sometimes moving the green band fixes it. Often it does not, because the person genuinely never settled: after the trunk rotations of this form that is the rule (10 of 12 windows), and it is a finding rather than a fault — those windows measure the transition into the next movement. After a single-leg stance's touch-down the windows mostly are quiet.

### Messages above the plots

**Errors** (red) block recalculation:

| error | what to do |
| --- | --- |
| *window … is out of order or outside the recording*, *… shorter than the minimum* | Fix the boundaries. |
| *the two windows are not the same movement* | The pair compares different movements. Delete the wrong one, re-detect, or drag it to where the outline is; **Confirm pair** only if you are sure they are the same movement. |
| *labelled with the left/right leg lifted, but the other foot is the higher one* | The stance has the wrong leg: press **Swap leg**. |

**Warnings** (amber) do not block, and are worth reading:

| warning | what to do |
| --- | --- |
| *event is only N s; smoothness metrics span very few cycles* | Consider whether the window really captures the whole movement. |
| *same movement, but the windows are cut differently* | The two windows cover the same movement with quite different boundaries (IoU below 0.5); check both edges. |
| *stabilization overlaps the event* | The green band starts before the amber one ends. Confirm on row 3. |
| *stabilization is busier than the recording's median moment* | The `unsettled` case above. |
| *the coordination-lag window is N s* | The lag search needs a window of at least 2 s; check the lag value. |

---

## 7. Recalculating and interpreting the results

Press **Recalculate metrics** — here, or **Recalculate** on the Pipeline page. It validates the windows, recomputes everything, rewrites the result CSVs and regenerates the four traceability figures, which appear below the table. Every table and figure is also on the **Results** page, and the Pipeline page's metrics card turns **done**.

**This overwrites the result files in place.** The previous values remain in git. Every CSV carries a `metrics_version` column: version 2 is the current definition; the numbers of the June write-up are version 1, reproduced by the git tag `metrics-v1`.

### What each metric means

The axes are **x = mediolateral, y = anteroposterior, z = vertical**. Sway is the lumbar acceleration with gravity removed, in the pelvis's heading frame; tilt is read against gravity, with no yaw in it; angular speeds have the gyroscope bias removed.

#### Trunk rotation

| metric | window | plain meaning | direction |
| --- | --- | --- | --- |
| `trunk_pelvis_lag_s` | event ±2 s | How far the chest's turn trails the pelvis's, from their turning angles. `trunk_pelvis_yaw_r` is the correlation behind it — near 1 for everyone, so it says the lag is meaningful, not who coordinates better. | Negative = the pelvis leads. Near zero = they turn together. |
| `trunk_yaw_sparc` | event | Smoothness of the trunk-on-pelvis turning speed (spectral arc length). | **Less negative = smoother.** |
| `lumbar_ml_acc_rms_mps2`, `lumbar_ap_acc_rms_mps2` | after the rotation | Side-to-side / fore-aft sway of the pelvis, m/s². | Lower = steadier. |
| `lumbar_frontal_tilt_sd_deg`, `lumbar_sagittal_tilt_sd_deg` | after | How much the pelvis tilts sideways / forward-backward. | Lower = more stable posture. |
| `lumbar_rms_angular_velocity_dps` | after | Overall rotational "busyness" of the pelvis. Needs no threshold. | Lower = quieter. |
| `lumbar_corrective_peak_rate_hz` | after | Sharp adjustments per second. Secondary: zero in most windows. | Lower = fewer corrections. |

#### Single-leg stance

| metric | window | plain meaning |
| --- | --- | --- |
| `support_duration_s`, `peak_lift_index` | support | How long on one leg, and how high the foot went. |
| `peak_knee_flexion_deg`, `knee_extension_while_lifted_deg`, `peak_hip_flexion_deg` | support | What the lifted leg did; the extension is large for a kick (60–100° here), small for a knee lift. |
| `support_ml_acc_rms_mps2`, `support_frontal_tilt_sd_deg`, `support_rms_angular_velocity_dps`, … | support | **Balance on one foot**: sway, tilt and angular activity of the pelvis while the base of support is a single foot. |
| `settle_…` | after touch-down | The same measures while settling. |
| `time_to_stabilization_s` | — | Touch-down to the start of settling. Sensitive to where the boundaries are (see below). |

`monopodal_stance_asymmetry_metrics.csv` compares **mirrored stances** — the k-th left lift with the k-th right lift (here: the two knee lifts, and the two lifts with a kick) — as signed left-minus-right differences, plus the mean absolute difference.

#### Sequence turns

See [§10](#what-each-metric-means-1).

#### Novice versus trained

`paired_comparison.csv` — the first table on the Results page — compares the two performers **pair by pair**: for every metric, each one's median, the median trained-minus-novice difference with its quartiles, and in what fraction of pairs the trained value is the higher. Because every pair is the same movement of the form, this is the comparison to read.

### How much to trust each number

Because you are placing windows by hand, `analysis/tools/boundary_robustness.py` jitters every boundary by up to ±0.25 s — roughly the precision of a manual drag — and reports how much each metric moved as an ICC: real between-event differences over the total. Above ~0.9 is dependable. On the current session almost every metric is 0.93–1.00; the exceptions are `time_to_stabilization_s` (0.51 — it is the gap between two boundaries) and the submovement rate (0.90). This is robustness to curation, **not reliability**: whether a number would come out the same if the person performed again needs a second recording.

**A lag of NaN** means the correlation peaked on the edge of the ±1 s search: there was no coordination lag to measure, and the log says so. During single-leg stances the trunk barely turns, and several of those lags are NaN — which is why they are not in the figures.

### What you can and cannot conclude

This dataset is **one novice and one trained practitioner, one recording each**. Differences between them are descriptive. You can legitimately say "in this recording, the trained participant's turns were smoother in 24 of 28 matched turns, at the same tempo"; you cannot attribute that to training, because with one participant per group there is no way to separate a training effect from two people simply being different.

Each movement of this form is performed once. The spread of a metric across the events of one recording therefore describes how varied the form's movements are, **not how consistently the person repeats a movement**; that needs repeated movements, which the next recordings are planned to include ([Recommendations_for_next_recordings.md](../Recommendations_for_next_recordings.md)).

---

## 8. Things worth knowing

**Yaw is the weak axis.** The sensors have accelerometers and gyroscopes but **no magnetometer**, so rotation about the vertical has no absolute reference and drifts. Each sensor drifts on its own — a relative angle between two sensors drifts too, it does not cancel — at a few tenths of a degree per second here. So:

- Tilt, joint flexion and everything gravity-referenced are fine.
- Yaw *angles* are de-drifted with a 0.05 Hz high-pass for display and detection; that high-pass shrinks slow movements (turns of 8–14 s by a quarter), which is why the turn metrics use the gyroscope's turning rate, integrated over the turn, instead.

The trunk detector works on yaw *speed* rather than yaw angle, which sidesteps drift.

**The gyroscope bias is removed when a recording loads**, from the stillest 10 s at its start and end. If a recording does not start and end with the participant standing still, the log warns that the two estimates disagree.

**Pairs come from the alignment**, and validation checks every pair against it; the partner outline and the overlap column show the check.

**Long yaw excursions inside a green band are real.** If row 1 ramps 20° during a stabilization window, the person is genuinely still turning; that is not drift.

---

## 9. A working routine

0. Start the app (`python3 app.py`). If any Pipeline card is not **done**, press **Run pipeline**
   and wait for it; then open the **Event editor**.
1. Open the **Trunk rotation** tab. Work down the event list in order.
2. For each event, zoom to it and check **row 2**: does the amber band start where the yaw speed
   rises and end where it settles? Drag the edges if not.
3. Check the **partner outline** and the overlap column: is the other recording's window the same
   movement? A `mismatch` must be fixed before recalculating.
4. Check **row 3**: is the green band in a trough near the dotted quiet baseline? If it sits on a
   hump, move it to the nearest genuine trough within a couple of seconds — or accept that the
   person did not settle.
5. On the **Single-leg stance** tab, judge the amber band against **row 4**: it should run from where
   the lift leaves zero to where it returns, on the side (+ left, − right) the event is labelled with.
6. On the **Sequence turns** tab, judge the bands against **row 6**: each should run from one peak of
   the chest yaw to the next. Here prefer re-detecting at a different **min turn from neutral** over
   correcting many boundaries by hand.
7. Press **Recalculate metrics**. The table shows the family whose tab is open; every family is
   recalculated regardless.
8. Read the log, and the figures below the table — the bands drawn there should match what you set.
9. Read the numbers on the **Results** page, starting with *Novice vs trained*. Commit `outputs/`
   together with the session file, so every number stays traceable to the windows it came from.

If the detector is systematically placing windows too wide or too narrow across many events, that is
a signal to adjust the **onset threshold** and re-detect rather than to correct twelve windows by
hand.

---

## 10. Sequence turns: the whole form

### What this tab is for

The other two tabs measure balance at a handful of moments. Six trunk rotations and four stances out
of a four-minute form leave most of the recording unmeasured, so nothing else describes how the form
*as a whole* is performed — whether it flows, and whether chest and pelvis turn together.

### What a turn is

The form is carried by large chest yaw rotations, so it is cut into **turns**: from one turning
point of the chest yaw to the next, a peak one way to a peak the other way. A turn starts and ends
with the chest momentarily still and turns one way throughout, which is what the smoothness measures
assume. (The earlier unit, a full back-and-forth cut where the chest crossed neutral, started and
ended at peak turning speed.) On these recordings there are 28–29 turns per recording, 2.5–13 s
long.

The turning points of the two recordings are matched through the alignment, and each turn is made
between two consecutive matched points in both recordings at once. An extra wiggle in one recording
therefore stays inside one turn, where the submovement rate and SPARC register it, instead of shifting
every later pair — which is what pairing by order did: both recordings had 13 full cycles, and from
the first one on the pairs were half a cycle apart.

Turns **abut** — one ends where the next begins. Genuine gaps appear where the form pauses; the
editor warns only if two turns *overlap* by more than 0.5 s.

### Working on this tab

Select a turn, judge it against **row 6**, drag the amber band or type into the two event boxes. The
stabilization boxes and the green band are greyed out, because there is nothing to settle from.

### What each metric means

Per turn, in `sequence_smoothness_metrics.csv`; `sequence_summary.csv` holds each recording's medians.

| metric | plain meaning | direction |
| --- | --- | --- |
| `chest_yaw_sparc` | Smoothness of the turning speed (spectral arc length), from the gyroscope. The primary smoothness measure. | Negative; **less negative = smoother**. |
| `chest_yaw_submovement_rate_hz` | How many separate speed peaks per second the turn is made of. One continuous turn has one peak; a hesitant one has more. The easiest to check by eye against row 6. | Lower = fewer interruptions. |
| `turn_duration_s`, `chest_yaw_excursion_deg`, `chest_yaw_peak_rate_dps` | How long, how far and how fast the chest turned. Report duration next to smoothness: slower movement is intrinsically less smooth. | — |
| `chest_pelvis_lag_s` | How far the chest's turn trails the pelvis's. | Negative = the pelvis leads. |
| `chest_pelvis_gain` | Pelvis turn ÷ chest turn. | 1 = the trunk turns as a unit; below 1 the chest turns on a comparatively still pelvis (~0.56 here). |
| `relative_yaw_range_deg`, `relative_yaw_sd_deg` | How much axial twist opens up inside the turn. | Larger = more dissociation between chest and pelvis. |
| `chest_yaw_log10_dimensionless_jerk` | Secondary, kept for comparability. | Lower = smoother, in principle. |

**Why jerk is secondary.** For movements this slow the jerk integral is dominated by noise near the
low-pass cutoff, which grows like duration⁶ / amplitude². On these recordings duration and amplitude
alone explain 96 % of log dimensionless jerk across the turns, so it measures the window more than
the movement. That, not a difference in what they measure, is why jerk and SPARC used to point in
opposite directions.

The figure's right-hand panel is the typical **time course of a turn** — the fraction of its own
excursion completed at each percent of its duration. Its spread shows how varied the form's turns
are; every turn being a different movement, it is not repetition variability.

---

The session file `outputs/analyses/<selection>/event_editor_session.json` is plain, readable JSON holding every boundary
in sample indices and seconds, with a note of whether each was set automatically or by hand. If the
interface ever disagrees with what you expect, that file is the authority.
