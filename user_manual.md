# Event editor — user manual

A guide to using the interactive event editor: reading the dashboard, adjusting event windows, tuning the detector, and interpreting what comes out.

See [README.md](README.md) for installation.

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
┌────────────────────┬───────────────────────────────────────────────────────┐
│ DETECTOR           │  Tai Chi event editor                                 │
│  8 numeric settings│  fs 370.37 Hz | Novice 241 s / Trained 236 s | ...    │
│  [Re-detect]       ├───────────────────────────────────────────────────────┤
│                    │  [ Trunk rotation ] [ Monopodal stance (knee > 60°) ] │
│ EVENTS             ├───────────────────────────────────────────────────────┤
│  id  what  window  │  warnings and errors for the current windows           │
│  ▸ trunk-01 …      ├───────────────────────────────────────────────────────┤
│    trunk-02 …      │  NOVICE                                               │
│  [Toggle] [Delete] │   1  trunk-pelvis yaw        ▓▓▓▓▓▓    ░░░░           │
│                    │   2  yaw speed envelope      ▓▓▓▓▓▓    ░░░░           │
│ BOUNDARIES (s)     │   3  lumbar + chest ang.vel. ▓▓▓▓▓▓    ░░░░           │
│  Novice   Trained  │   4  knee flexion            ▓▓▓▓▓▓    ░░░░           │
│  ev start ev start │  TRAINED                                              │
│  ev end   ev end   │   (same four plots, independent time axis)            │
│  stab st  stab st  ├───────────────────────────────────────────────────────┤
│  stab end stab end │  Recalculated metrics (table)                         │
│                    │  Regenerated figures                                  │
│ [Recalculate]      │                                                       │
└────────────────────┴───────────────────────────────────────────────────────┘
```

**Two tabs**, labelled `Trunk rotation` and `Monopodal stance (knee > 60 deg)`. Trunk-rotation events exist in both recordings as a matched pair — one event, two windows. Monopodal-stance events belong to a single recording, so when one is selected the other participant's boundary boxes grey out. That is expected.

**Two graphs.** Novice on top, Trained below. They pan and zoom **independently**, because the two recordings are not time-aligned — the same movement can happen at different clock times in each.

**Coloured bands.** Amber = movement window. Green = stabilization window. Faint grey = other events,

---

## 3. Reading the four plots

All four rows share one time axis, so a vertical line through them is the same instant. From top to bottom:

### Row 1 — trunk-pelvis yaw (deg)

Rotation of the chest **relative to the pelvis**, about the vertical axis. This is axial trunk twist: positive and negative correspond to turning one way and the other. Because it is a *relative* angle between two sensors, it is unaffected by the yaw drift that limits absolute heading (see §8).

This is the signal that defines a trunk-rotation event. A clean event looks like a ramp from one plateau to another, typically 40–70° on these recordings.

### Row 2 — yaw speed envelope (deg/s)

How fast that twist is changing, smoothed. **This is the row that determines the movement boundaries**, so it is the one to look at when judging whether a band is placed correctly.

A trunk rotation appears as a clear hump. The movement starts where the envelope rises off the baseline and ends where it settles back. The dotted horizontal line is the **onset/offset floor** — the level the detector walks out to. If a band looks too narrow or too wide, compare it against this line first; usually the floor is in the wrong place rather than the algorithm being confused.

### Row 3 — lumbar + chest angular velocity (deg/s)

Three traces: the lumbar sensor, the chest sensor, and their sum in bold. The sum is what the stabilization search minimises, so **this is the row for judging the green band**.

The dotted line is the **quiet baseline** — a low percentile of this signal across the whole recording, i.e. roughly "how still this person gets when they are being still". A good stabilization window sits in a trough near or below that line. A green band sitting on a hump means the person was still moving.

### Row 4 — knee flexion |x| (deg)

Absolute flexion angle of the left and right knee, with a dashed line at **60°**. Crossings of that line define monopodal-stance events. On the trunk-rotation tab this row is context: it tells you whether a trunk rotation coincided with a deep weight shift onto one leg.

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

**Type.** The four boxes under *Boundaries (s)* take times in seconds, to two decimals. Use the spinner arrows for 0.05 s nudges. Typing and dragging stay in sync: drag a band and the numbers update; type a number and the band moves.

If you enter a start later than its end, the editor swaps them rather than creating an invalid window.

### Other event actions

- **Toggle on/off** — excludes an event from recalculation without deleting it. Use this when an
  event is spurious but you want a record that you saw it. Disabled events show the flag `off` and
  are drawn faintly.
- **Delete** — removes it from the session.

All changes save immediately to `outputs/event_editor_session.json`. There is no separate save step,
and your work survives restarting the app.

---

## 5. The detector settings

These re-run automatic detection. **"Re-detect trunk events" replaces all trunk events and discards any manual edits to them**, so tune the settings first and hand-correct afterwards, not the other way round.

The most useful thing about these controls is not the numbers themselves — it is that row 2 and row 3 draw the thresholds the detector actually used, so you can see *why* a boundary landed where it did.

### Movement boundaries

| setting | default | sensible range | what it does |
| --- | --- | --- | --- |
| **onset threshold** (fraction of peak) | 0.15 | 0.10 – 0.30 | Where the movement is considered to start and stop, as a fraction of that event's own peak speed. Higher = tighter windows. Measured on these recordings: 0.15 → mean 3.8 s, 0.25 → 3.0 s, 0.40 → 2.9 s, 0.60 → 1.9 s (by then it is clipping into the movement). |
| **velocity floor** (percentile) | 40 | 25 – 55 | A hard lower bound on that threshold, taken from the whole recording, so the search cannot wander off through a quiet stretch. Below 25 it has no effect here; at 70 it starts truncating real movement (mean 2.5 s), at 85 more so (1.9 s). |
| **min event duration** (s) | 1.5 | 1.0 – 2.5 | Rejects blips. Keep at or above 1.0 s: shorter windows make the smoothness and coordination metrics unreliable (§7). |
| **min yaw excursion** (deg) | 8 | 10 – 40 | Rejects small wobbles. The real rotations here are 45–70°, so 8° is permissive. Raising to 20–40° filters noise without losing genuine events; above 60° almost everything is rejected. |
| **number of events** | 6 | — | How many of the highest-scoring events to keep, ranked by excursion × peak speed. |

The two thresholds interact: the effective bar is `max(onset threshold × peak speed, velocity floor)`, so whichever is higher wins. That is why lowering the floor below ~25 changes nothing — the relative threshold is already binding.

### Stabilization window

| setting | default | sensible range | what it does |
| --- | --- | --- | --- |
| **stabilization latency penalty** (λ) | 0.35 | 0.15 – 0.50 | Trades *quiet* against *soon*. λ = 0 picks the quietest window anywhere in the horizon, which here averaged 2.4 s after the movement — quiet, but arguably no longer a recovery from *that* event. λ = 0.7 and above always takes the window immediately after the movement, which is prompt but noisier (mean quiet ratio 1.61 vs 1.23). |
| **stabilization search horizon** (s) | 10 | 8 – 15 | How far after the movement to look. |
| **stabilization length** (s) | 3.0 | 3.0 – 4.0 | How long the window is. **This matters for reliability**: at 2 s, mediolateral sway has an ICC of 0.91 against realistic boundary error; at 3 s, 0.97. Longer scores marginally better again but starts overlapping the next movement. |

---

## 6. Flags and warnings

### Flags in the event list

| flag | meaning |
| --- | --- |
| `edited` | at least one boundary of this event was set by hand, not by the detector |
| `unsettled` | the stabilization window is more than 1.6× the recording's quiet baseline — **look at this one** |
| `off` | disabled, excluded from recalculation |

`unsettled` is the flag worth acting on. It means the software could not find a genuinely quiet window after the movement. Sometimes moving the green band fixes it. Sometimes it does not, because the person genuinely never settled — which on these recordings happens, especially for the novice, and is a finding rather than a fault. Either way, decide deliberately rather than accept the default.

### Messages above the plots

**Errors** (red) block recalculation. They mean a window is out of order, outside the recording, or too short to compute anything from.

**Warnings** (amber) do not block, and are worth reading:

| warning | what to do |
| --- | --- |
| *event is only N s; smoothness metrics span very few cycles* | Consider whether the window really captures the whole movement. |
| *event is N s; the coordination lag searches ±2 s and may saturate* | Usually fine — the lag is computed on a padded window (§7) — but check the lag value in the results. |
| *stabilization overlaps the event* | Legal and sometimes correct: if the movement ends the moment the person stops turning, recovery can begin immediately. Confirm on row 3. |
| *stabilization is N× the quiet baseline* | The `unsettled` case above. |

---

## 7. Recalculating and interpreting the results

Press **Recalculate metrics**. It validates the windows, recomputes everything, rewrites the result CSVs and regenerates the three figures, which appear below the table. 

**This overwrites the result files in place.** The previous values remain in git.

### What each metric means

Metrics fall into two groups by which window they use.

#### Computed over the movement window

| metric | plain meaning | direction |
| --- | --- | --- |
| `trunk_pelvis_lag_s` | How far the chest's rotation trails the pelvis's, in seconds. A coordination measure: in skilled movement the trunk tends to rotate as a coordinated unit. | Near zero = chest and pelvis turn together. Larger = more dissociated. |
| `trunk_pelvis_peak_cross_correlation` | How similarly the two rotate at that lag (0–1). | Near 1 = they follow the same shape. Low values mean the lag is not meaningful. |
| `weight_shift_log10_dimensionless_jerk` | Movement smoothness, log scale. Jerk is the rate of change of acceleration; a smooth movement has little of it. Log₁₀ because raw values span orders of magnitude. | **Lower = smoother.** A difference of 1 is a tenfold difference in jerk. |
| `peak_knee_flexion_deg` | Deepest knee flexion in a monopodal-stance event. | Deeper = more demanding stance. |
| `time_to_stabilization_s` | Delay from peak knee flexion to the start of the settled period. | Shorter = quicker recovery. |

#### Computed over the stabilization window

| metric | plain meaning | direction |
| --- | --- | --- |
| `lumbar_ml_acc_variance_g2` | **Mediolateral** (side-to-side) sway of the lower trunk. | Lower = steadier. Side-to-side control is usually the more demanding direction in single-leg stance. |
| `lumbar_ap_acc_variance_g2` | **Anteroposterior** (fore-aft) sway. | Lower = steadier. |
| `lumbar_orientation_variability_deg` | How much the lower trunk's orientation varies while settling. | Lower = more stable posture. |
| `corrective_peak_rate_hz` | Rate of corrective bursts per second — how often the trunk makes a sharp adjustment, measured against a threshold set from the whole recording. | Lower = fewer corrections needed. |
| `lumbar_rms_angular_velocity_dps` | Overall rotational "busyness" while settling. Needs no threshold, so it is the simplest of this group. | Lower = quieter. |

`monopodal_stance_asymmetry_metrics.csv` gives the absolute left-versus-right difference per
participant. Larger = more asymmetric between stance legs.

### How much to trust each number

Because you are placing windows by hand, each metric was tested by jittering every boundary by ±0.25 s — roughly the precision of a manual drag — and measuring how much it moved. Reliability is an ICC: real between-event differences divided by measurement noise. Above ~0.9 is dependable.

| metric | ICC | verdict |
| --- | --- | --- |
| `lumbar_rms_angular_velocity_dps` | 0.98 | dependable |
| `corrective_peak_rate_hz` | 0.97 | dependable |
| `lumbar_ml_acc_variance_g2` | 0.97 | dependable |
| `lumbar_ap_acc_variance_g2` | 0.96 | dependable |
| `corrective_lumbar_angular_velocity_peak_count` | 0.41 | **do not use** — superseded by the rate above; kept only so older results reproduce |

**Coordination lag has a failure mode with a specific signature.** The cross-correlation searches ±2 s. If it reports **±1.998 s**, it did not find a peak — it ran out of search range. That is a failure marker, not a measurement, and the log flags it. The lag is computed on a deliberately widened window to avoid this, but if you make a movement window very short you may still see it. Discard those values rather than interpreting them.

### What you can and cannot conclude

This dataset is **one novice and one trained practitioner, one recording each**. Differences between them are descriptive. You can legitimately say "in this recording, the trained participant's mediolateral sway was lower across these six events"; you cannot attribute that to training, because with one participant per group there is no way to separate a training effect from two people simply being different. Treat the per-event spread as a description of within-person consistency, not as a between-group statistic.

---

## 8. Things worth knowing

**Yaw is the weak axis.** The sensors have accelerometers and gyroscopes but **no magnetometer**, so
rotation about the vertical axis has no absolute reference and drifts slowly. Two consequences:

- *Relative* angles are fine. Trunk-pelvis yaw (row 1), knee flexion, ankle and hip angles all
  compare two sensors, so shared drift cancels. Roll and pitch are anchored by gravity and are fine.
- *Absolute* heading of a single sensor is not trustworthy over long spans.

The movement detector works on yaw *speed* rather than yaw angle, which sidesteps drift entirely —
differentiating removes a slowly accumulating offset.

**Novice and trained events are paired by order**, on the assumption that both perform the same form
in the same sequence. The software does not verify this. If event 3 in one recording is clearly not
the same movement as event 3 in the other, fix it by hand — that is exactly what the editor is for.

**Long yaw excursions inside a green band are real.** If row 1 ramps 20° during a stabilization
window, the person is genuinely still turning; that is not drift. Whole-recording drift here is about
0.07 deg/s, which is a fifth of a degree over a 3 s window.

---

## 9. A working routine

1. Open the **Trunk rotation** tab. Work down the event list in order.
2. For each event, zoom to it and check **row 2**: does the amber band start where the yaw speed
   rises and end where it settles? Drag the edges if not.
3. Check **row 3**: is the green band in a trough near the dotted quiet baseline? If it sits on a
   hump, move it to the nearest genuine trough within a couple of seconds.
4. Check the paired event in the other recording is actually the same movement.
5. Deal with every `unsettled` flag deliberately — fix it, or accept it as a real finding.
6. Repeat on the **Monopodal stance** tab. There, judge the amber band against row 4 and the 60°
   line.
7. Press **Recalculate metrics**.
8. Read the log. Discard any lag reported at ±1.998 s.
9. Check the regenerated figures below the table — the bands drawn there should match what you set.

If the detector is systematically placing windows too wide or too narrow across many events, that is
a signal to adjust the **onset threshold** and re-detect rather than to correct twelve windows by
hand.

---

## 10. If something looks wrong

| symptom | likely cause |
| --- | --- |
| Plots are blank | A figure was built from numpy arrays instead of lists — see the note in [README.md](README.md). |
| Zoom resets while editing | Should not happen; the view is preserved deliberately. Report it. |
| A boundary snaps back after you drag it | It was clamped for being invalid — check the red error message. |
| Recalculation refuses to run | There is an error (not a warning) somewhere in the list; the message names the event. |
| Your edits vanished | Check whether **Re-detect trunk events** was pressed; it replaces all trunk events. |
| Figures below the table look stale | They are regenerated on each recalculation; reload the page if the browser cached them. |

The session file `outputs/event_editor_session.json` is plain, readable JSON holding every boundary
in sample indices and seconds, with a note of whether each was set automatically or by hand. If the
interface ever disagrees with what you expect, that file is the authority.
