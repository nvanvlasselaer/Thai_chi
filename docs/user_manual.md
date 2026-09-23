# Event editor — user manual

A guide to using the interactive event editor: reading the dashboard, adjusting event windows, tuning the detector, and interpreting what comes out.

See [README.md](../README.md) for installation.

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
│ SESSION            │  Tai Chi event editor                                 │
│  [saved ▾] [Load]  │  fs 370.37 Hz | Novice 241 s / Trained 236 s | ...    │
│  [name___] [Save]  ├───────────────────────────────────────────────────────┤
│  [New session]     │                                                       │
│ DETECTOR — <tab>   │  [ Trunk rotation ] [ Monopodal stance ]              │
│  settings for the  │  [ General smoothness ]                               │
│  open tab, plus    ├───────────────────────────────────────────────────────┤
│  stabilization     │                                                       │
│  [Re-detect <fam>] │                                                       │
│ EVENTS             │  warnings and errors for the current windows          │
│  id  what  window  ├───────────────────────────────────────────────────────┤
│  ▸ trunk-01 …      │                                                       │
│    trunk-02 …      │  NOVICE                                               │
│  [Toggle] [Delete] │   1  trunk-pelvis yaw        ▓▓▓▓▓▓    ░░░░           │
│                    │   2  yaw speed envelope      ▓▓▓▓▓▓    ░░░░           │
│ ADD AN EVENT       │   3  lumbar + chest ang.vel. ▓▓▓▓▓▓    ░░░░           │
│  [trial▾] [leg▾]   │   4  knee flexion            ▓▓▓▓▓▓    ░░░░           │
│  [+ Add event here]│   5  chest + pelvis yaw      ▓▓▓▓▓▓    ░░░░           │
│                    │  TRAINED                                              │
│                    │   (same five plots, independent time axis)            │
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

**Three tabs**, labelled `Trunk rotation`, `Monopodal stance (knee > 60 deg)` and `General smoothness`. Events in all three families are **paired**: one event holds a window in each recording, so selecting it shows the same movement in the novice and the trained participant side by side, and every metric is reported for both.

The first two tabs hold **events** — moments picked out of the recording, each followed by a settling period. The third holds **segments**, which are different in kind: they tile the moving passages of the form end to end rather than being picked out of it, and they have no stabilization window, so the green band and its two boundary boxes grey out on that tab. See [§10](#10-general-smoothness-segmenting-the-whole-sequence).

Monopodal-stance events are paired **within each leg**, in time order — the k-th left-knee event in one recording corresponds to the k-th in the other. Pairing within a leg rather than across all events guarantees a pair always compares like with like, which matters because the stance leg drives the asymmetry metrics.

Where one recording has more events of a leg than the other, the surplus becomes a **single-sided** event: it keeps a window for the recording it was found in, and the other participant's boundary boxes grey out. That is expected, and such an event contributes only to the trial it has a window for.

**Two graphs.** Novice on top, Trained below. They pan and zoom **independently**, because the two recordings are not time-aligned — the same movement can happen at different clock times in each.

**Coloured bands.** Amber = movement window. Green = stabilization window. Faint grey = other events,

---

## 3. Reading the five plots

All five rows share one time axis

### Row 1 — trunk-pelvis yaw (deg)

Rotation of the chest **relative to the pelvis**, about the vertical axis. This is axial trunk twist: positive and negative correspond to turning one way and the other. It is a *relative* angle between two sensors.

This is the signal that defines a trunk-rotation event. A clean event looks like a ramp from one plateau to another, typically 40–70° on these recordings.

### Row 2 — yaw speed envelope (deg/s)

How fast that twist is changing, smoothed. **This is the row that determines the movement boundaries**, so it is the one to look at when judging whether a band is placed correctly.

A trunk rotation appears as a clear hump. The movement starts where the envelope rises off the baseline and ends where it settles back. The dotted horizontal line is the **onset/offset floor** — the level the detector walks out to. If a band looks too narrow or too wide, compare it against this line first; usually the floor is in the wrong place rather than the algorithm being confused.

### Row 3 — lumbar + chest angular velocity (deg/s)

Three traces: the lumbar sensor, the chest sensor, and their sum in bold. The sum is what the stabilization search minimises, so **this is the row for judging the green band**.

The dotted line is the **quiet baseline** — a low percentile of this signal across the whole recording, i.e. roughly "how still this person gets when they are being still". A good stabilization window sits in a trough near or below that line. A green band sitting on a hump means the person was still moving.

### Row 4 — knee flexion |x| (deg)

Absolute flexion angle of the left and right knee, with a dashed line at the **monopodal threshold** (60° by default). Crossings of that line define monopodal-stance events, and the line moves if you change the threshold on the *Monopodal stance* detector panel. On the trunk-rotation tab this row is context: it tells you whether a trunk rotation coincided with a deep weight shift onto one leg.

### Row 5 — chest and pelvis yaw (deg)

Absolute rotation of the chest and of the pelvis about the vertical, each relative to its own drifting baseline. Row 1 shows the *difference* between these two; this row shows them separately, so you can see whether a twist came from the chest turning, the pelvis staying, or both.

**This is the row that defines a sequence segment.** The dotted line at zero is where the chest passes through neutral, and the two dashed lines are the **minimum turn** a swing must reach to count (±10° by default). A segment boundary is placed where the chest crosses zero between two opposing turns that both reach those lines.

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

**Type.** The four boxes under *Boundaries (s)* take times in seconds, to two decimals. Use the spinner arrows for 0.05 s nudges. Typing and dragging stay in sync: drag a band and the numbers update; type a number and the band moves.

If you enter a start later than its end, the editor swaps them rather than creating an invalid window.

### Other event actions

- **Toggle on/off** — excludes an event from recalculation without deleting it. Use this when an
  event is spurious but you want a record that you saw it. Disabled events show the flag `off` and
  are drawn faintly.
- **Delete** — moves the event to the *Deleted events* list. It is not lost; see below.

All changes save immediately to `outputs/event_editor_session.json`. There is no separate save step,
and your work survives restarting the app.

### Adding an event the detector missed

Automatic detection keeps only the highest-scoring events, so a genuine movement can be left out —
particularly a gentle one, or one close in time to a larger movement.

1. **Zoom to where the event belongs.** The new event is placed at the centre of the currently
   visible time range, so frame the movement before adding it.
2. On the *Monopodal stance* tab, choose which **knee flexes**, and whether the event is created in
   `Both` recordings (the default, giving a paired event) or in just one — use a single recording
   when a movement genuinely appears in only one of them. On the *Trunk rotation* tab these
   dropdowns are disabled, because a trunk event is always created in both.
3. Press **+ Add event here**.

You get a default window — 3 s for a trunk rotation, 2 s for a monopodal stance, with a
stabilization window of the current *stabilization length* immediately after — which you then drag
to fit, exactly as for a detected event. A manually added event is marked `edited` from the start,
since none of its boundaries came from the detector.

The two windows of a paired event are placed independently, each at the centre of its own graph's
view. That is deliberate: the recordings are not time-aligned, so a single shared time would put one
of them in the wrong place. **Frame the movement in both graphs before adding**, or you will have to
drag one of the two windows a long way.

### Saving and loading sessions

Your editing is written continuously to the **working session**, so nothing is lost to a crash or a
restart. A *named* session is a copy of it kept alongside, which lets you hold several curations of
the same recordings — a conservative segmentation and a permissive one, say — and move between them
instead of overwriting one with the other.

- **Save** — type a name and press it. The name is tidied into a filename (`tight windows` becomes
  `tight-windows`), the copy is written to `outputs/sessions/`, and that name becomes the one you
  are editing. Saving again under the same name overwrites that copy.
- **Load** — pick from the dropdown. The session replaces everything on screen: events, deleted
  events and detector settings.
- **New session from detection** — throws the current curation away and re-detects both families
  from scratch, using the settings currently in the detector panels. The deleted-events list is
  emptied and the session becomes unnamed again. Use it to start over, or to see what the detector
  makes of a new set of thresholds without a half-edited session in the way.

*New* differs from *Re-detect* in scope: Re-detect replaces one family and leaves the other and the
deleted list alone, whereas New resets everything.

**Loading and New both park what they replaced in `_previous`.** Load that to undo, including when
you deliberately reload the session you are already on to throw away unsaved edits. Only one step of
undo is kept, so the next load overwrites it.

Named sessions live in `outputs/sessions/` as readable JSON and are tracked by git, so a curation can
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

These re-run automatic detection. **"Re-detect trunk events" replaces all trunk events and discards any manual edits to them**, so tune the settings first and hand-correct afterwards, not the other way round.

The most useful thing about these controls is not the numbers themselves — it is that row 2 and row 3 draw the thresholds the detector actually used, so you can see *why* a boundary landed where it did.

The panel follows the tab. On *Trunk rotation* you see the trunk settings and a **Re-detect trunk
events** button; on *Monopodal stance* you see the knee settings and **Re-detect stance events**.
Either button re-detects only its own family, so the other family's curated windows are never
replaced by a click meant for this one. The stabilization settings sit below both, because the
settling window is found the same way whichever movement precedes it.

Switching tabs does not reset anything — each panel keeps its values while hidden.

### Trunk rotation — movement boundaries

| setting | default | sensible range | what it does |
| --- | --- | --- | --- |
| **onset threshold** (fraction of peak) | 0.15 | 0.10 – 0.30 | Where the movement is considered to start and stop, as a fraction of that event's own peak speed. Higher = tighter windows. Measured on these recordings: 0.15 → mean 3.8 s, 0.25 → 3.0 s, 0.40 → 2.9 s, 0.60 → 1.9 s (by then it is clipping into the movement). |
| **velocity floor** (percentile) | 40 | 25 – 55 | A hard lower bound on that threshold, taken from the whole recording, so the search cannot wander off through a quiet stretch. Below 25 it has no effect here; at 70 it starts truncating real movement (mean 2.5 s), at 85 more so (1.9 s). |
| **min event duration** (s) | 1.5 | 1.0 – 2.5 | Rejects blips. Keep at or above 1.0 s: shorter windows make the smoothness and coordination metrics unreliable (§7). |
| **min yaw excursion** (deg) | 8 | 10 – 40 | Rejects small wobbles. The real rotations here are 45–70°, so 8° is permissive. Raising to 20–40° filters noise without losing genuine events; above 60° almost everything is rejected. |
| **number of events** | 6 | — | How many of the highest-scoring events to keep, ranked by excursion × peak speed. |

The two thresholds interact: the effective bar is `max(onset threshold × peak speed, velocity floor)`, so whichever is higher wins. That is why lowering the floor below ~25 changes nothing — the relative threshold is already binding.

### Monopodal stance — movement boundaries

| setting | default | sensible range | what it does |
| --- | --- | --- | --- |
| **knee flexion threshold** (deg) | 60 | 45 – 90 | How deeply the knee must flex for the stance to count as monopodal. This is the *definition* of the event, so changing it changes what is being measured, not just how well it is found. The dashed line on row 4 moves with it. Raised to 90° on these recordings, only the deepest stances survive. |
| **min event duration** (s) | 0.4 | 0.3 – 1.0 | Rejects brief dips past the threshold. |
| **merge gap** (s) | 0.2 | 0.1 – 0.5 | A momentary rise back above the threshold lasting less than this does not split one stance into two. Raise it if a single stance is being reported as two events. |

### General smoothness — segment boundaries

| setting | default | sensible range | what it does |
| --- | --- | --- | --- |
| **segment unit** | Full cycle | — | `Full back-and-forth cycle` gives one segment per complete oscillation — chest turns one way, back through neutral, then the other way. `Half cycle` gives one per single-direction excursion. On these recordings: 13 full cycles per participant at ~11 s each, or 27–28 half cycles at ~5.7 s. Full cycles match how the form is built; half cycles give you twice as many samples for the variability statistics, at the cost of each one being half a movement. Conclusions that only hold for one unit are worth distrusting. |
| **min turn from neutral** (deg) | 10 | 5 – 20 | How far the chest must swing for a turn to count. This is the setting that decides how many segments you get, because the real turns reach 40–70° and everything below this is the chest hovering near neutral. At 5° the still passages start producing segments; at 20° only the largest turns survive. |
| **min turn separation** (s) | 1.5 | 1.0 – 3.0 | Minimum spacing between successive turns, as for the trunk detector. |
| **min segment excursion** (deg) | 10 | 10 – 30 | Rejects a finished segment whose total yaw range is too small to score. Segments below ~10° give erratic SPARC values. |
| **min segment duration** (s) | 2.0 | 1.0 – 3.0 | Rejects blips. |
| **max segment duration** (s) | 20 | 15 – 25 | **The setting that keeps the pauses out.** Both recordings stand still for roughly the first 18 s, the last 35 s, and a ~40 s passage in the middle. Without this cap the turns on either side of a pause are joined into one 30 s "cycle" that is mostly not moving. With it, those spans are simply not segmented, which is why the bands on row 5 cover about 60 % of the recording rather than all of it. |

### Stabilization window (applies to trunk and stance events)

Sequence segments have no stabilization window, so this panel is hidden on the *General smoothness* tab.

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

`unsettled` is the flag worth acting on. It means the script could not find a genuinely quiet window after the movement. Sometimes moving the green band fixes it. Sometimes it does not, because the person genuinely never settled — which on these recordings happens, especially for the novice, and is a finding rather than a fault, or the sequences follow to close to each other. Either way, decide deliberately rather than accept the default.

### Messages above the plots

**Errors** (red) block recalculation. They mean a window is out of order, outside the recording, or too short to compute anything from.

**Warnings** (amber) do not block, and are worth reading:

| warning | what to do |
| --- | --- |
| *event is only N s; smoothness metrics span very few cycles* | Consider whether the window really captures the whole movement. |
| *event is N s; the coordination lag searches ±2 s and may saturate* | Usually fine — the lag is computed on a padded window (§7) — but check the lag value in the results. |
| *stabilization overlaps the event* | Legal and sometimes correct: if the movement ends the moment the person stops turning, recovery can begin immediately. Confirm on row 3. |
| *stabilization is N× the quiet baseline* | The `unsettled` case above. |
| *novice and trained windows are N s apart, against a typical M s* | Probably a mispairing. Pairing is by order, so one missing or spurious event shifts every later pair. Check the two windows really are the same movement; add the missing event or delete the spurious one to bring the rest back into step. |

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
| `lumbar_rms_angular_velocity_dps` | Overall rotational "busyness" while settling. Needs no threshold. | Lower = quieter. |

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

**Coordination lag has a failure mode with a specific signature.** The cross-correlation searches ±2 s. If it reports **±1.998 s**, it did not find a peak — it ran out of search range. That is a failure marker, not a measurement, and the log flags it. The lag is computed on a deliberately widened window to avoid this, but if you make a movement window very short you may still see it. Consider discarding those values rather than interpreting them.

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
in the same sequence. The script does not verify this. If event 3 in one recording is clearly not
the same movement as event 3 in the other, fix it by hand in the editor.

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
7. On the **General smoothness** tab, judge the bands against row 5: each should span one full
   oscillation of the chest yaw, with its edges on the dotted zero line. Here prefer re-detecting at
   a different **min turn from neutral** over correcting many boundaries by hand, and check that the
   two recordings produced the same number of segments (§10).
8. Press **Recalculate metrics**. The table shows the family whose tab is open; every family is
   recalculated regardless.
9. Read the log. Discard any lag reported at ±1.998 s.
10. Check the regenerated figures below the table — the bands drawn there should match what you set.

If the detector is systematically placing windows too wide or too narrow across many events, that is
a signal to adjust the **onset threshold** and re-detect rather than to correct twelve windows by
hand.

---

## 10. General smoothness: segmenting the whole sequence

### What this tab is for

The other two tabs measure balance at a handful of moments. Six trunk rotations and four stances out
of a four-minute form leaves most of the recording unmeasured, so nothing in the analysis describes
how the form *as a whole* is performed — whether it flows, and whether it is performed the same way
from one part to the next. That second question is the movement-variability question.

The form is carried by large chest yaw rotations, so the chest passing through neutral is a natural
place to cut. Each part is then scored for smoothness and for chest–pelvis coordination, and the
**spread of those scores across parts** is the variability measure.

### What a segment is

A segment runs from one neutral crossing of the chest yaw to another. With the default *Full cycle*
unit that is a complete back-and-forth — the chest turns one way, returns through neutral, turns the
other way — which is the ~11 s unit the form is built from on these recordings.

Boundaries are placed by finding the **turns** first and the crossings between them second, rather
than by looking for crossings directly. A turn reaching 40–70° is unambiguous; the signal near zero
is not, and debouncing crossings by size silently welds the neighbours of any rejected wobble into
one very long lobe. Finding the peaks first avoids that.

Segments **abut** — one ends where the next begins — unlike the other two families, whose events are
picked out of the recording with gaps between them. Genuine gaps do appear where the form pauses and
no segment is produced; the editor warns only if two segments *overlap* by more than 0.5 s, which
normally means a boundary was dragged past its neighbour.

### Working on this tab

It works like the others: select a segment, judge it against **row 5**, drag the amber band or type
into the two event boxes. The stabilization boxes and the green band are greyed out, because there is
nothing to settle from.

The thing to check is that each band spans one full oscillation and that its edges sit where the
chest yaw crosses the dotted zero line. Changing the **segment unit** or the **min turn from neutral**
and pressing *Re-detect segments* is usually faster than correcting many boundaries by hand.

Because a segment is defined by the signal rather than chosen, the detector finds the same number of
parts in both recordings when both perform the same form — 13 and 13 here — which makes the
order-based pairing more trustworthy than it is for the other families. A count that differs between
the two participants is itself worth looking at before you correct it.

### What each metric means

Per segment, in `outputs/sequence_smoothness_metrics.csv`:

| metric | plain meaning | direction |
| --- | --- | --- |
| `chest_yaw_log10_dimensionless_jerk` | Smoothness of the turn in the time domain, on the same definition and log scale as `weight_shift_log10_dimensionless_jerk`. | **Lower = smoother.** |
| `chest_yaw_sparc` | Smoothness from the shape of the speed spectrum (spectral arc length). Needs no amplitude or duration normalisation, so it does not inherit their sensitivity to where you put the edges. | Negative; **less negative = smoother**. |
| `chest_yaw_submovement_rate_hz` | How many separate speed peaks per second the turn is made of. One continuous turn has one peak; a hesitant or two-stage turn has more. The most directly checkable of the three against row 5. | Lower = fewer interruptions. |
| `chest_pelvis_lag_s` | How far the chest's rotation trails the pelvis's. | Near zero = they turn together. |
| `chest_pelvis_peak_cross_correlation` | How alike the two rotations are at that lag (0–1). | Near 1 = same shape; a low value means the lag is not meaningful. |
| `chest_pelvis_gain` | Pelvis yaw range ÷ chest yaw range. | 1 = the trunk turns as a unit; below 1 the chest turns on a comparatively still pelvis. ~0.57 here. |
| `relative_yaw_range_deg`, `relative_yaw_rms_deg` | How much axial twist opens up inside the segment. | Larger = more dissociation between chest and pelvis. |

Across segments, in `outputs/sequence_variability_summary.csv` — **this is the variability answer**:

| metric | plain meaning |
| --- | --- |
| `duration_cv`, `chest_yaw_excursion_cv` | How consistent the parts are in length and size. |
| `*_sd` (jerk, SPARC, submovement rate, lag) | How consistent the *quality* of the movement is from part to part, as opposed to how good it is on average. |
| `waveform_mean_sd_deg` | Every segment time-normalised to 0–100 %, sign-aligned and centred; this is the mean spread of the resulting corridor, in degrees. |
| `waveform_variance_ratio` | The same idea, dimensionless (the Kadaba ratio): within-cycle variance about the mean waveform over total variance. **Lower = more repeatable**, and it compares across participants where the SD in degrees does not. |

The corridor is drawn in `outputs/sequence_smoothness_figure.png` alongside the segmented signal.

### A caution on the two smoothness measures

Jerk and SPARC do not have to agree, and on these recordings they do not: the trained participant
scores slightly smoother on jerk and slightly *less* smooth on SPARC. That is not a bug in either.
They measure different things — jerk is dominated by the sharpest moment in the window, SPARC by how
much spectral content the movement has overall. Where they disagree, report both and say so, or fall
back on the submovement rate, which you can check by eye against row 5.

Remember the standing caution from §7: one novice and one trained participant is descriptive. The
within-person spread across 13 segments is a reasonable description of that person's consistency; the
difference between the two people is not a training effect.

---

The session file `outputs/event_editor_session.json` is plain, readable JSON holding every boundary
in sample indices and seconds, with a note of whether each was set automatically or by hand. If the
interface ever disagrees with what you expect, that file is the authority.
