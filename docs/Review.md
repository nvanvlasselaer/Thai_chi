# Movement-variability analysis in `Thai_chi` — code review and recommendations

**Repository:** https://github.com/nvanvlasselaer/Thai_chi (reviewed at commit `9e8987b`)
**Date:** 25 September 2026
**Basis:** the companion literature review *Motion variability metrics for Tai Chi* (Sept 2026). Every reference below was checked against PubMed; DOIs and PMIDs are listed at the end.
**Scope:** everything in `analysis/` that turns IMU data into variability, smoothness, coordination or balance metrics, plus the committed outputs in `outputs/` and the notes in `docs/`. The raw recordings are git-ignored, so I could not re-run the full pipeline. Findings come from reading the code, testing the metric functions on synthetic signals, and checking the committed CSVs and session file.

---

## TL;DR

The pipeline is well organised and unusually transparent: provenance is tracked, detector parameters are exposed, and lag saturation is flagged. But several metrics do not measure what their names say, and the headline "variability" output does not measure execution variability. In priority order:

1. **The cross-correlation lag is biased and its "correlation" can exceed 1.** Seven of the 12 committed `trunk_pelvis_peak_cross_correlation` values are above 1 (up to 1.044). In simulation, the median lag error was 0.17 s, against 0.02 s for a per-lag Pearson estimate. The novice-vs-trained lag difference reported in `docs/Tai Chi Balance Analysis.pdf` (−0.42 vs −0.18 s) is the same size as that error. → §3.1
2. **Dimensionless jerk here measures noise, not smoothness.** Across the 22 committed sequence segments, log₁₀(DJ) is explained almost entirely by segment duration and amplitude (R² = 0.968). The fitted exponents, D^5.9 and A^−1.9, match what a constant noise floor predicts (D⁶/A²). → §3.2
3. **Several novice–trained pairs are probably different movements.** The repository's own offset check flags 4 of 6 trunk pairs and 5 of 11 segment pairs. For at least `trunk-05` (offset −71.6 s against a typical +8.1 s) and `smooth-10`, cross-checking against the knee events shows different movements are being compared: the knee events place the two recordings within about 3 s of each other over the same stretch. → §3.3
4. **"Variability across segments" measures the choreography, not repetition.** Each segment is a *different* part of the form. Their spread in duration, excursion or waveform (CV, SD, variance ratio) mostly reflects how different the movements are, not how consistently one movement is repeated. → §3.4
5. **The AP/ML axis labels are probably swapped, and sway includes gravity.** Knee flexion appears on Euler x, so for the legs x is the mediolateral axis, not "forward" as documented. The trunk sensors' mounting matrices follow the same logic. The lumbar "AP"/"ML" acceleration variances are also computed in sensor axes without removing gravity, so a ±1° tilt alone adds about 1.5 × 10⁻⁴ g², the same order as the reported sway. → §3.5–3.6
6. **The monopodal-stance definition (knee flexion > 60°) misses extended-leg kicks.** It also cannot tell the lifted leg from a deeply flexed stance leg. The committed event `knee-05` peaks at 11° (novice) and 9° (trained). → §3.7
7. **Add metrics the literature supports that the current data can already provide:**
   - Inter-repetition variability of movements that repeat within the form, with amplitude and timing separated by landmark registration.
   - Coordination variability, via vector coding or Hilbert-based CRP.
   - SPARC computed on gyroscope angular velocity.
   - Sample entropy and multiscale entropy of trunk acceleration during the 35–40 s quiet-standing spans.
   → §4

---

## 1. What the repository computes now

| Family | Window | Metric (CSV column) | Function |
| --- | --- | --- | --- |
| Trunk rotation | event (+2 s pad for lag) | `trunk_pelvis_peak_cross_correlation`, `trunk_pelvis_lag_s` | `signals.cross_correlation_lag` |
| | event | `weight_shift_(log10_)dimensionless_jerk` (on lumbar Euler x) | `smoothness_metrics.dimensionless_jerk` |
| | stabilization (3 s) | `lumbar_ap_acc_variance_g2`, `lumbar_ml_acc_variance_g2` | `balance_metrics.lumbar_body_acceleration` + `np.var` |
| | stabilization | `lumbar_orientation_variability_deg` | pooled SD of lumbar Euler x, y, z |
| | stabilization | `corrective_peak_rate_hz`, `lumbar_rms_angular_velocity_dps` | `balance_metrics.corrective_activity` |
| Monopodal stance | knee > 60° | peak flexion, time to stabilization, same sway/orientation/corrective metrics, yaw and pitch lags | `balance_metrics.compute_knee_balance_metrics` |
| | per participant | left–right absolute differences | `compute_asymmetry_metrics` |
| Sequence segments | chest-yaw cycle | log₁₀ DJ, SPARC, submovement rate, chest–pelvis lag / r / gain, relative-yaw range and SD | `smoothness_metrics.compute_sequence_smoothness_metrics` |
| | across segments | `duration_cv`, `chest_yaw_excursion_cv`, `*_sd`, `waveform_mean_sd_deg`, `waveform_variance_ratio` | `summarize_sequence_variability`, `waveform_consistency` |

## 2. Verdict per metric

| Metric | Verdict | Main reason | Action |
| --- | --- | --- | --- |
| Trunk/chest–pelvis cross-correlation lag and r | **Fix, then consider replacing** | Biased lag; r > 1 possible; r sits at a ceiling (0.86–0.99 in 21 of 22 segments), so it cannot discriminate | Per-lag Pearson (§3.1), or replace with CRP / vector coding (§4.2) |
| Dimensionless jerk (weight shift, chest yaw) | **Drop, or demote to secondary** | Noise-dominated for slow, long movements (§3.2) | Use SPARC on gyroscope angular velocity (§4.3) |
| SPARC | **Keep, recompute** | Right metric (Balasubramanian et al., 2012, 2015), but full cycles contain a reversal, and pad level differs from the reference | Half-cycles, gyroscope input, `padlevel=4` (§3.8, §4.3) |
| Submovement rate | Keep as a descriptive check | Baseline depends on how many reversals a segment contains | Compute per half-cycle |
| Chest–pelvis gain, relative-yaw range | Keep as descriptors | Describe axial dissociation, not variability. Relative yaw is not detrended | Detrend relative yaw; rename `relative_yaw_rms_deg` (it is an SD) |
| Duration CV, excursion CV, `*_sd` across segments | **Re-scope** | Measure choreography heterogeneity (§3.4) | Compute across repetitions of the same movement (§4.1) |
| Waveform corridor and variance ratio | **Re-scope, register** | Formula is correct, but inputs are not repetitions, and linear normalisation inflates spread | Landmark registration on true repeats (§4.1) |
| Lumbar AP/ML acceleration variance | **Fix** | Axis labels likely swapped; gravity/tilt not removed; unfiltered | Gravity-compensated, heading-frame, low-passed RMS (§3.5–3.6, §4.5) |
| Lumbar orientation variability | **Fix** | Mixes yaw with tilt; Euler-based | Tilt-only angular deviation (§3.10) |
| Corrective peak rate | Demote | Zero in 8/12 trunk and 7/8 stance rows (floor effect); threshold is participant-specific | Keep RMS angular velocity (bias-corrected) |
| RMS lumbar angular velocity | **Keep** | Threshold-free; robust to boundary jitter | Subtract gyroscope bias (§3.12) |
| Monopodal detection (knee > 60°) | **Replace** | Misses kicks; ambiguous stance leg (§3.7) | Foot-contact detection from foot IMUs |
| Left–right asymmetry | Re-scope | Compares different movements (three left events vs one right) | Compare mirrored repeats of the same movement |
| Time to stabilization | Keep, with caution | Depends on the latency penalty and on manual placement | Report the detector setting and provenance with it |

---

## 3. Calculation issues

### 3.1 `cross_correlation_lag` — correlation above 1 and biased lag

**Where:** `analysis/signals.py:118–132`

```python
a = (a - mean(a)) / std(a)            # z-scored over the whole window
values = corr / np.maximum(overlap, 1.0)   # "unbiased": divided by overlap count
idx = int(np.argmax(np.abs(values)))       # |r|: anti-phase peaks accepted
```

**Problem.** Each lag's sum of products is divided by the number of overlapping samples. The signals, though, were normalised over the *whole* window, not the overlap. At non-zero lags the retained samples can have larger |z| than average, so the value is not a correlation: it is not bounded by 1, and its peak shifts depending on where the movement sits in the window. `argmax(abs(...))` also accepts negative (anti-phase) peaks.

**Evidence.**
- In `outputs/trunk_rotation_balance_metrics.csv`, 7 of the 12 `trunk_pelvis_peak_cross_correlation` values are above 1 (1.004–1.044).
- Simulation: pelvis yaw was a scaled copy of chest yaw leading by a known lag (0.05–0.6 s), in 6–11 s windows with 1–3 minimum-jerk turns and small random-walk noise, 300 windows.

  | Estimator | r > 1 | median \|lag error\| | 90th percentile |
  | --- | --- | --- | --- |
  | Current `cross_correlation_lag` | 29/300 | 0.166 s | 0.491 s |
  | Pearson r per lag, overlap ≥ 75 % | 0/300 | 0.017 s | 0.108 s |

- With a single clean turn and a true lag of 0.30 s, the current estimate ranged from 0.05 s to 0.25 s depending on where the turn sat in the window. Per-lag Pearson returned 0.300 s every time.
- In `monopodal_stance_balance_metrics.csv`, 2 of the 8 `trunk_pelvis_pitch_lag_s` values sit on the ±2 s bound (±1.998) and 2 more are within 0.13 s of it. Three `trunk_pelvis_lag_s` values are exactly 0.000 s.

**Fix.**

```python
def xcorr_lag_pearson(a, b, fs, max_lag_s=1.0, min_overlap=0.75, step=1):
    """Pearson r on the overlapping samples at each lag; most positive peak.
    lag_s < 0 means `a` leads `b` (same convention as the current function)."""
    a = np.asarray(a, float); b = np.asarray(b, float); n = len(a)
    best_r, best_k = -np.inf, 0
    for k in range(-int(max_lag_s * fs), int(max_lag_s * fs) + 1, step):
        x, y = (a[k:], b[:n - k]) if k >= 0 else (a[:n + k], b[-k:])
        if len(x) < min_overlap * n or np.std(x) == 0 or np.std(y) == 0:
            continue
        r = np.corrcoef(x, y)[0, 1]
        if r > best_r:
            best_r, best_k = r, k
    if not np.isfinite(best_r):
        return np.nan, np.nan
    if abs(best_k) >= int(max_lag_s * fs) - step:     # on the bound: not a measurement
        return best_r, np.nan
    return float(best_r), best_k / fs
```

- Downsample to about 50 Hz first (the signals are low-passed at ≤ 6 Hz) to make the loop cheap; use `step=1` at that rate.
- Restrict the search to a physiologically plausible range (±1 s) and return `NaN` at the bound. The current code writes ±1.998 s into the CSV as a number.
- Even when fixed, chest–pelvis r is near 1 for everyone (0.86–0.99 in 21 of the 22 committed segments; the other is 0.39), so it cannot tell participants apart. Coordination *variability* is better captured by §4.2.

### 3.2 Dimensionless jerk is dominated by noise

**Where:** `analysis/smoothness_metrics.py:49–64`, used for the chest yaw of segments (`:269`) and the lumbar Euler x of trunk events (`balance_metrics.py:122`).

**Problem.** DJ = D⁵/A² · ∫ jerk² dt. For a noise-free movement shape it does not change with duration: a synthetic smooth cycle gave log₁₀ DJ = 5.53 / 5.56 / 5.58 at D = 5 / 10 / 20 s. But noise at up to the 6 Hz cutoff adds a jerk power that is roughly constant per second. Its contribution therefore grows like D⁶/A². On a 15 s, ±40° cycle:

| Added angle noise (deg RMS) | log₁₀ DJ | SPARC |
| --- | --- | --- |
| 0 | 5.57 | −2.92 |
| 0.05 | 7.14 | −2.40 |
| 0.2 | 8.31 | −2.48 |
| 0.5 | 9.14 | −2.47 |

The committed values are 6.9–10.7 (sequence segments) and 6.4–8.5 (weight shift), which is the noise-dominated range. Regressing the 22 committed segment values:

`log10 DJ = 5.45 + 5.86·log10(duration) − 1.90·log10(excursion)`, R² = 0.968

The exponents are what a noise floor predicts (+6, −2). Duration and amplitude alone explain 97 % of the between-segment variance, which leaves almost nothing for smoothness. This also explains why jerk and SPARC "point in opposite directions" in the README.

The weight-shift DJ is computed on lumbar Euler x, whose amplitude within an event is only a few degrees. The A² in the denominator makes it even more noise-sensitive. It is also probably pelvic *pitch*, not a lateral weight shift (§3.5).

**Fix.**
- Use SPARC as the primary smoothness metric. It was designed to be robust to measurement noise (Balasubramanian et al., 2012).
- Compute SPARC on gyroscope angular velocity (§4.3). SPARC is valid on gyroscope angular velocity, whereas log dimensionless jerk is error-prone there (Melendez-Calderon et al., 2021).
- If a jerk measure is kept, report it only with a cutoff-sensitivity analysis. Always report movement duration alongside it: slow movements are intrinsically less smooth, and Tai Chi experts showed no smoothness advantage in a slow tracking task (Noy et al., 2024).

### 3.3 Novice–trained pairing mismatches movements

**Where:** `detection.pair_trunk_events` (`analysis/detection.py:540–586`) pairs by temporal order. `editor/validation.check_pairing` (`:122–156`) only *warns* at > 8 s from the median offset, so recomputation goes ahead.

**Evidence (from `outputs/event_editor_session.json`).** The table applies the repository's own `check_pairing` rule, which flags a pair more than 8 s from the family's median offset. An offset outlier is a warning sign, not proof: tempo differences between performers can shift offsets legitimately.

| Family | Typical offset (novice − trained) | Pairs off by > 8 s |
| --- | --- | --- |
| Monopodal stance | +3.4 s | none (offsets +2.2 to +4.0 s) |
| Trunk rotation | +8.1 s | trunk-01 (+28.7), trunk-02 (+25.6), trunk-05 (−71.6), trunk-06 (−20.7) |
| Sequence segments | −6.2 s | smooth-01 (+3.2), smooth-08 (+3.9), smooth-10 (−41.2), smooth-12 (−29.8), smooth-13 (−21.4) |

Independent confirmation comes from the knee events, which are unflagged and place both recordings within about 3 s of each other at 123–158 s.
- `smooth-10` pairs novice 126.5–147.7 s, which contains the novice's single-leg passage, with trained 167.7–182.1 s, which comes *after* the trained single-leg passage.
- `trunk-05` pairs novice 109.3 s with trained 180.8 s.

Per-event paired comparisons, the trunk traceability figure and the write-up's novice-vs-trained event means are therefore at least partly comparing different movements.

**Fix.**
- Align the two whole recordings once with banded DTW on slow, multichannel signals, then map every event through the warping path. Suitable signals are chest yaw (low-passed at 0.5 Hz), left and right knee flexion, and trunk–pelvis yaw, down-sampled to 5 Hz and z-scored. A whole-form DTW is far more distinctive than matching 2–4 s windows, which is why the v1 approach failed. A tested sketch is in §4.6.
- Make an offset outlier a blocking error, or at least exclude such pairs from paired statistics.
- Better still, label the movements of the form: the novice video exists, and the form has a known sequence. Pair by label.

### 3.4 Across-segment spread is not execution variability

**Where:** `summarize_sequence_variability` and `waveform_consistency` (`analysis/smoothness_metrics.py:168–216, 303–354`). The user manual calls this output "the variability answer".

**Problem.** The 11 segments are consecutive, different parts of the form. Their CV of duration (0.33–0.35) and of excursion, the SD of per-segment metrics, and the waveform corridor all mix two things:

- (a) how different the movements of the form are;
- (b) how consistently the participant executes a movement.

Only (b) is movement variability in the sense of the literature, which defines it as variation across repetitions of the same task (Cowin et al., 2022; Preatoni et al., 2013). In a synthetic test the variance ratio was 0.027 for six repetitions of one movement, and 0.15 for three repetitions each of two different movements, with identical timing jitter. The metric mainly reports heterogeneity.

Two further issues:

- **Linear time normalisation.** Segments differ in duration and internal timing, and normalising them linearly inflates the spread. In a synthetic test, the mean SD was 5.8° after linear normalisation but 0.4° after landmark registration, with noise of 0.5°. Page & Epifanio (2007) describe when linear normalisation increases variability. Registration (Sadeghi et al., 2000) and piecewise alignment (Helwig et al., 2011) separate timing from amplitude.
- **Variance ratio and range of motion.** The VR divides within-curve variance by total variance, so it depends on the range of motion, in the same way that the related CMC does (Røislien et al., 2012). Participants with smaller turns get a higher VR for the same absolute spread. The formula itself (`:206–208`) is implemented correctly.

**Fix.** See §4.1: compute variability across *repeats of the same movement*, after landmark registration, and report amplitude and timing variability separately.

### 3.5 Probable axis-label swap: x is mediolateral, not forward

**Where:** `orientation.mounting_matrix` (`analysis/orientation.py:150–167`); its docstring states "x = forward, y = right, z = downward".

**Evidence.**
- Knee flexion (60–104° peaks) is read from Euler **x** (`balance_metrics.py:159, 230`; `pipeline.py:157`).
- In the exported 50 Hz data, knee x spans up to −76°, while y stays within ±21°.
- Hip x likewise carries the large range (up to 59°).

If x were forward, flexion would appear on y. So for the thigh and shank sensors, x is the mediolateral axis.

Now read the matrices, assuming the Trigno z-axis is the face normal (the Delsys axis definition should be checked):
- **Leg sensors** sit on the lateral side. Their face-normal (z) axis maps to body x, so body x = mediolateral, matching the knee data.
- **Chest and lumbar sensors** sit on the front and back. Their face-normal axis maps to body y, so body y = anteroposterior. Body x (sensor x) is then mediolateral.

Under that assumption, body x is mediolateral and body y anteroposterior for all segments.

**Consequences if confirmed.**
- `lumbar_ap_acc_variance_g2` (column 0) is actually mediolateral, and `lumbar_ml_acc_variance_g2` is anteroposterior. The write-up's interpretation of AP vs ML sway would reverse.
- `weight_shift_dimensionless_jerk` (lumbar Euler x) measures pelvic pitch, not lateral tilt.

**Check.** Run `analysis/tools/plot_frame.py` on a frame during a known forward trunk bend and a lateral weight shift. Or add a functional calibration movement (pure trunk flexion, then a side step) to the protocol, as `docs/notes.txt` already proposes. Then fix the docstring or the matrices, and relabel the columns.

### 3.6 Lumbar "sway" variance includes gravity and rotates with the pelvis

**Where:** `balance_metrics.lumbar_body_acceleration` (`:79–82`) and `np.var` at `:114–115` and `:259–260`.

**Problem.** Raw accelerometer data are rotated by a *fixed* mounting matrix only.

- **Gravity leaks in.** Any trunk tilt projects gravity onto the "horizontal" axes: sin(1°) ≈ 0.017 g. A ±1° tilt oscillation alone contributes about 1.5 × 10⁻⁴ g² of variance, and ±2° about 6 × 10⁻⁴ g². The committed values are 0.6–27 × 10⁻⁴ g² ("ML") and 1.7–60 × 10⁻⁴ g² ("AP"), so tilt and linear acceleration cannot be separated.
- **The axes turn with the pelvis.** After a 40° trunk rotation, "AP" points elsewhere.
- **No low-pass filter** is applied, so sensor noise at 370 Hz enters the variance.

**Fix.** Rotate into the neutral (gravity-aligned) frame with the lumbar quaternion, subtract gravity, express the result in a pelvis-heading frame, then low-pass. A sketch is in §4.5. Report RMS (in m/s² or g) rather than variance, so values scale with amplitude.

### 3.7 Monopodal stance by knee flexion > 60°

**Where:** `detection.find_knee_flexion_windows` (`analysis/detection.py:320–345`) and `pipeline.py:157–158`.

**Problems.**
- **Extended-leg kicks are missed.** The write-up itself describes a variation in which the knee extends while the hip stays flexed. The committed event `knee-05` was placed by hand and peaks at **11.0°** (novice) and **8.8°** (trained), so it fails the family's own definition. Its sway and asymmetry numbers are pooled with the 84–104° events regardless.
- **The stance leg is inferred, not measured.** `stance_leg` is simply the side opposite the flexed knee. In low Tai Chi stances the *weight-bearing* knee can also exceed 60°, which would invert the labels.
- **Asymmetry compares different movements.** The left mean pools three events (including the kick), and the right mean is a single event.

**Fix.** Detect single-leg support from the foot IMUs directly. The stance foot is quasi-stationary: gyroscope magnitude below a threshold and accelerometer magnitude close to 1 g for a sustained period. The swing foot is not. Confirm against the novice video. Classify the lifted-leg configuration (knee flexed vs extended) as a *label*, not as the definition.

### 3.8 SPARC and submovements on full back-and-forth cycles 

**Where:** `spectral_arc_length` (`analysis/smoothness_metrics.py:67–112`) and `count_submovements` (`:122–151`).

- **The reversal is penalised as if it were jerkiness.** A full segment contains at least one reversal, where speed passes through zero. On perfectly smooth synthetic movements:

  | Movement | SPARC | Speed peaks |
  | --- | --- | --- |
  | Single turn | −1.40 | 1 |
  | Out-and-back | −2.15 | 2 |
  | Four-piece full cycle | −2.96 | 3 |

  SPARC differences between segments therefore partly reflect how many reversals each part of the choreography contains. SPARC was developed for discrete movements, and Balasubramanian et al. (2015) discuss the extension to other movement types. Use the existing `half_cycles=True` option, which gives one direction per segment, for smoothness.
- **The FFT is zero-padded less than in the reference.** `n_fft` (`:90`) is 4 × the next power of two, i.e. padlevel 2. The published algorithm uses padlevel 4. On an out-and-back movement: −2.147 (repository) vs −2.270 (padlevel 4). The rest of the implementation matches the reference, including the adaptive amplitude threshold. Use padlevel 4 so values are comparable with the literature.
- **Input is the derivative of high-passed Euler yaw.** Prefer the gyroscope's vertical angular velocity (§4.3). It has no drift and needs no high-pass.

### 3.9 High-pass filtering: attenuation of long segments, and exported joint angles

**Where:** `signals.highpass_detrend` (`analysis/signals.py:32–58`); joint-angle export at `analysis/data_io.py:197`.

- **Long segments are attenuated.** The docstring assumes Tai Chi content at 0.2–1 Hz. The committed segments last 6.6–21.2 s, so the full-cycle carrier sits at 0.05–0.15 Hz. The 0.05 Hz, 4th-order `sosfiltfilt` high-pass keeps the following share of a sinusoid's amplitude:

  | Frequency | Period | Amplitude kept |
  | --- | --- | --- |
  | 0.100 Hz | 10 s | 99.6 % |
  | 0.078 Hz | 12.8 s | 97.2 % |
  | 0.060 Hz | 16.7 s | 81 % |
  | 0.053 Hz | 18.9 s | 61 % |
  | 0.040 Hz | 25 s | 14 % |

  Segments of 17–21 s (novice smooth-10 and -12, trained smooth-02) are therefore attenuated and reshaped. Options: use gyroscope-based quantities; use relative (chest−pelvis) yaw, where common drift cancels; or remove a drift model anchored on the still periods at the start, middle and end.
- **Exported joint angles lose their level.** All joint angles in `kinematic_variables_*_50hz.csv` are high-passed, so every joint has zero mean. A mean knee flexion held for tens of seconds is removed, and the right knee appears to "hyperextend" by up to +36°. Export joint angles without the high-pass; they are gravity-referenced in flexion. Detrend only yaw channels, and state this in the CSV header or README.

### 3.10 `lumbar_orientation_variability_deg` mixes yaw with tilt

**Where:** `balance_metrics.py:123–124` and `:255–256`.

The pooled SD over Euler x, y and z includes raw (not detrended) yaw. The committed trunk-rotation values are 1.9–7.9° within 3 s "stabilization" windows, which is large for a settled posture and suggests continuing axial rotation. Report tilt only: the SD of the angle between the instantaneous lumbar longitudinal axis and its window mean, or the SDs of the two tilt components separately. Report yaw separately if it is needed.

### 3.11 Corrective peak rate

**Where:** `balance_metrics.corrective_activity` (`:58–76`) and `corrective_threshold` (`:31–35`).

- **Floor effect.** The rate is 0 in 8 of 12 trunk rows and 7 of 8 stance rows, so it carries little information.
- **Participant-specific threshold.** The threshold is each recording's own Tukey fence, so a participant who moves more overall is judged against a higher bar. Across participants this is self-referential, only at recording level rather than window level.

Keep `lumbar_rms_angular_velocity_dps` as the primary measure. If a peak count is wanted, use a fixed absolute threshold shared by all participants.

### 3.12 Angular-velocity magnitude includes gyroscope bias 

**Where:** `kinematics.py:152` and `:246`.

`omega_mag` is the norm of the *raw* gyroscope signal. The bias is removed only inside the Madgwick filter (`orientation.py:82`, from the first second). Subtract a per-sensor bias estimated in the neutral window before computing RMS angular velocity, the stabilization quiet ratio and corrective peaks. Estimate the Madgwick bias from the same neutral window rather than from the first second, which may include the sync movement.

### 3.13 Smaller items

- **Relative yaw is neither detrended nor named correctly** (`smoothness_metrics.py:267, 299`). `relative_yaw` is not high-passed, unlike every other yaw channel, and `relative_yaw_rms_deg` is `np.std`, i.e. an SD, not an RMS.
- **Bound lags are written as numbers.** `flag_saturated_lags` (`editor/recompute.py:45–60`) logs a warning but leaves ±1.998 s in the CSV. Write `NaN`.
- **The reliability ICCs cannot be reproduced.** The ICC values quoted in the README and docstrings come from boundary jitter, but the script is not in the repository, and the numbers disagree:
  - peak count ICC: 0.41 (README) vs 0.77 (`balance_metrics.py:63`);
  - 2 s → 3 s stabilization window: ML sway 0.91 → 0.97 (README) vs "sway metrics" 0.87 → 0.96 (`detection.py:127`).

  Window lengths may differ between these statements; the script would settle it.

  Commit the script. Jitter-ICC measures robustness to boundary placement, not test–retest reliability. Call it "boundary robustness", and plan a test–retest session to obtain the SEM and MDC.
- **`docs/literature/Literature.md` misdescribes Gow et al. (2017).** It says the study computed Lyapunov exponents from trunk accelerometers. It used **detrended fluctuation analysis of stride time** over 10-minute walks (Gow et al., 2017).
- **The Lyapunov plan in `docs/Implementation_strategy.md` is not feasible on this data.**
  - The plan itself asks for ≥ 200 cycles. λS estimates depend on series length, and detecting condition effects needed more than 150 strides (Bruijn et al., 2009).
  - The form yields about 11 cycles per recording, and it is not a stationary cyclic task.
  - For short, non-stationary data, recurrence quantification analysis is the more appropriate family (Webber & Zbilut, 1994; Riley et al., 1999). Keep LyE for a dedicated long cyclic recording, if at all.

---

## 4. Metrics to add or substitute

Every sketch below was run on synthetic data, except §4.5, which needs the raw recordings; the results are summarised in §6. Adapt names to the package conventions.

### 4.1 Inter-repetition variability with amplitude and timing separated (primary)

**Why.** This is the core magnitude measure in the review. Skill is characterised by *selectively* low variability in task-critical variables (Hiley et al., 2013), and most expertise studies find lower variability in more skilled performers (Marineau et al., 2024).

**Data available now.** Several movements of a standard form recur within one run. In the Yang 24-form, for example, "Part the Wild Horse's Mane", "Brush Knee" and "Repulse Monkey" repeat three to four times, and "Wave Hands Like Clouds" repeats several times. Confirm the form. Use the novice video to label these repeats, and map them to the trained recording via §4.6.

Three or four repeats are few: variability estimates stabilise only with more cycles (König et al., 2014; Galna et al., 2013). Report these results as descriptive, and plan 10–20 continuous repetitions in future recordings.

**Variables.** Chest and pelvis yaw, trunk–pelvis yaw, knee flexion, and hand-segment orientation or angular velocity.

```python
def landmark_register(curves, landmarks, n_points=101):
    """Piecewise-linear registration of repetitions onto mean landmark positions.
    curves: list of 1-D arrays; landmarks: list of index arrays (first=0, last=len-1),
    e.g. [start, turn extremum, neutral crossing, turn extremum, end]."""
    lm_pct = np.array([np.asarray(l, float) / (len(c) - 1) for c, l in zip(curves, landmarks)])
    target = lm_pct.mean(axis=0)
    grid = np.linspace(0, 1, n_points)
    registered, warps = [], []
    for c, p in zip(curves, lm_pct):
        warp = np.interp(grid, target, p)                 # registered time -> original time
        registered.append(np.interp(warp, np.linspace(0, 1, len(c)), c))
        warps.append(warp)
    return np.vstack(registered), np.vstack(warps), lm_pct

def amplitude_timing_variability(registered, warps, lm_pct):
    grid = np.linspace(0, 1, warps.shape[1])
    return {
        "amplitude_sd_deg": float(registered.std(axis=0, ddof=1).mean()),
        "timing_rms_pct": float(np.sqrt(((warps - grid) ** 2).mean(axis=1)).mean() * 100),
        "landmark_sd_pct": lm_pct.std(axis=0, ddof=1)[1:-1] * 100,
    }
```

Registration can go too far (Moudy et al., 2018), and timing shifts change whole-curve statistics (Honert & Pataky, 2021). Keep the number of landmarks small, and report both components.

### 4.2 Coordination variability: vector coding or Hilbert CRP (primary)

**Why.** This replaces the ceiling-bound chest–pelvis r and the biased lag. Coordination variability has already been measured in Tai Chi: long-term practitioners showed lower CRP deviation phase than controls during obstacle crossing (Kuo et al., 2021), and Tai Chi movements showed lower hip–knee deviation phase than walking (Zhao et al., 2023). One 12-week RCT in patients with ankle instability found no training effect on gait coordination variability (Li Y et al., 2026).

**Couplings.**
- Pelvis–chest yaw (the "waist leads" principle);
- Hip–knee flexion during weight shifts;
- Trunk–upper arm.

```python
# Modified vector coding (Chang et al., 2008; Needham et al., 2014)
def coupling_angle(proximal, distal):
    """proximal on x (e.g. pelvis yaw), distal on y (e.g. chest yaw); time-normalised curves."""
    return np.degrees(np.arctan2(np.diff(distal), np.diff(proximal))) % 360.0

PATTERN_BINS = {"in_phase": [(22.5, 67.5), (202.5, 247.5)],
                "anti_phase": [(112.5, 157.5), (292.5, 337.5)],
                "proximal_phase": [(0, 22.5), (157.5, 202.5), (337.5, 360.0)],
                "distal_phase": [(67.5, 112.5), (247.5, 292.5)]}

def coordination_pattern_share(gamma):
    return {k: float(np.mean(np.any([(gamma >= lo) & (gamma < hi) for lo, hi in v], axis=0)))
            for k, v in PATTERN_BINS.items()}

def coupling_angle_variability(gammas):
    """gammas: n_reps x n_points. Circular SD per point, averaged (deg)."""
    R = np.clip(np.abs(np.exp(1j * np.radians(gammas)).mean(axis=0)), 1e-12, 1.0)
    return float(np.degrees(np.sqrt(-2.0 * np.log(R))).mean())

# Continuous relative phase, Hilbert-based (Lamb & Stoeckl, 2014)
from scipy.signal import hilbert
def hilbert_phase(x):
    x = np.asarray(x, float); x = x - (x.max() + x.min()) / 2.0   # centre the amplitude
    return np.unwrap(np.angle(hilbert(x)))

def crp_deg(proximal, distal):
    return (np.degrees(hilbert_phase(distal) - hilbert_phase(proximal)) + 180.0) % 360.0 - 180.0

def marp_dp(crp_curves):          # n_reps x n_points, time-normalised; trim the edges
    return float(np.abs(crp_curves.mean(axis=0)).mean()), float(crp_curves.std(axis=0, ddof=1).mean())
```

**Notes.**
- **Choose one method.** CRP and vector coding give different variability magnitudes and peak timings (Miller et al., 2010).
- **CRP has pitfalls on non-sinusoidal movement.** It is arbitrary without handling frequency differences, and non-sinusoidal signals distort it (Peters et al., 2003). Vector coding is simpler to interpret on slow, non-sinusoidal Tai Chi movements.
- **How many repetitions.** About 8–10 cycles gave reliable coupling-angle variability in gait (Hafer & Boyer, 2017).
- **IMU validity.** IMU-derived coupling-angle waveforms agree well with optical capture, but their *variability* correlates only moderately (average r ≥ 0.6), with distal joint pairs worse (Yin et al., 2026). Prefer proximal couplings.
- **Interpretation.** Do not assume lower is better: higher coordinative variability is often the healthy state, within a window (Hamill et al., 2012).

### 4.3 SPARC on gyroscope angular velocity, per half-cycle (primary smoothness)

- **Vertical angular velocity.** Rotate each sample's gyroscope vector into the gravity-aligned frame with the sensor quaternion, and take the vertical component.
- **Whole-segment smoothness.** Alternatively, use the 3D angular-speed magnitude.
- **Per half-cycle.** Compute SPARC on each single-direction excursion.

This avoids yaw drift, the high-pass filter and differentiating Euler angles.

```python
def sparc(speed, fs, padlevel=4, fc=10.0, amp_th=0.05):
    """Spectral arc length (Balasubramanian et al., 2012, 2015); less negative = smoother."""
    speed = np.asarray(speed, float)
    nfft = int(2 ** (np.ceil(np.log2(len(speed))) + padlevel))
    f = np.arange(nfft // 2 + 1) * fs / nfft
    M = np.abs(np.fft.rfft(speed, nfft)); M = M / M.max()
    keep = f <= fc; f, M = f[keep], M[keep]
    idx = np.nonzero(M >= amp_th)[0]
    f, M = f[idx[0]: idx[-1] + 1], M[idx[0]: idx[-1] + 1]
    return float(-np.sum(np.hypot(np.diff(f) / (f[-1] - f[0]), np.diff(M))))
```

**Reporting.**
- Report the mean SPARC per movement, and its inter-repetition SD as smoothness *variability*.
- Always report movement duration next to it (Noy et al., 2024). Experts in one Tai Chi study moved with lower jerk and at lower frequency (Zorzi et al., 2015), so smoothness and tempo are confounded unless both are reported.

### 4.4 Postural complexity during quiet standing (primary balance marker)

**Why.** This is the best-replicated variability marker of Tai Chi expertise. Experts had higher multiscale-entropy complexity of sway, while traditional sway speed and magnitude did not differ (Wayne et al., 2014). Complexity also rose with training (Manor et al., 2013).

**Data available now.** The README reports still spans of about 18 s at the start, about 40 s in the middle and about 35 s at the end of both recordings. Use tilt-compensated lumbar acceleration (§4.5), or lumbar angular velocity, from the 35–40 s spans.

**Parameters.**
- Entropy is very sensitive to m, r and N when N ≤ 200 (Yentes et al., 2013); for COP, ≥ 60 s is recommended (Montesinos et al., 2018). The 35–40 s spans are therefore short: treat results as exploratory, and record 60 s holds in future.
- Down-sample to a fixed rate (e.g. 50 Hz) for everyone, because the sampling rate changes SampEn (Raffalt et al., 2019).
- Report sway frequency alongside entropy: sway SampEn tracked sway frequency at r = 0.92–0.98 in one study (Kuczyński et al., 2011).
- Trunk-acceleration MSE and RQA had excellent within-session reliability in walking (Riva et al., 2014).

```python
from scipy.spatial import cKDTree

def sample_entropy(x, m=2, r=0.2, sd=None):
    """Richman & Moorman (2000). Chebyshev distance, self-matches excluded,
    same template count for lengths m and m+1."""
    x = np.asarray(x, float); tol = r * (np.std(x) if sd is None else sd)
    n = len(x) - m
    def matches(mm):
        emb = np.lib.stride_tricks.sliding_window_view(x, mm)[:n]
        tree = cKDTree(emb)
        return tree.count_neighbors(tree, tol, p=np.inf) - n
    B, A = matches(m), matches(m + 1)
    return float(-np.log(A / B)) if A > 0 and B > 0 else np.nan

def multiscale_entropy(x, max_scale=10, m=2, r=0.15):
    """Costa et al. (2002, 2005): coarse-grain by non-overlapping means, tolerance fixed
    from the original series. Returns the curve and the complexity index (area)."""
    x = np.asarray(x, float); sd = np.std(x); out = []
    for s in range(1, max_scale + 1):
        n = len(x) // s
        cg = x[: n * s].reshape(n, s).mean(axis=1)
        out.append(sample_entropy(cg, m, r, sd=sd) if n > 200 else np.nan)
    out = np.array(out)
    return out, float(np.nansum(out))
```

### 4.5 Gravity-compensated sway in a pelvis-heading frame (replaces §3.6)

```python
from scipy.spatial.transform import Rotation as R

def lumbar_linear_acc_heading_frame(kin, trial, neutral: slice, fs, cutoff_hz=10.0):
    q = kin.quaternions["lumbar"]                         # body frame -> neutral body frame, wxyz
    rot = R.from_quat(q[:, [1, 2, 3, 0]])
    a_ref = rot.apply(lumbar_body_acceleration(kin, trial))   # in the (gravity-aligned) neutral frame
    a_ref -= np.median(a_ref[neutral], axis=0)            # removes gravity and any neutral-pose offset
    yaw = rot.as_euler("xyz")[:, 2]
    a_head = R.from_euler("z", -yaw).apply(a_ref)         # horizontal axes follow the pelvis heading
    return np.column_stack([lowpass_signal(a_head[:, i], fs, cutoff_hz) for i in range(3)])
```

**Using it.**
- **Validate first.** The mean should be about 0 over the neutral stand, and a known forward bend should appear on the axis you label AP (see §3.5).
- **Report per window:** RMS of the two horizontal components, plus the sway frequency.

### 4.6 Whole-recording alignment for pairing (replaces order-based pairing)

```python
def dtw_path(A, B, band_frac=0.2):
    """A: n x d, B: m x d (z-scored, low-passed, ~5 Hz). Banded DTW; returns the path."""
    n, m = len(A), len(B); band = max(int(band_frac * max(n, m)), abs(n - m) + 1)
    D = np.full((n + 1, m + 1), np.inf); D[0, 0] = 0.0
    for i in range(1, n + 1):
        j0 = max(1, int(i * m / n) - band); j1 = min(m, int(i * m / n) + band)
        cost = np.linalg.norm(B[j0 - 1: j1] - A[i - 1], axis=1)
        for jj, j in enumerate(range(j0, j1 + 1)):
            D[i, j] = cost[jj] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    i, j, path = n, m, [(n - 1, m - 1)]
    while i > 1 or j > 1:
        k = np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]])
        i, j = (i - 1, j - 1) if k == 0 else ((i - 1, j) if k == 1 else (i, j - 1))
        path.append((i - 1, j - 1))
    return np.array(path[::-1])

def map_time(path, t_a, fs_ds):
    js = path[path[:, 0] == int(round(t_a * fs_ds)), 1]
    return float(np.median(js)) / fs_ds
```

**Channels:** chest yaw (0.5 Hz low-pass), left and right knee flexion, trunk–pelvis yaw, and optionally hand angular speed.

**Other uses of the path.**
- **Tempo profile.** The slope of the path gives a local tempo ratio between performers.
- **Template distance.** The DTW distance of each movement to a reference execution is a secondary similarity metric. It separated Baduanjin novices from senior students using IMUs (Li H et al., 2020), and a DTW-based score correlated with expert ratings of a Tai Chi exergame (Yu & Xiong, 2019). With a single trained participant, that participant is not a valid "expert template"; use an instructor recording if one becomes available.

### 4.7 Needs new data (not feasible on the current recordings)

- **Uncontrolled manifold (UCM).** Split joint variance into the part that leaves a task variable (e.g. COM over the base of support) unchanged and the part that changes it (Scholz & Schöner, 1999). This needs about 40 repetitions for reliable estimates (Santos-Dias et al., 2026), and a COM estimate: optical capture, or a validated full-body IMU model.
- **Principal movements (PCA).** Whole-body synergies and dimensionality (Federolf, 2016). This has been applied to Xsens IMU data (Debertin et al., 2022), so it is possible with this sensor set once repetitions exist.
- **DFA and Lyapunov exponents.** These need hundreds of continuous cycles: about 600 strides for DFA (Damouras et al., 2010), and 3-minute trials were already unreliable (Marmelat & Meidinger, 2019). Keep them for separate gait tests, not the form.

---

## 5. Recommendations for the next recordings

1. **Fix the style, form and practice dose**, and document them (Wayne & Kaptchuk, 2008).
2. **Add repeated movements:** 10–20 continuous repetitions of 2–3 movements (e.g. Cloud Hands, Brush Knee, Push), plus about 3 runs of the form segment.
3. **Add quiet stance:** 3 × ≥ 60 s holds (standing posture and a single-leg posture) for entropy measures.
4. **Add a functional calibration:** pure trunk flexion, pure knee flexion and a side step after the neutral pose. This resolves the axis-labelling question in §3.5 and anchors yaw.
5. **Recruit ≥ 3 skill levels** (novice, intermediate, expert), and log practice hours. A test–retest session gives the SEM and MDC for every metric.
6. **Validate IMUs against optical** in a synchronised pilot, if available. IMU-derived *variability* measures are much less valid than means (Kobsar et al., 2020). IMU joint-angle error grows with task length and complexity (Robert-Lachaine et al., 2017; Poitras et al., 2019).

## 6. How the claims in this report were checked

| Check | Result |
| --- | --- |
| r > 1 in committed trunk-rotation lags | 7 of 12 rows (`trunk_rotation_balance_metrics.csv`) |
| Lag bias, current vs per-lag Pearson (300 simulated windows) | median error 0.166 s vs 0.017 s; r > 1 in 29/300 |
| DJ vs duration and amplitude (22 committed segments) | log₁₀ DJ = 5.45 + 5.86 log₁₀ D − 1.90 log₁₀ A, R² = 0.968 |
| DJ with added noise (synthetic 15 s cycle) | 5.57 → 7.14 → 8.31 → 9.14 for 0 / 0.05 / 0.2 / 0.5° noise |
| SPARC of smooth single vs out-and-back vs full cycle | −1.40 / −2.15 / −2.96 |
| SPARC pad level | −2.147 (repo) vs −2.270 (reference padlevel 4) on out-and-back |
| High-pass amplitude retention | 99.6 / 97.2 / 81 / 61 / 14 % at 0.10 / 0.078 / 0.06 / 0.053 / 0.04 Hz |
| Variance ratio: one movement repeated vs two movements mixed | 0.027 vs 0.15 |
| Linear normalisation vs landmark registration (8 synthetic repeats) | mean SD 5.8° vs 0.4° (true noise 0.5°) |
| Pairing offsets (session file, repo's 8 s rule) | trunk 4/6 and segments 5/11 flagged; knee 0/4; `trunk-05` and `smooth-10` contradicted by the knee-event alignment |
| `knee-05` peak knee flexion | 11.0° (novice), 8.8° (trained); both windows manual |
| Proposed functions (synthetic; the code blocks in this file were executed as written) | per-lag Pearson recovered a 0.300 s lag; SampEn of white noise 2.2 (m = 2, r = 0.2); MSE flat for 1/f noise and falling for white noise; CRP MARP 19° for an 18° phase lag; DTW mapped times with < 0.1 s error. §4.5 was not executed: it needs the raw recordings |

## 7. Suggested order of work

- [ ] Replace `cross_correlation_lag` with the per-lag Pearson version; write `NaN` at the bound (§3.1)
- [ ] Verify the axis convention; fix the `mounting_matrix` docstring and the AP/ML column labels (§3.5)
- [ ] Align the recordings with whole-form DTW; re-pair trunk events and segments; make offset outliers blocking (§3.3)
- [ ] Gravity-compensated heading-frame lumbar RMS; bias-corrected angular velocity (§3.6, §3.12, §4.5)
- [ ] SPARC from gyroscope on half-cycles with `padlevel=4`; demote DJ (§3.2, §3.8, §4.3)
- [ ] Label repeated movements in the form; landmark-registered amplitude and timing variability; coupling-angle variability (§4.1–4.2)
- [ ] SampEn and MSE on the quiet-standing spans (§4.4)
- [ ] Foot-contact-based single-leg stance detection (§3.7)
- [ ] Stop high-passing exported joint angles; detrend relative yaw (§3.9, §3.13)
- [ ] Commit the boundary-jitter script; correct `Literature.md` (Gow et al., 2017) and the LyE plan (§3.13)

---

## References

All verified against PubMed (PMID given).

- Balasubramanian S, Melendez-Calderon A, Burdet E (2012). A robust and sensitive metric for quantifying movement smoothness. *IEEE Trans Biomed Eng* 59(8):2126-36. https://doi.org/10.1109/TBME.2011.2179545 (PMID 22180502)
- Balasubramanian S, Melendez-Calderon A, Roby-Brami A, et al. (2015). On the analysis of movement smoothness. *J Neuroeng Rehabil* 12:112. https://doi.org/10.1186/s12984-015-0090-9 (PMID 26651329)
- Bruijn SM, van Dieën JH, Meijer OG, et al. (2009). Statistical precision and sensitivity of measures of dynamic gait stability. *J Neurosci Methods* 178(2):327-33. https://doi.org/10.1016/j.jneumeth.2008.12.015 (PMID 19135478)
- Chang R, Van Emmerik R, Hamill J (2008). Quantifying rearfoot-forefoot coordination in human walking. *J Biomech* 41(14):3101-5. https://doi.org/10.1016/j.jbiomech.2008.07.024 (PMID 18778823)
- Costa M, Goldberger AL, Peng CK (2002). Multiscale entropy analysis of complex physiologic time series. *Phys Rev Lett* 89(6):068102. https://doi.org/10.1103/PhysRevLett.89.068102 (PMID 12190613)
- Costa M, Goldberger AL, Peng CK (2005). Multiscale entropy analysis of biological signals. *Phys Rev E* 71(2 Pt 1):021906. https://doi.org/10.1103/PhysRevE.71.021906 (PMID 15783351)
- Cowin J, Nimphius S, Fell J, et al. (2022). A Proposed Framework to Describe Movement Variability within Sporting Tasks: A Scoping Review. *Sports Med Open* 8(1):85. https://doi.org/10.1186/s40798-022-00473-4 (PMID 35759128)
- Damouras S, Chang MD, Sejdić E, et al. (2010). An empirical examination of detrended fluctuation analysis for gait data. *Gait Posture* 31(3):336-40. https://doi.org/10.1016/j.gaitpost.2009.12.002 (PMID 20060298)
- Debertin D, Wachholz F, Mikut R, et al. (2022). Quantitative downhill skiing technique analysis according to ski instruction curricula: A proof-of-concept study applying principal component analysis on wearable sensor data. *Front Bioeng Biotechnol* 10:1003619. https://doi.org/10.3389/fbioe.2022.1003619 (PMID 36237214)
- Federolf PA (2016). A novel approach to study human posture control: "Principal movements" obtained from a principal component analysis of kinematic marker data. *J Biomech* 49(3):364-70. https://doi.org/10.1016/j.jbiomech.2015.12.030 (PMID 26768228)
- Galna B, Lord S, Rochester L (2013). Is gait variability reliable in older adults and Parkinson's disease? Towards an optimal testing protocol. *Gait Posture* 37(4):580-5. https://doi.org/10.1016/j.gaitpost.2012.09.025 (PMID 23103242)
- Gow BJ, Hausdorff JM, Manor B, et al. (2017). Can Tai Chi training impact fractal stride time dynamics, an index of gait health, in older adults? Cross-sectional and randomized trial studies. *PLoS One* 12(10):e0186212. https://doi.org/10.1371/journal.pone.0186212 (PMID 29020106)
- Hafer JF, Boyer KA (2017). Variability of segment coordination using a vector coding technique: Reliability analysis for treadmill walking and running. *Gait Posture* 51:222-227. https://doi.org/10.1016/j.gaitpost.2016.11.004 (PMID 27821354)
- Hamill J, Palmer C, Van Emmerik RE (2012). Coordinative variability and overuse injury. *Sports Med Arthrosc Rehabil Ther Technol* 4(1):45. https://doi.org/10.1186/1758-2555-4-45 (PMID 23186012)
- Helwig NE, Hong S, Hsiao-Wecksler ET, et al. (2011). Methods to temporally align gait cycle data. *J Biomech* 44(3):561-6. https://doi.org/10.1016/j.jbiomech.2010.09.015 (PMID 20887992)
- Hiley MJ, Zuevsky VV, Yeadon MR (2013). Is skilled technique characterized by high or low variability? An analysis of high bar giant circles. *Hum Mov Sci* 32(1):171-80. https://doi.org/10.1016/j.humov.2012.11.007 (PMID 23465724)
- Honert EC, Pataky TC (2021). Timing of gait events affects whole trajectory analyses: A statistical parametric mapping sensitivity analysis of lower limb biomechanics. *J Biomech* 119:110329. https://doi.org/10.1016/j.jbiomech.2021.110329 (PMID 33652238)
- Kobsar D, Charlton JM, Tse CTF, et al. (2020). Validity and reliability of wearable inertial sensors in healthy adult walking: a systematic review and meta-analysis. *J Neuroeng Rehabil* 17(1):62. https://doi.org/10.1186/s12984-020-00685-3 (PMID 32393301)
- König N, Singh NB, von Beckerath J, et al. (2014). Is gait variability reliable? An assessment of spatio-temporal parameters of gait variability during continuous overground walking. *Gait Posture* 39(1):615-7. https://doi.org/10.1016/j.gaitpost.2013.06.014 (PMID 23838361)
- Kuczyński M, Szymańska M, Bieć E (2011). Dual-task effect on postural control in high-level competitive dancers. *J Sports Sci* 29(5):539-45. https://doi.org/10.1080/02640414.2010.544046 (PMID 21294035)
- Kuo CC, Chen SC, Wang JY, et al. (2021). Effects of Tai-Chi Chuan Practice on Patterns and Stability of Lower Limb Inter-Joint Coordination During Obstructed Gait in the Elderly. *Front Bioeng Biotechnol* 9:739722. https://doi.org/10.3389/fbioe.2021.739722 (PMID 34993183)
- Lamb PF, Stöckl M (2014). On the use of continuous relative phase: Review of current approaches and outline for a new standard. *Clin Biomech* 29(5):484-93. https://doi.org/10.1016/j.clinbiomech.2014.03.008 (PMID 24726779)
- Li H, Khoo S, Yap HJ (2020). Differences in Motion Accuracy of Baduanjin between Novice and Senior Students on Inertial Sensor Measurement Systems. *Sensors* 20(21):6258. https://doi.org/10.3390/s20216258 (PMID 33147851)
- Li Y, Xu Y, Kang G, et al. (2026). Effects of Tai Chi training on lower-limb coordination pattern and variability during walking in patients with functional ankle instability: a pilot randomized controlled trial. *Sci Rep* 16(1). https://doi.org/10.1038/s41598-026-50403-7 (PMID 42036489)
- Manor B, Lipsitz LA, Wayne PM, et al. (2013). Complexity-based measures inform Tai Chi's impact on standing postural control in older adults with peripheral neuropathy. *BMC Complement Altern Med* 13:87. https://doi.org/10.1186/1472-6882-13-87 (PMID 23587193)
- Marineau E, Ducas J, Mathieu J, et al. (2024). From Novice to Expert: How Expertise Shapes Motor Variability in Sports Biomechanics — a Scoping Review. *Scand J Med Sci Sports* 34(8):e14706. https://doi.org/10.1111/sms.14706 (PMID 39049526)
- Marmelat V, Meidinger RL (2019). Fractal analysis of gait in people with Parkinson's disease: three minutes is not enough. *Gait Posture* 70:229-234. https://doi.org/10.1016/j.gaitpost.2019.02.023 (PMID 30909002)
- Melendez-Calderon A, Shirota C, Balasubramanian S (2021). Estimating Movement Smoothness From Inertial Measurement Units. *Front Bioeng Biotechnol* 8:558771. https://doi.org/10.3389/fbioe.2020.558771 (PMID 33520949)
- Miller RH, Chang R, Baird JL, et al. (2010). Variability in kinematic coupling assessed by vector coding and continuous relative phase. *J Biomech* 43(13):2554-60. https://doi.org/10.1016/j.jbiomech.2010.05.014 (PMID 20541759)
- Montesinos L, Castaldo R, Pecchia L (2018). On the use of approximate entropy and sample entropy with centre of pressure time-series. *J Neuroeng Rehabil* 15(1):116. https://doi.org/10.1186/s12984-018-0465-9 (PMID 30541587)
- Moudy S, Richter C, Strike S (2018). Landmark registering waveform data improves the ability to predict performance measures. *J Biomech* 78:109-117. https://doi.org/10.1016/j.jbiomech.2018.07.027 (PMID 30126719)
- Needham R, Naemi R, Chockalingam N (2014). Quantifying lumbar-pelvis coordination during gait using a modified vector coding technique. *J Biomech* 47(5):1020-6. https://doi.org/10.1016/j.jbiomech.2013.12.032 (PMID 24485511)
- Noy L, van der Wel R, Friedman J (2024). A slow limit: extensive motor training cannot overcome a limit on the production of slow and smooth motion. *J Neurophysiol* 132(6):1779-1792. https://doi.org/10.1152/jn.00208.2024 (PMID 39441212)
- Page A, Epifanio I (2007). A simple model to analyze the effectiveness of linear time normalization to reduce variability in human movement analysis. *Gait Posture* 25(1):153-6. https://doi.org/10.1016/j.gaitpost.2006.01.006 (PMID 16563770)
- Peters BT, Haddad JM, Heiderscheit BC, et al. (2003). Limitations in the use and interpretation of continuous relative phase. *J Biomech* 36(2):271-4. https://doi.org/10.1016/s0021-9290(02)00341-x (PMID 12547366)
- Poitras I, Dupuis F, Bielmann M, et al. (2019). Validity and Reliability of Wearable Sensors for Joint Angle Estimation: A Systematic Review. *Sensors* 19(7):1555. https://doi.org/10.3390/s19071555 (PMID 30935116)
- Preatoni E, Hamill J, Harrison AJ, et al. (2013). Movement variability and skills monitoring in sports. *Sports Biomech* 12(2):69-92. https://doi.org/10.1080/14763141.2012.738700 (PMID 23898682)
- Raffalt PC, McCamley J, Denton W, et al. (2019). Sampling frequency influences sample entropy of kinematics during walking. *Med Biol Eng Comput* 57(4):759-764. https://doi.org/10.1007/s11517-018-1920-2 (PMID 30392162)
- Richman JS, Moorman JR (2000). Physiological time-series analysis using approximate entropy and sample entropy. *Am J Physiol Heart Circ Physiol* 278(6):H2039-49. https://doi.org/10.1152/ajpheart.2000.278.6.H2039 (PMID 10843903)
- Riley MA, Balasubramaniam R, Turvey MT (1999). Recurrence quantification analysis of postural fluctuations. *Gait Posture* 9(1):65-78. https://doi.org/10.1016/s0966-6362(98)00044-7 (PMID 10575072)
- Riva F, Bisi MC, Stagni R (2014). Gait variability and stability measures: minimum number of strides and within-session reliability. *Comput Biol Med* 50:9-13. https://doi.org/10.1016/j.compbiomed.2014.04.001 (PMID 24792493)
- Robert-Lachaine X, Mecheri H, Larue C, et al. (2017). Validation of inertial measurement units with an optoelectronic system for whole-body motion analysis. *Med Biol Eng Comput* 55(4):609-619. https://doi.org/10.1007/s11517-016-1537-2 (PMID 27379397)
- Røislien J, Skare O, Opheim A, et al. (2012). Evaluating the properties of the coefficient of multiple correlation (CMC) for kinematic gait data. *J Biomech* 45(11):2014-8. https://doi.org/10.1016/j.jbiomech.2012.05.014 (PMID 22673759)
- Sadeghi H, Allard P, Shafie K, et al. (2000). Reduction of gait data variability using curve registration. *Gait Posture* 12(3):257-64. https://doi.org/10.1016/s0966-6362(00)00085-0 (PMID 11154937)
- Santos-Dias M, Russo-Junior DV, de Freitas PB (2026). Multi-joint synergies stabilize anteroposterior and vertical toe clearance during single-step ascent. *J Biomech* 205:113461. https://doi.org/10.1016/j.jbiomech.2026.113461 (PMID 42447767)
- Scholz JP, Schöner G (1999). The uncontrolled manifold concept: identifying control variables for a functional task. *Exp Brain Res* 126(3):289-306. https://doi.org/10.1007/s002210050738 (PMID 10382616)
- Wayne PM, Gow BJ, Costa MD, et al. (2014). Complexity-Based Measures Inform Effects of Tai Chi Training on Standing Postural Control: Cross-Sectional and Randomized Trial Studies. *PLoS One* 9(12):e114731. https://doi.org/10.1371/journal.pone.0114731 (PMID 25494333)
- Wayne PM, Kaptchuk TJ (2008). Challenges inherent to t'ai chi research: part II — defining the intervention and optimal study design. *J Altern Complement Med* 14(2):191-7. https://doi.org/10.1089/acm.2007.7170b (PMID 18446928)
- Webber CL, Zbilut JP (1994). Dynamical assessment of physiological systems and states using recurrence plot strategies. *J Appl Physiol* 76(2):965-73. https://doi.org/10.1152/jappl.1994.76.2.965 (PMID 8175612)
- Yentes JM, Hunt N, Schmid KK, et al. (2013). The appropriate use of approximate entropy and sample entropy with short data sets. *Ann Biomed Eng* 41(2):349-65. https://doi.org/10.1007/s10439-012-0668-3 (PMID 23064819)
- Yin L, Xu J, Chen P, et al. (2026). Validity and reliability of IMUs-based system in assessing lower extremity inter-joint coupling angles and variability during gait in older adults. *Gait Posture* 127:110153. https://doi.org/10.1016/j.gaitpost.2026.110153 (PMID 41812486)
- Yu X, Xiong S (2019). A Dynamic Time Warping Based Algorithm to Evaluate Kinect-Enabled Home-Based Physical Rehabilitation Exercises for Older People. *Sensors* 19(13):2882. https://doi.org/10.3390/s19132882 (PMID 31261746)
- Zhao J, Han W, Tang H (2023). Lower limbs inter-joint coordination and variability during typical Tai Chi movement in older female adults. *Front Physiol* 14:1164923. https://doi.org/10.3389/fphys.2023.1164923 (PMID 37200836)
- Zorzi E, Nardello F, Fracasso E, et al. (2015). A kinematic and metabolic analysis of the first Lu of Tai Chi in experts and beginners. *Appl Physiol Nutr Metab* 40(10):1082-5. https://doi.org/10.1139/apnm-2015-0064 (PMID 26352536)