## Recommendations for the next recordings

1. **Fix the style, form and practice dose**, and document them (Wayne & Kaptchuk, 2008).
2. **Add repeated movements:** 10–20 continuous repetitions of 2–3 movements (e.g. Cloud Hands, Brush Knee, Push), plus about 3 runs of the form segment.
3. **Add quiet stance:** 3 × ≥ 60 s holds (standing posture and a single-leg posture) for entropy measures.
4. **Add a functional calibration:** pure trunk flexion, pure knee flexion and a side step after the neutral pose. This resolves the axis-labelling question in §3.5 and anchors yaw.
5. **Recruit ≥ 3 skill levels** (novice, intermediate, expert), and log practice hours. A test–retest session gives the SEM and MDC for every metric.
6. **Validate IMUs against optical** in a synchronised pilot, if available. IMU-derived *variability* measures are much less valid than means (Kobsar et al., 2020). IMU joint-angle error grows with task length and complexity (Robert-Lachaine et al., 2017; Poitras et al., 2019).

## Needs new data (not feasible on the current recordings)

- **Uncontrolled manifold (UCM).** Split joint variance into the part that leaves a task variable (e.g. COM over the base of support) unchanged and the part that changes it (Scholz & Schöner, 1999). This needs about 40 repetitions for reliable estimates (Santos-Dias et al., 2026), and a COM estimate: optical capture, or a validated full-body IMU model.
- **Principal movements (PCA).** Whole-body synergies and dimensionality (Federolf, 2016). This has been applied to Xsens IMU data (Debertin et al., 2022), so it is possible with this sensor set once repetitions exist.
- **DFA and Lyapunov exponents.** These need hundreds of continuous cycles: about 600 strides for DFA (Damouras et al., 2010), and 3-minute trials were already unreliable (Marmelat & Meidinger, 2019). Keep them for separate gait tests, not the form.
## From the analysis of the current recordings

What checking the metrics against these two recordings showed the protocol needs:

- **Stand still for at least 20 s at the start and at the end.** The gyroscope bias is read from the
  stillest 10 s of each (breathing moves the sternum sensor at 0.13–0.33 Hz, so a 2 s window is too
  short), and it is large: about 6.6 °/s on the lumbar sensor and 15 °/s on the chest. The same spans
  would let the orientation filter itself be re-run with a better bias, which should cut the yaw drift.
- **Record the functional calibration** in item 4 before anything else: the axes were only worked out
  from indirect evidence (x = mediolateral, y = anteroposterior); the signs of each are still unknown.
- **Hold the quiet and single-leg stances as instructed postures**, not only as parts of the form:
  the only whole-body quiet spans now are ~15–25 s before and after the form, and the trunk never
  settles after a rotation within the form, so there is no stabilization phase to measure there.
- **Check the right-foot sensor.** Its gyroscope reads 16–21 °/s RMS while the participant stands
  still, against 7 °/s on the left foot, which rules out any stance-foot measure.
- **Note whether the participants follow a video.** Both did here, with a near-constant 3.4 s offset,
  so tempo was paced externally; self-paced performance would make duration a measure of the
  participant rather than of the video.
- **With repeated movements** (item 2), the landmark-based pairing already in place carries over: the
  turning points that pair the two performers can also align the repetitions of one performer, for
  amplitude and timing variability after registration and for coupling-angle (vector-coding)
  variability.
