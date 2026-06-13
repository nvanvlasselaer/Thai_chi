# Kinematic Analysis of Novice vs Trained Tai Chi Using Full-Body IMUs

## Aim

This analysis compares `data/IMU_Trial_1_RC_Novice.csv` and `data/IMU_Trial_3_RC_Trained.csv` to identify IMU-derived kinematic differences that may provide insight into balance control. The emphasis is on reconstructed segment orientation, relative segment motion, timing, smoothness, and stabilization behavior rather than raw accelerometer variability.

## Data Validation and Sensor Map

Both recordings were exported from Trigno Discover at 370.3704 Hz. The novice file contains 89425 synchronized samples over 241.447 s; the trained file contains 87515 synchronized samples over 236.291 s. All 14 expected body sensors were present in both files, with no missing accel/gyro values in the parsed sensor channels. The sensor inventory is saved in `outputs/sensor_inventory.csv`.

The body-segment map used for interpretation was: chestbone = thorax, lumbar = pelvis/lower-trunk proxy, thigh sensors = thighs, tibia sensors = shanks, foot sensors = feet, humerus sensors = upper arms, ulna sensors = forearms, and hand sensors = hands.

## Sensor Fusion and Kinematic Variables

For every sensor, 3-axis acceleration and 3-axis gyroscope data were fused with a 6-axis Madgwick orientation filter. The first second of each gyroscope signal was used for a simple static-bias estimate, and quaternion signs were made temporally continuous. Because the recordings do not include magnetometer channels, yaw should be interpreted as gyro-integrated heading within short windows rather than absolute compass heading.

Quaternions were retained for relative segment calculations. Euler roll, pitch, and yaw were extracted only for visualization and scalar metrics. The main variables were trunk-pelvis relative orientation (`inverse(q_lumbar) * q_chestbone`), left/right knee estimates (`inverse(q_thigh) * q_tibia`), left/right ankle estimates (`inverse(q_tibia) * q_foot`), and segment angular velocity magnitudes for lumbar, chestbone, and both feet. Kinematic analysis signals were filtered with zero-phase 4th-order Butterworth filters before event detection, resampling, cross-correlation, and jerk calculation. Orientation time series are saved as compressed NPZ files, anti-aliased kinematic variables are saved as 50 Hz CSVs, and lumbar/chest orientation traces are plotted in `outputs/orientation_validation.png`.

## Event Selection and Alignment

The selected movement module was a trunk-rotation transition. Selected 6 8 s trunk-rotation transitions based on combined trunk-pelvis relative-yaw excursion and chest angular-velocity peaks. This module is balance-relevant because Tai Chi transitions require controlled rotation of the thorax over the lower trunk while body mass is being transferred and stabilized.

The novice event window was 46.97-54.96 s. A 3-channel event signature was built from trunk-pelvis relative yaw, lumbar roll, and chest angular velocity. Each signature channel was anti-alias filtered at the source rate and interpolated to a 20 Hz grid before Dynamic Time Warping was used to search the trained recording for the closest 8 s signature. The matched trained event window was 42.97-50.97 s, with normalized DTW distance 0.787. Stabilization windows were selected after each event from periods of low combined lumbar and chest angular velocity lasting at least 2 s where available.

## Three Balance-Relevant Outputs

| Output | Novice | Trained | Interpretation |
|---|---:|---:|---|
| 1. Trunk-pelvis coordination | peak r = 0.966; lag = -0.262 s | peak r = 0.967; lag = -0.061 s | Higher absolute coupling and smaller absolute lag indicate more consistent timing between lower trunk and thorax during the transition. |
| 2. Weight-shift smoothness | log10 dimensionless jerk = 9.39 | log10 dimensionless jerk = 9.35 | Lower jerk indicates a smoother lateral lumbar-roll trajectory during the selected transition. |
| 3. Postural stabilization strategy | variability = 3.79 deg; corrective peaks = 0; largest peak = 30.4 deg/s | variability = 3.23 deg; corrective peaks = 0; largest peak = 23.1 deg/s | Variability and corrective velocity peaks describe regulation strategy, not simple good/bad performance. |

## Dataset-Specific Observation

The trained event showed a smaller absolute chest-lumbar yaw timing lag (-0.061 s vs -0.262 s). During the selected stabilization window, lumbar orientation variability was larger in the novice recording (3.79 deg) than in the trained recording (3.23 deg).

## Balance-Control Interpretation

The most useful IMU-based differences were the organization of trunk-pelvis timing, smoothness of lateral weight transfer, and the pattern of postural corrections after the transition. The trained execution showed the features expected from a more organized motor strategy when its event metrics had stronger trunk-lumbar coupling, smaller timing lag, smoother lumbar roll, or fewer large corrective peaks. These patterns suggest that the trained practitioner may regulate balance by coordinating trunk and pelvis rotation earlier and by distributing corrections over the movement rather than relying on abrupt late corrections.

The novice execution should not be interpreted only through larger or smaller variability. Some variability can represent adaptive control. The balance-relevant issue is whether variability appears as structured, small-amplitude regulation or as larger discrete corrections following a transition. In this dataset, the lumbar and chest IMUs were the most informative for event definition and stabilization assessment; distal limb sensors were useful context but were less direct for the selected balance-control question.

## Files Generated

- `outputs/sensor_inventory.csv`
- `outputs/event_windows.csv`
- `outputs/balance_metrics.csv`
- `outputs/kinematic_variables_novice_50hz.csv`
- `outputs/kinematic_variables_trained_50hz.csv`
- `outputs/orientation_novice.npz`
- `outputs/orientation_trained.npz`
- `outputs/orientation_validation.png`
- `outputs/traceability_figure.png`
