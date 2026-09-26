"""Tai Chi balance analysis from full-body IMU recordings.

The modules follow the processing chain, one stage each.  Each imports only
modules listed above it, so any stage can be used without the ones after it.

    config             paths, recordings and analysis-wide switches
    signals             filtering, resampling and other signal primitives
    orientation         quaternion algebra, Madgwick filter, sensor-mounting alignment
    recordings          the recordings in data/, the selection, and where their outputs go
    data_io             Delsys CSV parsing, sensor inventory, orientation cache, 50 Hz export
    kinematics          segment and joint kinematics of one recording; gyroscope bias and vertical reference
    gravity             tilt, turning rate, gravity-free sway and leg lift, from the vertical in each segment
    alignment           whole-recording DTW map between the novice and trained clocks; event matching
    detection           motion-driven event detection and novice/trained pairing through the alignment
    detection_v1        the original fixed-window detectors, kept for reproducibility
    smoothness_metrics  smoothness and chest-pelvis coordination of the sequence's turns
    balance_metrics     trunk-rotation and single-leg-stance balance metrics
    comparison          novice versus trained, paired event by event
    figures             the matplotlib output figures
    sessions            the session file: the event windows every metric is computed on
    validation          checks on those windows, run before every recalculation
    recompute           windows -> metrics, CSVs and figures, with their provenance
    pipeline            the stages, from raw recordings to metrics; also runs headless

    dashboard/          the app the analysis is run from (``python3 app.py``)
    tools/              standalone viewers and helpers, not part of the pipeline
"""
