"""Tai Chi balance analysis from full-body IMU recordings.

The modules follow the processing chain, one stage each.  Each imports only
modules listed above it, so any stage can be used without the ones after it.

    config             paths, recordings and analysis-wide switches
    signals             filtering, resampling and other signal primitives
    orientation         quaternion algebra, Madgwick filter, sensor-mounting alignment
    data_io             Delsys CSV parsing, sensor inventory, orientation cache, 50 Hz export
    kinematics          segment and joint kinematics of one recording
    detection           motion-driven event detection and novice/trained pairing
    detection_v1        the original fixed-window detectors, kept for reproducibility
    smoothness_metrics  smoothness, coordination and consistency of sequence segments
    balance_metrics     trunk-rotation and monopodal-stance balance metrics
    figures             the matplotlib output figures
    pipeline            batch entry point: runs everything and writes outputs/

    editor/             the interactive event-window editor (Dash)
    tools/              standalone viewers and helpers, not part of the pipeline
"""
