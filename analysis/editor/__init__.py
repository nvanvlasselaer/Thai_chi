"""Interactive editor for the event windows every metric is computed on.

    app         the Dash user interface (run this)
    sessions    the session file: seeding from the detectors, editing, named copies
    validation  sanity checks that gate recalculation
    recompute   run the curated windows through the metrics and write outputs/

Everything except ``app`` is free of Dash, so it can be driven from a script.
"""
