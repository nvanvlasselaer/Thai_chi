"""The dashboard: the app the analysis is run from.

    app              the Dash app: page navigation, the progress poll, the server (``python3 app.py``)
    pipeline_page    stage-by-stage status of the pipeline, and buttons to run it
    editor_page      check and correct the event windows every metric is computed on
    results_page     the metric tables and figures in outputs/
    tasks            the background jobs the buttons start, built from analysis.pipeline stages
    state            what the server holds between requests: the loaded recordings and the job

The pages only call into the analysis package; nothing below ``dashboard``
depends on Dash, so every stage can also be run from a script.
"""
