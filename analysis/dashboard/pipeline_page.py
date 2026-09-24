"""The Pipeline page: which recordings, what has been computed, and buttons to run it.

The first card chooses the recordings: every CSV in ``data/`` is offered for the
novice and the trained role, and either may be left empty.  Below it, one card
per stage of :mod:`analysis.pipeline`, each with its status and the button that
re-runs it, and a single "Run pipeline" button that runs whatever is missing or
out of date.  The progress of the running job and its log are shown
underneath, refreshed by the one-second poll.
"""

from __future__ import annotations

from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate

from analysis import config, pipeline, recordings, sessions
from analysis.dashboard import tasks
from analysis.dashboard.state import jobs, workspace
from analysis.recordings import Selection

PILL = {
    "done": ("✔ done", "#2a9d8f", "#e6f4f1"),
    "stale": ("⚠ out of date", "#9a6700", "#fff4d6"),
    "missing": ("○ not yet", "#666", "#eeeeee"),
    "loading": ("● working", "#1d6fa5", "#e3f0fa"),
}
BUTTON = {"padding": "5px 12px", "fontSize": "12px", "cursor": "pointer", "whiteSpace": "nowrap"}
PRIMARY = {**BUTTON, "padding": "8px 18px", "fontSize": "14px", "fontWeight": "600",
           "backgroundColor": "#2a9d8f", "color": "white", "border": "none", "borderRadius": "4px"}
ACTION_BUTTONS = ("btn-run-all", "btn-stage-preprocess", "btn-stage-load", "btn-stage-detect", "btn-stage-recalc")

STAGE_KEYS = ("recordings", "preprocess", "load", "session", "metrics")
CACHE_NOTE = {"done": "processed", "stale": "preprocess again", "missing": "not processed yet"}
"""Short state of a recording's stage-1 outputs; the card below says why one is stale."""


def recording_options() -> list[dict]:
    """Every CSV in data/ as a dropdown option; ones the parser cannot read are shown but disabled."""
    options = []
    for recording in recordings.scan():
        size = f"{recording.size / 1e6:.0f} MB"
        if recording.usable:
            note = CACHE_NOTE[recording.cache_state()[0]]
            options.append({"label": f"{recording.name}  ·  {size}  ·  {note}", "value": recording.name})
        else:
            options.append({"label": f"{recording.name}  ·  {recording.problem}", "value": recording.name,
                            "disabled": True})
    return options


def _role_picker(role: str, value: str | None, options: list[dict]) -> html.Div:
    return html.Div([
        html.Div(role, style={"fontSize": "11px", "color": "#555", "marginBottom": "2px"}),
        dcc.Dropdown(id=f"select-{role.lower()}", options=options, value=value, clearable=True,
                     placeholder=f"no {role.lower()} recording", optionHeight=30,
                     style={"width": "420px", "fontSize": "12px"}),
    ])


def _stages() -> list[tuple]:
    """(key, step label, title, what it does, actions) for every card, built at page load."""
    active, options = recordings.active(), recording_options()
    return [
        ("recordings", "Input", "Recordings",
         "Every CSV in data/ (subfolders included) is offered. Choose one for each role, or leave one "
         "empty to analyse a single recording. Each selection gets its own folder in "
         "outputs/analyses/, named after the recordings in it.",
         [_role_picker("Novice", active.novice, options),
          _role_picker("Trained", active.trained, options),
          html.Button("Use these recordings", id="btn-apply-selection", n_clicks=0,
                      style={**BUTTON, "alignSelf": "flex-end"}),
          html.Div(id="selection-hint", style={"flexBasis": "100%", "fontSize": "12px", "color": "#555"}),
          html.Div(id="catalogue-summary", style={"flexBasis": "100%", "fontSize": "11px", "color": "#888"})]),
        ("preprocess", "1", "Orientation and kinematics",
         "Parse each selected recording and run the Madgwick orientation filter on every sensor, in "
         "parallel. Writes the orientation cache, the 50 Hz kinematic CSV, the sensor inventory and the "
         "orientation validation figure to outputs/recordings/<recording>/, where every selection "
         "using that recording finds them. Nothing here depends on the event windows.",
         [html.Button("Run again", id="btn-stage-preprocess", n_clicks=0, style=BUTTON)]),
        ("load", "2", "Recordings loaded",
         "Rebuild the selected recordings' kinematics from their orientation caches, for the editor "
         "and the metrics.",
         [html.Button("Reload", id="btn-stage-load", n_clicks=0, style=BUTTON)]),
        ("session", "3", "Event windows",
         "The windows every metric is computed on, saved in this selection's folder as "
         "event_editor_session.json. Check and correct them in the event editor. A new session "
         "replaces the current one, which is kept in the folder's sessions/_previous.json.",
         [dcc.Dropdown(id="detect-source", value=config.DETECTOR, clearable=False,
                       options=[{"label": text, "value": key} for key, text in sessions.NEW_SESSION_SOURCES.items()],
                       style={"width": "300px", "fontSize": "12px"}),
          html.Button("New session", id="btn-stage-detect", n_clicks=0, style=BUTTON),
          html.Button("Open the event editor", id="btn-open-editor", n_clicks=0, style=BUTTON)]),
        ("metrics", "4", "Metrics and figures",
         "Run the windows through every metric and write the CSVs and figures to this selection's "
         "folder. Every table and figure names the recording files it came from.",
         [html.Button("Recalculate", id="btn-stage-recalc", n_clicks=0, style=BUTTON),
          html.Button("Open the results", id="btn-open-results", n_clicks=0, style=BUTTON)]),
    ]


def selection_hint(chosen: Selection) -> str:
    """Where the chosen recordings' outputs would go, and whether that is the current selection."""
    folder = f"outputs/analyses/{chosen.id}/" if chosen.names() else ""
    if chosen == recordings.active():
        return f"In use. Outputs are written to {folder}"
    problem = chosen.problem()
    if problem:
        return f"Cannot use this selection: {problem}."
    kind = "an existing analysis" if chosen.output_dir.exists() else "a new analysis"
    return f"Not in use yet: press Use these recordings to switch to {kind}, {folder}"


def catalogue_summary() -> str:
    found = recordings.scan()
    if not found:
        return f"No CSV files in {config.DATA_DIR}."
    usable = [r for r in found if r.usable]
    processed = sum(r.cache_state()[0] == "done" for r in usable)
    parts = [f"{processed} processed", f"{len(usable) - processed} to preprocess"]
    if len(found) > len(usable):
        parts.append(f"{len(found) - len(usable)} not a readable Delsys export")
    return f"data/ holds {len(found)} CSV file{'s' if len(found) != 1 else ''}: " + ", ".join(parts) + "."


def _card(key: str, step: str, title: str, description: str, actions: list) -> html.Div:
    return html.Div([
        html.Div(step, style={"width": "44px", "flex": "none", "fontSize": "12px", "color": "#888",
                              "fontWeight": "600", "paddingTop": "2px"}),
        html.Div([
            html.Div([
                html.Span(title, style={"fontWeight": "600", "fontSize": "14px"}),
                html.Span(id=f"stage-{key}-pill"),
            ], style={"display": "flex", "alignItems": "center", "gap": "10px"}),
            html.Div(description, style={"fontSize": "12px", "color": "#666", "margin": "3px 0 6px"}),
            html.Div(id=f"stage-{key}-detail", style={"fontSize": "12px", "color": "#222"}),
            html.Div(actions, style={"display": "flex", "gap": "8px", "marginTop": "8px",
                                     "alignItems": "center", "flexWrap": "wrap"}) if actions else None,
        ], style={"flex": 1, "minWidth": 0}),
    ], style={"display": "flex", "padding": "12px 14px", "border": "1px solid #e2e2e2",
              "borderRadius": "6px", "backgroundColor": "white", "marginBottom": "8px"})


def layout() -> html.Div:
    return html.Div(html.Div([
        html.H2("Pipeline", style={"margin": "0 0 4px"}),
        html.P("From the raw recordings to the metrics. Run everything with one click, or re-run "
               "a single stage. Every stage writes to outputs/.",
               style={"color": "#555", "fontSize": "13px", "margin": "0 0 14px"}),
        html.Div([
            html.Button("▶  Run pipeline", id="btn-run-all", n_clicks=0, style=PRIMARY),
            html.Span("runs every stage that is missing or out of date",
                      style={"fontSize": "12px", "color": "#666"}),
            html.Span(id="pipeline-action-msg", style={"fontSize": "12px", "color": "#9a6700"}),
        ], style={"display": "flex", "alignItems": "center", "gap": "12px", "marginBottom": "14px"}),

        *[_card(*stage) for stage in _stages()],

        html.Div(id="job-panel", children=[
            html.Div([
                html.Span(id="job-title", style={"fontWeight": "600", "fontSize": "13px"}),
                html.Span(id="job-message", style={"fontSize": "12px", "color": "#555"}),
            ], style={"display": "flex", "gap": "10px", "alignItems": "baseline", "marginBottom": "6px"}),
            html.Div(html.Div(id="job-bar", style={"height": "100%", "width": "0%",
                                                   "backgroundColor": "#2a9d8f"}),
                     style={"height": "6px", "backgroundColor": "#e6e6e6", "borderRadius": "3px",
                            "overflow": "hidden"}),
            html.Details([
                html.Summary("Log", style={"fontSize": "12px", "color": "#555", "cursor": "pointer"}),
                html.Pre(id="job-log", style={"fontSize": "11px", "backgroundColor": "#f6f6f6",
                                              "padding": "8px", "maxHeight": "260px", "overflowY": "auto",
                                              "whiteSpace": "pre-wrap", "margin": "6px 0 0"}),
            ], open=True, style={"marginTop": "8px"}),
        ], style={"display": "none"}),

        html.Div(id="pipeline-settings", style={"fontSize": "12px", "color": "#777", "marginTop": "16px"}),
    ], style={"maxWidth": "1000px", "margin": "0 auto", "padding": "20px 24px"}),
        style={"height": "100%", "overflowY": "auto", "backgroundColor": "#f7f7f5"})


def _pill(state: str):
    text, colour, background = PILL[state]
    return html.Span(text, style={"fontSize": "11px", "fontWeight": "600", "color": colour,
                                  "backgroundColor": background, "padding": "1px 8px", "borderRadius": "9px"})


def _load_status() -> tuple[str, str]:
    if workspace.ready:
        return "done", " · ".join(
            f"{label}: {loaded.n_samples:,} samples at {loaded.fs:.2f} Hz ({loaded.duration_s:.0f} s)"
            for label, loaded in workspace.trials.items()
        )
    return "missing", "not loaded"


# The stage a progress message belongs to, from how the pipeline words it.
STAGE_OF_MESSAGE = (
    (("Parsing", "Orientation filter", "Deriving", "Drawing"), "preprocess"),
    (("Loading the", "Preparing the event editor"), "load"),
    (("Selected", "Not preprocessed yet"), "recordings"),
    (("Detecting events", "Session migrated"), "session"),
    (("Recalculating", "Trunk rotation:", "Monopodal stance:", "Sequence smoothness:", "WARNING"), "metrics"),
)


def running_stage(snapshot: dict) -> str | None:
    if not snapshot["running"]:
        return None
    for prefixes, stage in STAGE_OF_MESSAGE:
        if snapshot["message"].startswith(prefixes):
            return stage
    return None


def _job_panel(snapshot: dict) -> tuple:
    if snapshot["name"] is None:
        return {"display": "none"}, "", "", {"width": "0%"}, ""
    if snapshot["running"]:
        title, colour = f"{snapshot['name']} ({snapshot['elapsed_s']:.0f} s)", "#2a9d8f"
        width = f"{100 * snapshot['fraction']:.0f}%" if snapshot["fraction"] is not None else "100%"
    elif snapshot["error"]:
        title, colour, width = f"{snapshot['name']} failed", "#c0392b", "100%"
    else:
        title, colour, width = f"{snapshot['name']} finished in {snapshot['elapsed_s']:.0f} s", "#2a9d8f", "100%"
    # A step of unknown length shows as a faded full bar.
    bar = {"height": "100%", "width": width, "backgroundColor": colour,
           "opacity": 0.35 if snapshot["running"] and snapshot["fraction"] is None else 1}
    panel = {"display": "block", "padding": "12px 14px", "border": "1px solid #e2e2e2",
             "borderRadius": "6px", "backgroundColor": "white", "marginTop": "14px"}
    return panel, title, snapshot["error"] or snapshot["message"], bar, "\n".join(snapshot["log"])


def register_callbacks(app: Dash) -> None:
    keys = STAGE_KEYS

    # One callback refreshes the whole page on every poll while it is open:
    # the status is read from a few small files, and edits made in the editor
    # change it without any job running.
    @app.callback(
        *[Output(f"stage-{key}-pill", "children") for key in keys],
        *[Output(f"stage-{key}-detail", "children") for key in keys],
        Output("pipeline-settings", "children"),
        Output("job-panel", "style"),
        Output("job-title", "children"),
        Output("job-message", "children"),
        Output("job-bar", "style"),
        Output("job-log", "children"),
        *[Output(button, "disabled") for button in ACTION_BUTTONS],
        Output("select-novice", "options"),
        Output("select-trained", "options"),
        Output("btn-apply-selection", "disabled"),
        Output("selection-hint", "children"),
        Output("catalogue-summary", "children"),
        Input("poll", "n_intervals"),
        Input("page-tabs", "value"),
        Input("store-revs", "data"),
        Input("select-novice", "value"),
        Input("select-trained", "value"),
        State("select-novice", "options"),
    )
    def refresh(_n, page, _revs, novice, trained, shown_options):
        if page != "pipeline":
            raise PreventUpdate
        snapshot = jobs.snapshot()
        stages = {key: (value.state, value.detail) for key, value in pipeline.status().items()}
        stages["load"] = _load_status()
        active = running_stage(snapshot)
        pills = [_pill("loading" if key == active else stages[key][0]) for key in keys]
        settings = (f"Settings in analysis/config.py: new sessions use the {config.DETECTOR} detectors · "
                    f"yaw high-pass de-drifting {'off' if config.IGNORE_HIGH_PASS_FILTER else 'on'}.")
        # Starting anything needs the recordings, so without them every button is off.
        blocked = snapshot["running"] or stages["recordings"][0] != "done"

        # The options are only resent when data/ has changed, so an open
        # dropdown is not redrawn under the pointer every second.
        options = recording_options()
        options_out = no_update if options == shown_options else options
        chosen = Selection(novice=novice or None, trained=trained or None)
        apply_disabled = (snapshot["running"] or chosen == recordings.active()
                          or chosen.problem() is not None)
        return (*pills, *[stages[key][1] for key in keys], settings, *_job_panel(snapshot),
                *[blocked] * len(ACTION_BUTTONS), options_out, options_out, apply_disabled,
                selection_hint(chosen), catalogue_summary())

    @app.callback(
        Output("pipeline-action-msg", "children"),
        *[Input(button, "n_clicks") for button in ACTION_BUTTONS],
        Input("btn-apply-selection", "n_clicks"),
        State("detect-source", "value"),
        State("select-novice", "value"),
        State("select-trained", "value"),
        prevent_initial_call=True,
    )
    def start_job(*args):
        source, novice, trained = args[-3:]
        start = {
            "btn-apply-selection": lambda: tasks.use_recordings(novice, trained),
            "btn-run-all": tasks.run_pipeline,
            "btn-stage-preprocess": tasks.preprocess,
            "btn-stage-load": tasks.load_recordings,
            "btn-stage-detect": lambda: tasks.new_session(source),
            "btn-stage-recalc": tasks.recalculate,
        }.get(ctx.triggered_id)
        if start is None:
            raise PreventUpdate
        return "" if start() else "Another job is still running."
