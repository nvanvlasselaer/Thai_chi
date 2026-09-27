"""The Sensor names card of the Pipeline page: which sensor in each recording is which.

The analysis knows its sensors by fixed names (:mod:`analysis.sensor_names`); a
recording labels them however they were set up in Trigno Discover.  For the
recording chosen at the top of the card, the table lists every label in the
file with the name proposed for it -- the one the label spells, one it reads as,
or the one its serial number has in another recording -- and a dropdown to
change it.  **Save these names** writes them to the recording's
``sensor_names.json``, and stage 1 processes the recording with them.

Beside the table, the neutral pose seen from the front shows which label each
spot on the body has been given, so a spot with no sensor, or with two, stands
out before anything is computed.  Whether the sensors really were on those
segments shows once they move, on the Kinematics check page.
"""

from __future__ import annotations

import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dash_table, dcc, html
from dash.exceptions import PreventUpdate

from analysis import recordings, sensor_names
from analysis.config import SENSOR_MAP
from analysis.dashboard.state import jobs
from analysis.skeleton import CONNECTIONS, pose, sensor_positions

UNUSED, UNDECIDED = "(not used)", ""
"""Table values for a label left out of the analysis, and for one not decided on."""
LEFT_COLOUR, RIGHT_COLOUR, CENTRE_COLOUR = "#3a86c8", "#d1495b", "#444444"
MISSING_COLOUR, DOUBLE_COLOUR = "#c0392b", "#b7791f"
BUTTON = {"padding": "5px 12px", "fontSize": "12px", "cursor": "pointer", "whiteSpace": "nowrap"}

NAME_OPTIONS = ([{"label": f"{name}  ·  {SENSOR_MAP[name].lower()}", "value": name} for name in sensor_names.NAMES]
                + [{"label": "not used", "value": UNUSED}])
LABEL_GAP = 0.14
"""Least vertical distance between two labels of the figure, in its units (m)."""
FIGURE_WIDTH_PX = 460


# ---------------------------------------------------------------------------
# The table's rows
# ---------------------------------------------------------------------------


def rows_for(recording: recordings.Recording) -> list[dict]:
    """One row per label in the file: the saved name where there is one, else the proposed one."""
    labels = recording.sensor_labels()
    saved = sensor_names.load(recording.sensor_names_path)
    proposals = sensor_names.suggest(labels, recordings.serial_history(exclude=recording))
    rows = []
    for label in labels:
        proposed, _, why = proposals[label]
        if saved is not None and label in saved:
            value = saved[label] or UNUSED
            how = "saved" if saved[label] == proposed else "saved, chosen by hand"
        else:
            value = proposed or UNDECIDED
            how = why if proposed else f"choose a name: {why}"
        rows.append({"label": label, "name": value, "how": how, "base": value, "base_how": how})
    return rows


def names_of(rows: list[dict]) -> dict[str, str | None]:
    """``{label: name or None}`` from the table, leaving out labels not decided on."""
    return {row["label"]: (None if row["name"] == UNUSED else row["name"])
            for row in rows if row["name"] != UNDECIDED}


def _side(sensor: str) -> str:
    return SENSOR_MAP.get(sensor, "").split(" ")[0]


# ---------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------


def naming_figure(rows: list[dict]) -> go.Figure:
    """The neutral pose seen from the front, each sensor spot labelled with what it has been given."""
    names = names_of(rows)
    holders = {name: [label for label, given in names.items() if given == name] for name in sensor_names.NAMES}
    positions, _ = pose({})
    spots = sensor_positions(positions)
    figure = go.Figure()

    # Seen from the front: the figure's right side is on the viewer's left.
    def joint_side(joint: str) -> str:
        return "right" if "right" in joint else "left" if "left" in joint else "centre"

    for side, colour in (("centre", CENTRE_COLOUR), ("left", LEFT_COLOUR), ("right", RIGHT_COLOUR)):
        x, y = [], []
        for a, b in CONNECTIONS:
            if joint_side(b) == side:
                x += [-float(positions[a][0]), -float(positions[b][0]), None]
                y += [float(positions[a][2]), float(positions[b][2]), None]
        figure.add_trace(go.Scatter(x=x, y=y, mode="lines", line={"color": colour, "width": 5},
                                    opacity=0.35, hoverinfo="skip"))
    neck = positions["neck"]
    figure.add_trace(go.Scatter(x=[-float(neck[0])], y=[float(neck[2]) + 0.11], mode="markers",
                                marker={"size": 26, "color": "#e3e3e3"}, hoverinfo="skip"))

    columns = {-1: [], +1: []}  # viewer's left: the figure's right side (and the pelvis); viewer's right: the rest
    for name in sensor_names.NAMES:
        columns[-1 if _side(name) == "Right" or name == "lumbar" else +1].append(name)
    colours, hovers = {}, {}
    for direction, names_here in columns.items():
        names_here.sort(key=lambda name: -spots[name][2])
        previous = None
        for name in names_here:
            spot = spots[name]
            y = float(spot[2]) if previous is None else min(float(spot[2]), previous - LABEL_GAP)
            previous = y
            given = holders[name]
            if len(given) == 1:
                text, colour = f"<b>{given[0]}</b>", "#222222"
            elif not given:
                text, colour = "<b>no sensor</b>", MISSING_COLOUR
            else:
                text, colour = f"<b>{len(given)} sensors</b>", DOUBLE_COLOUR  # named in the hover and the table
            colours[name] = colour
            hovers[name] = f"{name} ({SENSOR_MAP[name].lower()}): " + (", ".join(given) or "no sensor")
            anchor = 0.5 * direction
            figure.add_trace(go.Scatter(x=[-float(spot[0]), anchor], y=[float(spot[2]), y], mode="lines",
                                        line={"color": "#c8c8c8", "width": 1}, hoverinfo="skip"))
            figure.add_annotation(x=anchor, y=y, showarrow=False, xanchor="right" if direction < 0 else "left",
                                  text=f"{text}<br><span style='color:#999'>{name}</span>",
                                  align="right" if direction < 0 else "left", font={"size": 10, "color": colour})

    order = list(sensor_names.NAMES)
    figure.add_trace(go.Scatter(
        x=[-float(spots[name][0]) for name in order], y=[float(spots[name][2]) for name in order],
        mode="markers", text=[hovers[name] for name in order], hovertemplate="%{text}<extra></extra>",
        marker={"size": 9, "color": [colours[name] if colours[name] != "#222222" else
                                     (LEFT_COLOUR if _side(name) == "Left" else
                                      RIGHT_COLOUR if _side(name) == "Right" else CENTRE_COLOUR)
                                     for name in order],
                "line": {"width": 1, "color": "white"}},
    ))
    for x, text in ((-0.8, "right side"), (0.8, "left side")):
        figure.add_annotation(x=x, y=0.72, text=text, showarrow=False, font={"size": 11, "color": "#777"})
    # Ranges chosen so a metre is about as long across as up, at the figure's size, without locking
    # the aspect ratio -- a locked one shrinks the figure into the middle of its box.
    figure.update_layout(
        height=430, margin={"l": 0, "r": 0, "t": 24, "b": 0}, showlegend=False, plot_bgcolor="white",
        title={"text": "seen from the front", "x": 0.5, "font": {"size": 11, "color": "#777"}},
        xaxis={"range": [-1.35, 1.35], "visible": False, "fixedrange": True},
        yaxis={"range": [-0.98, 0.84], "visible": False, "fixedrange": True},
    )
    return figure


# ---------------------------------------------------------------------------
# Layout and callbacks
# ---------------------------------------------------------------------------


def layout() -> html.Div:
    """The card's controls: which recording, its labels and names, and the figure."""
    return html.Details([
        html.Summary("Show the sensors", style={"fontSize": "12px", "color": "#1d6fa5", "cursor": "pointer"}),
        html.Div([
            dcc.RadioItems(id="names-recording", inline=True, inputStyle={"marginRight": "4px"},
                           labelStyle={"marginRight": "16px", "fontSize": "12px", "cursor": "pointer"}),
            dcc.Store(id="names-drafts", data={}),
            html.Div([
                html.Div([
                    dash_table.DataTable(
                        id="names-table",
                        columns=[
                            {"name": "label in the recording", "id": "label", "editable": False},
                            {"name": "name in the analysis", "id": "name", "presentation": "dropdown",
                             "editable": True},
                            {"name": "how", "id": "how", "editable": False},
                        ],
                        dropdown={"name": {"options": NAME_OPTIONS, "clearable": False}},
                        style_cell={"fontSize": "11px", "padding": "3px 6px", "textAlign": "left",
                                    "fontFamily": "system-ui, sans-serif", "whiteSpace": "normal",
                                    "height": "auto"},
                        style_cell_conditional=[{"if": {"column_id": "label"}, "minWidth": "120px"},
                                                {"if": {"column_id": "name"}, "minWidth": "190px"},
                                                {"if": {"column_id": "how"}, "color": "#666"}],
                        style_header={"fontWeight": "bold", "backgroundColor": "#f0f0f0"},
                        style_data_conditional=[
                            {"if": {"filter_query": '{name} = ""'}, "backgroundColor": "#fdecea"},
                            {"if": {"filter_query": '{how} contains "not saved"'}, "backgroundColor": "#fff4d6"},
                        ],
                        css=[{"selector": ".Select-menu-outer", "rule": "display: block !important"}],
                    ),
                    html.Div(id="names-problems", style={"fontSize": "12px", "margin": "8px 0"}),
                    html.Div([
                        html.Button("Save these names", id="btn-names-save", n_clicks=0, style=BUTTON),
                        html.Button("Back to the suggestions", id="btn-names-reset", n_clicks=0, style=BUTTON),
                        html.Span(id="names-msg", style={"fontSize": "12px", "color": "#555"}),
                    ], style={"display": "flex", "gap": "8px", "alignItems": "center", "flexWrap": "wrap"}),
                ], style={"flex": "1 1 560px", "minWidth": 0}),
                dcc.Graph(id="names-figure", config={"displayModeBar": False},
                          style={"flex": f"0 0 {FIGURE_WIDTH_PX}px"}),
            ], style={"display": "flex", "gap": "14px", "flexWrap": "wrap", "marginTop": "8px"}),
        ], style={"marginTop": "8px"}),
    ], id="names-details", open=False, style={"flexBasis": "100%"})


def _recording(name: str | None) -> recordings.Recording | None:
    chosen = {recording.name: recording for recording in recordings.active().recordings().values() if recording}
    return chosen.get(name)


def register_callbacks(app: Dash) -> None:
    @app.callback(
        Output("names-recording", "options"),
        Output("names-recording", "value"),
        Output("names-details", "open"),
        Input("page-tabs", "value"),
        Input("store-revs", "data"),
        State("names-recording", "options"),
        State("names-recording", "value"),
    )
    def structure(page, _revs, shown, current):
        if page != "pipeline":
            raise PreventUpdate
        chosen = {role: r for role, r in recordings.active().recordings().items() if r and r.usable}
        options = [{"label": f"{role}: {recording.name}", "value": recording.name}
                   for role, recording in chosen.items()]
        if options == shown:  # the same recordings: leave the user's choice and the card as they are
            raise PreventUpdate
        unsettled = [recording.name for recording in chosen.values() if recording.naming_problems()]
        names = [option["value"] for option in options]
        value = current if current in names else (unsettled or names or [None])[0]
        return options, value, bool(unsettled)

    @app.callback(
        Output("names-table", "data"),
        Output("names-drafts", "data"),
        Output("names-problems", "children"),
        Output("names-figure", "figure"),
        Output("btn-names-save", "disabled"),
        Output("names-msg", "children"),
        Input("names-recording", "value"),
        Input("names-table", "data_timestamp"),
        Input("btn-names-reset", "n_clicks"),
        Input("btn-names-save", "n_clicks"),
        State("names-table", "data"),
        State("names-drafts", "data"),
    )
    def edit(name, _edited, _reset, _save, rows, drafts):
        recording = _recording(name)
        if recording is None:
            return [], drafts, "No recording selected.", go.Figure(), True, ""
        drafts = dict(drafts or {})
        trigger, message = ctx.triggered_id, ""
        if trigger == "names-table" and rows:
            for row in rows:
                row["how"] = row["base_how"] if row["name"] == row["base"] else "changed here, not saved yet"
            drafts[name] = rows
        elif trigger == "btn-names-save" and rows:
            labels = recording.sensor_labels()
            if jobs.snapshot()["running"]:
                message = "Wait for the running job to finish before saving."
            elif sensor_names.problems(names_of(rows), labels):
                message = "Not saved: sort out the problems above first."
            else:
                sensor_names.save(recording.sensor_names_path, names_of(rows), labels, recording.identity())
                drafts.pop(name, None)
                rows = rows_for(recording)
                message = ("Saved. Stage 1 processes the recording with these names: press Run pipeline."
                           if recording.cache_state()[0] != "done" else "Saved.")
        elif trigger == "btn-names-reset":
            drafts.pop(name, None)
            rows = rows_for(recording)
        else:
            rows = drafts.get(name) or rows_for(recording)

        names = names_of(rows)
        problems = sensor_names.problems(names, recording.sensor_labels())
        unsaved = names != recording.sensor_names()
        if problems:
            summary = html.Div([html.Div("To sort out before stage 1 can run:", style={"fontWeight": "600"}),
                                html.Ul([html.Li(problem) for problem in problems],
                                        style={"margin": "2px 0", "paddingLeft": "18px"})],
                               style={"color": MISSING_COLOUR})
        elif unsaved:
            summary = html.Div("Every sensor has a name. Save them to use them.", style={"color": DOUBLE_COLOUR})
        else:
            summary = html.Div(f"All {len(sensor_names.NAMES)} sensors have a name, and these are the ones "
                               "stage 1 uses.", style={"color": "#2a9d8f"})
        return rows, drafts, summary, naming_figure(rows), bool(problems) or not unsaved, message
