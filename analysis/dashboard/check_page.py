"""The Kinematics check page: are the sensors where the analysis assumes, and do the kinematics look right?

Opened from stage 1 of the Pipeline page, once the recordings are processed
and loaded, before any event is detected.  For the recording chosen at the
top it shows:

* the **sensor table** of :func:`analysis.kinematics_check.sensor_check` --
  what stage 1 also writes to ``sensor_check.csv`` -- with the sensors that
  need a look marked;
* a **stick figure** at any moment, scrubbed with the slider or jumped to one
  of the moments worth comparing with the video: at the highest left-foot lift
  the figure's left leg must be the one up, at the neutral pose it must stand
  upright;
* the **joint angles**, grouped by joint with left and right overlaid, the
  neutral pose and the quiet standing spans shaded.  Clicking a point moves
  the figure to that moment.

Everything shown is computed from the loaded recording once and cached, so
scrubbing only redraws the figure.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, Patch, State, ctx, dash_table, dcc, html, no_update
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots

from analysis import kinematics_check
from analysis.config import TRIALS
from analysis.dashboard.state import workspace
from analysis.skeleton import CONNECTIONS, SEGMENTS, pose

DISPLAY_FS = 25.0
"""Rate everything on this page is drawn at: plenty for movements this slow."""
LEFT_COLOUR, RIGHT_COLOUR, CENTRE_COLOUR = "#3a86c8", "#d1495b", "#444444"
COMPONENT_COLOURS = ("#1f77b4", "#e76f51", "#2a9d8f")
AXIS_COLOURS = {"x": "#d62728", "y": "#2ca02c", "z": "#1f77b4"}
HIDDEN, SHOWN = {"display": "none"}, {"display": "block"}
PANEL_HEIGHT_PX = 140
HEAD_OFFSET = np.array([0.0, 0.0, 0.16])
"""Where the head is drawn, above the neck on the chest's axis (m)."""
VIEWS = {
    # view: (camera, zoom)
    # Seen from in front and to the figure's right, in perspective: the view with the most depth.
    "diagonal": ({"eye": {"x": 1.1, "y": 1.3, "z": 0.35}, "projection": {"type": "perspective"}}, 1.0),
    # Straight views without perspective, so angles read true, as from a video camera there.  An
    # orthographic scene is zoomed through its aspect ratio: the eye's distance does not change it.
    "front": ({"eye": {"x": 0.0, "y": 2.0, "z": 0.03}, "projection": {"type": "orthographic"}}, 1.6),
    "side": ({"eye": {"x": 2.0, "y": 0.0, "z": 0.03}, "projection": {"type": "orthographic"}}, 1.6),
    "top": ({"eye": {"x": 0.0, "y": 0.0, "z": 2.0}, "up": {"x": 0.0, "y": 1.0, "z": 0.0},
             "projection": {"type": "orthographic"}}, 1.6),
}
"""Camera presets of the stick figure; the figure's right side is +x, its front +y."""

TABLE_COLUMNS = [
    ("sensor", "sensor"),
    ("segment", "segment"),
    ("mounting_tilt_deg", "mounting tilt (deg)"),
    ("gravity_at_rest_g", "|g| at rest"),
    ("gyro_bias_dps", "gyro bias (deg/s)"),
    ("still_rms_dps", "still (deg/s)"),
    ("knee_flexion_axis_offset_deg", "knee axis off x (deg)"),
    ("allowed_direction_pct", "moves the allowed way (%)"),
    ("missing_values", "missing"),
    ("status", "status"),
    ("notes", "notes"),
]


# ---------------------------------------------------------------------------
# What is drawn, per recording, computed once
# ---------------------------------------------------------------------------


_prepared: dict[str, dict] = {}


def prepared(label: str | None) -> dict | None:
    """The page's data for one loaded recording, cached by the file's content."""
    trials = workspace.trials or {}
    loaded = trials.get(label) if label else None
    if loaded is None:
        return None
    key = loaded.recording.sha256()
    if key not in _prepared:
        kin, fs = loaded.kin, loaded.fs
        rotations = kinematics_check.dedrifted_rotations(kin, fs)
        time, panels = kinematics_check.panel_series(kin, fs, DISPLAY_FS, rotations)
        grid, raw, dedrifted = kinematics_check.segment_orientations(kin, fs, DISPLAY_FS, rotations)
        neutral = kin.calibration.neutral
        _prepared[key] = {
            "check": kinematics_check.sensor_check(kin, loaded.trial, rotations),
            "time": time,
            "panels": panels,
            "grid": grid,
            "raw": raw,
            "dedrifted": dedrifted,
            "moments": kinematics_check.check_moments(kin, fs),
            "neutral_s": (neutral[0] / fs, neutral[1] / fs),
            "quiet_s": [(start / fs, end / fs) for start, end in kin.calibration.quiet_spans],
            "duration_s": loaded.duration_s,
        }
    return _prepared[key]


def _series(values) -> list:
    """plotly 6 sends numpy arrays as typed arrays the bundled plotly.js cannot read."""
    return np.round(np.asarray(values, dtype=float), 4).tolist()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _side(joint: str) -> str:
    return "right" if "right" in joint else "left" if "left" in joint else "centre"


def _lines(pairs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[list, list, list]:
    """Line segments as one trace's coordinates, broken by None."""
    x, y, z = [], [], []
    for a, b in pairs:
        x += [float(a[0]), float(b[0]), None]
        y += [float(a[1]), float(b[1]), None]
        z += [float(a[2]), float(b[2]), None]
    return x, y, z


def skeleton_figure(data: dict, time_s: float, options: list[str], view: str = "diagonal") -> go.Figure:
    """The stick figure at one moment: left blue, right red, and optionally its neutral pose and sensor axes."""
    camera, zoom = VIEWS.get(view, VIEWS["diagonal"])
    orientations = data["dedrifted"] if "dedrift" in options else data["raw"]
    index = int(np.clip(round((time_s - data["grid"][0]) * DISPLAY_FS), 0, len(data["grid"]) - 1))
    positions, rotations = pose(kinematics_check.rotations_at(orientations, index))

    figure = go.Figure()
    if "ghost" in options:
        rest, _ = pose({})
        x, y, z = _lines([(rest[a], rest[b]) for a, b in CONNECTIONS])
        figure.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", hoverinfo="skip",
                                      line={"color": "#cccccc", "width": 4}, name="neutral pose"))
    for side, colour in (("centre", CENTRE_COLOUR), ("left", LEFT_COLOUR), ("right", RIGHT_COLOUR)):
        bones = [(positions[a], positions[b]) for a, b in CONNECTIONS if _side(b) == side]
        x, y, z = _lines(bones)
        figure.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", hoverinfo="skip",
                                      line={"color": colour, "width": 7}, name=side))
    joints = list(positions)
    figure.add_trace(go.Scatter3d(
        x=[float(positions[j][0]) for j in joints], y=[float(positions[j][1]) for j in joints],
        z=[float(positions[j][2]) for j in joints], mode="markers", text=joints,
        hovertemplate="%{text}<extra></extra>", marker={"size": 3, "color": "#222222"}, name="joints",
    ))
    # No sensor on the head: it is drawn on the chest's axis, so the figure reads at a glance.
    head = positions["neck"] + rotations["neck"].apply(HEAD_OFFSET)
    figure.add_trace(go.Scatter3d(x=[float(head[0])], y=[float(head[1])], z=[float(head[2])], mode="markers",
                                  hoverinfo="skip", marker={"size": 16, "color": "#bbbbbb", "opacity": 0.8},
                                  name="head"))
    if "axes" in options:
        for axis, unit in zip("xyz", np.eye(3)):
            pairs = [(positions[j], positions[j] + 0.08 * rotations[j].apply(unit))
                     for j, (_, sensor, _) in SEGMENTS.items() if sensor is not None]
            x, y, z = _lines(pairs)
            figure.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", hoverinfo="skip",
                                          line={"color": AXIS_COLOURS[axis], "width": 3}, name=f"sensor {axis}"))
    figure.update_layout(
        height=620,
        margin={"l": 0, "r": 0, "t": 36, "b": 0},
        showlegend=False,
        title={"text": f"t = {data['grid'][index]:.2f} s", "x": 0.5, "font": {"size": 13}},
        # Keeps the camera where the user put it while the slider moves; a new view resets it.
        uirevision=f"skeleton-{view}",
        scene={
            "xaxis": {"range": [-0.9, 0.9], "title": "x right"},
            "yaxis": {"range": [-0.9, 0.9], "title": "y forward"},
            "zaxis": {"range": [-1.0, 1.0], "title": "z up"},
            "aspectmode": "manual",
            "aspectratio": {"x": 0.9 * zoom, "y": 0.9 * zoom, "z": 1.0 * zoom},
            "camera": camera,
        },
    )
    return figure


def _bands(data: dict, time_s: float | None, rows: int) -> list[dict]:
    """The neutral pose and the quiet spans shaded across every panel, and a line at ``time_s``."""
    xref = f"x{rows}" if rows > 1 else "x"
    shapes = [{"type": "rect", "xref": xref, "yref": "paper", "x0": start, "x1": end, "y0": 0, "y1": 1,
               "fillcolor": "#9aa0a6", "opacity": 0.15, "line": {"width": 0}, "layer": "below"}
              for start, end in data["quiet_s"]]
    start, end = data["neutral_s"]
    shapes.append({"type": "rect", "xref": xref, "yref": "paper", "x0": start, "x1": end, "y0": 0, "y1": 1,
                   "fillcolor": "#57a773", "opacity": 0.35, "line": {"width": 0}, "layer": "below"})
    if time_s is not None:
        shapes.append({"type": "line", "xref": xref, "yref": "paper", "x0": time_s, "x1": time_s,
                       "y0": 0, "y1": 1, "line": {"color": "#222222", "width": 1.2, "dash": "dot"}})
    return shapes


def angles_figure(data: dict, time_s: float | None) -> go.Figure:
    panels = data["panels"]
    rows = len(panels)
    figure = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.018,
                           subplot_titles=[panel["title"] for panel in panels])
    time = _series(data["time"])
    for row, panel in enumerate(panels, 1):
        for k, (label, side, values) in enumerate(panel["traces"]):
            colour = (LEFT_COLOUR if side == kinematics_check.LEFT else RIGHT_COLOUR if side == kinematics_check.RIGHT
                      else COMPONENT_COLOURS[k % len(COMPONENT_COLOURS)])
            figure.add_trace(go.Scattergl(
                x=time, y=_series(values), mode="lines", name=label, line={"width": 1.1, "color": colour},
                hovertemplate="%{y:.1f}<extra>" + label + "</extra>",
            ), row=row, col=1)
    figure.update_layout(
        height=PANEL_HEIGHT_PX * rows + 60,
        margin={"l": 50, "r": 10, "t": 30, "b": 30},
        showlegend=False,
        hovermode="x unified",
        plot_bgcolor="#fbfbfb",
        uirevision="angles",
        shapes=_bands(data, time_s, rows),
    )
    figure.update_xaxes(showgrid=True, gridcolor="#e8e8e8")
    figure.update_yaxes(showgrid=True, gridcolor="#e8e8e8")
    figure.update_xaxes(title_text="time (s)", row=rows, col=1)
    for annotation in figure.layout.annotations:
        annotation.font.size = 11
    return figure


# ---------------------------------------------------------------------------
# Layout and callbacks
# ---------------------------------------------------------------------------


def layout() -> html.Div:
    intro = ("Check what stage 1 produced before any event is placed on it. The table says how each "
             "sensor sits and reads; the stick figure and the joint angles show whether the movement looks "
             "like the person in the video. The quickest test of whether every sensor is on the segment "
             "and side it is labelled with: jump to each moment below and compare the figure with the video.")
    help_figure = ("Left segments are blue, right red, the grey outline is the neutral pose, and the head is "
                   "drawn on the chest's axis (there is no head sensor). Drag to rotate the figure, or pick a "
                   "view: front, side and top show angles without perspective. Headings are de-drifted by "
                   "default: tilt is anchored by gravity and exact, but each sensor's heading drifts on its "
                   "own, and without the de-drift the figure slowly twists apart. In the joint angles, green "
                   "is the neutral pose every angle is measured from, grey the quiet standing the gyroscope "
                   "bias is read from; click anywhere to move the figure there. The arms are shown as angles "
                   "between long axes, which ignore rotation about a segment's own axis.")
    return html.Div(html.Div([
        html.H2("Kinematics check", style={"margin": "0 0 4px"}),
        html.P(intro, style={"color": "#555", "fontSize": "13px", "margin": "0 0 12px", "maxWidth": "1100px"}),
        html.Div([
            html.P("The recordings are not loaded yet: run stage 1 on the Pipeline page, and they load by "
                   "themselves when it finishes.", style={"color": "#555", "fontSize": "13px"}),
            dcc.Link("Go to the Pipeline page", href="/pipeline", style={"fontSize": "13px"}),
        ], id="check-empty", style=SHOWN),
        html.Div([
            dcc.RadioItems(id="check-recording", inline=True, inputStyle={"marginRight": "4px"},
                           labelStyle={"marginRight": "16px", "fontSize": "13px", "cursor": "pointer"}),
            dcc.Store(id="check-data-rev"),
            html.Div(id="check-summary", style={"fontSize": "13px", "margin": "8px 0 6px", "fontWeight": "600"}),
            html.H3("Sensors", style={"margin": "12px 0 4px"}),
            html.Div("Mounting tilt: how far the axis the mounting assumes is vertical sits from vertical in "
                     "the neutral pose. Still: the sensor's angular speed while the participant stands still, "
                     "compared across sensors. Knee axis: how far each knee's flexion axis lies from x. Moves the "
                     "allowed way: of the time a thigh is raised or a knee bent, how often the thigh points forward "
                     "and the shank swings back, as the hip and knee allow; a leg sensor turned around on its "
                     "segment, or left and right swapped, makes it the other way.",
                     style={"fontSize": "12px", "color": "#666", "marginBottom": "6px"}),
            dash_table.DataTable(
                id="check-table",
                columns=[{"name": name, "id": key} for key, name in TABLE_COLUMNS],
                style_cell={"fontSize": "11px", "padding": "4px 6px", "textAlign": "left",
                            "fontFamily": "system-ui, sans-serif", "whiteSpace": "normal", "height": "auto"},
                style_cell_conditional=[{"if": {"column_id": "notes"}, "minWidth": "340px"}],
                style_header={"fontWeight": "bold", "backgroundColor": "#f0f0f0"},
                style_data_conditional=[
                    {"if": {"filter_query": '{status} = "check"'}, "backgroundColor": "#fff4d6"},
                ],
            ),
            html.H3("Stick figure and joint angles", style={"margin": "18px 0 4px"}),
            html.Div(help_figure, style={"fontSize": "12px", "color": "#666", "marginBottom": "8px",
                                         "maxWidth": "1100px"}),
            html.Div([
                dcc.Dropdown(id="check-moment", placeholder="jump to a moment worth comparing with the video…",
                             style={"width": "480px", "fontSize": "12px"}),
                dcc.Checklist(id="check-options", inline=True, value=["dedrift", "ghost"],
                              options=[{"label": " de-drift headings", "value": "dedrift"},
                                       {"label": " neutral pose", "value": "ghost"},
                                       {"label": " sensor axes", "value": "axes"}],
                              inputStyle={"marginRight": "3px"},
                              labelStyle={"marginRight": "14px", "fontSize": "12px"}),
                html.Div([
                    html.Span("view:", style={"fontSize": "12px", "color": "#666", "marginRight": "6px"}),
                    dcc.RadioItems(id="check-view", inline=True, value="diagonal",
                                   options=[{"label": f" {name}", "value": name} for name in VIEWS],
                                   inputStyle={"marginRight": "3px"},
                                   labelStyle={"marginRight": "12px", "fontSize": "12px"}),
                ], style={"display": "flex", "alignItems": "center"}),
            ], style={"display": "flex", "gap": "16px", "alignItems": "center", "flexWrap": "wrap"}),
            html.Div(dcc.Slider(id="check-time", min=0, max=1, step=1.0 / DISPLAY_FS, value=0, marks=None,
                                updatemode="drag", tooltip={"placement": "bottom", "always_visible": True}),
                     style={"margin": "10px 0 4px"}),
            html.Div([
                html.Div(dcc.Graph(id="check-skeleton", config={"displaylogo": False}),
                         style={"flex": "0 0 38%", "position": "sticky", "top": "0", "alignSelf": "flex-start",
                                "border": "1px solid #e2e2e2", "borderRadius": "6px", "backgroundColor": "white"}),
                html.Div(dcc.Graph(id="check-angles", config={"displaylogo": False, "scrollZoom": True}),
                         style={"flex": "1", "minWidth": 0}),
            ], style={"display": "flex", "gap": "12px"}),
        ], id="check-body", style=HIDDEN),
    ], style={"maxWidth": "1500px", "margin": "0 auto", "padding": "20px 24px"}),
        style={"height": "100%", "overflowY": "auto"})


def register_callbacks(app: Dash) -> None:
    @app.callback(
        Output("check-empty", "style"),
        Output("check-body", "style"),
        Output("check-recording", "options"),
        Output("check-recording", "value"),
        Output("check-data-rev", "data"),
        Input("page-tabs", "value"),
        Input("store-revs", "data"),
        State("check-recording", "value"),
        State("check-data-rev", "data"),
    )
    def structure(page, revs, current, seen):
        if page != "check":
            raise PreventUpdate
        trials = workspace.trials or {}
        roles = [label for label in TRIALS if label in trials]
        if not roles:
            return SHOWN, HIDDEN, [], None, no_update
        options = [{"label": f"{label}: {trials[label].recording.name}", "value": label} for label in roles]
        data_rev = (revs or {}).get("data")
        return (HIDDEN, SHOWN, options, current if current in roles else roles[0],
                data_rev if data_rev != seen else no_update)

    @app.callback(
        Output("check-summary", "children"),
        Output("check-table", "data"),
        Output("check-moment", "options"),
        Output("check-moment", "value"),
        Output("check-time", "max"),
        Output("check-angles", "figure"),
        Input("check-recording", "value"),
        Input("check-data-rev", "data"),
    )
    def content(label, _data_rev):
        data = prepared(label)
        if data is None:
            raise PreventUpdate
        check = data["check"]
        summary = f"{label}: {kinematics_check.summary(check)}."
        records = check.astype(object).where(check.notna(), "").to_dict("records")
        moments = [{"label": f"{text} ({time_s:.1f} s)", "value": round(time_s, 2)} for text, time_s in data["moments"]]
        return (summary, records, moments, None, float(data["grid"][-1]),
                angles_figure(data, data["moments"][0][1]))

    @app.callback(
        Output("check-time", "value"),
        Input("check-recording", "value"),
        Input("check-moment", "value"),
        Input("check-angles", "clickData"),
        State("check-time", "value"),
    )
    def choose_time(label, moment, click, current):
        data = prepared(label)
        if data is None:
            raise PreventUpdate
        trigger = ctx.triggered_id
        if trigger == "check-moment":
            if moment is None:
                raise PreventUpdate
            return float(moment)
        if trigger == "check-angles":
            points = (click or {}).get("points") or []
            if not points:
                raise PreventUpdate
            return float(points[0]["x"])
        return round(float(data["moments"][0][1]), 2)  # a new recording opens at its neutral pose

    @app.callback(
        Output("check-skeleton", "figure"),
        Input("check-time", "value"),
        Input("check-options", "value"),
        Input("check-view", "value"),
        Input("check-recording", "value"),
    )
    def draw_skeleton(time_s, options, view, label):
        data = prepared(label)
        if data is None or time_s is None:
            raise PreventUpdate
        return skeleton_figure(data, float(time_s), options or [], view or "diagonal")

    @app.callback(
        Output("check-angles", "figure", allow_duplicate=True),
        Input("check-time", "value"),
        State("check-recording", "value"),
        prevent_initial_call=True,
    )
    def mark_time(time_s, label):
        data = prepared(label)
        if data is None or time_s is None:
            raise PreventUpdate
        patch = Patch()
        patch["layout"]["shapes"] = _bands(data, float(time_s), len(data["panels"]))
        return patch
