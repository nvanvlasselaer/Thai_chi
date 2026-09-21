#!/usr/bin/env python3
"""Interactive editor for Tai Chi balance-event windows.

Automatic detection gets most events roughly right and some badly wrong, and
every variability metric is computed *on* those windows, so a bad window
silently produces a bad number.  This app plots the kinematics on a scrollable,
zoomable timeline, lets the boundaries be dragged into place, and recalculates
the metrics through the pipeline's own functions.

Four boundaries per event are editable: the start and end of the movement, and
the start and end of the stabilization period that follows it.  The subplot
showing the yaw-velocity envelope also draws the threshold the detector used, so
a boundary can be judged against the signal that produced it rather than taken
on trust.

Run it with::

    python3 analysis/event_editor.py [--port 8051] [--reseed auto|v1]
                                     [--recompute-orientation] [--debug]
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

# The pipeline binds a matplotlib backend at import time, and the default here
# is "macosx", which cannot render from a Dash worker thread.  This must run
# before tai_chi_trunk_and_knee is imported, directly or indirectly.
import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import flask
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, Patch, State, ctx, dash_table, dcc, html, no_update
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots

import editor_data as edt
import event_detection as ed
import tai_chi_trunk_and_knee as pipeline

DISPLAY_FS = 50.0
EVENT_COLOUR = "#f2b134"
STAB_COLOUR = "#57a773"
CONTEXT_COLOUR = "#9aa0a6"
ROW_TITLES = [
    "trunk-pelvis yaw (deg)",
    "yaw speed envelope (deg/s)",
    "lumbar + chest angular velocity (deg/s)",
    "knee flexion |x| (deg)",
]
N_ROWS = len(ROW_TITLES)
RECOMPUTE_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Display traces
#
# plotly 6 serialises numpy arrays as base64 typed arrays, which the plotly.js
# bundled with dash 2.14 cannot decode -- traces render blank with no error.
# Every array reaching a figure goes through _series().
# ---------------------------------------------------------------------------


def _series(values: np.ndarray) -> list[float]:
    return np.round(np.asarray(values, dtype=float), 4).tolist()


def build_display(loaded: edt.LoadedTrial) -> dict:
    """Resample every plotted channel to 50 Hz once, at startup."""
    kin, fs = loaded.kin, loaded.fs

    def resample(signal: np.ndarray) -> np.ndarray:
        _, values = pipeline.resample_filtered_full(kin.t, signal, fs, DISPLAY_FS)
        return values

    time_s, _ = pipeline.resample_filtered_full(kin.t, kin.t, fs, DISPLAY_FS)

    trunk_diag = ed.trunk_yaw_envelope(kin, fs)
    stab_diag = ed.combined_omega(kin, fs)

    return {
        "time": _series(time_s),
        "yaw": _series(resample(trunk_diag.yaw_deg)),
        "envelope": _series(resample(trunk_diag.envelope_dps)),
        "envelope_floor": float(trunk_diag.floor_dps),
        "envelope_peak_height": float(trunk_diag.peak_height_dps),
        "lumbar_omega": _series(resample(kin.omega_mag["lumbar"])),
        "chest_omega": _series(resample(kin.omega_mag["chestbone"])),
        "combined_omega": _series(resample(stab_diag.combined_omega_dps)),
        "omega_baseline": float(stab_diag.baseline_dps),
        "left_knee": _series(resample(np.abs(pipeline.lowpass_signal(kin.left_knee_deg[:, 0], fs, cutoff_hz=6.0)))),
        "right_knee": _series(resample(np.abs(pipeline.lowpass_signal(kin.right_knee_deg[:, 0], fs, cutoff_hz=6.0)))),
        "duration_s": loaded.duration_s,
    }


def base_figure(label: str, display: dict) -> go.Figure:
    """Four linked-x subplots for one recording."""
    figure = make_subplots(
        rows=N_ROWS,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        subplot_titles=ROW_TITLES,
    )
    time = display["time"]

    def add(y, name, row, colour, width=1.1, dash=None):
        figure.add_trace(
            go.Scattergl(
                x=time, y=y, name=name, mode="lines",
                line={"width": width, "color": colour, "dash": dash},
                hovertemplate="%{x:.2f}s  %{y:.2f}<extra>" + name + "</extra>",
            ),
            row=row, col=1,
        )

    add(display["yaw"], "trunk-pelvis yaw", 1, "#1f77b4", width=1.4)
    add(display["envelope"], "yaw speed", 2, "#7b4173", width=1.2)
    add(display["lumbar_omega"], "lumbar w", 3, "#2a9d8f", width=0.9)
    add(display["chest_omega"], "chest w", 3, "#8ecae6", width=0.9)
    add(display["combined_omega"], "lumbar + chest w", 3, "#264653", width=1.3)
    add(display["left_knee"], "left knee", 4, "#d1495b", width=1.1)
    add(display["right_knee"], "right knee", 4, "#f4a261", width=1.1)

    # The thresholds the detectors actually used, so a boundary is explainable.
    figure.add_hline(
        y=display["envelope_floor"], row=2, col=1,
        line={"color": "#7b4173", "width": 1, "dash": "dot"},
        annotation_text="onset/offset floor", annotation_font_size=9,
    )
    figure.add_hline(
        y=display["omega_baseline"], row=3, col=1,
        line={"color": "#264653", "width": 1, "dash": "dot"},
        annotation_text="quiet baseline", annotation_font_size=9,
    )
    figure.add_hline(
        y=edt.KNEE_THRESHOLD_DEG, row=4, col=1,
        line={"color": "#444444", "width": 1, "dash": "dash"},
        annotation_text="60 deg", annotation_font_size=9,
    )

    figure.update_layout(
        height=760,
        margin={"l": 60, "r": 20, "t": 40, "b": 40},
        dragmode="pan",
        hovermode="x unified",
        showlegend=False,
        # Without a stable uirevision every store update would reset the user's
        # zoom, which is exactly what they need to keep while nudging an edge.
        uirevision=f"{label}-static",
        plot_bgcolor="#fbfbfb",
    )
    figure.update_xaxes(showgrid=True, gridcolor="#e8e8e8")
    figure.update_yaxes(showgrid=True, gridcolor="#e8e8e8")
    figure.update_xaxes(title_text="time (s)", row=N_ROWS, col=1)
    for annotation in figure.layout.annotations:
        annotation.font.size = 11
    return figure


# ---------------------------------------------------------------------------
# Shapes
#
# make_subplots(shared_xaxes=True) makes the bottom axis the driver and matches
# the rest to it, so a shape on "x4" with yref="paper" spans all four rows.
# Four boundaries therefore need only two editable rectangles, always at
# indices 0 and 1, which makes decoding a relayout a fixed lookup.
# ---------------------------------------------------------------------------


DRIVING_XREF = f"x{N_ROWS}"


def _rect(x0: float, x1: float, colour: str, editable: bool, opacity: float) -> dict:
    return {
        "type": "rect",
        "xref": DRIVING_XREF,
        "yref": "paper",
        "x0": x0,
        "x1": x1,
        "y0": 0,
        "y1": 1,
        "fillcolor": colour,
        "opacity": opacity,
        "line": {"width": 1.5 if editable else 0, "color": colour},
        "editable": editable,
        "layer": "above" if editable else "below",
    }


def build_shapes(session: dict, selection: dict, label: str, fs: float) -> list[dict]:
    """Two editable rects for the selected event, then dimmed context rects."""
    selected = find_event(session, selection)
    shapes: list[dict] = []

    window = window_for(selected, label) if selected else None
    if window is not None:
        shapes.append(_rect(window["event_start"] / fs, window["event_end"] / fs, EVENT_COLOUR, True, 0.22))
        shapes.append(_rect(window["stab_start"] / fs, window["stab_end"] / fs, STAB_COLOUR, True, 0.20))
    else:
        # Keep indices 0 and 1 occupied so a stale relayout cannot address a
        # context rect. Placed off-screen rather than omitted.
        shapes.append(_rect(-1, -1, EVENT_COLOUR, False, 0.0))
        shapes.append(_rect(-1, -1, STAB_COLOUR, False, 0.0))

    selected_id = selection.get("event_id")
    for event in all_events(session, selection.get("family", "trunk")):
        if event["event_id"] == selected_id or not event.get("enabled", True):
            continue
        other = window_for(event, label)
        if other is None:
            continue
        shapes.append(_rect(other["event_start"] / fs, other["event_end"] / fs, CONTEXT_COLOUR, False, 0.10))
    return shapes


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------


def all_events(session: dict, family: str) -> list[dict]:
    return session.get("trunk_events" if family == "trunk" else "knee_events", [])


def find_event(session: dict, selection: dict) -> dict | None:
    for event in all_events(session, selection.get("family", "trunk")):
        if event["event_id"] == selection.get("event_id"):
            return event
    return None


def window_for(event: dict | None, label: str) -> dict | None:
    """Return the boundary dict a trial contributes to an event, or None.

    Trunk events hold one window per trial; a monopodal-stance event belongs to
    a single trial and contributes nothing to the other graph.
    """
    if event is None:
        return None
    if "windows" in event:
        return event["windows"].get(label)
    return event if event.get("trial") == label else None


def set_boundaries(event: dict, label: str, band: str, start: int, end: int) -> bool:
    """Write a boundary pair, returning whether anything actually changed."""
    window = window_for(event, label)
    if window is None:
        return False
    start_key, end_key = ("event_start", "event_end") if band == "event" else ("stab_start", "stab_end")
    if window[start_key] == start and window[end_key] == end:
        return False
    window[start_key], window[end_key] = start, end
    window.setdefault("source", {})[band if band == "event" else "stab"] = "manual"
    return True


def decode_relayout(relayout: dict | None) -> dict[str, tuple[float, float]]:
    """Read a shape drag out of relayoutData, in either form plotly emits.

    y0/y1 are ignored: the drag handles on a paper-referenced rect can distort
    the vertical extent, and the next render restores it.
    """
    if not relayout:
        return {}
    result: dict[str, tuple[float, float]] = {}
    shapes = relayout.get("shapes")
    if isinstance(shapes, list):
        for index, band in ((0, "event"), (1, "stab")):
            if index < len(shapes) and "x0" in shapes[index] and "x1" in shapes[index]:
                result[band] = (shapes[index]["x0"], shapes[index]["x1"])
        return result
    for index, band in ((0, "event"), (1, "stab")):
        x0 = relayout.get(f"shapes[{index}].x0")
        x1 = relayout.get(f"shapes[{index}].x1")
        if x0 is not None and x1 is not None:
            result[band] = (x0, x1)
    return result


def event_table_rows(session: dict, family: str, trials: dict[str, edt.LoadedTrial]) -> list[dict]:
    rows = []
    for event in all_events(session, family):
        if family == "trunk":
            novice = event["windows"]["Novice"]
            trained = event["windows"]["Trained"]
            fs_n = trials["Novice"].fs
            fs_t = trials["Trained"].fs
            rows.append({
                "event_id": event["event_id"],
                "detail": "trunk rotation",
                "novice": f"{novice['event_start']/fs_n:.1f}-{novice['event_end']/fs_n:.1f}",
                "trained": f"{trained['event_start']/fs_t:.1f}-{trained['event_end']/fs_t:.1f}",
                "dur": f"{(novice['event_end']-novice['event_start'])/fs_n:.1f}s",
                "flags": event_flags(event),
            })
        else:
            fs = trials[event["trial"]].fs
            rows.append({
                "event_id": event["event_id"],
                "detail": f"{event['trial'][:4]} {event['flexed_leg'][:1]}-knee",
                "novice": f"{event['event_start']/fs:.1f}-{event['event_end']/fs:.1f}",
                "trained": f"stab {event['stab_start']/fs:.1f}-{event['stab_end']/fs:.1f}",
                "dur": f"{(event['event_end']-event['event_start'])/fs:.1f}s",
                "flags": event_flags(event),
            })
    return rows


def event_flags(event: dict) -> str:
    """Short markers: edited, and low-confidence stabilization."""
    windows = event["windows"].values() if "windows" in event else [event]
    flags = []
    if any("manual" in (w.get("source") or {}).values() for w in windows):
        flags.append("edited")
    ratios = [w.get("stab_quiet_ratio") for w in windows]
    if any(r is not None and r > edt.LOW_CONFIDENCE_RATIO for r in ratios):
        flags.append("unsettled")
    if not event.get("enabled", True):
        flags.append("off")
    return ", ".join(flags)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def build_app(trials: dict[str, edt.LoadedTrial], session: dict) -> Dash:
    displays = {label: build_display(loaded) for label, loaded in trials.items()}
    figures = {label: base_figure(label, displays[label]) for label in edt.TRIALS}

    app = Dash(__name__, title="Tai Chi event editor")

    @app.server.route("/outputs/<path:name>")
    def _serve_output(name: str):
        return flask.send_from_directory(pipeline.OUTPUT_DIR, name, max_age=0)

    def number(component_id: str, label_text: str):
        return html.Div([
            html.Label(label_text, style={"fontSize": "11px", "color": "#555"}),
            dcc.Input(id=component_id, type="number", step=0.05, debounce=True,
                      style={"width": "100%", "fontSize": "12px"}),
        ], style={"marginBottom": "4px"})

    def detector_input(component_id: str, label_text: str, value, step):
        return html.Div([
            html.Label(label_text, style={"fontSize": "11px", "color": "#555"}),
            dcc.Input(id=component_id, type="number", value=value, step=step, debounce=True,
                      style={"width": "100%", "fontSize": "12px"}),
        ], style={"marginBottom": "4px"})

    defaults = ed.TrunkDetectorParams()
    stab_defaults = ed.StabilizationParams()

    sidebar = html.Div([
        html.H4("Detector", style={"margin": "4px 0"}),
        html.Div([
            detector_input("p-rel-thr", "onset threshold (fraction of peak)", defaults.rel_threshold, 0.01),
            detector_input("p-floor", "velocity floor (percentile)", defaults.floor_percentile, 1),
            detector_input("p-min-dur", "min event duration (s)", defaults.min_duration_s, 0.1),
            detector_input("p-min-exc", "min yaw excursion (deg)", defaults.min_excursion_deg, 1),
            detector_input("p-n-events", "number of events", defaults.n_events, 1),
            detector_input("p-lambda", "stabilization latency penalty", stab_defaults.latency_weight, 0.05),
            detector_input("p-horizon", "stabilization search horizon (s)", stab_defaults.horizon_s, 1),
            detector_input("p-stab-dur", "stabilization length (s)", stab_defaults.min_duration_s, 0.5),
        ]),
        html.Button("Re-detect trunk events", id="btn-redetect", n_clicks=0,
                    style={"width": "100%", "marginTop": "6px"}),
        html.Div("Replaces all trunk events and discards their edits.",
                 style={"fontSize": "10px", "color": "#888", "marginBottom": "10px"}),

        html.Hr(),
        html.H4("Events", style={"margin": "4px 0"}),
        dash_table.DataTable(
            id="event-table",
            columns=[
                {"name": "id", "id": "event_id"},
                {"name": "what", "id": "detail"},
                {"name": "window", "id": "novice"},
                {"name": "paired / stab", "id": "trained"},
                {"name": "dur", "id": "dur"},
                {"name": "flags", "id": "flags"},
            ],
            row_selectable="single",
            selected_rows=[0],
            page_size=14,
            style_cell={"fontSize": "11px", "padding": "3px", "textAlign": "left"},
            style_header={"fontWeight": "bold", "backgroundColor": "#f0f0f0"},
            style_data_conditional=[
                {"if": {"filter_query": '{flags} contains "unsettled"'}, "backgroundColor": "#fff4e0"},
                {"if": {"filter_query": '{flags} contains "off"'}, "color": "#aaa"},
            ],
        ),
        html.Div([
            html.Button("Toggle on/off", id="btn-toggle", n_clicks=0, style={"flex": 1, "fontSize": "11px"}),
            html.Button("Delete", id="btn-delete", n_clicks=0, style={"flex": 1, "fontSize": "11px"}),
        ], style={"display": "flex", "gap": "4px", "marginTop": "6px"}),

        html.Hr(),
        html.H4("Boundaries (s)", style={"margin": "4px 0"}),
        html.Div(id="boundary-title", style={"fontSize": "11px", "color": "#666", "marginBottom": "4px"}),
        html.Div([
            html.Div([
                html.Strong("Novice", style={"fontSize": "11px"}),
                number("n-ev-start", "event start"),
                number("n-ev-end", "event end"),
                number("n-st-start", "stab start"),
                number("n-st-end", "stab end"),
            ], style={"flex": 1}),
            html.Div([
                html.Strong("Trained", style={"fontSize": "11px"}),
                number("t-ev-start", "event start"),
                number("t-ev-end", "event end"),
                number("t-st-start", "stab start"),
                number("t-st-end", "stab end"),
            ], style={"flex": 1}),
        ], style={"display": "flex", "gap": "8px"}),

        html.Hr(),
        html.Div(id="status", style={"fontSize": "11px", "whiteSpace": "pre-wrap", "color": "#444"}),
        dcc.Loading(html.Div(id="recalc-status",
                             style={"fontSize": "11px", "whiteSpace": "pre-wrap", "color": "#444"})),
        html.Button("Recalculate metrics", id="btn-recalc", n_clicks=0,
                    style={"width": "100%", "marginTop": "8px", "padding": "8px",
                           "fontWeight": "bold", "backgroundColor": "#2a9d8f", "color": "white",
                           "border": "none", "cursor": "pointer"}),
    ], style={"width": "330px", "padding": "10px", "overflowY": "auto", "height": "100vh",
              "borderRight": "1px solid #ddd", "boxSizing": "border-box"})

    main = html.Div([
        html.Div([
            html.H3("Tai Chi event editor", style={"margin": "0 0 2px 0"}),
            html.Div(id="header-meta", style={"fontSize": "11px", "color": "#666"}),
        ], style={"padding": "8px 12px", "borderBottom": "1px solid #ddd"}),

        dcc.Tabs(id="family-tabs", value="trunk", children=[
            dcc.Tab(label="Trunk rotation", value="trunk"),
            dcc.Tab(label="Monopodal stance (knee > 60 deg)", value="knee"),
        ]),

        html.Div(id="validation", style={"fontSize": "11px", "padding": "6px 12px"}),

        html.Div([
            html.Div([
                html.Div("Novice", style={"fontWeight": "bold", "fontSize": "12px", "padding": "4px 12px"}),
                dcc.Graph(id="graph-Novice", figure=figures["Novice"], config=GRAPH_CONFIG),
            ]),
            html.Div([
                html.Div("Trained", style={"fontWeight": "bold", "fontSize": "12px", "padding": "4px 12px"}),
                dcc.Graph(id="graph-Trained", figure=figures["Trained"], config=GRAPH_CONFIG),
            ]),
        ], style={"overflowY": "auto"}),

        html.Div([
            html.H4("Recalculated metrics", style={"margin": "8px 12px 4px"}),
            dash_table.DataTable(
                id="metrics-table", page_size=14,
                style_cell={"fontSize": "10px", "padding": "3px"},
                style_header={"fontWeight": "bold", "backgroundColor": "#f0f0f0"},
                style_table={"overflowX": "auto"},
            ),
            html.Div(id="figure-gallery", style={"display": "flex", "gap": "8px",
                                                 "flexWrap": "wrap", "padding": "8px 12px"}),
        ]),
    ], style={"flex": 1, "overflowY": "auto", "height": "100vh", "boxSizing": "border-box"})

    app.layout = html.Div([
        dcc.Store(id="store-session", data=session),
        dcc.Store(id="store-selection", data={"family": "trunk", "event_id": _first_id(session, "trunk")}),
        dcc.Store(id="store-figrev", data=0),
        sidebar,
        main,
    ], style={"display": "flex", "fontFamily": "system-ui, sans-serif", "margin": 0})

    register_callbacks(app, trials, displays)
    return app


GRAPH_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "doubleClick": "reset",
    "edits": {"shapePosition": True},
    "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
}


def _first_id(session: dict, family: str) -> str | None:
    events = all_events(session, family)
    return events[0]["event_id"] if events else None


def register_callbacks(app: Dash, trials: dict[str, edt.LoadedTrial], displays: dict[str, dict]) -> None:
    n_samples = {label: loaded.n_samples for label, loaded in trials.items()}
    fs_of = {label: loaded.fs for label, loaded in trials.items()}

    BOUNDARY_INPUTS = [
        ("n-ev-start", "Novice", "event", 0), ("n-ev-end", "Novice", "event", 1),
        ("n-st-start", "Novice", "stab", 0), ("n-st-end", "Novice", "stab", 1),
        ("t-ev-start", "Trained", "event", 0), ("t-ev-end", "Trained", "event", 1),
        ("t-st-start", "Trained", "stab", 0), ("t-st-end", "Trained", "stab", 1),
    ]

    @app.callback(
        Output("store-selection", "data"),
        Input("family-tabs", "value"),
        Input("event-table", "selected_rows"),
        State("store-session", "data"),
        State("store-selection", "data"),
    )
    def update_selection(family, selected_rows, session, selection):
        if ctx.triggered_id == "family-tabs":
            return {"family": family, "event_id": _first_id(session, family)}
        events = all_events(session, family)
        if not selected_rows or selected_rows[0] >= len(events):
            raise PreventUpdate
        candidate = {"family": family, "event_id": events[selected_rows[0]]["event_id"]}
        if candidate == selection:
            raise PreventUpdate
        return candidate

    # One callback both mutates and renders.  Dash rejects a dependency cycle
    # that spans two callbacks, and the boundary inputs are necessarily both
    # read (the user types) and written (the user drags), so the read/write pair
    # has to live inside a single callback, which Dash does allow.
    @app.callback(
        Output("store-session", "data"),
        Output("graph-Novice", "figure"),
        Output("graph-Trained", "figure"),
        Output("event-table", "data"),
        Output("validation", "children"),
        Output("boundary-title", "children"),
        Output("status", "children"),
        *[Output(component_id, "value") for component_id, *_ in BOUNDARY_INPUTS],
        *[Output(component_id, "disabled") for component_id, *_ in BOUNDARY_INPUTS],
        Input("graph-Novice", "relayoutData"),
        Input("graph-Trained", "relayoutData"),
        *[Input(component_id, "value") for component_id, *_ in BOUNDARY_INPUTS],
        Input("btn-redetect", "n_clicks"),
        Input("btn-toggle", "n_clicks"),
        Input("btn-delete", "n_clicks"),
        Input("store-selection", "data"),
        State("store-session", "data"),
        State("p-rel-thr", "value"), State("p-floor", "value"),
        State("p-min-dur", "value"), State("p-min-exc", "value"),
        State("p-n-events", "value"), State("p-lambda", "value"),
        State("p-horizon", "value"), State("p-stab-dur", "value"),
    )
    def edit_and_render(novice_relayout, trained_relayout, *args):
        values = list(args[: len(BOUNDARY_INPUTS)])
        rest = args[len(BOUNDARY_INPUTS):]
        (_redetect, _toggle, _delete, selection, session,
         rel_thr, floor_pct, min_dur, min_exc, n_events, lam, horizon, stab_dur) = rest

        trigger = ctx.triggered_id
        session = dict(session)
        message = no_update
        session_out: object = no_update

        if trigger == "btn-redetect":
            trunk_params = ed.TrunkDetectorParams(
                n_events=int(n_events or 6),
                rel_threshold=float(rel_thr or 0.15),
                floor_percentile=float(floor_pct or 40),
                min_duration_s=float(min_dur or 1.5),
                min_excursion_deg=float(min_exc or 8.0),
            )
            stab_params = ed.StabilizationParams(
                latency_weight=float(lam or 0.35),
                horizon_s=float(horizon or 10.0),
                min_duration_s=float(stab_dur or 2.0),
            )
            fresh = edt.seed_session(trials, trunk_params, stab_params)
            session["trunk_events"] = fresh["trunk_events"]
            session["detector"] = fresh["detector"]
            durations = [
                (e["windows"]["Novice"]["event_end"] - e["windows"]["Novice"]["event_start"]) / fs_of["Novice"]
                for e in session["trunk_events"]
            ]
            message = (
                f"Detected {len(durations)} trunk events, "
                f"{min(durations):.1f}-{max(durations):.1f} s "
                f"(mean {np.mean(durations):.1f} s)." if durations else "No trunk events found."
            )
            session_out = session
            if selection.get("family") == "trunk":
                selection = {**selection, "event_id": _first_id(session, "trunk")}

        event = find_event(session, selection)

        if event is not None and trigger == "btn-toggle":
            event["enabled"] = not event.get("enabled", True)
            session_out = session
            message = f"{event['event_id']} {'enabled' if event['enabled'] else 'disabled'}."

        elif event is not None and trigger == "btn-delete":
            key = "trunk_events" if selection.get("family") == "trunk" else "knee_events"
            session[key] = [e for e in session[key] if e["event_id"] != event["event_id"]]
            message = f"Deleted {event['event_id']}."
            session_out = session
            selection = {**selection, "event_id": _first_id(session, selection.get("family", "trunk"))}
            event = find_event(session, selection)

        elif event is not None and trigger in ("graph-Novice", "graph-Trained"):
            label = "Novice" if trigger == "graph-Novice" else "Trained"
            bands = decode_relayout(novice_relayout if label == "Novice" else trained_relayout)
            if not bands:
                raise PreventUpdate  # a pan or zoom, not an edit
            fs, n = fs_of[label], n_samples[label]
            changed = False
            for band, (x0, x1) in bands.items():
                start = edt.sec_to_idx(min(x0, x1), fs, n)
                end = edt.sec_to_idx(max(x0, x1), fs, n)
                changed |= set_boundaries(event, label, band, start, end)
            # Quantising to integer samples means a re-render that echoes the
            # same coordinates back produces no change and stops here, which is
            # what keeps dragging and typing from driving each other in a loop.
            if not changed:
                raise PreventUpdate
            session_out = session

        elif event is not None and isinstance(trigger, str) and trigger in {c for c, *_ in BOUNDARY_INPUTS}:
            lookup = {cid: (lbl, bnd, slt) for cid, lbl, bnd, slt in BOUNDARY_INPUTS}
            label, band, slot = lookup[trigger]
            window = window_for(event, label)
            value = values[[cid for cid, *_ in BOUNDARY_INPUTS].index(trigger)]
            if window is None or value is None:
                raise PreventUpdate
            fs, n = fs_of[label], n_samples[label]
            keys = ("event_start", "event_end") if band == "event" else ("stab_start", "stab_end")
            pair = [window[keys[0]], window[keys[1]]]
            pair[slot] = edt.sec_to_idx(value, fs, n)
            if not set_boundaries(event, label, band, min(pair), max(pair)):
                raise PreventUpdate
            session_out = session

        if session_out is not no_update:
            edt.save_session(edt.refresh_seconds(session, trials))

        # --- render -------------------------------------------------------
        patches = []
        for label in edt.TRIALS:
            patch = Patch()
            patch["layout"]["shapes"] = build_shapes(session, selection, label, fs_of[label])
            patches.append(patch)

        rows = event_table_rows(session, selection.get("family", "trunk"), trials)
        banner = render_issues(edt.validate_session(session, trials))
        title = describe_event(event, selection)

        out_values, out_disabled = [], []
        for _cid, label, band, slot in BOUNDARY_INPUTS:
            window = window_for(event, label)
            if window is None:
                out_values.append(None)
                out_disabled.append(True)
                continue
            keys = ("event_start", "event_end") if band == "event" else ("stab_start", "stab_end")
            out_values.append(round(window[keys[slot]] / fs_of[label], 2))
            out_disabled.append(False)

        return (session_out, *patches, rows, banner, title, message, *out_values, *out_disabled)

    @app.callback(
        Output("metrics-table", "data"),
        Output("metrics-table", "columns"),
        Output("figure-gallery", "children"),
        Output("store-figrev", "data"),
        Output("recalc-status", "children"),
        Input("btn-recalc", "n_clicks"),
        State("store-session", "data"),
        State("store-figrev", "data"),
        prevent_initial_call=True,
    )
    def recalculate(_clicks, session, figrev):
        try:
            with RECOMPUTE_LOCK:
                result = edt.recompute(session, trials, write=True)
        except Exception as error:  # surfaced in the UI instead of killing the callback
            return no_update, no_update, no_update, no_update, f"Recalculation failed:\n{error}"

        table = result.trunk_metrics.round(4)
        columns = [{"name": c, "id": c} for c in table.columns]
        figrev = (figrev or 0) + 1
        gallery = [
            html.Img(src=f"/outputs/{name}?v={figrev}", style={"width": "48%", "border": "1px solid #ddd"})
            for name in (
                "trunk_traceability_figure.png",
                "monopodal_stance_traceability_figure.png",
                "monopodal_stance_overview_figure.png",
            )
        ]
        message = "Wrote:\n  " + "\n  ".join(result.written) + "\n\n" + "\n".join(result.log)
        return table.to_dict("records"), columns, gallery, figrev, message

    @app.callback(Output("header-meta", "children"), Input("store-session", "data"))
    def header(session):
        parts = [
            f"fs {trials['Novice'].fs:.2f} Hz",
            f"Novice {trials['Novice'].duration_s:.0f} s / Trained {trials['Trained'].duration_s:.0f} s",
            f"high-pass {'disabled' if session.get('ignore_high_pass_filter') else 'enabled'}",
            f"session {edt.SESSION_PATH.name}",
        ]
        return " | ".join(parts)


def describe_event(event: dict | None, selection: dict) -> str:
    if event is None:
        return "No event selected."
    if "windows" in event:
        return f"{event['event_id']} - trunk rotation, paired across both trials."
    return f"{event['event_id']} - {event['trial']}, {event['flexed_leg']} knee flexed ({event['stance_leg']} leg stance)."


def render_issues(issues: list[edt.Issue]) -> list:
    if not issues:
        return [html.Span("No issues.", style={"color": "#2a9d8f"})]
    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]
    children = []
    for issue in errors[:6]:
        children.append(html.Div(f"ERROR  {issue.event_id} ({issue.trial}): {issue.message}",
                                 style={"color": "#c0392b"}))
    for issue in warnings[:6]:
        children.append(html.Div(f"warning  {issue.event_id} ({issue.trial}): {issue.message}",
                                 style={"color": "#b9770e"}))
    hidden = len(issues) - min(len(errors), 6) - min(len(warnings), 6)
    if hidden > 0:
        children.append(html.Div(f"... and {hidden} more", style={"color": "#888"}))
    return children


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8051, help="port to serve on (default 8051)")
    parser.add_argument("--recompute-orientation", action="store_true",
                        help="re-run the Madgwick filter instead of using outputs/orientation_*.npz")
    parser.add_argument("--reseed", choices=["auto", "v1"], default=None,
                        help="discard the saved session and reseed from the detectors or the committed v1 CSVs")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print("Loading recordings ...")
    trials = edt.load_all_trials(recompute_orientation=args.recompute_orientation)
    for label, loaded in trials.items():
        print(f"  {label}: {loaded.n_samples} samples at {loaded.fs:.4f} Hz ({loaded.duration_s:.1f} s)")

    session = None if args.reseed else edt.load_session()
    if session is None:
        if args.reseed == "v1":
            print("Seeding from the committed v1 window CSVs ...")
            session = edt.seed_session_from_v1_csvs(trials)
        else:
            print("Detecting events ...")
            session = edt.seed_session(trials)
        edt.save_session(edt.refresh_seconds(session, trials))
    print(f"  {len(session['trunk_events'])} trunk events, {len(session['knee_events'])} monopodal-stance events")

    app = build_app(trials, session)
    print(f"\nEditor running at http://127.0.0.1:{args.port}\n")
    app.run(debug=args.debug, port=args.port)


if __name__ == "__main__":
    main()
