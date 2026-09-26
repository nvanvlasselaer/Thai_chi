"""The event editor page: check and correct the windows every metric is computed on.

Automatic detection gets most events roughly right and some badly wrong, and
every variability metric is computed *on* those windows, so a bad window
silently produces a bad number.  This page plots the kinematics on a
scrollable, zoomable timeline, lets the boundaries be dragged into place, and
recalculates the metrics through the pipeline's own functions.

Four boundaries per event are editable: the start and end of the movement, and
the start and end of the stabilization period that follows it.  Each subplot
draws the threshold its detector used, so a boundary can be judged against the
signal that produced it rather than taken on trust, and the selected event's
partner in the other recording is outlined where the whole-recording alignment
puts it, so a pair that is not the same movement shows at a glance.

The recordings come from the shared workspace (:mod:`analysis.dashboard.state`),
which the Pipeline page loads; until it has, the page shows a placeholder.  With
only one recording selected, only its graph is shown and every event has one
window.  Every edit is written to the session file straight away.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, Patch, State, ctx, dash_table, dcc, html, no_update
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots

from analysis import alignment, config, detection, recordings, sessions
from analysis.config import TRIALS
from analysis.dashboard.results_page import output_url
from analysis.dashboard.state import WORK_LOCK, workspace
from analysis.kinematics import LoadedTrial
from analysis.recompute import recompute
from analysis.signals import lowpass_signal, resample_filtered_full, sec_to_idx
from analysis.validation import MIN_PAIR_OVERLAP, Issue, pair_overlap, validate_session

DISPLAY_FS = 50.0
EVENT_COLOUR = "#f2b134"
STAB_COLOUR = "#57a773"
CONTEXT_COLOUR = "#9aa0a6"
ROW_TITLES = [
    "trunk-pelvis yaw (deg)",
    "yaw speed envelope (deg/s)  -  dotted: onset/offset floor",
    "lumbar + chest angular speed, gyro bias removed (deg/s)  -  dotted: quiet baseline, dashed: median",
    "leg lift (thigh lengths): + left foot up, - right foot up  -  dashed: minimum lift",
    "knee flexion |x| (deg)  -  dashed: legacy knee-threshold method",
    "chest and pelvis yaw (deg)  -  dashed: minimum turn from neutral",
]
N_ROWS = len(ROW_TITLES)
HIDDEN = {"display": "none"}
BODY_STYLE = {"display": "flex", "height": "100%"}
EMPTY_STYLE = {"padding": "40px"}
SHOWN = {"display": "block"}


def loaded_roles() -> list[str]:
    trials = workspace.trials or {}
    return [label for label in TRIALS if label in trials]


def graph_title(label: str) -> str:
    trials = workspace.trials or {}
    return f"{label}: {trials[label].recording.name}" if label in trials else label


def add_trial_options(labels: list[str]) -> tuple[list[str], str | None]:
    """Where a new event can be placed: both recordings, or one of them."""
    if len(labels) < 2:
        return labels, (labels[0] if labels else None)
    return ["Both", *labels], "Both"


# ---------------------------------------------------------------------------
# Display traces
#
# plotly 6 serialises numpy arrays as base64 typed arrays, which the plotly.js
# bundled with dash 2.14 cannot decode -- traces render blank with no error.
# Every array reaching a figure goes through _series().
# ---------------------------------------------------------------------------


def _series(values: np.ndarray) -> list[float]:
    return np.round(np.asarray(values, dtype=float), 4).tolist()


def build_display(loaded: LoadedTrial) -> dict:
    """Resample every plotted channel to 50 Hz once, at startup."""
    kin, fs = loaded.kin, loaded.fs

    def resample(signal: np.ndarray) -> np.ndarray:
        _, values = resample_filtered_full(kin.t, signal, fs, DISPLAY_FS)
        return values

    time_s, _ = resample_filtered_full(kin.t, kin.t, fs, DISPLAY_FS)

    trunk_diag = detection.trunk_yaw_envelope(kin, fs)
    stab_diag = detection.combined_omega(kin, fs)
    segment_diag = detection.chest_yaw_signals(kin, fs)
    lift_diag = detection.lift_signal(kin, fs)

    return {
        "time": _series(time_s),
        "chest_yaw": _series(resample(segment_diag.chest_yaw_deg)),
        "pelvis_yaw": _series(resample(segment_diag.pelvis_yaw_deg)),
        "yaw": _series(resample(trunk_diag.yaw_deg)),
        "envelope": _series(resample(trunk_diag.envelope_dps)),
        "envelope_floor": float(trunk_diag.floor_dps),
        "envelope_peak_height": float(trunk_diag.peak_height_dps),
        "lumbar_omega": _series(resample(kin.omega_mag["lumbar"])),
        "chest_omega": _series(resample(kin.omega_mag["chestbone"])),
        "combined_omega": _series(resample(stab_diag.combined_omega_dps)),
        "omega_baseline": float(stab_diag.baseline_dps),
        "omega_median": float(stab_diag.baseline_dps + stab_diag.scale_dps),
        "lift": _series(resample(lift_diag.lift_index)),
        "left_knee": _series(resample(np.abs(lowpass_signal(kin.left_knee_deg[:, 0], fs, cutoff_hz=6.0)))),
        "right_knee": _series(resample(np.abs(lowpass_signal(kin.right_knee_deg[:, 0], fs, cutoff_hz=6.0)))),
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
    add(display["lift"], "leg lift", 4, "#444444", width=1.2)
    add(display["left_knee"], "left knee", 5, "#d1495b", width=1.1)
    add(display["right_knee"], "right knee", 5, "#f4a261", width=1.1)
    add(display["chest_yaw"], "chest yaw", 6, "#1f77b4", width=1.3)
    add(display["pelvis_yaw"], "pelvis yaw", 6, "#8ecae6", width=1.0)

    figure.update_layout(
        height=1080,
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


def _threshold_line(y: float, row: int, colour: str, dash: str) -> dict:
    """A horizontal reference line across one subplot, at a detector threshold."""
    return {
        "type": "line",
        "xref": "paper", "x0": 0, "x1": 1,
        "yref": f"y{row}" if row > 1 else "y",
        "y0": y, "y1": y,
        "line": {"color": colour, "width": 1, "dash": dash},
        "editable": False,
        "layer": "below",
    }


def threshold_lines(display: dict, knee_threshold: float, min_lobe_deg: float,
                    lift_min_peak: float) -> list[dict]:
    """The thresholds the detectors used, so a boundary stays explainable."""
    return [
        _threshold_line(display["envelope_floor"], 2, "#7b4173", "dot"),
        _threshold_line(display["omega_baseline"], 3, "#264653", "dot"),
        _threshold_line(display["omega_median"], 3, "#264653", "dash"),
        _threshold_line(0.0, 4, "#888888", "dot"),
        _threshold_line(lift_min_peak, 4, "#444444", "dash"),
        _threshold_line(-lift_min_peak, 4, "#444444", "dash"),
        _threshold_line(knee_threshold, 5, "#444444", "dash"),
        _threshold_line(0.0, 6, "#888888", "dot"),
        _threshold_line(min_lobe_deg, 6, "#444444", "dash"),
        _threshold_line(-min_lobe_deg, 6, "#444444", "dash"),
    ]


def _partner_outline(x0: float, x1: float, same_movement: bool) -> dict:
    """Where the selected event's other window lands on this recording's clock."""
    colour = "#2a9d8f" if same_movement else "#c0392b"
    return {
        "type": "rect", "xref": DRIVING_XREF, "yref": "paper", "x0": x0, "x1": x1, "y0": 0, "y1": 1,
        "fillcolor": "rgba(0,0,0,0)", "line": {"width": 2, "color": colour, "dash": "dash"},
        "editable": False, "layer": "above",
    }


def partner_positions(event: dict | None, trials: dict[str, LoadedTrial]) -> dict[str, tuple[float, float, bool]]:
    """Per recording, the selected event's *other* window carried onto its clock.

    Returns ``{label: (x0, x1, same_movement)}`` for the graph each outline goes
    on; empty for an unpaired event or a single recording.
    """
    aligned = alignment.for_trials(trials)
    windows = (event or {}).get("windows", {})
    if aligned is None or not all(label in windows for label in TRIALS):
        return {}
    overlap = pair_overlap(event, trials)
    same = overlap is not None and overlap[0] >= MIN_PAIR_OVERLAP
    out = {}
    for here, other in (("Novice", "Trained"), ("Trained", "Novice")):
        window, fs = windows[other], trials[other].fs
        x0 = float(aligned.map(window["event_start"] / fs, other, here))
        x1 = float(aligned.map(window["event_end"] / fs, other, here))
        out[here] = (x0, x1, same)
    return out


def build_shapes(session: dict, selection: dict, label: str, fs: float,
                 display: dict, knee_threshold: float, min_lobe_deg: float, lift_min_peak: float,
                 partner: tuple[float, float, bool] | None = None) -> list[dict]:
    """Two editable rects for the selected event, then references and context."""
    selected = find_event(session, selection)
    shapes: list[dict] = []

    window = window_for(selected, label) if selected else None
    if window is not None:
        shapes.append(_rect(window["event_start"] / fs, window["event_end"] / fs, EVENT_COLOUR, True, 0.22))
    else:
        # Keep indices 0 and 1 occupied so a stale relayout cannot address a
        # context rect. Placed off-screen rather than omitted.
        shapes.append(_rect(-1, -1, EVENT_COLOUR, False, 0.0))
    # Sequence segments have no stabilization band, so index 1 is held by an
    # off-screen rect for them -- the fixed index contract decode_relayout
    # depends on holds whatever family is open.
    if window is not None and "stab_start" in window:
        shapes.append(_rect(window["stab_start"] / fs, window["stab_end"] / fs, STAB_COLOUR, True, 0.20))
    else:
        shapes.append(_rect(-1, -1, STAB_COLOUR, False, 0.0))

    # Appended after the editable pair so their indices 0 and 1 stay fixed.
    shapes += threshold_lines(display, knee_threshold, min_lobe_deg, lift_min_peak)
    if partner is not None:
        shapes.append(_partner_outline(*partner))

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
    return session.get(sessions.FAMILY_KEY.get(family, "trunk_events"), [])


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
    if start_key not in window:
        return False  # this family has no such band -- sequence segments have no stabilization
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


EVENT_DETAIL = {
    "trunk": lambda event: "trunk rotation",
    "knee": lambda event: f"{event.get('lifted_leg', '?')[:1]} leg up",
    "smooth": lambda event: f"turn {'+' if event.get('direction', 0) > 0 else '-'}",
}


def event_table_rows(session: dict, family: str, trials: dict[str, LoadedTrial],
                     low_confidence_z: float = 1.0) -> list[dict]:
    """One row per event, for whichever family the open tab shows.

    Every family may now be single-sided -- a stance found in one recording
    only, or a part of the form the detector found in one of them -- so the span
    of each trial is read defensively rather than indexed.
    """
    rows = []
    detail_of = EVENT_DETAIL.get(family, lambda event: family)
    for event in all_events(session, family):
        windows = event.get("windows", {})

        def span(label: str) -> str:
            window = windows.get(label)
            if window is None:
                return "--"
            fs = trials[label].fs
            return f"{window['event_start']/fs:.1f}-{window['event_end']/fs:.1f}"

        present = [label for label in trials if label in windows]
        ref = windows.get(present[0]) if present else None
        dur = (f"{(ref['event_end']-ref['event_start'])/trials[present[0]].fs:.1f}s"
               if ref else "--")
        overlap = pair_overlap(event, trials)
        rows.append({
            "event_id": event["event_id"],
            "detail": detail_of(event),
            "novice": span("Novice"),
            "trained": span("Trained"),
            "dur": dur,
            "overlap": f"{overlap[0]:.0%}" if overlap is not None else "--",
            "flags": event_flags(event, overlap, low_confidence_z),
        })
    return rows


def event_flags(event: dict, overlap: tuple[float, float] | None, low_confidence_z: float) -> str:
    """Short markers: edited, mismatched pair, low-confidence stabilization."""
    windows = event["windows"].values() if "windows" in event else [event]
    flags = []
    if any("manual" in (w.get("source") or {}).values() for w in windows):
        flags.append("edited")
    if overlap is not None and overlap[0] < MIN_PAIR_OVERLAP:
        flags.append("confirmed" if event.get("pair_confirmed") else "mismatch")
    scores = [w.get("stab_quiet_z") for w in windows]
    if any(z is not None and z > low_confidence_z for z in scores):
        flags.append("unsettled")
    if not event.get("enabled", True):
        flags.append("off")
    return ", ".join(flags)


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------


def layout() -> html.Div:
    """The editor page, filled from the workspace if the recordings are loaded.

    Built on every page load, so refreshing the browser always shows the
    session as it is saved on disk.
    """
    ready = workspace.ready
    displays = workspace.displays if ready else None
    session = sessions.load_session() if ready else None
    present = loaded_roles()
    figures = {label: base_figure(label, displays[label]) if label in present else {} for label in TRIALS}
    add_options, add_value = add_trial_options(present)

    def number(component_id: str, label_text: str):
        return html.Div([
            html.Label(label_text, style={"fontSize": "11px", "color": "#555"}),
            dcc.Input(id=component_id, type="number", step=0.01, debounce=True,
                      style={"width": "100%", "fontSize": "12px", "boxSizing": "border-box"}),
        ], style={"marginBottom": "4px"})

    def detector_input(component_id: str, label_text: str, value, step):
        return html.Div([
            html.Label(label_text, style={"fontSize": "11px", "color": "#555"}),
            dcc.Input(id=component_id, type="number", value=value, step=step, debounce=True,
                      style={"width": "100%", "fontSize": "12px", "boxSizing": "border-box"}),
        ], style={"marginBottom": "4px"})

    defaults = detection.TrunkDetectorParams()
    stab_defaults = detection.StabilizationParams()

    knee_defaults = detection.KneeDetectorParams()
    segment_defaults = detection.SegmentDetectorParams()

    row_style = {"display": "flex", "gap": "4px", "marginTop": "3px"}

    sidebar = html.Div([
        html.H4("Session", style={"margin": "4px 0"}),
        html.Div(id="session-current", style={"fontSize": "11px", "color": "#666"}),
        html.Div([
            dcc.Dropdown(id="session-select", options=[], placeholder="saved sessions",
                         style={"flex": 1, "fontSize": "11px"}),
            html.Button("Load", id="btn-load-session", n_clicks=0,
                        style={"fontSize": "11px", "whiteSpace": "nowrap"}),
        ], style=row_style),
        html.Div([
            dcc.Input(id="session-name", placeholder="save as…", debounce=True,
                      style={"flex": 1, "fontSize": "11px"}),
            html.Button("Save", id="btn-save-session", n_clicks=0,
                        style={"fontSize": "11px", "whiteSpace": "nowrap"}),
        ], style=row_style),
        html.Button("New session from detection", id="btn-new-session", n_clicks=0,
                    style={"width": "100%", "fontSize": "11px", "marginTop": "4px"}),
        html.Div("Editing is always saved to the working session; a named save keeps a copy. "
                 "Load and New park what they replace in '_previous'.",
                 style={"fontSize": "10px", "color": "#888", "margin": "3px 0 10px"}),

        html.H4(id="detector-title", style={"margin": "4px 0", "borderTop": "1px solid #eee",
                                            "paddingTop": "8px"}),

        # Only the panel for the active tab is shown, so the settings on screen
        # are always the ones the Re-detect button will use.
        html.Div(id="trunk-params", children=[
            detector_input("p-rel-thr", "onset threshold (fraction of peak)", defaults.rel_threshold, 0.01),
            detector_input("p-floor", "velocity floor (percentile)", defaults.floor_percentile, 1),
            detector_input("p-min-dur", "min event duration (s)", defaults.min_duration_s, 0.1),
            detector_input("p-min-exc", "min yaw excursion (deg)", defaults.min_excursion_deg, 1),
            detector_input("p-n-events", "number of events", defaults.n_events, 1),
        ]),
        html.Div(id="knee-params", children=[
            html.Label("method", style={"fontSize": "11px", "color": "#555"}),
            dcc.Dropdown(
                id="p-knee-method",
                options=[{"label": "Leg lift (one foot higher)", "value": "lift"},
                         {"label": "Knee flexion threshold (v1)", "value": "knee_threshold"}],
                value=knee_defaults.method, clearable=False,
                style={"fontSize": "11px", "marginBottom": "4px"},
            ),
            detector_input("p-lift-peak", "min lift (thigh lengths)", knee_defaults.lift_min_peak, 0.05),
            detector_input("p-lift-rel", "onset threshold (fraction of peak lift)",
                           knee_defaults.lift_rel_threshold, 0.01),
            detector_input("p-lift-dur", "min support duration (s)", knee_defaults.lift_min_duration_s, 0.1),
            html.Div("Knee-threshold method only:", style={"fontSize": "10px", "color": "#888", "marginTop": "4px"}),
            detector_input("p-knee-thr", "knee flexion threshold (deg)", knee_defaults.threshold_deg, 1),
            detector_input("p-knee-dur", "min event duration (s)", knee_defaults.min_duration_s, 0.1),
            detector_input("p-knee-gap", "merge gap (s)", knee_defaults.merge_gap_s, 0.05),
        ]),
        html.Div(id="smooth-params", children=[
            html.Div("Each turn runs from one turning point of the chest yaw to the next.",
                     style={"fontSize": "10px", "color": "#888", "marginBottom": "4px"}),
            detector_input("p-smooth-lobe", "min turn from neutral (deg)", segment_defaults.min_lobe_deg, 1),
            detector_input("p-smooth-sep", "min turning-point separation (s)",
                           segment_defaults.min_lobe_separation_s, 0.1),
            detector_input("p-smooth-exc", "min turn excursion (deg)", segment_defaults.min_excursion_deg, 1),
            detector_input("p-smooth-min-dur", "min turn duration (s)", segment_defaults.min_duration_s, 0.5),
            detector_input("p-smooth-max-dur", "max turn duration (s)", segment_defaults.max_duration_s, 1),
        ]),

        html.Div(id="stab-params", children=[
            html.Div("Stabilization — applies to trunk and stance events",
                     style={"fontSize": "10px", "color": "#888", "borderTop": "1px solid #eee",
                            "paddingTop": "6px", "marginTop": "6px"}),
            detector_input("p-lambda", "latency penalty (per s)", stab_defaults.latency_weight, 0.05),
            detector_input("p-horizon", "search horizon (s)", stab_defaults.horizon_s, 1),
            detector_input("p-stab-dur", "length (s)", stab_defaults.min_duration_s, 0.5),
        ]),

        html.Button(id="btn-redetect", n_clicks=0, style={"width": "100%", "marginTop": "6px"}),
        html.Div(id="redetect-hint",
                 style={"fontSize": "10px", "color": "#888", "marginBottom": "10px"}),

        html.Hr(),
        html.H4("Events", style={"margin": "4px 0"}),
        dash_table.DataTable(
            id="event-table",
            columns=[
                {"name": "id", "id": "event_id"},
                {"name": "what", "id": "detail"},
                {"name": "novice", "id": "novice"},
                {"name": "trained", "id": "trained"},
                {"name": "dur", "id": "dur"},
                {"name": "overlap", "id": "overlap"},
                {"name": "flags", "id": "flags"},
            ],
            row_selectable="single",
            selected_rows=[0],
            page_size=14,
            style_cell={"fontSize": "11px", "padding": "3px", "textAlign": "left"},
            style_table={"overflowX": "auto"},
            style_header={"fontWeight": "bold", "backgroundColor": "#f0f0f0"},
            style_data_conditional=[
                {"if": {"filter_query": '{flags} contains "unsettled"'}, "backgroundColor": "#fff4e0"},
                {"if": {"filter_query": '{flags} contains "mismatch"'}, "backgroundColor": "#fde2e0"},
                {"if": {"filter_query": '{flags} contains "off"'}, "color": "#aaa"},
            ],
        ),
        html.Div([
            html.Button("Toggle on/off", id="btn-toggle", n_clicks=0, style={"flex": 1, "fontSize": "11px"}),
            html.Button("Delete", id="btn-delete", n_clicks=0, style={"flex": 1, "fontSize": "11px"}),
        ], style={"display": "flex", "gap": "4px", "marginTop": "6px"}),
        html.Div([
            html.Button("Confirm pair", id="btn-confirm-pair", n_clicks=0,
                        title="Accept a pair the alignment says is not the same movement",
                        style={"flex": 1, "fontSize": "11px"}),
            html.Button("Swap leg", id="btn-swap-leg", n_clicks=0,
                        title="Single-leg stance: exchange the lifted and the stance leg",
                        style={"flex": 1, "fontSize": "11px"}),
        ], style={"display": "flex", "gap": "4px", "marginTop": "4px"}),

        html.Div([
            html.Strong("Add an event", style={"fontSize": "11px"}),
            html.Div("Placed at the centre of the visible time range, then drag to fit.",
                     style={"fontSize": "10px", "color": "#888", "margin": "2px 0 4px"}),
            html.Div([
                dcc.Dropdown(id="add-trial", options=add_options, value=add_value,
                             clearable=False, style={"flex": 1, "fontSize": "11px"}),
                dcc.Dropdown(id="add-leg", options=[{"label": "Left leg up", "value": "Left"},
                                                    {"label": "Right leg up", "value": "Right"}],
                             value="Left", clearable=False, style={"flex": 1, "fontSize": "11px"}),
            ], style={"display": "flex", "gap": "4px"}),
            html.Div(id="add-hint", style={"fontSize": "10px", "color": "#888", "margin": "2px 0"}),
            html.Button("+ Add event here", id="btn-add", n_clicks=0,
                        style={"width": "100%", "fontSize": "11px", "marginTop": "2px"}),
        ], style={"marginTop": "10px"}),

        html.Div([
            html.Strong("Deleted events", style={"fontSize": "11px"}),
            dcc.Dropdown(id="trash-select", options=[], placeholder="nothing deleted",
                         style={"fontSize": "11px", "marginTop": "2px"}),
            html.Button("Restore selected", id="btn-restore", n_clicks=0,
                        style={"width": "100%", "fontSize": "11px", "marginTop": "4px"}),
        ], style={"marginTop": "10px"}),

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
    ], style={"width": "330px", "padding": "10px", "overflowY": "auto", "height": "100%",
              "borderRight": "1px solid #ddd", "boxSizing": "border-box"})

    main = html.Div([
        html.Div(header_text(), id="header-meta",
                 style={"fontSize": "11px", "color": "#666", "padding": "6px 12px", "borderBottom": "1px solid #ddd"}),

        dcc.Tabs(id="family-tabs", value="trunk", children=[
            dcc.Tab(label="Trunk rotation", value="trunk"),
            dcc.Tab(label="Single-leg stance", value="knee"),
            dcc.Tab(label="Sequence turns", value="smooth"),
        ]),

        html.Div(id="validation", style={"fontSize": "11px", "padding": "6px 12px"}),

        html.Div([
            # Both graphs are always in the page, since the callbacks name them;
            # a role with no recording selected is hidden.
            *[html.Div([
                html.Div(graph_title(label), id=f"graph-title-{label}",
                         style={"fontWeight": "bold", "fontSize": "12px", "padding": "4px 12px"}),
                dcc.Graph(id=f"graph-{label}", figure=figures[label], config=GRAPH_CONFIG),
            ], id=f"graph-section-{label}", style=SHOWN if label in present else HIDDEN) for label in TRIALS],
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
    ], style={"flex": 1, "overflowY": "auto", "height": "100%", "boxSizing": "border-box"})

    empty = html.Div([
        html.H3("No recordings loaded yet", style={"margin": "0 0 8px"}),
        html.P("The editor needs both recordings loaded. When the orientation cache exists they "
               "load by themselves a few seconds after the server starts; otherwise run the "
               "pipeline first.", style={"maxWidth": "560px", "color": "#555", "fontSize": "13px"}),
        html.Button("Go to the Pipeline page", id="btn-goto-pipeline", n_clicks=0,
                    style={"padding": "6px 12px", "fontSize": "13px"}),
    ], id="editor-empty", style=HIDDEN if ready else EMPTY_STYLE)

    return html.Div([
        dcc.Store(id="store-session", data=session),
        dcc.Store(id="store-selection", data={"family": "trunk", "event_id": _first_id(session, "trunk")}),
        dcc.Store(id="store-figrev", data=0),
        dcc.Store(id="store-view", data={}),
        # The workspace revisions this page last loaded, so a finished job's
        # changes are picked up once and only once.
        dcc.Store(id="store-editor-revs", data=workspace.revision()),
        empty,
        html.Div([sidebar, main], id="editor-body", style=BODY_STYLE if ready else HIDDEN),
    ], style={"height": "100%"})


GRAPH_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "doubleClick": "reset",
    "edits": {"shapePosition": True},
    "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
}


def _first_id(session: dict | None, family: str) -> str | None:
    events = all_events(session, family) if session else []
    return events[0]["event_id"] if events else None


def register_callbacks(app: Dash) -> None:
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
        if session is None:
            raise PreventUpdate
        if ctx.triggered_id == "family-tabs":
            return {"family": family, "event_id": _first_id(session, family)}
        events = all_events(session, family)
        if not selected_rows or selected_rows[0] >= len(events):
            raise PreventUpdate
        candidate = {"family": family, "event_id": events[selected_rows[0]]["event_id"]}
        if candidate == selection:
            raise PreventUpdate
        return candidate

    DETECTOR_PANEL = {
        "trunk": ("Detector — trunk rotation", "Re-detect trunk events",
                  "Replaces all trunk-rotation events and discards their edits."),
        "knee": ("Detector — single-leg stance", "Re-detect stance events",
                 "Replaces all single-leg-stance events and discards their edits."),
        "smooth": ("Detector — sequence turns", "Re-detect turns",
                   "Replaces all sequence turns and discards their edits."),
    }

    @app.callback(
        Output("trunk-params", "style"),
        Output("knee-params", "style"),
        Output("smooth-params", "style"),
        Output("stab-params", "style"),
        Output("detector-title", "children"),
        Output("btn-redetect", "children"),
        Output("redetect-hint", "children"),
        Input("family-tabs", "value"),
    )
    def switch_detector_panel(family):
        shown, hidden = {"display": "block"}, {"display": "none"}
        # The panels all stay in the DOM so their States are always resolvable;
        # only which one is visible changes.  Sequence segments have no
        # stabilization window, so that block hides with them.
        styles = [shown if family == key else hidden for key in ("trunk", "knee", "smooth")]
        styles.append(hidden if family == "smooth" else shown)
        return (*styles, *DETECTOR_PANEL.get(family, DETECTOR_PANEL["trunk"]))

    # The visible time range per graph, so a new event can be placed where the
    # user is looking.  make_subplots links the rows to the bottom axis, so zoom
    # and pan are reported as xaxis{N}.range[*] on the driving axis.
    @app.callback(
        Output("store-view", "data"),
        Input("graph-Novice", "relayoutData"),
        Input("graph-Trained", "relayoutData"),
        State("store-view", "data"),
        prevent_initial_call=True,
    )
    def track_view(novice_relayout, trained_relayout, view):
        trigger = ctx.triggered_id
        relayout = novice_relayout if trigger == "graph-Novice" else trained_relayout
        if not relayout:
            raise PreventUpdate
        label = "Novice" if trigger == "graph-Novice" else "Trained"
        axis = f"xaxis{N_ROWS}"
        lo, hi = relayout.get(f"{axis}.range[0]"), relayout.get(f"{axis}.range[1]")
        if lo is None or hi is None:
            if not relayout.get(f"{axis}.autorange"):
                raise PreventUpdate
            lo = hi = None  # reset to full view
        view = dict(view or {})
        view[label] = None if lo is None else [float(lo), float(hi)]
        return view

    def view_centre(view: dict, label: str) -> float:
        """Mid-point of the visible range, or of the whole recording."""
        span = (view or {}).get(label)
        if not span:
            return workspace.trials[label].duration_s / 2
        return 0.5 * (span[0] + span[1])

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
        Output("trash-select", "options"),
        Output("trash-select", "value"),
        Output("add-hint", "children"),
        Output("add-trial", "disabled"),
        Output("add-leg", "disabled"),
        Output("session-select", "options"),
        Output("session-current", "children"),
        Output("session-name", "value"),
        *[Output(component_id, "value") for component_id, *_ in BOUNDARY_INPUTS],
        *[Output(component_id, "disabled") for component_id, *_ in BOUNDARY_INPUTS],
        Input("graph-Novice", "relayoutData"),
        Input("graph-Trained", "relayoutData"),
        *[Input(component_id, "value") for component_id, *_ in BOUNDARY_INPUTS],
        Input("btn-redetect", "n_clicks"),
        Input("btn-toggle", "n_clicks"),
        Input("btn-delete", "n_clicks"),
        Input("btn-confirm-pair", "n_clicks"),
        Input("btn-swap-leg", "n_clicks"),
        Input("btn-add", "n_clicks"),
        Input("btn-restore", "n_clicks"),
        Input("btn-save-session", "n_clicks"),
        Input("btn-load-session", "n_clicks"),
        Input("btn-new-session", "n_clicks"),
        Input("store-selection", "data"),
        Input("store-editor-revs", "data"),
        State("store-session", "data"),
        State("p-rel-thr", "value"), State("p-floor", "value"),
        State("p-min-dur", "value"), State("p-min-exc", "value"),
        State("p-n-events", "value"), State("p-lambda", "value"),
        State("p-horizon", "value"), State("p-stab-dur", "value"),
        State("p-knee-method", "value"), State("p-lift-peak", "value"),
        State("p-lift-rel", "value"), State("p-lift-dur", "value"),
        State("p-knee-thr", "value"), State("p-knee-dur", "value"), State("p-knee-gap", "value"),
        State("p-smooth-lobe", "value"),
        State("p-smooth-sep", "value"), State("p-smooth-exc", "value"),
        State("p-smooth-min-dur", "value"), State("p-smooth-max-dur", "value"),
        State("store-view", "data"),
        State("add-trial", "value"), State("add-leg", "value"),
        State("trash-select", "value"),
        State("session-name", "value"), State("session-select", "value"),
    )
    def edit_and_render(novice_relayout, trained_relayout, *args):
        values = list(args[: len(BOUNDARY_INPUTS)])
        rest = args[len(BOUNDARY_INPUTS):]
        (_redetect, _toggle, _delete, _confirm, _swap, _add, _restore, _save_session, _load_session,
         _new_session, selection, _editor_revs, session,
         rel_thr, floor_pct, min_dur, min_exc, n_events, lam, horizon, stab_dur,
         knee_method, lift_peak, lift_rel, lift_dur, knee_thr, knee_dur, knee_gap,
         smooth_lobe, smooth_sep, smooth_exc, smooth_min_dur, smooth_max_dur,
         view, add_trial, add_leg, trash_index, session_name, session_choice) = rest

        trials, displays = workspace.trials, workspace.displays
        if trials is None or session is None:
            raise PreventUpdate
        fs_of = {label: loaded.fs for label, loaded in trials.items()}
        n_samples = {label: loaded.n_samples for label, loaded in trials.items()}

        trigger = ctx.triggered_id
        session = dict(session)
        message = no_update
        session_out: object = no_update
        family = selection.get("family", "trunk")

        clear_name = no_update

        def current_detector_params():
            """The settings shown in the panels, which every detector acts on."""
            return (
                detection.TrunkDetectorParams(
                    n_events=int(n_events or 6),
                    rel_threshold=float(rel_thr or 0.15),
                    floor_percentile=float(floor_pct or 40),
                    min_duration_s=float(min_dur or 1.5),
                    min_excursion_deg=float(min_exc or 8.0),
                ),
                detection.StabilizationParams(
                    latency_weight=float(lam or 0.65),
                    horizon_s=float(horizon or 10.0),
                    min_duration_s=float(stab_dur or 3.0),
                ),
                detection.KneeDetectorParams(
                    method=knee_method or "lift",
                    lift_min_peak=float(lift_peak or 0.25),
                    lift_rel_threshold=float(lift_rel or 0.15),
                    lift_min_duration_s=float(lift_dur or 0.5),
                    threshold_deg=float(knee_thr or 60.0),
                    min_duration_s=float(knee_dur or 0.4),
                    merge_gap_s=float(knee_gap or 0.2),
                ),
                detection.SegmentDetectorParams(
                    min_lobe_deg=float(smooth_lobe or 10.0),
                    min_lobe_separation_s=float(smooth_sep or 1.5),
                    min_excursion_deg=float(smooth_exc or 10.0),
                    min_duration_s=float(smooth_min_dur or 1.0),
                    max_duration_s=float(smooth_max_dur or 15.0),
                ),
            )

        if trigger == "btn-new-session":
            session = sessions.start_new_session(trials, *current_detector_params())
            session_out = session
            family = selection.get("family", "trunk")
            selection = {"family": family, "event_id": _first_id(session, family)}
            message = (
                f"New session: {len(session['trunk_events'])} trunk, "
                f"{len(session['knee_events'])} monopodal-stance and "
                f"{len(session['smooth_events'])} sequence segments detected at the current "
                f"settings. The session it replaced is in '{sessions.PREVIOUS_SLOT}' — load that to undo."
            )

        elif trigger == "btn-save-session":
            try:
                path = sessions.save_session_as(session, session_name)
            except ValueError as error:
                message = str(error)
            else:
                session["session_name"] = path.stem
                session_out = session
                clear_name = ""
                message = f"Saved session as '{path.stem}'."

        elif trigger == "btn-load-session":
            if not session_choice:
                message = "Choose a saved session to load."
            else:
                try:
                    session = sessions.load_session_named(session_choice, trials)
                except (FileNotFoundError, ValueError) as error:
                    message = str(error)
                else:
                    session_out = session
                    family = selection.get("family", "trunk")
                    selection = {"family": family, "event_id": _first_id(session, family)}
                    loaded_name = session.get("session_name", session_choice)
                    counts = (f"{len(session.get('trunk_events', []))} trunk, "
                              f"{len(session.get('knee_events', []))} monopodal-stance, "
                              f"{len(session.get('smooth_events', []))} sequence segments")
                    message = (
                        f"Loaded '{loaded_name}': {counts}. "
                        + (f"'{sessions.PREVIOUS_SLOT}' now holds what this replaced, so loading it "
                           "again swaps the two back."
                           if loaded_name == sessions.PREVIOUS_SLOT else
                           f"The session it replaced is in '{sessions.PREVIOUS_SLOT}' — load that to undo.")
                    )

        if trigger == "btn-redetect":
            trunk_params, stab_params, knee_params, segment_params = current_detector_params()
            # Re-detect only the family whose tab is open, so the other families'
            # curated windows are never replaced by a click meant for this one.
            if family == "trunk":
                session["trunk_events"] = sessions.seed_trunk_events(trials, trunk_params, stab_params)
                noun, prefixes = "trunk-rotation", ("trunk_", "stab_")
            elif family == "knee":
                session["knee_events"] = sessions.seed_knee_events(trials, stab_params, knee_params)
                noun, prefixes = "single-leg-stance", ("knee_", "stab_")
            else:
                session["smooth_events"] = sessions.seed_smooth_events(trials, segment_params)
                noun, prefixes = "sequence-turn", ("smooth_",)
            spans = [
                (w["event_end"] - w["event_start"]) / fs_of[label]
                for e in all_events(session, family) for label, w in e["windows"].items()
            ]

            # Only the settings the re-detected family was detected with are
            # recorded; the others still describe the events they produced.
            everything = detection.params_to_dict(trunk_params, stab_params, knee_params, segment_params)
            session["detector"] = {
                **(session.get("detector") or {}),
                **{key: value for key, value in everything.items() if key.startswith(prefixes)},
            }
            count = len(all_events(session, family))
            message = (
                f"Detected {count} {noun} events, windows "
                f"{min(spans):.1f}-{max(spans):.1f} s (mean {np.mean(spans):.1f} s)."
                if spans else f"No {noun} events found at these settings."
            )
            session_out = session
            selection = {**selection, "event_id": _first_id(session, family)}

        if trigger == "btn-add":
            stab_len = float(stab_dur or 3.0)
            centres = {label: view_centre(view, label) for label in trials}
            if family == "trunk":
                new = sessions.add_trunk_event(session, trials, centres, stab_len_s=stab_len)
            elif family == "knee":
                new = sessions.add_knee_event(
                    session, trials, add_trial or "Both", add_leg or "Left",
                    centres, stab_len_s=stab_len,
                )
            else:
                new = sessions.add_smooth_event(session, trials, add_trial or "Both", centres)
            where = ", ".join(f"{label} {centres[label]:.1f}s" for label in sorted(new["windows"]))
            session_out = session
            selection = {**selection, "event_id": new["event_id"]}
            message = f"Added {new['event_id']} at {where}. Drag its edges to fit the movement."

        elif trigger == "btn-restore":
            entry = (sessions.trash_of(session)[trash_index]
                     if trash_index is not None and 0 <= trash_index < len(sessions.trash_of(session))
                     else None)
            # Read the family from the trash entry: every family now stores its
            # windows the same way, so the shape of the event cannot say which
            # one it came from.
            restored_family = entry.get("family") if entry else None
            restored = sessions.restore_event(session, trash_index, trials) if entry else None
            if restored is None:
                message = "Select a deleted event to restore."
            else:
                session_out = session
                family = restored_family or family
                selection = {"family": family, "event_id": restored["event_id"]}
                message = f"Restored {restored['event_id']}."

        event = find_event(session, selection)

        if event is not None and trigger == "btn-toggle":
            event["enabled"] = not event.get("enabled", True)
            session_out = session
            message = f"{event['event_id']} {'enabled' if event['enabled'] else 'disabled'}."

        elif event is not None and trigger == "btn-confirm-pair":
            event["pair_confirmed"] = not event.get("pair_confirmed", False)
            session_out = session
            message = (f"{event['event_id']}: pairing confirmed by hand." if event["pair_confirmed"]
                       else f"{event['event_id']}: pairing back under the alignment check.")

        elif event is not None and trigger == "btn-swap-leg":
            if family != "knee":
                message = "Swap leg applies to single-leg-stance events."
            else:
                lifted = "Right" if event.get("lifted_leg") == "Left" else "Left"
                event["lifted_leg"], event["stance_leg"] = lifted, ("Right" if lifted == "Left" else "Left")
                session_out = session
                message = f"{event['event_id']}: {lifted.lower()} leg lifted."

        elif event is not None and trigger == "btn-delete":
            sessions.delete_event(session, family, event["event_id"])
            message = f"Deleted {event['event_id']}. Restore it from the Deleted events list."
            session_out = session
            selection = {**selection, "event_id": _first_id(session, family)}
            event = find_event(session, selection)

        elif event is not None and trigger in ("graph-Novice", "graph-Trained"):
            label = "Novice" if trigger == "graph-Novice" else "Trained"
            bands = decode_relayout(novice_relayout if label == "Novice" else trained_relayout)
            if not bands:
                raise PreventUpdate  # a pan or zoom, not an edit
            fs, n = fs_of[label], n_samples[label]
            changed = False
            for band, (x0, x1) in bands.items():
                start = sec_to_idx(min(x0, x1), fs, n)
                end = sec_to_idx(max(x0, x1), fs, n)
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
            pair[slot] = sec_to_idx(value, fs, n)
            if not set_boundaries(event, label, band, min(pair), max(pair)):
                raise PreventUpdate
            session_out = session

        if session_out is not no_update:
            sessions.save_session(sessions.refresh_seconds(session, trials))

        # --- render -------------------------------------------------------
        _, shown_stab, shown_knee, shown_segment = detection.params_from_dict(session.get("detector") or {})
        partners = partner_positions(event, trials)
        patches = []
        for label in TRIALS:
            if label not in trials:
                patches.append(no_update)  # no recording in this role: its graph is hidden
                continue
            patch = Patch()
            patch["layout"]["shapes"] = build_shapes(
                session, selection, label, fs_of[label], displays[label],
                shown_knee.threshold_deg, shown_segment.min_lobe_deg, shown_knee.lift_min_peak,
                partner=partners.get(label),
            )
            patches.append(patch)

        rows = event_table_rows(session, selection.get("family", "trunk"), trials,
                                low_confidence_z=shown_stab.low_confidence_z)
        banner = render_issues(validate_session(session, trials))
        title = describe_event(event, selection)

        trash_options = [
            {"label": sessions.describe_trashed(entry, trials), "value": i}
            for i, entry in enumerate(sessions.trash_of(session))
        ]
        trash_value = None if trigger == "btn-restore" else no_update

        open_family = selection.get("family", "trunk")
        # A trunk event always exists in both recordings, so its trial dropdown
        # is fixed; the leg dropdown only means anything for a stance.
        lock_trial = open_family == "trunk"
        lock_leg = open_family != "knee"
        add_hint = (
            "A trunk event is created in both recordings at once." if lock_trial
            else "Both = a paired event; pick one recording for a single-sided one."
        )

        out_values, out_disabled = [], []
        for _cid, label, band, slot in BOUNDARY_INPUTS:
            window = window_for(event, label)
            keys = ("event_start", "event_end") if band == "event" else ("stab_start", "stab_end")
            # A family with no stabilization band greys out its two boxes rather
            # than showing boundaries that do not exist.
            if window is None or keys[slot] not in window:
                out_values.append(None)
                out_disabled.append(True)
                continue
            out_values.append(round(window[keys[slot]] / fs_of[label], 2))
            out_disabled.append(False)

        current_name = session.get("session_name")
        current_label = (
            f"editing: {current_name}" if current_name
            else "editing: working session (unnamed)"
        )
        session_options = [{"label": n, "value": n} for n in sessions.list_sessions()]

        return (session_out, *patches, rows, banner, title, message,
                trash_options, trash_value, add_hint, lock_trial, lock_leg,
                session_options, current_label, clear_name,
                *out_values, *out_disabled)

    @app.callback(
        Output("metrics-table", "data"),
        Output("metrics-table", "columns"),
        Output("figure-gallery", "children"),
        Output("store-figrev", "data"),
        Output("recalc-status", "children"),
        Input("btn-recalc", "n_clicks"),
        State("store-session", "data"),
        State("store-figrev", "data"),
        State("family-tabs", "value"),
        prevent_initial_call=True,
    )
    def recalculate(_clicks, session, figrev, family):
        trials = workspace.trials
        if trials is None or session is None:
            raise PreventUpdate
        # A pipeline job holds the lock while it writes outputs/; wait for it
        # rather than interleave two sets of CSVs.
        if not WORK_LOCK.acquire(blocking=False):
            return (no_update, no_update, no_update, no_update,
                    "A pipeline job is running. Recalculate once it has finished.")
        try:
            result = recompute(session, trials, write=True)
        except Exception as error:  # surfaced in the UI instead of killing the callback
            return no_update, no_update, no_update, no_update, f"Recalculation failed:\n{error}"
        finally:
            WORK_LOCK.release()
        workspace.bump("results")

        # Every family is recalculated, but the table shows the one whose tab is
        # open -- the numbers you were just curating windows for.
        frames = {
            "trunk": result.trunk_metrics,
            "knee": result.knee_metrics,
            "smooth": result.smooth_metrics,
        }
        table = frames.get(family, result.trunk_metrics).round(4)
        columns = [{"name": c, "id": c} for c in table.columns]
        figrev = (figrev or 0) + 1
        gallery = [
            html.Img(src=f"{output_url(recordings.analysis_dir() / name)}?v={figrev}",
                     style={"width": "48%", "border": "1px solid #ddd"})
            for name in (
                "trunk_traceability_figure.png",
                "monopodal_stance_traceability_figure.png",
                "monopodal_stance_overview_figure.png",
                "sequence_smoothness_figure.png",
            )
        ]
        message = "Wrote:\n  " + "\n  ".join(result.written) + "\n\n" + "\n".join(result.log)
        return table.to_dict("records"), columns, gallery, figrev, message

    @app.callback(Output("header-meta", "children"), Input("store-session", "data"))
    def header(_session):
        return header_text()


    @app.callback(
        Output("graph-Novice", "figure", allow_duplicate=True),
        Output("graph-Trained", "figure", allow_duplicate=True),
        Output("store-session", "data", allow_duplicate=True),
        Output("store-selection", "data", allow_duplicate=True),
        Output("store-editor-revs", "data"),
        Output("editor-empty", "style"),
        Output("editor-body", "style"),
        *[Output(f"graph-section-{label}", "style") for label in TRIALS],
        *[Output(f"graph-title-{label}", "children") for label in TRIALS],
        Output("add-trial", "options"),
        Output("add-trial", "value"),
        Input("store-revs", "data"),
        State("store-editor-revs", "data"),
        State("store-selection", "data"),
        prevent_initial_call=True,
    )
    def sync_with_server(revs, seen, selection):
        """Reload whatever a finished job changed: the recordings, the session, or both.

        The session is read back from disk, which is safe because every edit
        made on this page is written there as it happens.  When a different
        selection has been made and not loaded yet, the page goes back to its
        placeholder.
        """
        if not revs:
            raise PreventUpdate
        seen = seen or {}
        data_changed = revs.get("data") != seen.get("data")
        if not data_changed and revs.get("session") == seen.get("session"):
            raise PreventUpdate
        family = (selection or {}).get("family", "trunk")

        present = loaded_roles()
        sections = [SHOWN if label in present else HIDDEN for label in TRIALS]
        titles = [graph_title(label) for label in TRIALS]
        add_options, add_value = add_trial_options(present)
        if not workspace.ready:
            return ({}, {}, None, {"family": family, "event_id": None}, revs, EMPTY_STYLE, HIDDEN,
                    *sections, *titles, add_options, add_value)

        displays = workspace.displays
        figures = ([base_figure(label, displays[label]) if label in present else {} for label in TRIALS]
                   if data_changed else [no_update, no_update])
        session = sessions.load_session()
        kept = (selection or {}).get("event_id")
        if not any(event["event_id"] == kept for event in all_events(session, family)):
            kept = _first_id(session, family)
        # store-editor-revs is also an input of edit_and_render, so setting it
        # re-renders the bands, table and boundary boxes for the reloaded session.
        return (*figures, session, {"family": family, "event_id": kept}, revs,
                HIDDEN, BODY_STYLE, *sections, *titles, add_options, add_value)


def header_text() -> str:
    trials = workspace.trials
    if trials is None:
        return ""
    first = next(iter(trials.values()))
    parts = [
        f"fs {first.fs:.2f} Hz",
        " / ".join(f"{label} {loaded.duration_s:.0f} s" for label, loaded in trials.items()),
        # Read the live module flag, not the stored one: the header must
        # describe the code about to run, not the code that seeded the file.
        f"high-pass {'disabled' if config.IGNORE_HIGH_PASS_FILTER else 'enabled'}",
        f"session outputs/analyses/{recordings.active().id}/{sessions.SESSION_FILENAME}",
    ]
    return " | ".join(parts)


def describe_event(event: dict | None, selection: dict) -> str:
    if event is None:
        return "No event selected."
    present = sorted(event.get("windows", {}))
    where = " + ".join(present) if len(present) > 1 else (present[0] if present else "no recording")
    family = selection.get("family", "trunk")
    if family == "trunk":
        return f"{event['event_id']} - trunk rotation  [{where}]"
    if family == "smooth":
        return f"{event['event_id']} - turn  [{where}]"
    confirmed = "  pairing confirmed by hand" if event.get("pair_confirmed") else ""
    return (f"{event['event_id']} - {event.get('lifted_leg', '?')} leg lifted "
            f"({event.get('stance_leg', '?')} leg stance)  [{where}]{confirmed}")


def render_issues(issues: list[Issue]) -> list:
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
