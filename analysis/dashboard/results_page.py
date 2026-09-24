"""The Results page: the active selection's tables and figures, and where they came from.

Everything shown is read back from the files on disk -- the selection's folder
in outputs/analyses/ and its recordings' folders in outputs/recordings/ -- so
this page shows exactly what a colleague opening those folders would see.
"""

from __future__ import annotations

from urllib.parse import quote

import pandas as pd
from dash import Dash, Input, Output, dash_table, dcc, html
from dash.dash_table.Format import Format, Scheme
from dash.exceptions import PreventUpdate

from analysis import config, pipeline, recordings
from analysis.recompute import read_provenance

TABLES = {
    "trunk": ("Trunk rotation", "trunk_rotation_balance_metrics.csv"),
    "knee": ("Monopodal stance", "monopodal_stance_balance_metrics.csv"),
    "asymmetry": ("Stance asymmetry", "monopodal_stance_asymmetry_metrics.csv"),
    "smooth": ("Sequence segments", "sequence_smoothness_metrics.csv"),
    "variability": ("Sequence variability", "sequence_variability_summary.csv"),
    "trunk_windows": ("Trunk windows", "trunk_rotation_event_windows.csv"),
    "knee_windows": ("Stance windows", "monopodal_stance_event_windows.csv"),
    "inventory": ("Sensor inventory", None),
}
FIGURES = [
    ("trunk_traceability_figure.png", "Trunk rotation: windows and metrics"),
    ("monopodal_stance_traceability_figure.png", "Monopodal stance: windows and metrics"),
    ("monopodal_stance_overview_figure.png", "Monopodal stance: knee flexion and trunk response"),
    ("sequence_smoothness_figure.png", "Sequence segments: smoothness and consistency"),
]
NUMBER = Format(precision=4, scheme=Scheme.decimal_or_exponent)


def layout() -> html.Div:
    return html.Div(html.Div([
        html.H2("Results", style={"margin": "0 0 4px"}),
        html.Div(id="results-meta", style={"fontSize": "13px", "color": "#555", "marginBottom": "14px"}),
        dcc.RadioItems(
            id="results-table-choice", value="trunk", inline=True,
            options=[{"label": title, "value": key} for key, (title, _) in TABLES.items()],
            inputStyle={"marginRight": "4px"},
            labelStyle={"marginRight": "14px", "fontSize": "13px", "cursor": "pointer"},
        ),
        html.Div(id="results-table-file", style={"fontSize": "11px", "color": "#888", "margin": "6px 0"}),
        dash_table.DataTable(
            id="results-table", page_size=15, sort_action="native",
            style_table={"overflowX": "auto"},
            style_cell={"fontSize": "11px", "padding": "4px 6px", "fontFamily": "system-ui, sans-serif",
                        "whiteSpace": "nowrap"},
            style_header={"fontWeight": "bold", "backgroundColor": "#f0f0f0"},
        ),
        html.H3("Figures", style={"margin": "24px 0 8px"}),
        html.Div(id="results-figures", style={"display": "grid", "gap": "14px",
                                              "gridTemplateColumns": "repeat(auto-fill, minmax(440px, 1fr))"}),
    ], style={"maxWidth": "1400px", "margin": "0 auto", "padding": "20px 24px"}),
        style={"height": "100%", "overflowY": "auto"})


def _relative(path) -> str:
    return path.relative_to(config.ROOT).as_posix()


def _url(path) -> str:
    """Address of a file under outputs/, served by the app's /outputs route."""
    return "/outputs/" + quote(path.relative_to(config.OUTPUT_DIR).as_posix())


def _table(key: str) -> tuple[list[dict], list[dict], str]:
    title, name = TABLES[key]
    if name is None:  # the sensor inventory is kept per recording
        paths = [r.inventory_path for r in recordings.active().recordings().values() if r]
        present = [path for path in paths if path.exists()]
        if not present:
            return [], [], "No sensor inventory yet: preprocess the recordings first."
        frame = pd.concat([pd.read_csv(path) for path in present], ignore_index=True)
        source = " + ".join(_relative(path) for path in present)
    else:
        path = recordings.analysis_dir() / name
        if not path.exists():
            return [], [], f"{_relative(path)} has not been written yet."
        frame = pd.read_csv(path)
        source = _relative(path)
    columns = [
        {"name": column, "id": column, "type": "numeric", "format": NUMBER}
        if pd.api.types.is_numeric_dtype(frame[column]) else {"name": column, "id": column}
        for column in frame.columns
    ]
    return frame.to_dict("records"), columns, f"{source} · {len(frame)} rows"


def _figures() -> list:
    folder = recordings.analysis_dir()
    entries = [(folder / name, caption) for name, caption in FIGURES]
    entries += [(recording.validation_figure_path, f"Orientation validation: {role.lower()}, {recording.name}")
                for role, recording in recordings.active().recordings().items() if recording]
    cards = []
    for path, caption in entries:
        if not path.exists():
            continue
        # The modification time busts the browser cache exactly when the file changes.
        src = f"{_url(path)}?v={path.stat().st_mtime_ns}"
        cards.append(html.Figure([
            html.A(html.Img(src=src, style={"width": "100%", "display": "block"}), href=src, target="_blank"),
            html.Figcaption(caption, style={"fontSize": "12px", "color": "#555", "marginTop": "4px"}),
        ], style={"margin": 0, "border": "1px solid #e2e2e2", "borderRadius": "6px", "padding": "8px",
                  "backgroundColor": "white"}))
    return cards or [html.Div("No figures yet: run the pipeline first.", style={"color": "#777"})]


def _meta() -> list:
    selection = recordings.active()
    provenance = read_provenance()
    metrics = pipeline.status()["metrics"]
    if provenance is None:
        state = f"Metrics: {metrics.detail}."
    else:
        name = provenance.get("session_name") or "the working session"
        state = f"Computed from {name}. {metrics.detail[0].upper()}{metrics.detail[1:]}."
    return [html.Div(f"{selection.describe()}  →  {_relative(selection.output_dir)}/"), html.Div(state)]


def register_callbacks(app: Dash) -> None:
    @app.callback(
        Output("results-table", "data"),
        Output("results-table", "columns"),
        Output("results-table-file", "children"),
        Output("results-figures", "children"),
        Output("results-meta", "children"),
        Input("results-table-choice", "value"),
        Input("page-tabs", "value"),
        Input("store-revs", "data"),
    )
    def render(choice, page, _revs):
        if page != "results":
            raise PreventUpdate
        data, columns, file_note = _table(choice)
        return data, columns, file_note, _figures(), _meta()
