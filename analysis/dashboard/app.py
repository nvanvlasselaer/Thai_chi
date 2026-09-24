"""The Tai Chi balance analysis dashboard: run the pipeline, curate the windows, read the results.

Start it from the repository root and open http://127.0.0.1:8051::

    python3 app.py [--port 8051] [--no-browser] [--debug]

Three pages:

    Pipeline      what has been computed, what is out of date, and buttons to run it
    Event editor  check and correct the windows every metric is computed on
    Results       the metric tables and figures in outputs/

Long work runs as a background job (:mod:`analysis.dashboard.tasks`), and every
page polls for its progress once a second.  When the selected recordings are
already preprocessed, they are loaded as soon as the server starts.
"""

from __future__ import annotations

import argparse
import threading
import webbrowser

import flask
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update

from analysis import config, pipeline, recordings
from analysis.dashboard import editor_page, pipeline_page, results_page, tasks
from analysis.dashboard.state import jobs, workspace

HEADER_HEIGHT_PX = 48
PAGES = {
    "pipeline": ("Pipeline", pipeline_page),
    "editor": ("Event editor", editor_page),
    "results": ("Results", results_page),
}
JUMP_BUTTONS = {"btn-goto-pipeline": "pipeline", "btn-open-editor": "editor", "btn-open-results": "results"}
TAB_STYLE = {"padding": "6px 16px", "fontSize": "13px", "border": "none", "backgroundColor": "transparent",
             "color": "#555", "lineHeight": "22px", "width": "auto", "whiteSpace": "nowrap"}
TAB_SELECTED_STYLE = {**TAB_STYLE, "color": "#1d3557", "fontWeight": "600",
                      "borderBottom": "3px solid #2a9d8f", "backgroundColor": "transparent"}


def serve_layout() -> html.Div:
    """The whole app.  Called on every page load, so each browser tab starts from what is on disk."""
    header = html.Div([
        html.Div("Tai Chi balance analysis",
                 style={"fontWeight": "600", "fontSize": "15px", "color": "#1d3557", "whiteSpace": "nowrap"}),
        dcc.Tabs(
            id="page-tabs", value="pipeline",
            children=[dcc.Tab(label=title, value=key, style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE)
                      for key, (title, _) in PAGES.items()],
            style={"height": f"{HEADER_HEIGHT_PX - 12}px"}, parent_style={"flex": "none"},
        ),
        html.Div(id="job-banner", style={"fontSize": "12px", "color": "#555", "marginLeft": "auto",
                                         "whiteSpace": "nowrap", "overflow": "hidden",
                                         "textOverflow": "ellipsis", "minWidth": 0}),
    ], style={"display": "flex", "alignItems": "center", "gap": "24px", "padding": "0 16px",
              "height": f"{HEADER_HEIGHT_PX}px", "borderBottom": "1px solid #ddd",
              "boxSizing": "border-box", "backgroundColor": "#fafafa"})

    # Every page stays in the DOM and is only shown or hidden, so the callbacks
    # of a page that is not on screen still find their components.
    pages = [
        html.Div(module.layout(), id=f"page-{key}",
                 style={"display": "block" if key == "pipeline" else "none", "height": "100%"})
        for key, (_, module) in PAGES.items()
    ]
    return html.Div([
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="store-revs", data=workspace.revision()),
        dcc.Interval(id="poll", interval=1000),
        header,
        html.Div(pages, style={"height": f"calc(100vh - {HEADER_HEIGHT_PX}px)"}),
    ], style={"fontFamily": "system-ui, -apple-system, sans-serif", "margin": 0, "color": "#222"})


def job_banner(snapshot: dict) -> list:
    """One line in the header saying what the background job is doing."""
    if snapshot["name"] is None:
        return []
    if snapshot["running"]:
        percent = "" if snapshot["fraction"] is None else f" {snapshot['fraction']:.0%} ·"
        return [html.Span("● ", style={"color": "#2a9d8f"}),
                f"{snapshot['name']}:{percent} {snapshot['message']}"]
    if snapshot["error"]:
        return [html.Span("✕ ", style={"color": "#c0392b"}),
                f"{snapshot['name']} failed: {snapshot['error']}"]
    return [html.Span("✔ ", style={"color": "#2a9d8f"}),
            f"{snapshot['name']} finished in {snapshot['elapsed_s']:.0f} s"]


def register_callbacks(app: Dash) -> None:
    @app.callback(
        *[Output(f"page-{key}", "style") for key in PAGES],
        Input("page-tabs", "value"),
    )
    def show_page(page):
        return [{"display": "block" if key == page else "none", "height": "100%"} for key in PAGES]

    # Each page has its own address (/pipeline, /editor, /results), so a page
    # can be bookmarked and a refresh stays on it.  The tab and the address are
    # kept in step by one callback, since Dash forbids a cycle across two.
    @app.callback(
        Output("page-tabs", "value"),
        Output("url", "pathname"),
        Input("url", "pathname"),
        Input("page-tabs", "value"),
        Input("btn-goto-pipeline", "n_clicks"),
        Input("btn-open-editor", "n_clicks"),
        Input("btn-open-results", "n_clicks"),
    )
    def navigate(pathname, tab, *_clicks):
        trigger = ctx.triggered_id
        if trigger == "page-tabs":
            page = tab
        elif trigger in JUMP_BUTTONS:
            page = JUMP_BUTTONS[trigger]
        else:  # page load, or the address changed
            page = (pathname or "/").strip("/") or "pipeline"
            if page not in PAGES:
                page = "pipeline"
        return (page if page != tab else no_update), (f"/{page}" if pathname != f"/{page}" else no_update)

    @app.callback(
        Output("store-revs", "data"),
        Output("job-banner", "children"),
        Input("poll", "n_intervals"),
        State("store-revs", "data"),
    )
    def poll(_n, seen):
        # Only a changed revision is sent on, so the pages reload once per
        # finished job rather than once a second.
        revs = workspace.revision()
        return (revs if revs != seen else no_update), job_banner(jobs.snapshot())


def create_app() -> Dash:
    # update_title=None: otherwise the tab title flickers to "Updating..." on every poll.
    app = Dash(__name__, title="Tai Chi balance analysis", update_title=None)

    @app.server.route("/outputs/<path:name>")
    def serve_output(name: str):
        return flask.send_from_directory(config.OUTPUT_DIR, name, max_age=0)

    app.layout = serve_layout
    register_callbacks(app)
    editor_page.register_callbacks(app)
    pipeline_page.register_callbacks(app)
    results_page.register_callbacks(app)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the Tai Chi balance analysis dashboard.")
    parser.add_argument("--port", type=int, default=8051, help="port to serve on (default 8051)")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    parser.add_argument("--debug", action="store_true", help="Dash debug mode: error overlay and dev tools")
    args = parser.parse_args()

    app = create_app()
    selection = recordings.active()
    if selection.problem() is None and not pipeline.pending_preprocessing(selection):
        tasks.load_recordings()

    url = f"http://127.0.0.1:{args.port}"
    print(f"\nDashboard running at {url}  (Ctrl+C to stop)\n")
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    # No reloader: it would run the app twice, each with its own loaded
    # recordings and background job.
    app.run(debug=args.debug, port=args.port, use_reloader=False)
