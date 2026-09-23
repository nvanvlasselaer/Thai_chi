#!/usr/bin/env python3
"""Dash dashboard for browsing the raw accelerometer and gyroscope channels."""
import sys
from pathlib import Path

import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if __package__ in (None, ""):
    # Run as a script rather than with -m: make the ``analysis`` package importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis import config

# ============================================================
# 1. Load file safely
# ============================================================
file_path = config.NOVICE_CSV

with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

header_idx = next(
    (i for i, line in enumerate(lines) if line.startswith("ACC X Time Series")),
    None
)

if header_idx is None:
    raise ValueError("Header not found")

rows = [r.strip().split(",") for r in lines[header_idx:]]
max_len = max(len(r) for r in rows)
rows = [r + [None] * (max_len - len(r)) for r in rows]

df = pd.DataFrame(rows).apply(pd.to_numeric, errors="coerce")

# ============================================================
# 2. Sensor structure
# ============================================================
sensors = [
    "L_humerus", "lumbar", "L_thigh", "R_thigh", "R_humerus",
    "R_foot", "R_ulna", "L_foot", "R_tibia", "L_hand",
    "chestbone", "R_hand", "L_tibia", "L_ulna"
]

block_size = 12

sensor_map = {}
for i, s in enumerate(sensors):
    start = i * block_size
    sensor_map[s] = df.iloc[:, start:start + block_size]

cols = [
    "ACC_X_t", "ACC_X",
    "ACC_Y_t", "ACC_Y",
    "ACC_Z_t", "ACC_Z",
    "GYRO_X_t", "GYRO_X",
    "GYRO_Y_t", "GYRO_Y",
    "GYRO_Z_t", "GYRO_Z"
]

for s in sensors:
    sensor_map[s].columns = cols

# ============================================================
# 3. App UI (FIXED)
# ============================================================
app = Dash(__name__)

app.layout = html.Div([
    html.H2("IMU Dashboard (ACC + GYRO per sensor)"),

    html.Div([
        html.Label("Sensor 1"),
        dcc.Dropdown(sensors, "lumbar", id="sensor1"),

        html.Br(),

        html.Label("Sensor 2"),
        dcc.Dropdown(sensors, None, id="sensor2"),
    ], style={"width": "30%", "display": "inline-block"}),

    dcc.Graph(id="imu_plot")
])

# ============================================================
# 4. Plot builder
# ============================================================
def add_sensor(fig, sensor, acc_row, gyro_row, label):

    if sensor is None:
        return

    d = sensor_map[sensor]

    # ACC
    fig.add_trace(go.Scatter(
        x=d["ACC_X_t"], y=d["ACC_X"], name=f"{label} ACC X"
    ), row=acc_row, col=1)

    fig.add_trace(go.Scatter(
        x=d["ACC_X_t"], y=d["ACC_Y"], name=f"{label} ACC Y"
    ), row=acc_row, col=1)

    fig.add_trace(go.Scatter(
        x=d["ACC_X_t"], y=d["ACC_Z"], name=f"{label} ACC Z"
    ), row=acc_row, col=1)

    # GYRO
    fig.add_trace(go.Scatter(
        x=d["GYRO_X_t"], y=d["GYRO_X"], name=f"{label} GYRO X"
    ), row=gyro_row, col=1)

    fig.add_trace(go.Scatter(
        x=d["GYRO_X_t"], y=d["GYRO_Y"], name=f"{label} GYRO Y"
    ), row=gyro_row, col=1)

    fig.add_trace(go.Scatter(
        x=d["GYRO_X_t"], y=d["GYRO_Z"], name=f"{label} GYRO Z"
    ), row=gyro_row, col=1)

# ============================================================
# 5. Callback
# ============================================================
@app.callback(
    Output("imu_plot", "figure"),
    Input("sensor1", "value"),
    Input("sensor2", "value"),
)
def update(sensor1, sensor2):

    selected = [s for s in [sensor1, sensor2] if s]

    if len(selected) == 0:
        return go.Figure()

    rows = 2 * len(selected)

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=[
            f"{s} - ACC" if i % 2 == 0 else f"{s} - GYRO"
            for s in selected
            for i in (0, 1)
        ]
    )

    for i, s in enumerate(selected):
        add_sensor(fig, s, 2*i + 1, 2*i + 2, s)

    fig.update_layout(
        height=320 * rows,
        hovermode="x unified",
        title="Sensor comparison (ACC + GYRO)"
    )

    return fig

# ============================================================
# 6. Run
# ============================================================
if __name__ == "__main__":
    app.run_server(debug=True)