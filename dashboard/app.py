"""Main Plotly Dash application.

Renders a four-panel synchronized dashboard: LOB heatmap with regime
overlay, regime state probabilities, 3D depth surface, and toxicity
diagnostics.

Run with:
    python -m dashboard.app
"""

from __future__ import annotations

import os

from dash import Dash, dcc, html

from dashboard._mock_data import generate_all
from dashboard.callbacks import register_callbacks
from dashboard.components.heatmap import create_heatmap_figure
from dashboard.components.regime_probs import create_regime_probs_figure
from dashboard.components.depth_surface import create_depth_surface_figure
from dashboard.components.diagnostics import create_diagnostics_figure

# ---------------------------------------------------------------------------
# Generate initial data for the layout (callbacks will regenerate on update)
# ---------------------------------------------------------------------------
_initial = generate_all(n_timestamps=3600)

_snap = _initial["snapshots"]
_feat = _initial["features"]
_hmm = _initial["hmm"]
_pnl = _initial["cumulative_pnl"]

# ---------------------------------------------------------------------------
# Build initial figures
# ---------------------------------------------------------------------------
_init_heatmap = create_heatmap_figure(_snap, _hmm["states"])
_init_regime = create_regime_probs_figure(
    _feat["timestamp"].values, _hmm["state_probs"], _hmm["transition_matrix"]
)
_init_depth = create_depth_surface_figure(_snap, _hmm["states"])
_init_diag = create_diagnostics_figure(_feat, _hmm["states"], _pnl)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Dash(
    __name__,
    title="LOB Regime Scanner",
    assets_folder=os.path.join(os.path.dirname(__file__), "assets"),
)

server = app.server  # For deployment (gunicorn / waitress)

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
app.layout = html.Div(
    id="app-container",
    children=[
        # ── Header ──
        html.Div(
            className="dashboard-header",
            children=[
                html.H1("LOB Regime Scanner"),
                html.Div(
                    className="header-meta",
                    children=[
                        html.Span([
                            html.Span("Instrument", className="label"),
                            " BTCUSDT Perp",
                        ]),
                        html.Span([
                            html.Span("Date", className="label"),
                            " 2025-01-15",
                        ]),
                        html.Span([
                            html.Span("Model", className="label"),
                            " GaussianHMM (3 states, full cov)",
                        ]),
                        html.Span([
                            html.Span("Source", className="label"),
                            " Mock / Synthetic Data",
                        ]),
                    ],
                ),
            ],
        ),
        # ── Controls bar ──
        html.Div(
            className="controls-bar",
            children=[
                html.Div(
                    className="slider-container",
                    children=[
                        html.Div("Date Range", className="slider-label"),
                        dcc.RangeSlider(
                            id="date-range-slider",
                            min=0,
                            max=len(_snap) - 1,
                            step=1,
                            value=[0, len(_snap) - 1],
                            marks={
                                0: "Start",
                                len(_snap) // 4: "25%",
                                len(_snap) // 2: "50%",
                                3 * len(_snap) // 4: "75%",
                                len(_snap) - 1: "End",
                            },
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                    ],
                ),
                html.Div(
                    className="regime-filters",
                    children=[
                        html.Span("Regimes:", className="filter-label"),
                        html.Button(
                            "Quiet",
                            id="regime-quiet-btn",
                            className="regime-btn regime-btn-quiet",
                            n_clicks=0,
                        ),
                        html.Button(
                            "Trending",
                            id="regime-trending-btn",
                            className="regime-btn regime-btn-trending",
                            n_clicks=0,
                        ),
                        html.Button(
                            "Toxic",
                            id="regime-toxic-btn",
                            className="regime-btn regime-btn-toxic",
                            n_clicks=0,
                        ),
                    ],
                ),
            ],
        ),
        # ── 2×2 Panel grid ──
        html.Div(
            className="panel-grid",
            children=[
                html.Div(
                    className="panel",
                    children=[
                        dcc.Graph(
                            id="heatmap-panel",
                            figure=_init_heatmap,
                            config={"displayModeBar": True, "scrollZoom": True},
                        ),
                    ],
                ),
                html.Div(
                    className="panel",
                    children=[
                        dcc.Graph(
                            id="regime-probs-panel",
                            figure=_init_regime,
                            config={"displayModeBar": True},
                        ),
                    ],
                ),
                html.Div(
                    className="panel",
                    children=[
                        dcc.Graph(
                            id="depth-surface-panel",
                            figure=_init_depth,
                            config={
                                "displayModeBar": True,
                                "scrollZoom": True,
                            },
                        ),
                    ],
                ),
                html.Div(
                    className="panel",
                    children=[
                        dcc.Graph(
                            id="diagnostics-panel",
                            figure=_init_diag,
                            config={"displayModeBar": True},
                        ),
                    ],
                ),
            ],
        ),
    ],
)

# ---------------------------------------------------------------------------
# Register callbacks
# ---------------------------------------------------------------------------
register_callbacks(app)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
