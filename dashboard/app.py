"""Main Plotly Dash application.

Renders a four-panel synchronized dashboard: LOB heatmap with regime
overlay, regime state probabilities, 3D depth surface, and toxicity
diagnostics.

Run with:
    python -m dashboard.app --symbol BTCUSDT --start 2025-01-01 --end 2025-01-14
    python -m dashboard.app --demo   # use synthetic mock data
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import pandas as pd
from dash import Dash, dcc, html

from dashboard._constants import PANEL_DESCRIPTIONS
from dashboard.callbacks import register_callbacks
from dashboard.components.heatmap import create_heatmap_figure
from dashboard.components.regime_probs import create_regime_probs_figure
from dashboard.components.depth_surface import create_depth_surface_figure
from dashboard.components.diagnostics import create_diagnostics_figure

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_slider_marks(
    timestamps: pd.Series,
    n_marks: int = 6,
) -> dict[int, str]:
    """Create evenly-spaced slider marks with human-readable time labels.

    If all timestamps fall on the same date, marks show only the time
    (e.g. "09:15"). When the range spans multiple days the date is
    included (e.g. "Jan 15 09:15").
    """
    n = len(timestamps)
    if n == 0:
        return {}

    indices = [int(i * (n - 1) / (n_marks - 1)) for i in range(n_marks)]

    ts_start = pd.Timestamp(timestamps.iloc[0])
    ts_end = pd.Timestamp(timestamps.iloc[-1])
    same_day = ts_start.date() == ts_end.date()

    marks: dict[int, str] = {}
    for idx in indices:
        ts = pd.Timestamp(timestamps.iloc[idx])
        if same_day:
            marks[idx] = ts.strftime("%-H:%M")
        else:
            marks[idx] = ts.strftime("%b %-d %H:%M")
    return marks


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the dashboard."""
    parser = argparse.ArgumentParser(
        description="LOB Regime Scanner Dashboard",
    )
    parser.add_argument(
        "--symbol",
        default="BTCUSDT",
        help="Trading pair symbol (default: BTCUSDT)",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Start date ISO format, e.g. 2025-01-01",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End date ISO format, e.g. 2025-01-14",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use synthetic mock data instead of the real pipeline",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port to listen on (default: 8050)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Dash debug mode",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(args: argparse.Namespace) -> dict:
    """Load data from the real pipeline or fall back to mock data.

    Returns the standard dict with keys: snapshots, features, hmm, cumulative_pnl.
    """
    if args.demo:
        logger.info("Running in demo mode with synthetic data")
        from dashboard._mock_data import generate_all
        return generate_all(n_timestamps=3600)

    # Try the real pipeline
    from dashboard.pipeline import NoDataError, run_pipeline

    try:
        return run_pipeline(
            symbol=args.symbol,
            start=args.start,
            end=args.end,
        )
    except NoDataError as exc:
        print(
            f"\n*** No data available ***\n\n{exc}\n\n"
            "To use synthetic data instead, run with --demo:\n"
            "  python -m dashboard.app --demo\n",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(args: argparse.Namespace | None = None) -> Dash:
    """Build and return the Dash app with layout and callbacks registered.

    If *args* is ``None``, defaults to ``--demo`` mode (for tests / imports).
    """
    if args is None:
        args = parse_args(["--demo"])

    data = load_data(args)

    snap = data["snapshots"]
    feat = data["features"]
    hmm = data["hmm"]
    pnl = data["cumulative_pnl"]

    # Build initial figures
    init_heatmap = create_heatmap_figure(snap, hmm["states"])
    init_regime = create_regime_probs_figure(
        feat["timestamp"].values, hmm["state_probs"], hmm["transition_matrix"]
    )
    init_depth = create_depth_surface_figure(snap, hmm["states"])
    init_diag = create_diagnostics_figure(feat, hmm["states"], pnl)

    # Determine data source label
    if args.demo:
        source_label = "Mock / Synthetic Data"
    else:
        source_label = f"Real Data ({args.symbol})"

    date_label = ""
    if args.start and args.end:
        date_label = f"{args.start} – {args.end}"
    elif args.start:
        date_label = f"{args.start} – ..."
    elif args.end:
        date_label = f"... – {args.end}"
    else:
        date_label = "Full range"

    # Build app
    dash_app = Dash(
        __name__,
        title="LOB Regime Scanner",
        assets_folder=os.path.join(os.path.dirname(__file__), "assets"),
    )

    dash_app.layout = html.Div(
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
                                f" {args.symbol} Perp",
                            ]),
                            html.Span([
                                html.Span("Date", className="label"),
                                f" {date_label}",
                            ]),
                            html.Span([
                                html.Span("Model", className="label"),
                                " GaussianHMM (3 states, full cov)",
                            ]),
                            html.Span([
                                html.Span("Source", className="label"),
                                f" {source_label}",
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
                            html.Div("Time Window", className="slider-label"),
                            dcc.RangeSlider(
                                id="date-range-slider",
                                min=0,
                                max=len(snap) - 1,
                                step=1,
                                value=[0, len(snap) - 1],
                                marks=_build_slider_marks(snap["timestamp"]),
                                tooltip={
                                    "placement": "bottom",
                                    "always_visible": False,
                                },
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
                            html.Div(className="panel-header", children=[
                                html.Div("Order-Book Heatmap with Regime Overlay",
                                         className="panel-title"),
                                html.Div(PANEL_DESCRIPTIONS["heatmap"],
                                         className="panel-description"),
                            ]),
                            dcc.Graph(
                                id="heatmap-panel",
                                figure=init_heatmap,
                                config={
                                    "displayModeBar": "hover",
                                    "scrollZoom": True,
                                    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                                },
                            ),
                        ],
                    ),
                    html.Div(
                        className="panel",
                        children=[
                            html.Div(className="panel-header", children=[
                                html.Div("Regime Posterior Probabilities",
                                         className="panel-title"),
                                html.Div(PANEL_DESCRIPTIONS["regime_probs"],
                                         className="panel-description"),
                            ]),
                            dcc.Graph(
                                id="regime-probs-panel",
                                figure=init_regime,
                                config={"displayModeBar": "hover"},
                            ),
                        ],
                    ),
                    html.Div(
                        className="panel",
                        children=[
                            html.Div(className="panel-header", children=[
                                html.Div("3-D Order-Book Depth Surface",
                                         className="panel-title"),
                                html.Div(PANEL_DESCRIPTIONS["depth_surface"],
                                         className="panel-description"),
                            ]),
                            dcc.Graph(
                                id="depth-surface-panel",
                                figure=init_depth,
                                config={
                                    "displayModeBar": "hover",
                                    "scrollZoom": True,
                                },
                            ),
                        ],
                    ),
                    html.Div(
                        className="panel",
                        children=[
                            html.Div(className="panel-header", children=[
                                html.Div("Microstructure Diagnostics",
                                         className="panel-title"),
                                html.Div(PANEL_DESCRIPTIONS["diagnostics"],
                                         className="panel-description"),
                            ]),
                            dcc.Graph(
                                id="diagnostics-panel",
                                figure=init_diag,
                                config={"displayModeBar": "hover"},
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )

    # Register interactive callbacks (passes data through closure)
    register_callbacks(dash_app, data)

    return dash_app


# ---------------------------------------------------------------------------
# Module-level app for import-based usage (``from dashboard.app import app``)
# Uses demo mode by default to avoid requiring data files on import.
# ---------------------------------------------------------------------------
app = create_app()
server = app.server  # For deployment (gunicorn / waitress)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cli_args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    cli_app = create_app(cli_args)
    cli_app.run(debug=cli_args.debug, host=cli_args.host, port=cli_args.port)
