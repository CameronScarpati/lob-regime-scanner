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

import dash_mantine_components as dmc
import pandas as pd
from dash import Dash, dcc, html

from dashboard._constants import PANEL_DESCRIPTIONS, REGIME_COLORS
from dashboard.callbacks import register_callbacks
from dashboard.components.heatmap import create_heatmap_figure
from dashboard.components.regime_probs import create_regime_probs_figure
from dashboard.components.depth_surface import create_depth_surface_figure
from dashboard.components.diagnostics import create_diagnostics_figure

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_time_options(
    timestamps: pd.Series,
) -> list[dict[str, str]]:
    """Build Select options from all available timestamps.

    The pipeline already caps data at ~3600 points, so showing every
    timestamp keeps the searchable dropdown usable while giving
    precise time-window control.
    """
    n = len(timestamps)
    if n == 0:
        return []

    ts_start = pd.Timestamp(timestamps.iloc[0])
    ts_end = pd.Timestamp(timestamps.iloc[-1])
    same_day = ts_start.date() == ts_end.date()

    options: list[dict[str, str]] = []
    for idx in range(n):
        ts = pd.Timestamp(timestamps.iloc[idx])
        if same_day:
            label = ts.strftime("%-H:%M:%S")
        else:
            label = ts.strftime("%b %-d %H:%M:%S")
        options.append({"label": label, "value": str(idx)})
    return options


def _make_panel(title: str, description: str, graph_id: str, figure, config: dict):
    """Build a single chart panel with DMC Paper + title + dcc.Graph."""
    return dmc.Paper(
        radius="md",
        withBorder=True,
        p=0,
        children=[
            dmc.Stack(
                gap=2,
                p="sm",
                pb=0,
                children=[
                    dmc.Text(title, fw=600, size="sm", c="gray.2"),
                    dmc.Text(description, size="xs", c="dimmed", lh=1.5),
                ],
            ),
            dcc.Graph(id=graph_id, figure=figure, config=config),
        ],
    )


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
        "--sample-interval",
        type=int,
        default=100,
        help="Snapshot subsampling interval in milliseconds (default: 100). "
        "Lower values capture more microstructure detail but use more memory. "
        "Try 1000 for faster loading, 10 for tick-level resolution.",
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
            sample_interval_us=args.sample_interval * 1000,
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

    # Build time picker options (sampled to ~80 entries)
    time_options = _build_time_options(snap["timestamp"])

    # Determine data source label
    if args.demo:
        source_label = "Mock / Synthetic"
    else:
        source_label = f"Live ({args.symbol})"

    date_label = ""
    if args.start and args.end:
        date_label = f"{args.start} - {args.end}"
    elif args.start:
        date_label = f"{args.start} - ..."
    elif args.end:
        date_label = f"... - {args.end}"
    else:
        date_label = "Full range"

    # Build app
    dash_app = Dash(
        __name__,
        title="LOB Regime Scanner",
        assets_folder=os.path.join(os.path.dirname(__file__), "assets"),
    )

    dash_app.layout = dmc.MantineProvider(
        id="mantine-provider",
        forceColorScheme="dark",
        theme={
            "primaryColor": "blue",
            "fontFamily": "Inter, Helvetica Neue, Helvetica, Arial, sans-serif",
            "colors": {
                "dark": [
                    "#c1c7d0",  # dark.0 - lightest text
                    "#a6adb8",  # dark.1
                    "#8892a0",  # dark.2
                    "#5c6775",  # dark.3
                    "#3a4654",  # dark.4
                    "#283444",  # dark.5
                    "#1a2332",  # dark.6 - paper bg
                    "#111a26",  # dark.7 - card bg
                    "#0d1420",  # dark.8 - body bg
                    "#080c12",  # dark.9 - deepest
                ],
            },
        },
        children=[
            dmc.Container(
                id="app-container",
                size="1860px",
                px="md",
                py="sm",
                children=dmc.Stack(
                    gap="sm",
                    children=[
                        # -- Header --
                        dmc.Paper(
                            radius="md",
                            withBorder=True,
                            p="sm",
                            px="lg",
                            children=dmc.Group(
                                justify="space-between",
                                children=[
                                    dmc.Title(
                                        "Limit Order Book Regime Scanner",
                                        order=4,
                                        tt="uppercase",
                                        lts="0.1em",
                                    ),
                                    dmc.Group(
                                        gap="lg",
                                        children=[
                                            dmc.Group(gap=6, children=[
                                                dmc.Badge("Instrument", size="xs", variant="light"),
                                                dmc.Text(f"{args.symbol} Perp", size="sm", c="dimmed"),
                                            ]),
                                            dmc.Group(gap=6, children=[
                                                dmc.Badge("Date", size="xs", variant="light"),
                                                dmc.Text(date_label, size="sm", c="dimmed"),
                                            ]),
                                            dmc.Group(gap=6, children=[
                                                dmc.Badge("Model", size="xs", variant="light"),
                                                dmc.Text("GaussianHMM (3 states)", size="sm", c="dimmed"),
                                            ]),
                                            dmc.Group(gap=6, children=[
                                                dmc.Badge("Interval", size="xs", variant="light"),
                                                dmc.Text(f"{args.sample_interval}ms", size="sm", c="dimmed"),
                                            ]),
                                            dmc.Group(gap=6, children=[
                                                dmc.Badge("Source", size="xs", variant="light"),
                                                dmc.Text(source_label, size="sm", c="dimmed"),
                                            ]),
                                        ],
                                    ),
                                ],
                            ),
                        ),

                        # -- Controls bar --
                        dmc.Paper(
                            radius="md",
                            withBorder=True,
                            p="xs",
                            px="lg",
                            children=dmc.Group(
                                justify="space-between",
                                children=[
                                    # Time window pickers
                                    dmc.Group(
                                        gap="sm",
                                        children=[
                                            dmc.Text(
                                                "Time Window",
                                                size="xs",
                                                fw=600,
                                                c="dimmed",
                                                tt="uppercase",
                                                style={"letterSpacing": "0.08em"},
                                            ),
                                            dmc.Select(
                                                id="time-start-select",
                                                data=time_options,
                                                value=time_options[0]["value"],
                                                searchable=True,
                                                clearable=False,
                                                w=130,
                                                size="xs",
                                            ),
                                            dmc.Text("to", size="sm", c="dimmed"),
                                            dmc.Select(
                                                id="time-end-select",
                                                data=time_options,
                                                value=time_options[-1]["value"],
                                                searchable=True,
                                                clearable=False,
                                                w=130,
                                                size="xs",
                                            ),
                                        ],
                                    ),

                                    # Regime filter chips
                                    dmc.Group(
                                        gap="sm",
                                        children=[
                                            dmc.Text(
                                                "Regimes",
                                                size="xs",
                                                fw=600,
                                                c="dimmed",
                                                tt="uppercase",
                                                style={"letterSpacing": "0.08em"},
                                            ),
                                            dmc.ChipGroup(
                                                id="regime-chip-group",
                                                value=["quiet", "trending", "toxic"],
                                                multiple=True,
                                                children=[
                                                    dmc.Chip(
                                                        "Quiet",
                                                        value="quiet",
                                                        color=REGIME_COLORS[0],
                                                        variant="outline",
                                                        size="xs",
                                                    ),
                                                    dmc.Chip(
                                                        "Trending",
                                                        value="trending",
                                                        color=REGIME_COLORS[1],
                                                        variant="outline",
                                                        size="xs",
                                                    ),
                                                    dmc.Chip(
                                                        "Toxic",
                                                        value="toxic",
                                                        color=REGIME_COLORS[2],
                                                        variant="outline",
                                                        size="xs",
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ),

                        # -- 2x2 Panel grid --
                        dmc.SimpleGrid(
                            cols=2,
                            spacing="sm",
                            children=[
                                _make_panel(
                                    "Order-Book Heatmap with Regime Overlay",
                                    PANEL_DESCRIPTIONS["heatmap"],
                                    "heatmap-panel",
                                    init_heatmap,
                                    {
                                        "displayModeBar": "hover",
                                        "scrollZoom": True,
                                        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                                    },
                                ),
                                _make_panel(
                                    "Regime Posterior Probabilities",
                                    PANEL_DESCRIPTIONS["regime_probs"],
                                    "regime-probs-panel",
                                    init_regime,
                                    {"displayModeBar": "hover"},
                                ),
                                _make_panel(
                                    "3-D Order-Book Depth Surface",
                                    PANEL_DESCRIPTIONS["depth_surface"],
                                    "depth-surface-panel",
                                    init_depth,
                                    {"displayModeBar": "hover", "scrollZoom": True},
                                ),
                                _make_panel(
                                    "Microstructure Diagnostics",
                                    PANEL_DESCRIPTIONS["diagnostics"],
                                    "diagnostics-panel",
                                    init_diag,
                                    {"displayModeBar": "hover"},
                                ),
                            ],
                        ),
                    ],
                ),
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
