"""Panel 3: 3D order book depth surface visualization.

Renders two 3-D surfaces (bid and ask) colored by side, showing resting
volume across price offsets from mid over time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dashboard._constants import PLOTLY_LAYOUT_DEFAULTS


def _build_depth_grids(
    snapshots: pd.DataFrame,
    n_levels: int = 10,
    n_time_samples: int = 120,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build separate bid and ask volume grids for the 3-D surface.

    Returns (time_indices, price_offsets, bid_volume_grid, ask_volume_grid).
    price_offsets are relative to mid-price so the surface is centered.
    """
    step = max(1, len(snapshots) // n_time_samples)
    sub = snapshots.iloc[::step].reset_index(drop=True)
    n_t = len(sub)

    n_bins = 2 * n_levels
    price_offsets = np.linspace(-n_levels * 0.5, n_levels * 0.5, n_bins)
    bin_range = price_offsets[-1] - price_offsets[0]

    bid_grid = np.zeros((n_t, n_bins))
    ask_grid = np.zeros((n_t, n_bins))

    for t_idx in range(n_t):
        mid = sub.loc[t_idx, "mid_price"]
        for lvl in range(1, n_levels + 1):
            bp = sub.loc[t_idx, f"bid_price_{lvl}"]
            bv = sub.loc[t_idx, f"bid_qty_{lvl}"]
            offset = bp - mid
            bin_idx = int((offset - price_offsets[0]) / bin_range * (n_bins - 1))
            if 0 <= bin_idx < n_bins:
                bid_grid[t_idx, bin_idx] += bv

            ap = sub.loc[t_idx, f"ask_price_{lvl}"]
            av = sub.loc[t_idx, f"ask_qty_{lvl}"]
            offset = ap - mid
            bin_idx = int((offset - price_offsets[0]) / bin_range * (n_bins - 1))
            if 0 <= bin_idx < n_bins:
                ask_grid[t_idx, bin_idx] += av

    time_indices = np.arange(n_t)
    return time_indices, price_offsets, bid_grid, ask_grid


def create_depth_surface_figure(
    snapshots: pd.DataFrame,
    regimes: np.ndarray,
) -> go.Figure:
    """Create the 3-D order book depth surface with bid/ask differentiation.

    Parameters
    ----------
    snapshots : DataFrame with book_reconstructor schema.
    regimes : 1-D array of regime labels aligned to snapshots.
    """
    time_idx, price_offsets, bid_grid, ask_grid = _build_depth_grids(snapshots)

    # Bid colorscale: dark to green (buy side)
    bid_colorscale = [
        [0.0, "rgba(12,16,22,0.9)"],
        [0.3, "rgba(20,80,60,0.9)"],
        [0.6, "rgba(40,140,90,0.9)"],
        [1.0, "rgba(76,175,130,0.95)"],
    ]

    # Ask colorscale: dark to red (sell side)
    ask_colorscale = [
        [0.0, "rgba(12,16,22,0.9)"],
        [0.3, "rgba(80,30,30,0.9)"],
        [0.6, "rgba(160,60,60,0.9)"],
        [1.0, "rgba(239,108,108,0.95)"],
    ]

    # Minimal scene axes
    _scene_axis_common = dict(
        backgroundcolor="rgba(0,0,0,0)",
        gridcolor="rgba(255,255,255,0.04)",
        showbackground=False,
        tickfont=dict(size=9, color="#5a6575"),
        showspikes=False,
    )

    _lighting = dict(
        ambient=0.55,
        diffuse=0.65,
        specular=0.08,
        roughness=0.70,
        fresnel=0.05,
    )

    fig = go.Figure()

    # Bid surface (green)
    fig.add_trace(
        go.Surface(
            x=price_offsets,
            y=time_idx,
            z=bid_grid,
            colorscale=bid_colorscale,
            opacity=0.88,
            showscale=False,
            name="Bids",
            lighting=_lighting,
            lightposition=dict(x=100, y=200, z=400),
            contours=dict(z=dict(show=False)),
            hovertemplate=(
                "Bid<br>Offset: %{x:.2f}<br>Time: %{y}<br>Volume: %{z:.2f}<extra></extra>"
            ),
        )
    )

    # Ask surface (red)
    fig.add_trace(
        go.Surface(
            x=price_offsets,
            y=time_idx,
            z=ask_grid,
            colorscale=ask_colorscale,
            opacity=0.88,
            showscale=False,
            name="Asks",
            lighting=_lighting,
            lightposition=dict(x=100, y=200, z=400),
            contours=dict(z=dict(show=False)),
            hovertemplate=(
                "Ask<br>Offset: %{x:.2f}<br>Time: %{y}<br>Volume: %{z:.2f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        height=540,
        margin=dict(l=0, r=0, t=8, b=0),
        scene=dict(
            xaxis=dict(
                title=dict(text="Price Offset from Mid", font=dict(size=10, color="#6b7685")),
                nticks=6,
                **_scene_axis_common,
            ),
            yaxis=dict(
                title=dict(text="Time", font=dict(size=10, color="#6b7685")),
                nticks=6,
                **_scene_axis_common,
            ),
            zaxis=dict(
                title=dict(text="Resting Volume", font=dict(size=10, color="#6b7685")),
                nticks=5,
                **_scene_axis_common,
            ),
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=0.65),
                up=dict(x=0, y=0, z=1),
            ),
            aspectmode="manual",
            aspectratio=dict(x=1.2, y=1.5, z=0.6),
        ),
    )

    return fig
