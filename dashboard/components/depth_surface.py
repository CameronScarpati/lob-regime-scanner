"""Panel 3: 3D order book depth surface visualization.

Renders a 3-D surface (time x price level x volume) with bid/ask
sides as separate surfaces in green/red for clean visual separation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter

from dashboard._constants import PLOTLY_LAYOUT_DEFAULTS


def _build_depth_grid(
    snapshots: pd.DataFrame,
    n_levels: int = 10,
    n_time_samples: int = 150,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build bid and ask volume grids.

    Returns (time_indices, level_offsets, bid_grid, ask_grid).
    """
    step = max(1, len(snapshots) // n_time_samples)
    sub = snapshots.iloc[::step].reset_index(drop=True)
    n_t = len(sub)

    # Level offsets: 1..n_levels for both sides
    level_offsets = np.arange(1, n_levels + 1, dtype=float)

    bid_grid = np.zeros((n_t, n_levels))
    ask_grid = np.zeros((n_t, n_levels))

    for t_idx in range(n_t):
        for lvl in range(1, n_levels + 1):
            bid_grid[t_idx, lvl - 1] = sub.iloc[t_idx][f"bid_qty_{lvl}"]
            ask_grid[t_idx, lvl - 1] = sub.iloc[t_idx][f"ask_qty_{lvl}"]

    # Light smoothing to remove tick noise while preserving structure
    sigma_t = max(0.8, n_t / 150)
    sigma_p = 0.5
    bid_grid = gaussian_filter(bid_grid, sigma=[sigma_t, sigma_p])
    ask_grid = gaussian_filter(ask_grid, sigma=[sigma_t, sigma_p])

    time_indices = np.arange(n_t)
    return time_indices, level_offsets, bid_grid, ask_grid


def create_depth_surface_figure(
    snapshots: pd.DataFrame,
    regimes: np.ndarray,
) -> go.Figure:
    """Create the 3-D order book depth surface.

    Parameters
    ----------
    snapshots : DataFrame with book_reconstructor schema.
    regimes : 1-D array of regime labels aligned to snapshots.
    """
    time_idx, level_offsets, bid_grid, ask_grid = _build_depth_grid(snapshots)

    # Shared lighting for a clean, modern look
    _lighting = dict(
        ambient=0.45,
        diffuse=0.70,
        specular=0.15,
        roughness=0.50,
        fresnel=0.10,
    )
    _lightpos = dict(x=-50, y=-200, z=500)

    # Bid colorscale: dark to green
    bid_colorscale = [
        [0.00, "rgba(10, 25, 18, 0.85)"],
        [0.25, "rgba(30, 80, 55, 0.90)"],
        [0.50, "rgba(50, 140, 90, 0.92)"],
        [0.75, "rgba(65, 185, 115, 0.95)"],
        [1.00, "rgba(90, 220, 145, 1.0)"],
    ]

    # Ask colorscale: dark to red
    ask_colorscale = [
        [0.00, "rgba(25, 12, 12, 0.85)"],
        [0.25, "rgba(80, 30, 30, 0.90)"],
        [0.50, "rgba(160, 55, 55, 0.92)"],
        [0.75, "rgba(210, 80, 80, 0.95)"],
        [1.00, "rgba(240, 110, 110, 1.0)"],
    ]

    # Minimal scene axes
    _scene_axis = dict(
        backgroundcolor="rgba(0,0,0,0)",
        gridcolor="rgba(255,255,255,0.06)",
        showbackground=False,
        tickfont=dict(size=9, color="#5a6575"),
        showspikes=False,
    )

    fig = go.Figure()

    # Bid surface (negative x-offset to place on the left)
    fig.add_trace(
        go.Surface(
            x=-level_offsets,
            y=time_idx,
            z=bid_grid,
            surfacecolor=bid_grid,
            colorscale=bid_colorscale,
            showscale=False,
            opacity=0.92,
            lighting=_lighting,
            lightposition=_lightpos,
            contours=dict(
                z=dict(show=True, color="rgba(76,175,130,0.15)", width=1),
            ),
            hovertemplate=(
                "Bid Level %{x:.0f}<br>"
                "Time: %{y}<br>"
                "Volume: %{z:.2f}<extra></extra>"
            ),
            name="Bids",
        )
    )

    # Ask surface (positive x-offset to place on the right)
    fig.add_trace(
        go.Surface(
            x=level_offsets,
            y=time_idx,
            z=ask_grid,
            surfacecolor=ask_grid,
            colorscale=ask_colorscale,
            showscale=False,
            opacity=0.92,
            lighting=_lighting,
            lightposition=_lightpos,
            contours=dict(
                z=dict(show=True, color="rgba(239,108,108,0.15)", width=1),
            ),
            hovertemplate=(
                "Ask Level %{x:.0f}<br>"
                "Time: %{y}<br>"
                "Volume: %{z:.2f}<extra></extra>"
            ),
            name="Asks",
        )
    )

    fig.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        height=540,
        margin=dict(l=0, r=0, t=8, b=0),
        scene=dict(
            xaxis=dict(
                title=dict(
                    text="\u2190 Bid Levels | Ask Levels \u2192",
                    font=dict(size=10, color="#6b7685"),
                ),
                nticks=8,
                zeroline=True,
                zerolinecolor="rgba(255,255,255,0.20)",
                zerolinewidth=1,
                **_scene_axis,
            ),
            yaxis=dict(
                title=dict(text="Time", font=dict(size=10, color="#6b7685")),
                nticks=6,
                **_scene_axis,
            ),
            zaxis=dict(
                title=dict(text="Resting Volume", font=dict(size=10, color="#6b7685")),
                nticks=5,
                **_scene_axis,
            ),
            camera=dict(
                eye=dict(x=1.6, y=-1.8, z=0.8),
                up=dict(x=0, y=0, z=1),
            ),
            aspectmode="manual",
            aspectratio=dict(x=1.3, y=1.6, z=0.55),
        ),
    )

    return fig
