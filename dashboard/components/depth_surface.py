"""Panel 3: 3D order book depth surface visualization.

Renders a single continuous 3-D surface (time x price offset x volume)
with a green-to-red diverging colorscale for bid/ask sides.
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
    """Build a unified volume grid and side grid for coloring.

    Returns (time_indices, price_offsets, volume_grid, side_grid).
    Bids occupy negative offsets, asks occupy positive offsets, joined
    at the mid-price so there is no gap.
    """
    step = max(1, len(snapshots) // n_time_samples)
    sub = snapshots.iloc[::step].reset_index(drop=True)
    n_t = len(sub)

    # Offsets: -n_levels..-1 for bids, +1..+n_levels for asks
    n_bins = 2 * n_levels
    price_offsets = np.concatenate([
        np.arange(-n_levels, 0, dtype=float),
        np.arange(1, n_levels + 1, dtype=float),
    ])

    volume_grid = np.zeros((n_t, n_bins))
    side_grid = np.zeros((n_t, n_bins))

    for t_idx in range(n_t):
        row = sub.iloc[t_idx]
        for lvl in range(1, n_levels + 1):
            # Bids: level 1 closest to mid (bin n_levels-1), level 10 farthest (bin 0)
            bid_bin = n_levels - lvl
            volume_grid[t_idx, bid_bin] = row[f"bid_qty_{lvl}"]
            side_grid[t_idx, bid_bin] = -1.0

            # Asks: level 1 closest to mid (bin n_levels), level 10 farthest (bin 2*n_levels-1)
            ask_bin = n_levels + lvl - 1
            volume_grid[t_idx, ask_bin] = row[f"ask_qty_{lvl}"]
            side_grid[t_idx, ask_bin] = 1.0

    # Light smoothing
    sigma_t = max(0.8, n_t / 150)
    sigma_p = 0.6
    volume_grid = gaussian_filter(volume_grid, sigma=[sigma_t, sigma_p])

    time_indices = np.arange(n_t)
    return time_indices, price_offsets, volume_grid, side_grid


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
    time_idx, price_offsets, vol_grid, side_grid = _build_depth_grid(snapshots)

    # Green-to-red diverging colorscale through a neutral mid
    colorscale = [
        [0.00, "#22c55e"],  # green (bids)
        [0.40, "#166534"],  # dark green
        [0.50, "#1e293b"],  # neutral slate
        [0.60, "#7f1d1d"],  # dark red
        [1.00, "#ef4444"],  # red (asks)
    ]

    # Scene axis style
    _scene_axis = dict(
        backgroundcolor="rgba(0,0,0,0)",
        gridcolor="rgba(255,255,255,0.08)",
        showbackground=False,
        tickfont=dict(size=9, color="#94a3b8"),
        showspikes=False,
    )

    fig = go.Figure()

    fig.add_trace(
        go.Surface(
            x=price_offsets,
            y=time_idx,
            z=vol_grid,
            surfacecolor=side_grid,
            colorscale=colorscale,
            cmin=-1,
            cmax=1,
            showscale=False,
            hovertemplate=(
                "Level: %{x:.0f}<br>"
                "Time: %{y}<br>"
                "Volume: %{z:.2f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        height=540,
        margin=dict(l=0, r=0, t=8, b=0),
        scene=dict(
            xaxis=dict(
                title=dict(
                    text="\u2190 Bids | Asks \u2192",
                    font=dict(size=10, color="#94a3b8"),
                ),
                nticks=8,
                **_scene_axis,
            ),
            yaxis=dict(
                title=dict(text="Time", font=dict(size=10, color="#94a3b8")),
                nticks=6,
                **_scene_axis,
            ),
            zaxis=dict(
                title=dict(text="Volume", font=dict(size=10, color="#94a3b8")),
                nticks=5,
                **_scene_axis,
            ),
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=0.7),
                up=dict(x=0, y=0, z=1),
            ),
            aspectmode="manual",
            aspectratio=dict(x=1.2, y=1.5, z=0.6),
        ),
    )

    return fig
