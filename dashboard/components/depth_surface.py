"""Panel 3: 3D order book depth surface visualization.

Renders a single smoothed 3-D surface (time x price offset x volume)
with a diverging bid/ask colorscale: green for bids (negative offset),
red for asks (positive offset).
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
    n_time_samples: int = 120,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a smoothed volume grid and a side-indicator grid for coloring.

    Returns (time_indices, price_offsets, volume_grid, side_grid).
    side_grid encodes bid (-1) vs ask (+1) for the diverging colorscale.
    """
    step = max(1, len(snapshots) // n_time_samples)
    sub = snapshots.iloc[::step].reset_index(drop=True)
    n_t = len(sub)

    # Place bid levels at negative offsets, ask levels at positive offsets,
    # evenly spaced by level index rather than raw price — avoids degenerate
    # binning when spreads are tiny relative to level spacing.
    n_bins = 2 * n_levels
    price_offsets = np.linspace(-n_levels, n_levels, n_bins)

    volume_grid = np.zeros((n_t, n_bins))
    side_grid = np.zeros((n_t, n_bins))

    for t_idx in range(n_t):
        for lvl in range(1, n_levels + 1):
            # Bid levels: map to negative offset bins
            bid_bin = n_levels - lvl  # level 1 -> bin 9, level 10 -> bin 0
            bv = sub.iloc[t_idx][f"bid_qty_{lvl}"]
            if 0 <= bid_bin < n_bins:
                volume_grid[t_idx, bid_bin] += bv
                side_grid[t_idx, bid_bin] = -1.0

            # Ask levels: map to positive offset bins
            ask_bin = n_levels + lvl - 1  # level 1 -> bin 10, level 10 -> bin 19
            av = sub.iloc[t_idx][f"ask_qty_{lvl}"]
            if 0 <= ask_bin < n_bins:
                volume_grid[t_idx, ask_bin] += av
                side_grid[t_idx, ask_bin] = 1.0

    # Smooth the volume to remove jaggedness (sigma in time and price dims)
    sigma_t = max(1.0, n_t / 60)
    sigma_p = 0.8
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

    # Diverging colorscale: green (bids) ← dark center → red (asks)
    colorscale = [
        [0.00, "#4CAF82"],  # strong bid (green)
        [0.35, "#1a3a2e"],  # fading bid
        [0.50, "#0c1016"],  # neutral center (dark)
        [0.65, "#3a1a1a"],  # fading ask
        [1.00, "#EF6C6C"],  # strong ask (red)
    ]

    # Minimal scene axes
    _scene_axis = dict(
        backgroundcolor="rgba(0,0,0,0)",
        gridcolor="rgba(255,255,255,0.04)",
        showbackground=False,
        tickfont=dict(size=9, color="#5a6575"),
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
            opacity=0.92,
            showscale=False,
            lighting=dict(
                ambient=0.55,
                diffuse=0.65,
                specular=0.08,
                roughness=0.70,
                fresnel=0.05,
            ),
            lightposition=dict(x=100, y=200, z=400),
            contours=dict(z=dict(show=False)),
            hovertemplate=(
                "Offset: %{x:.1f} levels<br>Time Step: %{y}<br>Volume: %{z:.2f}<extra></extra>"
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
                    text="Price Offset (levels from mid)",
                    font=dict(size=10, color="#6b7685"),
                ),
                nticks=6,
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
                eye=dict(x=1.5, y=-1.5, z=0.65),
                up=dict(x=0, y=0, z=1),
            ),
            aspectmode="manual",
            aspectratio=dict(x=1.2, y=1.5, z=0.6),
        ),
    )

    return fig
