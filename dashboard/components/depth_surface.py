"""Panel 3: 3D order book depth surface visualization.

Renders a single continuous 3-D surface (time x price offset x volume)
with solid green (bids) and red (asks) coloring.
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
    n_time_samples: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a unified volume grid and side indicator for coloring.

    Returns (time_indices, price_offsets, volume_grid, side_grid).
    side_grid is -1 for bids, +1 for asks.
    """
    step = max(1, len(snapshots) // n_time_samples)
    sub = snapshots.iloc[::step].reset_index(drop=True)
    n_t = len(sub)

    n_bins = 2 * n_levels
    price_offsets = np.concatenate(
        [
            np.arange(-n_levels, 0, dtype=float),
            np.arange(1, n_levels + 1, dtype=float),
        ]
    )

    volume_grid = np.zeros((n_t, n_bins))
    side_grid = np.zeros((n_t, n_bins))

    for t_idx in range(n_t):
        row = sub.iloc[t_idx]
        for lvl in range(1, n_levels + 1):
            bid_bin = n_levels - lvl
            volume_grid[t_idx, bid_bin] = row[f"bid_qty_{lvl}"]
            side_grid[t_idx, bid_bin] = -1.0

            ask_bin = n_levels + lvl - 1
            volume_grid[t_idx, ask_bin] = row[f"ask_qty_{lvl}"]
            side_grid[t_idx, ask_bin] = 1.0

    # Light smoothing to remove tick noise
    sigma_t = max(0.6, n_t / 200)
    volume_grid = gaussian_filter(volume_grid, sigma=[sigma_t, 0.4])

    time_indices = np.arange(n_t)
    return time_indices, price_offsets, volume_grid, side_grid


def create_depth_surface_figure(
    snapshots: pd.DataFrame,
    regimes: np.ndarray,
) -> go.Figure:
    """Create the 3-D order book depth surface."""
    time_idx, price_offsets, vol_grid, side_grid = _build_depth_grid(snapshots)

    # Hard two-tone colorscale: green for bids (-1), red for asks (+1)
    # No white/light middle — just a narrow transition
    colorscale = [
        [0.00, "#22c55e"],
        [0.45, "#22c55e"],
        [0.50, "#334155"],
        [0.55, "#ef4444"],
        [1.00, "#ef4444"],
    ]

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
            lighting=dict(
                ambient=0.5,
                diffuse=0.6,
                specular=0.25,
                roughness=0.5,
                fresnel=0.15,
            ),
            lightposition=dict(x=0, y=0, z=800),
            hovertemplate=("Level: %{x:.0f}<br>Time: %{y}<br>Volume: %{z:.2f}<extra></extra>"),
        )
    )

    _axis = dict(
        backgroundcolor="#0f1724",
        gridcolor="rgba(255,255,255,0.06)",
        showbackground=True,
        tickfont=dict(size=10, color="#94a3b8"),
        title_font=dict(size=11, color="#94a3b8"),
        showspikes=False,
    )

    fig.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        height=540,
        margin=dict(l=0, r=0, t=8, b=0),
        scene=dict(
            xaxis=dict(title="\u2190 Bids | Asks \u2192", nticks=8, **_axis),
            yaxis=dict(title="Time", nticks=6, **_axis),
            zaxis=dict(title="Resting Volume", nticks=5, **_axis),
            camera=dict(
                eye=dict(x=1.4, y=-1.6, z=0.7),
                up=dict(x=0, y=0, z=1),
            ),
            aspectmode="manual",
            aspectratio=dict(x=1.2, y=1.5, z=0.55),
        ),
    )

    return fig
