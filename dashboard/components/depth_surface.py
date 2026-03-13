"""Panel 3: 3D order book depth surface visualization.

Renders a single continuous 3-D surface (time x price offset x volume)
colored by signed volume — green for bids, red for asks, with intensity
proportional to resting depth.
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
    """Build a unified volume grid and signed-volume grid for coloring.

    Returns (time_indices, price_offsets, volume_grid, signed_volume).
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

    for t_idx in range(n_t):
        row = sub.iloc[t_idx]
        for lvl in range(1, n_levels + 1):
            bid_bin = n_levels - lvl
            volume_grid[t_idx, bid_bin] = row[f"bid_qty_{lvl}"]

            ask_bin = n_levels + lvl - 1
            volume_grid[t_idx, ask_bin] = row[f"ask_qty_{lvl}"]

    # Light smoothing — just enough to remove 1-tick jitter
    sigma_t = max(0.6, n_t / 200)
    volume_grid = gaussian_filter(volume_grid, sigma=[sigma_t, 0.4])

    # Signed volume: negative for bids, positive for asks.
    # This gives smooth color gradients that follow the actual volume shape.
    signed_vol = volume_grid.copy()
    signed_vol[:, :n_levels] *= -1.0

    time_indices = np.arange(n_t)
    return time_indices, price_offsets, volume_grid, signed_vol


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
    time_idx, price_offsets, vol_grid, signed_vol = _build_depth_grid(snapshots)

    # Diverging colorscale: green (bids, negative) → white center → red (asks, positive)
    colorscale = [
        [0.00, "#16a34a"],
        [0.30, "#4ade80"],
        [0.45, "#bbf7d0"],
        [0.50, "#e2e8f0"],
        [0.55, "#fecaca"],
        [0.70, "#f87171"],
        [1.00, "#dc2626"],
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Surface(
            x=price_offsets,
            y=time_idx,
            z=vol_grid,
            surfacecolor=signed_vol,
            colorscale=colorscale,
            cmid=0,
            showscale=False,
            lighting=dict(
                ambient=0.5,
                diffuse=0.6,
                specular=0.3,
                roughness=0.4,
                fresnel=0.2,
            ),
            lightposition=dict(x=0, y=0, z=1000),
            contours=dict(
                z=dict(
                    show=True,
                    usecolormap=False,
                    color="rgba(255,255,255,0.08)",
                    width=1,
                    highlightcolor="rgba(255,255,255,0.15)",
                ),
            ),
            hovertemplate=(
                "Level: %{x:.0f}<br>"
                "Time: %{y}<br>"
                "Volume: %{z:.2f}<extra></extra>"
            ),
        )
    )

    # Clean scene styling
    _axis = dict(
        backgroundcolor="rgba(0,0,0,0)",
        gridcolor="rgba(255,255,255,0.06)",
        showbackground=True,
        backgroundcolor_='#0f1724',
        tickfont=dict(size=10, color="#94a3b8"),
        title_font=dict(size=11, color="#94a3b8"),
        showspikes=False,
    )
    # backgroundcolor_ is not a real key — use the correct one
    _axis_clean = dict(
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
            xaxis=dict(
                title="\u2190 Bids | Asks \u2192",
                nticks=8,
                **_axis_clean,
            ),
            yaxis=dict(
                title="Time",
                nticks=6,
                **_axis_clean,
            ),
            zaxis=dict(
                title="Resting Volume",
                nticks=5,
                **_axis_clean,
            ),
            camera=dict(
                eye=dict(x=1.4, y=-1.6, z=0.7),
                up=dict(x=0, y=0, z=1),
            ),
            aspectmode="manual",
            aspectratio=dict(x=1.2, y=1.5, z=0.55),
        ),
    )

    return fig
