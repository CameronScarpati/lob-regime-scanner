"""Panel 3: 3D order book depth surface visualization.

Renders a 3-D surface (time x price x volume) colored by regime state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dashboard._constants import REGIME_COLORS


def _build_depth_grid(
    snapshots: pd.DataFrame,
    n_levels: int = 10,
    n_time_samples: int = 120,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build 2-D grids for the 3-D surface.

    Returns (time_indices, price_offsets, volume_grid).
    price_offsets are relative to mid-price so the surface is centered.
    """
    step = max(1, len(snapshots) // n_time_samples)
    sub = snapshots.iloc[::step].reset_index(drop=True)
    n_t = len(sub)

    n_bins = 2 * n_levels
    price_offsets = np.linspace(-n_levels * 0.5, n_levels * 0.5, n_bins)

    volume_grid = np.zeros((n_t, n_bins))

    for t_idx in range(n_t):
        mid = sub.loc[t_idx, "mid_price"]
        for lvl in range(1, n_levels + 1):
            bp = sub.loc[t_idx, f"bid_price_{lvl}"]
            bv = sub.loc[t_idx, f"bid_qty_{lvl}"]
            offset = bp - mid
            bin_idx = int(
                (offset - price_offsets[0])
                / (price_offsets[-1] - price_offsets[0])
                * (n_bins - 1)
            )
            if 0 <= bin_idx < n_bins:
                volume_grid[t_idx, bin_idx] += bv

            ap = sub.loc[t_idx, f"ask_price_{lvl}"]
            av = sub.loc[t_idx, f"ask_qty_{lvl}"]
            offset = ap - mid
            bin_idx = int(
                (offset - price_offsets[0])
                / (price_offsets[-1] - price_offsets[0])
                * (n_bins - 1)
            )
            if 0 <= bin_idx < n_bins:
                volume_grid[t_idx, bin_idx] += av

    time_indices = np.arange(n_t)
    return time_indices, price_offsets, volume_grid


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
    time_idx, price_offsets, vol_grid = _build_depth_grid(snapshots)

    step = max(1, len(regimes) // len(time_idx))
    sub_regimes = regimes[::step][: len(time_idx)]

    regime_surface = np.tile(sub_regimes.reshape(-1, 1), (1, len(price_offsets)))

    colorscale = [
        [0.0, REGIME_COLORS[0]],
        [0.5, REGIME_COLORS[1]],
        [1.0, REGIME_COLORS[2]],
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Surface(
            x=price_offsets,
            y=time_idx,
            z=vol_grid,
            surfacecolor=regime_surface,
            colorscale=colorscale,
            cmin=0,
            cmax=2,
            opacity=0.85,
            showscale=False,
            hovertemplate=(
                "Price Offset: %{x:.2f}<br>"
                "Time Step: %{y}<br>"
                "Volume: %{z:.2f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="3D Order Book Depth Surface",
        template="plotly_dark",
        height=480,
        margin=dict(l=10, r=10, t=40, b=10),
        scene=dict(
            xaxis_title="Price Offset from Mid",
            yaxis_title="Time",
            zaxis_title="Volume",
            camera=dict(eye=dict(x=1.5, y=-1.8, z=0.8)),
            bgcolor="#111111",
        ),
    )

    return fig
