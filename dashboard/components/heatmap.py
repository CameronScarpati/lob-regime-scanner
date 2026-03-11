"""Panel 1: Bookmap-style LOB heatmap with regime overlay.

Renders a heatmap of resting volume (price vs time), a mid-price line,
and color-coded regime overlay bands at the top.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard._constants import (
    AXIS_STYLE,
    PLOTLY_LAYOUT_DEFAULTS,
    REGIME_COLORS,
    REGIME_NAMES,
    XAXIS_STYLE,
)


def _build_volume_matrix(
    snapshots: pd.DataFrame,
    n_levels: int = 10,
    max_time_steps: int = 2000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a 2-D volume matrix (price_levels x time) from snapshot data.

    Returns (price_axis, time_axis, volume_matrix).
    Subsamples to *max_time_steps* if the data is larger.
    """
    # Subsample for performance
    if len(snapshots) > max_time_steps:
        step = len(snapshots) // max_time_steps
        snapshots = snapshots.iloc[::step].reset_index(drop=True)

    n_t = len(snapshots)

    bid_prices = np.column_stack(
        [snapshots[f"bid_price_{i}"].values for i in range(1, n_levels + 1)]
    )
    bid_vols = np.column_stack([snapshots[f"bid_qty_{i}"].values for i in range(1, n_levels + 1)])
    ask_prices = np.column_stack(
        [snapshots[f"ask_price_{i}"].values for i in range(1, n_levels + 1)]
    )
    ask_vols = np.column_stack([snapshots[f"ask_qty_{i}"].values for i in range(1, n_levels + 1)])

    all_prices = np.concatenate([bid_prices.ravel(), ask_prices.ravel()])
    p_min, p_max = np.nanpercentile(all_prices, [1, 99])
    n_price_bins = 80
    price_axis = np.linspace(p_min, p_max, n_price_bins)
    bin_width = price_axis[1] - price_axis[0]

    volume_matrix = np.zeros((n_price_bins, n_t))

    for t in range(n_t):
        for lvl in range(n_levels):
            bp = bid_prices[t, lvl]
            idx = int((bp - p_min) / bin_width)
            if 0 <= idx < n_price_bins:
                volume_matrix[idx, t] += bid_vols[t, lvl]
            ap = ask_prices[t, lvl]
            idx = int((ap - p_min) / bin_width)
            if 0 <= idx < n_price_bins:
                volume_matrix[idx, t] += ask_vols[t, lvl]

    return price_axis, snapshots["timestamp"].values, volume_matrix


def create_heatmap_figure(
    snapshots: pd.DataFrame,
    regimes: np.ndarray,
) -> go.Figure:
    """Create the Bookmap-style LOB heatmap with regime overlay.

    Parameters
    ----------
    snapshots : DataFrame with book_reconstructor schema.
    regimes : 1-D array of regime labels (0, 1, 2) aligned to snapshots.
    """
    price_axis, time_axis, vol_matrix = _build_volume_matrix(snapshots)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.05, 0.95],
        vertical_spacing=0.012,
    )

    # --- Regime overlay band (top strip) ---
    for regime_id, color in REGIME_COLORS.items():
        mask = regimes == regime_id
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=np.where(mask, 1, 0),
                fill="tozeroy",
                fillcolor=color,
                line=dict(width=0),
                opacity=0.75,
                name=REGIME_NAMES[regime_id],
                showlegend=True,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

    # --- LOB heatmap ---
    # Professional colorscale: dark base -> deep blue -> teal -> warm gold
    bookmap_colorscale = [
        [0.00, "rgba(12,16,22,1)"],
        [0.04, "rgba(10,28,54,1)"],
        [0.12, "rgba(12,50,100,1)"],
        [0.25, "rgba(0,95,130,1)"],
        [0.42, "rgba(0,152,115,1)"],
        [0.60, "rgba(140,165,50,1)"],
        [0.80, "rgba(220,185,50,1)"],
        [1.00, "rgba(255,248,210,1)"],
    ]

    fig.add_trace(
        go.Heatmap(
            x=time_axis,
            y=price_axis,
            z=vol_matrix,
            colorscale=bookmap_colorscale,
            zsmooth="best",
            colorbar=dict(
                title=dict(
                    text="Volume",
                    font=dict(size=11, color="#98a2ae"),
                    side="right",
                ),
                len=0.78,
                y=0.42,
                thickness=10,
                tickfont=dict(size=10, color="#7a8490"),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
                outlinewidth=0,
            ),
            showlegend=False,
            hovertemplate=("Time: %{x}<br>Price: $%{y:,.2f}<br>Volume: %{z:.2f}<extra></extra>"),
        ),
        row=2,
        col=1,
    )

    # --- Mid-price line ---
    fig.add_trace(
        go.Scatter(
            x=time_axis,
            y=snapshots["mid_price"].values,
            mode="lines",
            line=dict(color="rgba(255,255,255,0.60)", width=1.3, dash="dot"),
            name="Mid Price",
            showlegend=True,
            hovertemplate="Mid: $%{y:,.2f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # --- Large trade markers ---
    trade_sizes = snapshots["last_trade_qty"].values
    finite_sizes = trade_sizes[np.isfinite(trade_sizes)]
    if len(finite_sizes) > 0:
        threshold = np.percentile(finite_sizes, 90)
        large_mask = trade_sizes > threshold
    else:
        large_mask = np.zeros(len(trade_sizes), dtype=bool)
    if large_mask.any():
        colors = [
            "#4CAF82" if s == "buy" else "#EF6C6C"
            for s in snapshots.loc[large_mask, "last_trade_side"]
        ]
        fig.add_trace(
            go.Scatter(
                x=time_axis[large_mask],
                y=snapshots.loc[large_mask, "last_trade_price"].values,
                mode="markers",
                marker=dict(
                    size=np.clip(trade_sizes[large_mask] * 5, 4, 11),
                    color=colors,
                    opacity=0.85,
                    line=dict(width=0.6, color="rgba(255,255,255,0.35)"),
                    symbol="diamond",
                ),
                name="Large Trades",
                showlegend=True,
                hovertemplate=("Trade: $%{y:,.2f}<br>Size: %{marker.size:.1f}<extra></extra>"),
            ),
            row=2,
            col=1,
        )

    fig.update_yaxes(
        row=1,
        col=1,
        showticklabels=False,
        showgrid=False,
        zeroline=False,
    )
    fig.update_yaxes(
        title_text="Price (USD)",
        row=2,
        col=1,
        **AXIS_STYLE,
    )
    fig.update_xaxes(
        row=1,
        col=1,
        showticklabels=False,
        showgrid=False,
    )
    fig.update_xaxes(
        title_text="",
        row=2,
        col=1,
        **XAXIS_STYLE,
    )

    fig.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        height=540,
        margin=dict(l=60, r=20, t=8, b=36),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#98a2ae"),
            bgcolor="rgba(0,0,0,0)",
            itemsizing="constant",
            tracegroupgap=4,
        ),
    )

    return fig
