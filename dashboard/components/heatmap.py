"""Panel 1: Bookmap-style LOB heatmap with regime overlay.

Renders a heatmap of resting volume (price vs time), a mid-price line,
and color-coded regime overlay bands at the top.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard._constants import REGIME_COLORS, REGIME_NAMES


def _build_volume_matrix(
    snapshots: pd.DataFrame,
    n_levels: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a 2-D volume matrix (price_levels x time) from snapshot data.

    Returns (price_axis, time_axis, volume_matrix).
    """
    n_t = len(snapshots)

    bid_prices = np.column_stack(
        [snapshots[f"bid_price_{i}"].values for i in range(1, n_levels + 1)]
    )
    bid_vols = np.column_stack(
        [snapshots[f"bid_qty_{i}"].values for i in range(1, n_levels + 1)]
    )
    ask_prices = np.column_stack(
        [snapshots[f"ask_price_{i}"].values for i in range(1, n_levels + 1)]
    )
    ask_vols = np.column_stack(
        [snapshots[f"ask_qty_{i}"].values for i in range(1, n_levels + 1)]
    )

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
        row_heights=[0.08, 0.92],
        vertical_spacing=0.02,
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
                opacity=0.7,
                name=REGIME_NAMES[regime_id],
                showlegend=True,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

    # --- LOB heatmap ---
    fig.add_trace(
        go.Heatmap(
            x=time_axis,
            y=price_axis,
            z=vol_matrix,
            colorscale="Inferno",
            zsmooth="best",
            colorbar=dict(title="Volume", len=0.85, y=0.42),
            showlegend=False,
            hovertemplate=(
                "Time: %{x}<br>Price: $%{y:,.2f}<br>Volume: %{z:.2f}"
                "<extra></extra>"
            ),
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
            line=dict(color="#f1c40f", width=1.5),
            name="Mid Price",
            showlegend=True,
            hovertemplate="Mid: $%{y:,.2f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # --- Large trade markers ---
    trade_sizes = snapshots["last_trade_qty"].values
    large_mask = trade_sizes > np.percentile(trade_sizes, 90)
    if large_mask.any():
        colors = [
            "#2ecc71" if s == "buy" else "#e74c3c"
            for s in snapshots.loc[large_mask, "last_trade_side"]
        ]
        fig.add_trace(
            go.Scatter(
                x=time_axis[large_mask],
                y=snapshots.loc[large_mask, "last_trade_price"].values,
                mode="markers",
                marker=dict(
                    size=np.clip(trade_sizes[large_mask] * 5, 3, 12),
                    color=colors,
                    opacity=0.7,
                    line=dict(width=0.5, color="white"),
                ),
                name="Large Trades",
                showlegend=True,
                hovertemplate=(
                    "Trade: %{y:,.2f}<br>Qty: %{marker.size:.1f}"
                    "<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

    fig.update_yaxes(title_text="Regime", row=1, col=1, showticklabels=False)
    fig.update_yaxes(title_text="Price (USD)", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)

    fig.update_layout(
        title="Order Book Heatmap with Regime Overlay",
        template="plotly_dark",
        height=520,
        margin=dict(l=60, r=20, t=40, b=40),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
    )

    return fig
