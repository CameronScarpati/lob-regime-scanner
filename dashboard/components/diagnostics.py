"""Panel 4: Toxicity diagnostics — VPIN, OFI, spread, and cumulative PnL.

Four vertically stacked subplots with regime-colored backgrounds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard._constants import REGIME_COLORS, REGIME_NAMES


def _add_regime_backgrounds(
    fig: go.Figure,
    timestamps: np.ndarray,
    regimes: np.ndarray,
    row: int,
    col: int = 1,
) -> None:
    """Add semi-transparent regime-colored background rectangles to a subplot."""
    if len(regimes) == 0:
        return

    changes = np.where(np.diff(regimes) != 0)[0] + 1
    starts = np.concatenate([[0], changes])
    ends = np.concatenate([changes, [len(regimes)]])

    for s, e in zip(starts, ends):
        regime = regimes[s]
        fig.add_vrect(
            x0=timestamps[s],
            x1=timestamps[min(e, len(timestamps) - 1)],
            fillcolor=REGIME_COLORS[int(regime)],
            opacity=0.10,
            layer="below",
            line_width=0,
            row=row,
            col=col,
        )


def create_diagnostics_figure(
    features: pd.DataFrame,
    regimes: np.ndarray,
    cumulative_pnl: np.ndarray,
) -> go.Figure:
    """Create the diagnostics multi-subplot panel.

    Parameters
    ----------
    features : DataFrame with feature columns (VPIN, OFI_1, spread_bps, etc.).
    regimes : 1-D array of regime labels aligned to features.
    cumulative_pnl : 1-D array of cumulative PnL values.
    """
    timestamps = features["timestamp"].values

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("VPIN", "OFI (Normalized)", "Spread (bps)", "Cumulative PnL"),
    )

    # --- Row 1: VPIN ---
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=features["VPIN"].values,
            mode="lines",
            line=dict(color="#e67e22", width=1.2),
            name="VPIN",
            hovertemplate="VPIN: %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="#e74c3c",
        opacity=0.6,
        row=1,
        col=1,
    )
    _add_regime_backgrounds(fig, timestamps, regimes, row=1)

    # --- Row 2: OFI ---
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=features["OFI_1"].values,
            mode="lines",
            line=dict(color="#3498db", width=1.2),
            name="OFI",
            hovertemplate="OFI: %{y:.3f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    for regime_id, color in REGIME_COLORS.items():
        mask = regimes == regime_id
        if mask.any():
            mean_val = features.loc[mask, "OFI_1"].mean()
            fig.add_hline(
                y=mean_val,
                line_dash="dot",
                line_color=color,
                opacity=0.7,
                row=2,
                col=1,
            )
    _add_regime_backgrounds(fig, timestamps, regimes, row=2)

    # --- Row 3: Spread ---
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=features["spread_bps"].values,
            mode="lines",
            line=dict(color="#9b59b6", width=1.2),
            name="Spread (bps)",
            hovertemplate="Spread: %{y:.2f} bps<extra></extra>",
        ),
        row=3,
        col=1,
    )
    _add_regime_backgrounds(fig, timestamps, regimes, row=3)

    # --- Row 4: Cumulative PnL ---
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=cumulative_pnl,
            mode="lines",
            line=dict(color="#2ecc71", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(46, 204, 113, 0.15)",
            name="Cum. PnL",
            hovertemplate="PnL: %{y:.4f}<extra></extra>",
        ),
        row=4,
        col=1,
    )
    _add_regime_backgrounds(fig, timestamps, regimes, row=4)

    fig.update_yaxes(title_text="VPIN", row=1, col=1)
    fig.update_yaxes(title_text="OFI", row=2, col=1)
    fig.update_yaxes(title_text="bps", row=3, col=1)
    fig.update_yaxes(title_text="PnL", row=4, col=1)
    fig.update_xaxes(title_text="Time", row=4, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=520,
        margin=dict(l=60, r=20, t=30, b=40),
        showlegend=False,
    )

    return fig
