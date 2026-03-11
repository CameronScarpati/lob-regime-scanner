"""Panel 4: Toxicity diagnostics — VPIN, OFI, spread, and cumulative PnL.

Four vertically stacked subplots with regime-colored backgrounds.
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
    REGIME_FILLS,
    REGIME_NAMES,
    XAXIS_STYLE,
)


def _add_regime_backgrounds(
    fig: go.Figure,
    timestamps: np.ndarray,
    regimes: np.ndarray,
    row: int,
    col: int = 1,
    max_vrects: int = 100,
) -> None:
    """Add semi-transparent regime-colored background rectangles to a subplot."""
    if len(regimes) == 0:
        return

    changes = np.where(np.diff(regimes) != 0)[0] + 1
    n_segments = len(changes) + 1

    if n_segments > max_vrects:
        return  # too many transitions — skip for performance

    starts = np.concatenate([[0], changes])
    ends = np.concatenate([changes, [len(regimes)]])

    for s, e in zip(starts, ends):
        regime = int(regimes[s])
        fig.add_vrect(
            x0=timestamps[s],
            x1=timestamps[min(e, len(timestamps) - 1)],
            fillcolor=REGIME_COLORS[regime],
            opacity=0.07,
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
        vertical_spacing=0.055,
        subplot_titles=(
            "VPIN  (Volume-Synchronised Probability of Informed Trading)",
            "Order Flow Imbalance  (Normalised)",
            "Quoted Spread  (basis points)",
            "Cumulative Strategy PnL",
        ),
        row_heights=[0.25, 0.25, 0.22, 0.28],
    )

    # Style subplot titles — left-aligned, readable
    for annotation in fig.layout.annotations:
        annotation.font = dict(size=12, color="#7a8490")
        annotation.xanchor = "left"
        annotation.x = 0.01

    # --- Row 1: VPIN ---
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=features["VPIN"].values,
            mode="lines",
            line=dict(color="#E6A817", width=1.2),
            name="VPIN",
            hovertemplate="VPIN: %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="#EF6C6C",
        line_width=0.8,
        opacity=0.50,
        annotation_text="Alert threshold",
        annotation_font_size=10,
        annotation_font_color="#EF6C6C",
        annotation_position="top right",
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
            line=dict(color="#5C9CF5", width=1.2),
            name="OFI",
            hovertemplate="OFI: %{y:.3f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_hline(
        y=0,
        line_dash="solid",
        line_color="rgba(255,255,255,0.10)",
        line_width=0.5,
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
                line_width=0.8,
                opacity=0.55,
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
            line=dict(color="#AB6DD6", width=1.2),
            name="Spread",
            fill="tozeroy",
            fillcolor="rgba(171,109,214,0.08)",
            hovertemplate="Spread: %{y:.2f} bps<extra></extra>",
        ),
        row=3,
        col=1,
    )
    _add_regime_backgrounds(fig, timestamps, regimes, row=3)

    # --- Row 4: Cumulative PnL ---
    pnl_color = "#4CAF82" if cumulative_pnl[-1] >= 0 else "#EF6C6C"
    pnl_fill = (
        "rgba(76,175,130,0.10)" if cumulative_pnl[-1] >= 0
        else "rgba(239,108,108,0.10)"
    )
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=cumulative_pnl,
            mode="lines",
            line=dict(color=pnl_color, width=1.3),
            fill="tozeroy",
            fillcolor=pnl_fill,
            name="Cum. PnL",
            hovertemplate="PnL: %{y:.4f}<extra></extra>",
        ),
        row=4,
        col=1,
    )
    fig.add_hline(
        y=0,
        line_dash="solid",
        line_color="rgba(255,255,255,0.10)",
        line_width=0.5,
        row=4,
        col=1,
    )
    _add_regime_backgrounds(fig, timestamps, regimes, row=4)

    # Apply consistent axis styles
    for row_num in range(1, 5):
        fig.update_yaxes(row=row_num, col=1, **AXIS_STYLE)
        fig.update_xaxes(row=row_num, col=1, **XAXIS_STYLE)

    fig.update_yaxes(title_text="VPIN", row=1, col=1)
    fig.update_yaxes(title_text="OFI", row=2, col=1)
    fig.update_yaxes(title_text="bps", row=3, col=1)
    fig.update_yaxes(title_text="PnL", row=4, col=1)
    fig.update_xaxes(title_text="", row=4, col=1)

    fig.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        height=540,
        margin=dict(l=60, r=20, t=24, b=36),
        showlegend=False,
    )

    return fig
