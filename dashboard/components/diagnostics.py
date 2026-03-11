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
    REGIME_NAMES,
)


def _add_regime_backgrounds(
    fig: go.Figure,
    timestamps: np.ndarray,
    regimes: np.ndarray,
    row: int,
    col: int = 1,
    max_vrects: int = 100,
) -> None:
    """Add semi-transparent regime-colored background rectangles to a subplot.

    Uses vrect shapes for a small number of segments. When there are too many
    regime transitions, skips background coloring for performance.
    """
    if len(regimes) == 0:
        return

    changes = np.where(np.diff(regimes) != 0)[0] + 1
    n_segments = len(changes) + 1

    if n_segments > max_vrects:
        return  # too many transitions — skip for performance

    starts = np.concatenate([[0], changes])
    ends = np.concatenate([changes, [len(regimes)]])

    for s, e in zip(starts, ends):
        regime = regimes[s]
        fig.add_vrect(
            x0=timestamps[s],
            x1=timestamps[min(e, len(timestamps) - 1)],
            fillcolor=REGIME_COLORS[int(regime)],
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
        vertical_spacing=0.05,
        subplot_titles=("VPIN", "OFI (Normalized)", "Spread (bps)", "Cumulative PnL"),
        row_heights=[0.25, 0.25, 0.22, 0.28],
    )

    # Style subplot titles
    for annotation in fig.layout.annotations:
        annotation.font = dict(size=11, color="#6e7681")
        annotation.xanchor = "left"
        annotation.x = 0.01

    # --- Row 1: VPIN ---
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=features["VPIN"].values,
            mode="lines",
            line=dict(color="#e6a817", width=1.0),
            name="VPIN",
            hovertemplate="VPIN: %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="#ff5252",
        line_width=0.8,
        opacity=0.5,
        annotation_text="Alert",
        annotation_font_size=9,
        annotation_font_color="#ff5252",
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
            line=dict(color="#448aff", width=1.0),
            name="OFI",
            hovertemplate="OFI: %{y:.3f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    # Add a zero line for reference
    fig.add_hline(
        y=0,
        line_dash="solid",
        line_color="rgba(255,255,255,0.1)",
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
                opacity=0.6,
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
            line=dict(color="#ab47bc", width=1.0),
            name="Spread",
            fill="tozeroy",
            fillcolor="rgba(171,71,188,0.08)",
            hovertemplate="Spread: %{y:.2f} bps<extra></extra>",
        ),
        row=3,
        col=1,
    )
    _add_regime_backgrounds(fig, timestamps, regimes, row=3)

    # --- Row 4: Cumulative PnL ---
    # Color the fill based on whether PnL is positive or negative
    pnl_color = "#00c853" if cumulative_pnl[-1] >= 0 else "#ff5252"
    pnl_fill = (
        "rgba(0,200,83,0.12)" if cumulative_pnl[-1] >= 0 else "rgba(255,82,82,0.12)"
    )
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=cumulative_pnl,
            mode="lines",
            line=dict(color=pnl_color, width=1.2),
            fill="tozeroy",
            fillcolor=pnl_fill,
            name="Cum. PnL",
            hovertemplate="PnL: %{y:.4f}<extra></extra>",
        ),
        row=4,
        col=1,
    )
    # Zero line for PnL
    fig.add_hline(
        y=0,
        line_dash="solid",
        line_color="rgba(255,255,255,0.12)",
        line_width=0.5,
        row=4,
        col=1,
    )
    _add_regime_backgrounds(fig, timestamps, regimes, row=4)

    # Apply consistent axis styles
    for row_num in range(1, 5):
        fig.update_yaxes(row=row_num, col=1, **AXIS_STYLE)
        fig.update_xaxes(row=row_num, col=1, **AXIS_STYLE)

    fig.update_yaxes(title_text="VPIN", row=1, col=1)
    fig.update_yaxes(title_text="OFI", row=2, col=1)
    fig.update_yaxes(title_text="bps", row=3, col=1)
    fig.update_yaxes(title_text="PnL", row=4, col=1)
    fig.update_xaxes(title_text="", row=4, col=1)

    fig.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        height=560,
        margin=dict(l=55, r=16, t=24, b=32),
        showlegend=False,
    )

    return fig
