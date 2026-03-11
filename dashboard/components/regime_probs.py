"""Panel 2: HMM regime state probabilities (stacked area chart) and transition matrix."""

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


def create_regime_probs_figure(
    timestamps: pd.DatetimeIndex | np.ndarray,
    state_probs: np.ndarray,
    transition_matrix: np.ndarray,
) -> go.Figure:
    """Create the regime probability stacked area chart + transition matrix.

    Parameters
    ----------
    timestamps : array-like of datetime values.
    state_probs : (T, 3) posterior probabilities.
    transition_matrix : (3, 3) learned transition probabilities.
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.68, 0.32],
        horizontal_spacing=0.10,
        subplot_titles=("Regime Posterior Probabilities", "Transition Matrix"),
    )

    # --- Stacked area chart ---
    # Muted fill colors with stronger edge lines
    fill_opacity = 0.55
    for state_id in range(3):
        base_color = REGIME_COLORS[state_id]
        # Convert hex to rgba for fill
        r, g, b = int(base_color[1:3], 16), int(base_color[3:5], 16), int(base_color[5:7], 16)
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=state_probs[:, state_id],
                mode="lines",
                line=dict(width=0.8, color=base_color),
                fillcolor=f"rgba({r},{g},{b},{fill_opacity})",
                stackgroup="one",
                name=REGIME_NAMES[state_id],
                hovertemplate=(
                    f"{REGIME_NAMES[state_id]}: "
                    "%{y:.1%}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    # --- Transition matrix heatmap ---
    state_labels = [REGIME_NAMES[i] for i in range(3)]
    text_matrix = [
        [f"{transition_matrix[i, j]:.3f}" for j in range(3)] for i in range(3)
    ]

    # Custom colorscale: dark navy -> muted blue -> bright for high values
    tm_colorscale = [
        [0.0, "#0c1620"],
        [0.3, "#122a45"],
        [0.6, "#1a5276"],
        [0.8, "#2e86c1"],
        [1.0, "#5dade2"],
    ]

    fig.add_trace(
        go.Heatmap(
            z=transition_matrix,
            x=state_labels,
            y=state_labels,
            text=text_matrix,
            texttemplate="%{text}",
            textfont=dict(size=12, color="#d4d8df"),
            colorscale=tm_colorscale,
            showscale=False,
            xgap=2,
            ygap=2,
            hovertemplate=(
                "From: %{y}<br>To: %{x}<br>P: %{z:.3f}<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )

    fig.update_yaxes(
        title_text="Probability", range=[0, 1], row=1, col=1,
        **AXIS_STYLE,
    )
    fig.update_xaxes(
        title_text="", row=1, col=1,
        **AXIS_STYLE,
    )
    fig.update_yaxes(
        autorange="reversed", row=1, col=2,
        tickfont=dict(size=10, color="#8b949e"),
    )
    fig.update_xaxes(
        row=1, col=2,
        tickfont=dict(size=10, color="#8b949e"),
        side="bottom",
    )

    # Style the subplot titles
    for annotation in fig.layout.annotations:
        annotation.font = dict(size=12, color="#8b949e")

    fig.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        height=560,
        margin=dict(l=55, r=16, t=36, b=32),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=0.66,
            font=dict(size=10, color="#8b949e"),
            bgcolor="rgba(0,0,0,0)",
        ),
    )

    return fig
