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
        column_widths=[0.66, 0.34],
        horizontal_spacing=0.10,
        subplot_titles=(
            "Posterior Probabilities",
            "Transition Matrix",
        ),
    )

    # --- Stacked area chart ---
    fill_opacity = 0.50
    for state_id in range(3):
        base_color = REGIME_COLORS[state_id]
        r, g, b = int(base_color[1:3], 16), int(base_color[3:5], 16), int(base_color[5:7], 16)
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=state_probs[:, state_id],
                mode="lines",
                line=dict(width=0.9, color=base_color),
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

    # Clean monochrome colorscale for the matrix
    tm_colorscale = [
        [0.0, "#0c1620"],
        [0.25, "#132c48"],
        [0.50, "#1a4a72"],
        [0.75, "#2a7ab0"],
        [1.0, "#5AAFE6"],
    ]

    fig.add_trace(
        go.Heatmap(
            z=transition_matrix,
            x=state_labels,
            y=state_labels,
            text=text_matrix,
            texttemplate="%{text}",
            textfont=dict(size=13, color="#e0e4ea", family="monospace"),
            colorscale=tm_colorscale,
            showscale=False,
            xgap=3,
            ygap=3,
            hovertemplate=(
                "From: %{y}<br>To: %{x}<br>P = %{z:.3f}<extra></extra>"
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
        tickfont=dict(size=11, color="#98a2ae"),
    )
    fig.update_xaxes(
        row=1, col=2,
        tickfont=dict(size=11, color="#98a2ae"),
        side="bottom",
    )

    # Style the subplot titles
    for annotation in fig.layout.annotations:
        annotation.font = dict(size=13, color="#98a2ae")

    fig.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        height=540,
        margin=dict(l=60, r=20, t=36, b=36),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=0.64,
            font=dict(size=11, color="#98a2ae"),
            bgcolor="rgba(0,0,0,0)",
            itemsizing="constant",
        ),
    )

    return fig
