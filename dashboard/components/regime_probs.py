"""Panel 2: HMM regime state probabilities (stacked area chart) and transition matrix."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard._constants import REGIME_COLORS, REGIME_NAMES


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
        column_widths=[0.70, 0.30],
        horizontal_spacing=0.08,
        subplot_titles=("Regime Posterior Probabilities", "Transition Matrix"),
    )

    # --- Stacked area chart ---
    for state_id in range(3):
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=state_probs[:, state_id],
                mode="lines",
                line=dict(width=0.5, color=REGIME_COLORS[state_id]),
                fillcolor=REGIME_COLORS[state_id],
                stackgroup="one",
                name=REGIME_NAMES[state_id],
                hovertemplate=(
                    f"{REGIME_NAMES[state_id]}: "
                    "%{y:.2%}<extra></extra>"
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

    fig.add_trace(
        go.Heatmap(
            z=transition_matrix,
            x=state_labels,
            y=state_labels,
            text=text_matrix,
            texttemplate="%{text}",
            colorscale=[
                [0.0, "#1a1a2e"],
                [0.5, "#16537e"],
                [1.0, "#e74c3c"],
            ],
            showscale=False,
            hovertemplate=(
                "From: %{y}<br>To: %{x}<br>P: %{z:.3f}<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )

    fig.update_yaxes(title_text="Probability", range=[0, 1], row=1, col=1)
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=2)

    fig.update_layout(
        template="plotly_dark",
        height=350,
        margin=dict(l=60, r=20, t=40, b=40),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=0.68
        ),
    )

    return fig
