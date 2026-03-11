"""Dash callbacks for dashboard interactivity.

Handles time range selection and regime filtering via DMC components.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dash import Dash, Input, Output, State, callback_context, no_update

from dashboard.components.heatmap import create_heatmap_figure
from dashboard.components.regime_probs import create_regime_probs_figure
from dashboard.components.depth_surface import create_depth_surface_figure
from dashboard.components.diagnostics import create_diagnostics_figure


def register_callbacks(app: Dash, data: dict | None = None) -> None:
    """Register all Dash callbacks on the given app instance.

    Parameters
    ----------
    app : Dash
        The Dash application.
    data : dict or None
        Pipeline output dict with keys: snapshots, features, hmm, cumulative_pnl.
        If None, falls back to mock data for backwards compatibility.
    """
    if data is None:
        from dashboard._mock_data import generate_all
        data = generate_all(n_timestamps=3600)

    _data = data

    @app.callback(
        [
            Output("heatmap-panel", "figure"),
            Output("regime-probs-panel", "figure"),
            Output("depth-surface-panel", "figure"),
            Output("diagnostics-panel", "figure"),
        ],
        [
            Input("time-start-select", "value"),
            Input("time-end-select", "value"),
            Input("regime-chip-group", "value"),
        ],
    )
    def update_panels(start_val, end_val, active_regimes):
        snapshots = _data["snapshots"]
        features = _data["features"]
        hmm = _data["hmm"]
        cum_pnl = _data["cumulative_pnl"]

        # Values are string indices from Select options
        start_idx = int(start_val) if start_val is not None else 0
        end_idx = int(end_val) if end_val is not None else len(snapshots) - 1
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx

        sl = slice(start_idx, end_idx + 1)

        snap_sub = snapshots.iloc[sl].reset_index(drop=True)
        feat_sub = features.iloc[sl].reset_index(drop=True)
        states_sub = hmm["states"][sl]
        probs_sub = hmm["state_probs"][sl]
        pnl_sub = cum_pnl[sl]
        timestamps_sub = feat_sub["timestamp"].values

        display_states = states_sub.copy()

        heatmap_fig = create_heatmap_figure(snap_sub, display_states)
        regime_fig = create_regime_probs_figure(
            timestamps_sub, probs_sub, hmm["transition_matrix"]
        )
        depth_fig = create_depth_surface_figure(snap_sub, display_states)
        diag_fig = create_diagnostics_figure(feat_sub, display_states, pnl_sub)

        return heatmap_fig, regime_fig, depth_fig, diag_fig
