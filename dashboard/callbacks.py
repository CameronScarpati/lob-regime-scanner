"""Dash callbacks for dashboard interactivity.

Handles synchronized crosshairs, date range selection, regime filtering,
and play/pause animation controls.
"""

from __future__ import annotations

import numpy as np
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
            Input("date-range-slider", "value"),
            Input("regime-quiet-btn", "n_clicks"),
            Input("regime-trending-btn", "n_clicks"),
            Input("regime-toxic-btn", "n_clicks"),
        ],
        [
            State("regime-quiet-btn", "className"),
            State("regime-trending-btn", "className"),
            State("regime-toxic-btn", "className"),
        ],
    )
    def update_panels(
        date_range,
        quiet_clicks,
        trending_clicks,
        toxic_clicks,
        quiet_cls,
        trending_cls,
        toxic_cls,
    ):
        snapshots = _data["snapshots"]
        features = _data["features"]
        hmm = _data["hmm"]
        cum_pnl = _data["cumulative_pnl"]

        # Apply date range slider
        start_idx, end_idx = date_range if date_range else (0, len(snapshots) - 1)
        sl = slice(start_idx, end_idx + 1)

        snap_sub = snapshots.iloc[sl].reset_index(drop=True)
        feat_sub = features.iloc[sl].reset_index(drop=True)
        states_sub = hmm["states"][sl]
        probs_sub = hmm["state_probs"][sl]
        pnl_sub = cum_pnl[sl]
        timestamps_sub = feat_sub["timestamp"].values

        # Determine which regimes are active from button toggle state
        ctx = callback_context
        active_regimes = {0, 1, 2}

        if quiet_cls and "inactive" in quiet_cls:
            active_regimes.discard(0)
        if trending_cls and "inactive" in trending_cls:
            active_regimes.discard(1)
        if toxic_cls and "inactive" in toxic_cls:
            active_regimes.discard(2)

        # Handle button clicks to toggle
        if ctx.triggered:
            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
            toggle_map = {
                "regime-quiet-btn": 0,
                "regime-trending-btn": 1,
                "regime-toxic-btn": 2,
            }
            if trigger_id in toggle_map:
                rid = toggle_map[trigger_id]
                if rid in active_regimes:
                    active_regimes.discard(rid)
                else:
                    active_regimes.add(rid)

        # For display, keep original regimes (always show all for context)
        display_states = states_sub.copy()

        heatmap_fig = create_heatmap_figure(snap_sub, display_states)
        regime_fig = create_regime_probs_figure(
            timestamps_sub, probs_sub, hmm["transition_matrix"]
        )
        depth_fig = create_depth_surface_figure(snap_sub, display_states)
        diag_fig = create_diagnostics_figure(feat_sub, display_states, pnl_sub)

        return heatmap_fig, regime_fig, depth_fig, diag_fig

    @app.callback(
        Output("regime-quiet-btn", "className"),
        Input("regime-quiet-btn", "n_clicks"),
        State("regime-quiet-btn", "className"),
        prevent_initial_call=True,
    )
    def toggle_quiet(n_clicks, current_cls):
        if current_cls and "inactive" in current_cls:
            return "regime-btn regime-btn-quiet"
        return "regime-btn regime-btn-quiet inactive"

    @app.callback(
        Output("regime-trending-btn", "className"),
        Input("regime-trending-btn", "n_clicks"),
        State("regime-trending-btn", "className"),
        prevent_initial_call=True,
    )
    def toggle_trending(n_clicks, current_cls):
        if current_cls and "inactive" in current_cls:
            return "regime-btn regime-btn-trending"
        return "regime-btn regime-btn-trending inactive"

    @app.callback(
        Output("regime-toxic-btn", "className"),
        Input("regime-toxic-btn", "n_clicks"),
        State("regime-toxic-btn", "className"),
        prevent_initial_call=True,
    )
    def toggle_toxic(n_clicks, current_cls):
        if current_cls and "inactive" in current_cls:
            return "regime-btn regime-btn-toxic"
        return "regime-btn regime-btn-toxic inactive"
