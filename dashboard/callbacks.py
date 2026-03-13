"""Dash callbacks for dashboard interactivity.

Handles time-range selection via DMC components.
"""

from __future__ import annotations

from dash import Dash, Input, Output

from dashboard.components.depth_surface import create_depth_surface_figure
from dashboard.components.diagnostics import create_diagnostics_figure
from dashboard.components.heatmap import create_heatmap_figure
from dashboard.components.regime_probs import create_regime_probs_figure


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
            Input("time-range-slider", "value"),
        ],
    )
    def update_panels(time_range):
        snapshots = _data["snapshots"]
        features = _data["features"]
        hmm = _data["hmm"]
        cum_pnl = _data["cumulative_pnl"]

        # Slice data to the selected time range
        t_start, t_end = time_range
        snapshots = snapshots.iloc[t_start : t_end + 1].reset_index(drop=True)
        features = features.iloc[t_start : t_end + 1].reset_index(drop=True)
        states = hmm["states"][t_start : t_end + 1]
        probs = hmm["state_probs"][t_start : t_end + 1]
        sliced_pnl = cum_pnl[t_start : t_end + 1]

        # Guard against empty selection
        if len(snapshots) < 2:
            snapshots = _data["snapshots"].iloc[:2].reset_index(drop=True)
            features = _data["features"].iloc[:2].reset_index(drop=True)
            states = hmm["states"][:2]
            probs = hmm["state_probs"][:2]
            sliced_pnl = cum_pnl[:2]

        timestamps = features["timestamp"].values

        heatmap_fig = create_heatmap_figure(snapshots, states)
        regime_fig = create_regime_probs_figure(timestamps, probs, hmm["transition_matrix"])
        depth_fig = create_depth_surface_figure(snapshots, states)
        diag_fig = create_diagnostics_figure(features, states, sliced_pnl)

        return heatmap_fig, regime_fig, depth_fig, diag_fig
