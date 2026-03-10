"""Tests for the Plotly Dash dashboard components and mock data."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from dashboard._mock_data import (
    REGIME_COLORS,
    REGIME_NAMES,
    generate_all,
    generate_cumulative_pnl,
    generate_features,
    generate_hmm_output,
    generate_snapshots,
    generate_transition_matrix,
)
from dashboard.components.heatmap import create_heatmap_figure
from dashboard.components.regime_probs import create_regime_probs_figure
from dashboard.components.depth_surface import create_depth_surface_figure
from dashboard.components.diagnostics import create_diagnostics_figure


# ---------------------------------------------------------------------------
# Mock data tests
# ---------------------------------------------------------------------------

class TestMockData:
    def test_transition_matrix_shape_and_rows_sum_to_one(self):
        tm = generate_transition_matrix()
        assert tm.shape == (3, 3)
        np.testing.assert_allclose(tm.sum(axis=1), [1.0, 1.0, 1.0], atol=1e-10)

    def test_snapshots_schema(self):
        df = generate_snapshots(n_timestamps=50)
        assert len(df) == 50
        assert "timestamp" in df.columns
        assert "mid_price" in df.columns
        assert "spread" in df.columns
        for i in range(1, 11):
            assert f"bid_price_{i}" in df.columns
            assert f"bid_qty_{i}" in df.columns
            assert f"ask_price_{i}" in df.columns
            assert f"ask_qty_{i}" in df.columns
        assert "last_trade_price" in df.columns
        assert "last_trade_qty" in df.columns
        assert "last_trade_side" in df.columns
        assert set(df["last_trade_side"].unique()).issubset({"buy", "sell"})

    def test_snapshots_prices_ordered(self):
        df = generate_snapshots(n_timestamps=100)
        # bid_price_1 should be below mid, ask_price_1 above mid
        assert (df["bid_price_1"] < df["mid_price"]).all()
        assert (df["ask_price_1"] > df["mid_price"]).all()

    def test_features_schema(self):
        df = generate_features(n_timestamps=50)
        assert len(df) == 50
        expected_cols = [
            "timestamp", "OFI_1", "OFI_5", "OFI_10", "OFI_velocity",
            "VPIN", "book_imbalance", "weighted_mid", "spread_bps",
            "kyle_lambda", "trade_aggression", "cancel_ratio",
            "realized_vol_1s", "realized_vol_10s", "realized_vol_60s",
            "realized_vol_300s",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_features_vpin_in_range(self):
        df = generate_features(n_timestamps=200)
        assert (df["VPIN"] >= 0).all()
        assert (df["VPIN"] <= 1).all()

    def test_hmm_output_shapes(self):
        hmm = generate_hmm_output(n_timestamps=100)
        assert hmm["states"].shape == (100,)
        assert hmm["state_probs"].shape == (100, 3)
        assert hmm["transition_matrix"].shape == (3, 3)
        assert set(np.unique(hmm["states"])).issubset({0, 1, 2})

    def test_hmm_probs_sum_to_one(self):
        hmm = generate_hmm_output(n_timestamps=100)
        row_sums = hmm["state_probs"].sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_cumulative_pnl_shape(self):
        pnl = generate_cumulative_pnl(n_timestamps=100)
        assert pnl.shape == (100,)

    def test_generate_all_keys(self):
        data = generate_all(n_timestamps=50)
        assert "snapshots" in data
        assert "features" in data
        assert "hmm" in data
        assert "cumulative_pnl" in data
        assert len(data["snapshots"]) == 50
        assert len(data["features"]) == 50

    def test_regime_constants(self):
        assert len(REGIME_NAMES) == 3
        assert len(REGIME_COLORS) == 3
        for i in range(3):
            assert i in REGIME_NAMES
            assert i in REGIME_COLORS


# ---------------------------------------------------------------------------
# Component figure tests
# ---------------------------------------------------------------------------

class TestHeatmapPanel:
    @pytest.fixture()
    def data(self):
        return generate_all(n_timestamps=100)

    def test_returns_figure(self, data):
        fig = create_heatmap_figure(data["snapshots"], data["hmm"]["states"])
        assert isinstance(fig, go.Figure)

    def test_has_traces(self, data):
        fig = create_heatmap_figure(data["snapshots"], data["hmm"]["states"])
        # At least: 3 regime overlays + 1 heatmap + 1 mid-price + maybe trades
        assert len(fig.data) >= 5

    def test_dark_template(self, data):
        fig = create_heatmap_figure(data["snapshots"], data["hmm"]["states"])
        assert fig.layout.template.layout.paper_bgcolor is not None or \
            "plotly_dark" in str(fig.layout.template)


class TestRegimeProbsPanel:
    @pytest.fixture()
    def data(self):
        return generate_all(n_timestamps=100)

    def test_returns_figure(self, data):
        fig = create_regime_probs_figure(
            data["features"]["timestamp"].values,
            data["hmm"]["state_probs"],
            data["hmm"]["transition_matrix"],
        )
        assert isinstance(fig, go.Figure)

    def test_has_stacked_and_matrix(self, data):
        fig = create_regime_probs_figure(
            data["features"]["timestamp"].values,
            data["hmm"]["state_probs"],
            data["hmm"]["transition_matrix"],
        )
        # 3 stacked area traces + 1 heatmap for transition matrix
        assert len(fig.data) == 4


class TestDepthSurfacePanel:
    @pytest.fixture()
    def data(self):
        return generate_all(n_timestamps=100)

    def test_returns_figure(self, data):
        fig = create_depth_surface_figure(data["snapshots"], data["hmm"]["states"])
        assert isinstance(fig, go.Figure)

    def test_has_surface_trace(self, data):
        fig = create_depth_surface_figure(data["snapshots"], data["hmm"]["states"])
        assert len(fig.data) >= 1
        assert isinstance(fig.data[0], go.Surface)


class TestDiagnosticsPanel:
    @pytest.fixture()
    def data(self):
        return generate_all(n_timestamps=100)

    def test_returns_figure(self, data):
        fig = create_diagnostics_figure(
            data["features"], data["hmm"]["states"], data["cumulative_pnl"]
        )
        assert isinstance(fig, go.Figure)

    def test_has_four_traces(self, data):
        fig = create_diagnostics_figure(
            data["features"], data["hmm"]["states"], data["cumulative_pnl"]
        )
        # VPIN + OFI + Spread + PnL = 4 scatter traces
        assert len(fig.data) == 4
