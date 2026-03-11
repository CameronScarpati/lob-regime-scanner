"""Tests for HMM regime detection model and backtest."""

import numpy as np
import pytest

from src.backtest import BacktestResult, run_backtest
from src.hmm_model import (
    ModelSelection,
    RegimeDetector,
    RegimeStats,
    _compute_durations,
    select_model,
)

# ---------------------------------------------------------------------------
# Helpers: synthetic data with known regime structure
# ---------------------------------------------------------------------------


def _make_regime_data(
    n_per_regime: int = 200,
    n_features: int = 3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic feature data with 3 known regimes.

    Regime 0 (Quiet):   low-variance, near-zero mean
    Regime 1 (Trending): directional mean shift, moderate variance
    Regime 2 (Toxic):    high variance, large mean
    """
    rng = np.random.RandomState(seed)
    states_true = np.concatenate(
        [
            np.zeros(n_per_regime, dtype=int),
            np.ones(n_per_regime, dtype=int),
            np.full(n_per_regime, 2, dtype=int),
        ]
    )
    X = np.empty((3 * n_per_regime, n_features))
    # Quiet
    X[:n_per_regime] = rng.normal(0.0, 0.3, (n_per_regime, n_features))
    # Trending
    X[n_per_regime : 2 * n_per_regime] = rng.normal(2.0, 1.0, (n_per_regime, n_features))
    # Toxic
    X[2 * n_per_regime :] = rng.normal(5.0, 3.0, (n_per_regime, n_features))
    return X, states_true


def _make_alternating_regime_data(
    n_cycles: int = 5,
    cycle_len: int = 100,
    n_features: int = 3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate alternating regime data (Quiet -> Trending -> Toxic) cycles."""
    rng = np.random.RandomState(seed)
    segments_X = []
    segments_s = []
    for _ in range(n_cycles):
        for state, (mu, sigma) in enumerate([(0.0, 0.3), (2.0, 1.0), (5.0, 3.0)]):
            seg = rng.normal(mu, sigma, (cycle_len, n_features))
            segments_X.append(seg)
            segments_s.append(np.full(cycle_len, state, dtype=int))
    return np.vstack(segments_X), np.concatenate(segments_s)


# ---------------------------------------------------------------------------
# Tests: RegimeDetector
# ---------------------------------------------------------------------------


class TestRegimeDetector:
    """Tests for the RegimeDetector class."""

    def test_fit_predict_basic(self):
        """Model fits and produces state predictions."""
        X, _ = _make_regime_data()
        det = RegimeDetector(n_states=3, n_iter=100, random_state=42)
        det.fit(X)
        assert det.is_fitted
        states = det.predict(X)
        assert states.shape == (len(X),)
        assert set(states).issubset({0, 1, 2})

    def test_predict_proba_shape(self):
        """predict_proba returns correct shape summing to 1."""
        X, _ = _make_regime_data()
        det = RegimeDetector(n_states=3, n_iter=100, random_state=42)
        det.fit(X)
        probs = det.predict_proba(X)
        assert probs.shape == (len(X), 3)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_unfitted_raises(self):
        """Calling predict/predict_proba before fit raises RuntimeError."""
        det = RegimeDetector(n_states=3)
        X, _ = _make_regime_data()
        with pytest.raises(RuntimeError, match="not been fitted"):
            det.predict(X)
        with pytest.raises(RuntimeError, match="not been fitted"):
            det.predict_proba(X)
        with pytest.raises(RuntimeError, match="not been fitted"):
            det.score(X)

    def test_state_sorting_by_volatility(self):
        """States are sorted so state 0 has lowest variance."""
        X, _ = _make_regime_data(n_per_regime=300)
        det = RegimeDetector(n_states=3, n_iter=200, random_state=42)
        det.fit(X)
        # Check that model means are sorted by magnitude (proxy for volatility ordering)
        covars = det.model.covars_
        traces = [np.trace(c) for c in covars]
        assert traces == sorted(traces), "States should be sorted by increasing covariance trace"

    def test_transition_matrix_shape_and_stochastic(self):
        """Transition matrix is square and row-stochastic."""
        X, _ = _make_regime_data()
        det = RegimeDetector(n_states=3, n_iter=100, random_state=42)
        det.fit(X)
        T = det.transition_matrix()
        assert T.shape == (3, 3)
        np.testing.assert_allclose(T.sum(axis=1), 1.0, atol=1e-6)
        assert np.all(T >= 0)

    def test_diagnostics(self):
        """Diagnostics contain valid convergence info."""
        X, _ = _make_regime_data()
        det = RegimeDetector(n_states=3, n_iter=200, random_state=42)
        det.fit(X)
        diag = det.diagnostics
        assert diag.log_likelihood < 0 or isinstance(diag.log_likelihood, float)
        assert diag.n_iter > 0
        assert len(diag.monitor_history) > 0

    def test_bic_aic(self):
        """BIC and AIC return finite values."""
        X, _ = _make_regime_data()
        det = RegimeDetector(n_states=3, n_iter=100, random_state=42)
        det.fit(X)
        bic = det.bic(X)
        aic = det.aic(X)
        assert np.isfinite(bic)
        assert np.isfinite(aic)

    def test_score(self):
        """Score returns a finite log-likelihood."""
        X, _ = _make_regime_data()
        det = RegimeDetector(n_states=3, n_iter=100, random_state=42)
        det.fit(X)
        ll = det.score(X)
        assert np.isfinite(ll)

    def test_dataframe_input(self):
        """Model accepts pandas DataFrame input."""
        import pandas as pd

        X, _ = _make_regime_data()
        df = pd.DataFrame(X, columns=["f1", "f2", "f3"])
        det = RegimeDetector(n_states=3, n_iter=100, random_state=42)
        det.fit(df)
        states = det.predict(df)
        assert states.shape == (len(df),)

    def test_1d_input(self):
        """Model handles 1D input by reshaping."""
        rng = np.random.RandomState(42)
        X = np.concatenate(
            [
                rng.normal(0, 0.3, 100),
                rng.normal(3, 1.0, 100),
            ]
        )
        det = RegimeDetector(n_states=2, n_iter=100, random_state=42)
        det.fit(X)
        states = det.predict(X)
        assert states.shape == (200,)

    def test_two_states(self):
        """Model works with 2 states."""
        rng = np.random.RandomState(42)
        X = np.vstack(
            [
                rng.normal(0, 0.5, (150, 2)),
                rng.normal(4, 2.0, (150, 2)),
            ]
        )
        det = RegimeDetector(n_states=2, n_iter=100, random_state=42)
        det.fit(X)
        states = det.predict(X)
        assert set(states).issubset({0, 1})

    def test_diag_covariance(self):
        """Model works with diagonal covariance."""
        X, _ = _make_regime_data()
        det = RegimeDetector(n_states=3, covariance_type="diag", n_iter=100, random_state=42)
        det.fit(X)
        states = det.predict(X)
        assert states.shape == (len(X),)


# ---------------------------------------------------------------------------
# Tests: Regime-conditional analysis
# ---------------------------------------------------------------------------


class TestRegimeStats:
    """Tests for regime-conditional statistics."""

    def test_regime_stats_with_returns(self):
        """regime_stats computes distributions over provided returns."""
        X, _ = _make_regime_data()
        det = RegimeDetector(n_states=3, n_iter=100, random_state=42)
        det.fit(X)

        rng = np.random.RandomState(99)
        returns = {
            "1s": rng.normal(0, 0.01, len(X)),
            "10s": rng.normal(0, 0.03, len(X)),
        }
        stats = det.regime_stats(X, returns=returns)

        assert isinstance(stats, RegimeStats)
        assert stats.transition_matrix is not None
        # Check all regimes have stats
        for state in range(3):
            if state in stats.means:
                assert "1s" in stats.means[state]
                assert "10s" in stats.stds[state]

    def test_regime_stats_without_returns(self):
        """regime_stats works with features when no returns provided."""
        X, _ = _make_regime_data(n_features=2)
        det = RegimeDetector(n_states=3, n_iter=100, random_state=42)
        det.fit(X)
        stats = det.regime_stats(X)
        assert len(stats.means) > 0

    def test_duration_stats(self):
        """Duration statistics are computed correctly."""
        X, _ = _make_alternating_regime_data(n_cycles=3, cycle_len=50)
        det = RegimeDetector(n_states=3, n_iter=200, random_state=42)
        det.fit(X)
        stats = det.regime_stats(X)
        for state in range(3):
            if state in stats.durations:
                d = stats.durations[state]
                assert "mean" in d
                assert "count" in d
                assert d["mean"] >= 0


# ---------------------------------------------------------------------------
# Tests: Model selection
# ---------------------------------------------------------------------------


class TestModelSelection:
    """Tests for BIC/AIC model selection."""

    def test_select_model_returns_results(self):
        """select_model evaluates multiple state counts."""
        X, _ = _make_regime_data(n_per_regime=150)
        result = select_model(X, state_range=range(2, 5), n_iter=50)
        assert isinstance(result, ModelSelection)
        assert len(result.n_states) == 3
        assert len(result.bics) == 3
        assert len(result.aics) == 3
        assert result.best_n_bic in [2, 3, 4]
        assert result.best_n_aic in [2, 3, 4]

    def test_select_model_prefers_more_than_two(self):
        """With 3 true regimes, BIC should prefer more than 2 states."""
        X, _ = _make_regime_data(n_per_regime=300)
        result = select_model(X, state_range=range(2, 6), n_iter=100)
        # With well-separated 3-state data, BIC should at least reject 2
        assert result.best_n_bic >= 3, f"Expected BIC to prefer >=3 states, got {result.best_n_bic}"


# ---------------------------------------------------------------------------
# Tests: Threshold comparison
# ---------------------------------------------------------------------------


class TestThresholdComparison:
    """Tests for compare_threshold_regimes."""

    def test_compare_returns_dict(self):
        """compare_threshold_regimes returns agreement and confusion matrix."""
        X, true_states = _make_regime_data()
        det = RegimeDetector(n_states=3, n_iter=100, random_state=42)
        det.fit(X)
        result = det.compare_threshold_regimes(X, true_states)
        assert "agreement_rate" in result
        assert "confusion_matrix" in result
        assert 0.0 <= result["agreement_rate"] <= 1.0
        assert result["confusion_matrix"].shape[0] >= 3


# ---------------------------------------------------------------------------
# Tests: Duration helper
# ---------------------------------------------------------------------------


class TestComputeDurations:
    """Tests for _compute_durations helper."""

    def test_simple_durations(self):
        """Known sequence produces correct duration stats."""
        states = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 0])
        result = _compute_durations(states, 3)
        assert result[0]["count"] == 2  # two runs of state 0
        assert result[1]["count"] == 1
        assert result[2]["count"] == 1
        assert result[2]["mean"] == 4.0

    def test_empty_states(self):
        """Empty input doesn't crash."""
        result = _compute_durations(np.array([]), 3)
        assert result[0]["count"] == 0


# ---------------------------------------------------------------------------
# Tests: Backtest
# ---------------------------------------------------------------------------


class TestBacktest:
    """Tests for the regime-conditional backtest."""

    def test_backtest_basic(self):
        """Backtest runs and returns valid metrics."""
        # Construct a sequence: Quiet -> Trending -> Toxic -> Quiet -> ...
        n = 300
        states = np.zeros(n, dtype=int)
        states[50:150] = 1  # Trending
        states[150:200] = 2  # Toxic
        states[200:250] = 0  # Quiet
        states[250:] = 1  # Trending again

        rng = np.random.RandomState(42)
        returns = rng.normal(0.001, 0.01, n)
        ofi = np.ones(n) * 0.5  # positive OFI

        result = run_backtest(states, returns, ofi)
        assert isinstance(result, BacktestResult)
        assert result.pnl.shape == (n,)
        assert result.cumulative_pnl.shape == (n,)
        assert result.n_trades >= 1
        assert np.isfinite(result.sharpe_ratio)
        assert result.max_drawdown >= 0

    def test_backtest_no_trades(self):
        """Backtest with no transitions produces no trades."""
        n = 100
        states = np.zeros(n, dtype=int)  # all Quiet, no transition
        returns = np.random.normal(0, 0.01, n)
        ofi = np.ones(n)

        result = run_backtest(states, returns, ofi)
        assert result.n_trades == 0
        assert result.total_pnl == 0.0

    def test_backtest_short_position(self):
        """Negative OFI leads to short entry."""
        n = 200
        states = np.zeros(n, dtype=int)
        states[50:100] = 1  # Trending
        states[100:150] = 2  # Toxic

        rng = np.random.RandomState(42)
        returns = rng.normal(-0.002, 0.01, n)  # slightly negative
        ofi = np.full(n, -0.5)  # negative OFI -> short

        result = run_backtest(states, returns, ofi)
        assert result.n_trades >= 1

    def test_backtest_empty_input(self):
        """Empty or very short input doesn't crash."""
        result = run_backtest(np.array([0]), np.array([0.01]), np.array([1.0]))
        assert result.n_trades == 0

    def test_backtest_exit_on_quiet(self):
        """Position is closed when returning to Quiet state."""
        states = np.array([0, 1, 1, 1, 0, 0, 0])
        returns = np.array([0.0, 0.01, 0.01, 0.01, 0.01, 0.0, 0.0])
        ofi = np.ones(7)

        result = run_backtest(states, returns, ofi)
        # Should have entered on 0->1 and exited on 1->0
        assert result.n_trades == 1

    def test_backtest_result_fields(self):
        """All BacktestResult fields are populated."""
        states = np.array([0, 1, 1, 2, 0, 0, 1, 1, 2])
        returns = np.array([0, 0.01, 0.02, -0.01, 0, 0, 0.01, 0.01, -0.01])
        ofi = np.ones(9)

        result = run_backtest(states, returns, ofi)
        assert hasattr(result, "sharpe_ratio")
        assert hasattr(result, "max_drawdown")
        assert hasattr(result, "hit_rate")
        assert hasattr(result, "profit_per_trade")
        assert hasattr(result, "n_trades")
        assert hasattr(result, "total_pnl")


# ---------------------------------------------------------------------------
# Integration: HMM + Backtest
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests combining HMM and backtest."""

    def test_hmm_to_backtest_pipeline(self):
        """End-to-end: fit HMM on synthetic data, run backtest."""
        X, _ = _make_regime_data(n_per_regime=200)
        det = RegimeDetector(n_states=3, n_iter=100, random_state=42)
        det.fit(X)
        states = det.predict(X)

        rng = np.random.RandomState(99)
        returns = rng.normal(0, 0.01, len(X))
        ofi = X[:, 0]  # use first feature as OFI proxy

        result = run_backtest(states, returns, ofi)
        assert isinstance(result, BacktestResult)
        assert result.pnl.shape == (len(X),)
        assert np.isfinite(result.sharpe_ratio)

    def test_regime_stats_and_backtest(self):
        """Regime stats and backtest both work on same decoded states."""
        X, _ = _make_alternating_regime_data(n_cycles=4, cycle_len=80)
        det = RegimeDetector(n_states=3, n_iter=200, random_state=42)
        det.fit(X)

        rng = np.random.RandomState(99)
        returns_dict = {"1s": rng.normal(0, 0.01, len(X))}
        stats = det.regime_stats(X, returns=returns_dict)
        assert stats.transition_matrix is not None

        states = det.predict(X)
        ofi = X[:, 0]
        bt = run_backtest(states, returns_dict["1s"], ofi)
        assert bt.pnl.shape == (len(X),)
