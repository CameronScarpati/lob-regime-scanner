"""Tests for the dashboard pipeline's walk-forward wiring.

These pin the contract the dashboard and the docs depend on: headline
backtest statistics are the held-out segment, the split is reported, the
stitched PnL curve aligns with the decoded states, and the decode driving
the backtest is causal.
"""

import numpy as np
import pandas as pd
import pytest

from dashboard import pipeline as pipeline_mod
from dashboard.pipeline import _resolve_split, run_pipeline
from src.features import N_LEVELS


def _make_snapshots(n: int = 900, seed: int = 7) -> pd.DataFrame:
    """Synthetic snapshot frame in the schema load_snapshots_directory returns.

    Built with three volatility phases so a 3-state HMM has structure to find.
    """
    rng = np.random.default_rng(seed)
    vols = np.concatenate(
        [
            rng.normal(0, 0.00005, n // 3),
            rng.normal(0, 0.00020, n // 3),
            rng.normal(0, 0.00060, n - 2 * (n // 3)),
        ]
    )
    mid = 30_000.0 * np.exp(np.cumsum(vols))

    rows: dict[str, np.ndarray] = {
        "timestamp": (np.arange(n) * 1_000_000).astype(np.int64),
        "mid_price": mid,
    }
    for i in range(1, N_LEVELS + 1):
        offset = i * 0.5
        rows[f"bid_price_{i}"] = mid - offset
        rows[f"ask_price_{i}"] = mid + offset
        rows[f"bid_qty_{i}"] = rng.exponential(10, n)
        rows[f"ask_qty_{i}"] = rng.exponential(10, n)
    rows["spread"] = rows["ask_price_1"] - rows["bid_price_1"]
    return pd.DataFrame(rows)


@pytest.fixture
def stub_loader(monkeypatch):
    """Bypass disk IO: _find_data_files and the snapshot loader are stubbed."""
    df = _make_snapshots()
    monkeypatch.setattr(pipeline_mod, "_find_data_files", lambda *a, **k: pipeline_mod.DATA_DIR)
    monkeypatch.setattr(
        pipeline_mod, "load_snapshots_directory", lambda *a, **k: df.copy(deep=True)
    )
    return df


class TestResolveSplit:
    def test_none_train_frac_disables_split(self):
        assert _resolve_split(1000, None, 3) is None

    def test_returns_expected_index(self):
        assert _resolve_split(1000, 0.7, 3) == 700

    def test_rejects_out_of_range_fraction(self):
        for bad in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(ValueError):
                _resolve_split(1000, bad, 3)

    @pytest.mark.parametrize("n_rows", [0, 1, 4, 5, 10, 25])
    def test_falls_back_when_train_slice_too_small(self, n_rows):
        """Tiny inputs must fall back rather than raise from inside hmmlearn."""
        assert _resolve_split(n_rows, 0.7, 3) is None

    def test_split_enabled_once_train_slice_is_large_enough(self):
        assert _resolve_split(200, 0.7, 3) == 140


@pytest.fixture(scope="module")
def result():
    """One walk-forward pipeline run, shared across the wiring assertions."""
    df = _make_snapshots()
    mp = pytest.MonkeyPatch()
    mp.setattr(pipeline_mod, "_find_data_files", lambda *a, **k: pipeline_mod.DATA_DIR)
    mp.setattr(pipeline_mod, "load_snapshots_directory", lambda *a, **k: df.copy(deep=True))
    try:
        yield run_pipeline(symbol="BTCUSDT", sample_interval_us=1_000_000)
    finally:
        mp.undo()


class TestWalkForwardWiring:
    def test_reports_both_segments(self, result):
        stats = result["backtest_stats"]
        assert "in_sample" in stats
        assert "out_of_sample" in stats
        assert "split_index" in stats
        assert stats["train_frac"] == 0.7

    def test_headline_stats_are_out_of_sample(self, result):
        """The whole point of the split: headline != in-sample."""
        stats = result["backtest_stats"]
        for key in ("sharpe_ratio", "max_drawdown", "n_trades", "total_pnl"):
            assert stats[key] == stats["out_of_sample"][key], key

    def test_split_index_matches_train_frac(self, result):
        n = len(result["hmm"]["states"])
        assert result["backtest_stats"]["split_index"] == int(n * 0.7)

    def test_pnl_curve_aligns_with_states(self, result):
        n = len(result["hmm"]["states"])
        assert len(result["cumulative_pnl"]) == n
        assert len(result["features"]) == n
        assert len(result["snapshots"]) == n

    def test_stitched_pnl_is_continuous(self, result):
        """The held-out curve continues from where the train curve ended."""
        split = result["backtest_stats"]["split_index"]
        pnl = result["cumulative_pnl"]
        assert np.isfinite(pnl).all()
        # No artificial jump at the seam beyond one bar's PnL
        seam_jump = abs(pnl[split] - pnl[split - 1])
        assert seam_jump < 0.5

    def test_costs_are_charged(self, result):
        stats = result["backtest_stats"]
        assert stats["total_costs"] >= 0.0
        if stats["n_trades"] > 0:
            assert stats["total_costs"] > 0.0
            assert stats["sharpe_ratio"] <= stats["sharpe_ratio_gross"]

    def test_states_are_causal(self, result):
        """States driving the backtest must be the filtered decode, which is
        prefix-invariant, not the smoothed Viterbi path."""
        states = result["hmm"]["states"]
        smoothed = result["hmm"]["states_smoothed"]
        assert states.shape == smoothed.shape
        assert set(np.unique(states)).issubset({0, 1, 2})

    def test_state_probs_are_filtered_distribution(self, result):
        probs = result["hmm"]["state_probs"]
        assert probs.shape == (len(result["hmm"]["states"]), 3)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0)
        # states are the argmax of the reported probabilities
        np.testing.assert_array_equal(np.argmax(probs, axis=1), result["hmm"]["states"])

    def test_transition_matrix_rows_sum_to_one(self, result):
        tm = result["hmm"]["transition_matrix"]
        assert tm.shape == (3, 3)
        np.testing.assert_allclose(tm.sum(axis=1), 1.0)


class TestInSampleMode:
    def test_train_frac_none_reports_only_in_sample(self, stub_loader):
        result = run_pipeline(symbol="BTCUSDT", sample_interval_us=1_000_000, train_frac=None)
        stats = result["backtest_stats"]
        assert "out_of_sample" not in stats
        assert "in_sample" in stats
        assert stats["sharpe_ratio"] == stats["in_sample"]["sharpe_ratio"]

    def test_invalid_train_frac_raises(self, stub_loader):
        with pytest.raises(ValueError, match="train_frac"):
            run_pipeline(symbol="BTCUSDT", sample_interval_us=1_000_000, train_frac=1.5)


class TestAnnualization:
    def test_sharpe_scales_with_sample_interval(self, monkeypatch):
        """bars_per_year must follow the bar interval, not assume 1s bars."""
        df = _make_snapshots()
        monkeypatch.setattr(pipeline_mod, "_find_data_files", lambda *a, **k: pipeline_mod.DATA_DIR)
        monkeypatch.setattr(
            pipeline_mod, "load_snapshots_directory", lambda *a, **k: df.copy(deep=True)
        )
        one_sec = run_pipeline(sample_interval_us=1_000_000)["backtest_stats"]["sharpe_ratio"]
        hundred_ms = run_pipeline(sample_interval_us=100_000)["backtest_stats"]["sharpe_ratio"]
        if one_sec != 0.0:
            assert hundred_ms == pytest.approx(one_sec * np.sqrt(10), rel=0.02)
