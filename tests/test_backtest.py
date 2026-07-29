"""Tests for the regime-conditional backtest: execution timing, transaction
costs, Sharpe annualization, and drawdown semantics."""

import numpy as np
import pytest

from src.backtest import (
    BacktestResult,
    _annualized_sharpe,
    _max_drawdown_fraction,
    run_backtest,
)

QUIET, TRENDING, TOXIC = 0, 1, 2


def _run(states, returns, ofi, **kwargs):
    defaults = {
        "cooldown_bars": 0,
        "ofi_smooth_window": 1,
        "fee_bps": 0.0,
        "slippage_bps": 0.0,
    }
    defaults.update(kwargs)
    return run_backtest(
        np.asarray(states),
        np.asarray(returns, dtype=float),
        np.asarray(ofi, dtype=float),
        **defaults,
    )


class TestBasics:
    def test_empty_input_returns_default(self):
        result = run_backtest(np.array([]), np.array([]), np.array([]))
        assert isinstance(result, BacktestResult)
        assert result.n_trades == 0
        assert result.sharpe_ratio == 0.0

    def test_all_quiet_never_trades(self):
        n = 100
        result = _run([QUIET] * n, np.full(n, 0.001), np.ones(n))
        assert result.n_trades == 0
        assert np.all(result.pnl == 0.0)
        assert result.total_pnl == 0.0

    def test_trending_long_direction_follows_ofi(self):
        # Positive OFI -> long; positive returns while trending -> profit
        states = [QUIET, TRENDING, TRENDING, TRENDING, QUIET, QUIET]
        returns = [0.0, 0.01, 0.01, 0.01, 0.01, 0.0]
        result = _run(states, returns, np.ones(6))
        assert result.n_trades == 1
        assert result.total_pnl > 0

    def test_trending_short_direction_follows_ofi(self):
        states = [QUIET, TRENDING, TRENDING, TRENDING, QUIET, QUIET]
        returns = [0.0, -0.01, -0.01, -0.01, -0.01, 0.0]
        result = _run(states, returns, -np.ones(6))
        assert result.n_trades == 1
        assert result.total_pnl > 0


class TestExecutionTiming:
    def test_no_pnl_on_signal_bar(self):
        """The entry bar's return must not be earned (next-bar execution)."""
        # Signal fires at t=1 (transition into Trending). A large return on
        # t=1 must not appear in PnL; the position earns from t=2 on.
        states = [QUIET, TRENDING, TRENDING, QUIET, QUIET]
        returns = [0.0, 1.0, 0.01, 0.01, 0.0]
        result = _run(states, returns, np.ones(5))
        # Bar 1 return (1.0) excluded; bars 2 and 3 earned (position still
        # held into bar 3, where the exit to Quiet is observed).
        assert result.pnl[1] == 0.0
        assert result.pnl[2] == pytest.approx(0.01)
        assert result.total_pnl == pytest.approx(0.02)

    def test_exit_bar_return_is_earned(self):
        """Position decided at t-1 is held over bar t, including the exit bar."""
        states = [QUIET, TRENDING, TRENDING, TOXIC, QUIET]
        returns = [0.0, 0.0, 0.01, -0.02, 0.05]
        result = _run(states, returns, np.ones(5))
        # Long entered at t=1, earns bars 2 and 3 (toxic observed at 3 ->
        # flat after 3), never earns bar 4.
        assert result.pnl[3] == pytest.approx(-0.02)
        assert result.pnl[4] == 0.0
        assert result.total_pnl == pytest.approx(0.01 - 0.02)


class TestTransactionCosts:
    def test_zero_cost_matches_gross(self):
        states = [QUIET, TRENDING, TRENDING, QUIET, QUIET]
        returns = [0.0, 0.0, 0.01, 0.01, 0.0]
        result = _run(states, returns, np.ones(5))
        assert result.total_costs == 0.0
        assert result.total_pnl == pytest.approx(result.total_pnl_gross)
        assert result.sharpe_ratio == pytest.approx(result.sharpe_ratio_gross)

    def test_costs_charged_per_side(self):
        states = [QUIET, TRENDING, TRENDING, QUIET, QUIET]
        returns = [0.0, 0.0, 0.01, 0.01, 0.0]
        result = _run(states, returns, np.ones(5), fee_bps=5.0, slippage_bps=5.0)
        # One round trip at 10 bps per side = 20 bps total
        assert result.total_costs == pytest.approx(2 * 10.0 / 10_000.0)
        assert result.total_pnl == pytest.approx(result.total_pnl_gross - result.total_costs)

    def test_costs_reduce_hit_rate_marginal_trade(self):
        # A trade that gains less than the round-trip cost is a net loss
        states = [QUIET, TRENDING, TRENDING, QUIET, QUIET]
        returns = [0.0, 0.0, 0.0001, 0.0001, 0.0]
        gross = _run(states, returns, np.ones(5))
        net = _run(states, returns, np.ones(5), fee_bps=5.5, slippage_bps=0.5)
        assert gross.hit_rate == 1.0
        assert net.hit_rate == 0.0


class TestStopLossAndCooldown:
    def test_stop_loss_exits(self):
        states = [QUIET] + [TRENDING] * 20
        returns = [0.0] + [-0.001] * 20
        result = _run(states, returns, np.ones(21), stop_loss=0.003)
        # The stop cuts the position, so the losing streak becomes several
        # small trades instead of one large one, each cut within one bar
        # of breaching the stop.
        assert result.n_trades >= 2
        assert result.profit_per_trade >= -(0.003 + 0.0011)

    def test_cooldown_blocks_reentry(self):
        # Trending -> Quiet -> Trending quickly; large cooldown blocks trade 2
        states = [QUIET, TRENDING, QUIET, TRENDING, TRENDING, QUIET]
        returns = [0.0, 0.0, 0.01, 0.0, 0.01, 0.01]
        no_cooldown = _run(states, returns, np.ones(6), cooldown_bars=0)
        cooldown = _run(states, returns, np.ones(6), cooldown_bars=100)
        assert no_cooldown.n_trades == 2
        assert cooldown.n_trades == 1


class TestMetrics:
    def test_sharpe_annualization_scales_with_bars_per_year(self):
        states = [QUIET] + [TRENDING] * 50 + [QUIET]
        rng = np.random.default_rng(0)
        returns = np.concatenate([[0.0], rng.normal(0.001, 0.01, 50), [0.0]])
        slow = _run(states, returns, np.ones(52), bars_per_year=3600 * 24 * 365)
        fast = _run(states, returns, np.ones(52), bars_per_year=3600 * 24 * 365 * 10)
        assert fast.sharpe_ratio == pytest.approx(slow.sharpe_ratio * np.sqrt(10))

    def test_annualized_sharpe_zero_when_flat(self):
        assert _annualized_sharpe(np.zeros(100), 1000.0) == 0.0

    def test_max_drawdown_is_fraction_of_peak(self):
        # Equity: up 10%, down to -20% from peak, recover
        pnl = np.array([np.log(1.1), np.log(0.8), np.log(1.5)])
        dd = _max_drawdown_fraction(pnl)
        assert dd == pytest.approx(0.2)

    def test_max_drawdown_in_unit_interval(self):
        states = [QUIET] + [TRENDING] * 50 + [QUIET]
        rng = np.random.default_rng(1)
        returns = np.concatenate([[0.0], rng.normal(0.0, 0.02, 50), [0.0]])
        result = _run(states, returns, np.ones(52))
        assert 0.0 <= result.max_drawdown < 1.0

    def test_cumulative_pnl_is_cumsum_of_pnl(self):
        states = [QUIET, TRENDING, TRENDING, QUIET, QUIET]
        returns = [0.0, 0.0, 0.01, -0.005, 0.0]
        result = _run(states, returns, np.ones(5), fee_bps=1.0)
        np.testing.assert_allclose(result.cumulative_pnl, np.cumsum(result.pnl))

    def test_open_trade_closed_at_end(self):
        states = [QUIET, TRENDING, TRENDING, TRENDING]
        returns = [0.0, 0.0, 0.01, 0.01]
        result = _run(states, returns, np.ones(4), fee_bps=5.0)
        assert result.n_trades == 1
        # Exit cost charged on the forced close
        assert result.total_costs == pytest.approx(2 * 5.0 / 10_000.0)
