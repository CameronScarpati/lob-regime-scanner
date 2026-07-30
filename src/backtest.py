"""Simple regime-conditional strategy backtest.

Computes Sharpe ratio, max drawdown, hit rate, and profit per trade for a
simple regime-conditional rule, to visualize how the detected regimes behave
over time.

Execution model
---------------
- **Next-bar execution.** The position decided from information at bar *t*
  (regime state, smoothed OFI) is held over bar *t + 1*, so PnL at *t* is
  ``position[t-1] * returns[t]``. The signal never earns the bar it fired on.
- **Transaction costs.** Each unit of turnover is charged
  ``(fee_bps + slippage_bps) / 10_000``. All reported metrics are net of
  costs; gross variants are reported alongside for comparison.
- **Drawdown.** Computed on a compounded equity curve ``exp(cumsum(pnl))``
  and reported as a fraction of the running peak (a true percentage
  drawdown), not as raw PnL units.
- **Annualization.** ``bars_per_year`` must reflect the actual bar interval
  of the input series. The caller computes it (e.g. 100ms bars in a 24/7
  crypto market give ``365 * 24 * 3600 * 10`` bars per year).

Even with these mechanics the backtest remains illustrative: run it on a
held-out segment (see ``dashboard/pipeline.py``) for the numbers to be
out-of-sample, and treat any Sharpe as a diagnostic, not evidence of a
tradeable edge.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Bars per year for 1-second bars in a 24/7 crypto market.
SECONDS_PER_YEAR = 365.0 * 24 * 3600


@dataclass
class BacktestResult:
    """Results from a regime-conditional backtest.

    ``pnl`` and ``cumulative_pnl`` are net of transaction costs and in
    log-return units on unit notional. ``max_drawdown`` is a fraction of
    peak equity (0.05 = 5% below peak).
    """

    pnl: np.ndarray = field(default_factory=lambda: np.array([]))
    cumulative_pnl: np.ndarray = field(default_factory=lambda: np.array([]))
    sharpe_ratio: float = 0.0
    sharpe_ratio_gross: float = 0.0
    max_drawdown: float = 0.0
    hit_rate: float = 0.0
    profit_per_trade: float = 0.0
    n_trades: int = 0
    total_pnl: float = 0.0
    total_pnl_gross: float = 0.0
    total_costs: float = 0.0


def _annualized_sharpe(pnl: np.ndarray, bars_per_year: float) -> float:
    """Annualized Sharpe of a per-bar PnL series (0.0 if degenerate)."""
    std = np.std(pnl)
    if std <= 0:
        return 0.0
    return float((np.mean(pnl) / std) * np.sqrt(bars_per_year))


def _max_drawdown_fraction(pnl: np.ndarray) -> float:
    """Max drawdown as a fraction of peak equity, from log-return PnL.

    The equity curve is anchored at 1.0 before the first bar, so a series
    that loses from bar one is measured against that starting capital
    rather than against its own already-depressed first value.
    """
    if len(pnl) == 0:
        return 0.0
    equity = np.exp(np.concatenate([[0.0], np.cumsum(pnl)]))
    peak = np.maximum.accumulate(equity)
    drawdown = 1.0 - equity / peak
    return float(np.max(drawdown))


def run_backtest(
    states: np.ndarray,
    returns: np.ndarray,
    ofi: np.ndarray,
    quiet_state: int = 0,
    trending_state: int = 1,
    toxic_state: int = 2,
    bars_per_year: float = SECONDS_PER_YEAR,
    ofi_smooth_window: int = 120,
    cooldown_bars: int = 30,
    stop_loss: float = 0.002,
    fee_bps: float = 5.5,
    slippage_bps: float = 0.5,
) -> BacktestResult:
    """Run a simple regime-conditional strategy.

    Strategy:
    - Enter on any transition into Trending in OFI direction.
    - Hold while Trending persists.
    - Flatten on Toxic detection, return to Quiet, or stop-loss.
    - Wait cooldown_bars after each exit before re-entering.

    Parameters
    ----------
    states : ndarray of shape (n_samples,)
        Decoded regime states (integers).
    returns : ndarray of shape (n_samples,)
        Per-bar log returns, where ``returns[t]`` is the return realized
        over bar *t* (from *t-1* to *t*).
    ofi : ndarray of shape (n_samples,)
        Order flow imbalance signal (sign determines direction).
    quiet_state : int
        State label for Quiet regime.
    trending_state : int
        State label for Trending regime.
    toxic_state : int
        State label for Toxic/Stressed regime.
    bars_per_year : float
        Number of bars in a year at the input series' bar interval, used to
        annualize the Sharpe ratio. Default assumes 1-second bars in a 24/7
        crypto market; pass the correct value for other intervals (e.g.
        ``SECONDS_PER_YEAR * 10`` for 100ms bars).
    ofi_smooth_window : int
        EMA-like smoothing window for OFI direction signal.
    cooldown_bars : int
        Minimum bars to wait after an exit before re-entering.
    stop_loss : float
        Maximum cumulative net loss per trade before forced exit.
    fee_bps : float
        Exchange taker fee per unit of turnover, in basis points.
    slippage_bps : float
        Assumed slippage per unit of turnover, in basis points.

    Returns
    -------
    BacktestResult with performance metrics (net of costs unless a field
    says gross).
    """
    n = len(states)
    if n < 2:
        return BacktestResult()

    cost_rate = (fee_bps + slippage_bps) / 10_000.0

    position = 0.0  # +1 long, -1 short, 0 flat; held over the NEXT bar
    pnl = np.zeros(n)  # net of costs
    costs = np.zeros(n)
    trades: list[float] = []  # net PnL per closed trade
    current_trade_pnl = 0.0
    in_trade = False
    bars_since_exit = cooldown_bars  # start ready to trade

    # Smooth OFI with EMA for reliable direction signal
    alpha = 2.0 / (ofi_smooth_window + 1)
    ofi_ema = np.zeros(n)
    ofi_ema[0] = ofi[0]
    for i in range(1, n):
        ofi_ema[i] = alpha * ofi[i] + (1 - alpha) * ofi_ema[i - 1]

    def close_trade(t: int) -> None:
        nonlocal position, in_trade, bars_since_exit, current_trade_pnl
        exit_cost = abs(position) * cost_rate
        costs[t] += exit_cost
        pnl[t] -= exit_cost
        current_trade_pnl -= exit_cost
        trades.append(current_trade_pnl)
        position = 0.0
        in_trade = False
        bars_since_exit = 0

    for t in range(1, n):
        # 1) Realize PnL on the position carried into bar t (decided at t-1).
        if position != 0.0:
            pnl[t] = position * returns[t]
            current_trade_pnl += pnl[t]
        else:
            bars_since_exit += 1

        prev_state = states[t - 1]
        curr_state = states[t]

        # 2) Update the position from information available at bar t.
        if position != 0.0:
            # Exit: Toxic detection or return to Quiet
            if curr_state in (toxic_state, quiet_state):
                close_trade(t)
                continue
            # Stop-loss: exit if cumulative net trade loss exceeds threshold
            if current_trade_pnl < -stop_loss:
                close_trade(t)
                continue

        # Entry: transition into Trending, with cooldown respected
        if (
            position == 0.0
            and curr_state == trending_state
            and prev_state != toxic_state
            and bars_since_exit >= cooldown_bars
        ):
            position = 1.0 if ofi_ema[t] > 0 else -1.0
            in_trade = True
            entry_cost = abs(position) * cost_rate
            costs[t] += entry_cost
            pnl[t] -= entry_cost
            current_trade_pnl = -entry_cost

    # Close any open trade at the end (charged an exit cost, conservatively)
    if in_trade and position != 0.0:
        close_trade(n - 1)

    gross_pnl = pnl + costs
    cumulative_pnl = np.cumsum(pnl)
    n_trades = len(trades)
    trades_arr = np.array(trades) if trades else np.array([0.0])

    hit_rate = float(np.mean(trades_arr > 0)) if n_trades > 0 else 0.0
    ppt = float(np.mean(trades_arr)) if n_trades > 0 else 0.0

    return BacktestResult(
        pnl=pnl,
        cumulative_pnl=cumulative_pnl,
        sharpe_ratio=_annualized_sharpe(pnl, bars_per_year),
        sharpe_ratio_gross=_annualized_sharpe(gross_pnl, bars_per_year),
        max_drawdown=_max_drawdown_fraction(pnl),
        hit_rate=hit_rate,
        profit_per_trade=ppt,
        n_trades=n_trades,
        total_pnl=float(cumulative_pnl[-1]) if len(cumulative_pnl) > 0 else 0.0,
        total_pnl_gross=float(np.sum(gross_pnl)),
        total_costs=float(np.sum(costs)),
    )
