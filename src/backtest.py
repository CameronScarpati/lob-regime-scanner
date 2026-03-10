"""Simple regime-conditional trading strategy backtest.

Validates that detected regimes contain actionable information by
computing Sharpe ratio, max drawdown, hit rate, and profit per trade.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BacktestResult:
    """Results from a regime-conditional backtest."""

    pnl: np.ndarray = field(default_factory=lambda: np.array([]))
    cumulative_pnl: np.ndarray = field(default_factory=lambda: np.array([]))
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    hit_rate: float = 0.0
    profit_per_trade: float = 0.0
    n_trades: int = 0
    total_pnl: float = 0.0


def run_backtest(
    states: np.ndarray,
    returns: np.ndarray,
    ofi: np.ndarray,
    quiet_state: int = 0,
    trending_state: int = 1,
    toxic_state: int = 2,
    annualization_factor: float = 252.0 * 6.5 * 3600,
) -> BacktestResult:
    """Run a simple regime-conditional strategy.

    Strategy:
    - Enter on Quiet -> Trending transition in OFI direction.
    - Flatten on Toxic detection.

    Parameters
    ----------
    states : ndarray of shape (n_samples,)
        Decoded regime states (integers).
    returns : ndarray of shape (n_samples,)
        Forward returns at each timestep.
    ofi : ndarray of shape (n_samples,)
        Order flow imbalance signal (sign determines direction).
    quiet_state : int
        State label for Quiet regime.
    trending_state : int
        State label for Trending regime.
    toxic_state : int
        State label for Toxic/Stressed regime.
    annualization_factor : float
        Factor to annualize Sharpe ratio (default assumes 1s bars,
        ~252 days * 6.5h * 3600s).

    Returns
    -------
    BacktestResult with performance metrics.
    """
    n = len(states)
    if n < 2:
        return BacktestResult()

    position = 0.0  # +1 long, -1 short, 0 flat
    pnl = np.zeros(n)
    trades: list[float] = []
    current_trade_pnl = 0.0
    in_trade = False

    for t in range(1, n):
        prev_state = states[t - 1]
        curr_state = states[t]

        # Entry: Quiet -> Trending transition
        if prev_state == quiet_state and curr_state == trending_state and position == 0.0:
            position = 1.0 if ofi[t] > 0 else -1.0
            in_trade = True
            current_trade_pnl = 0.0

        # Exit: Toxic detection
        if curr_state == toxic_state and position != 0.0:
            pnl[t] = position * returns[t]
            current_trade_pnl += pnl[t]
            trades.append(current_trade_pnl)
            position = 0.0
            in_trade = False
            continue

        # Also exit if we go back to Quiet while in a trade
        if curr_state == quiet_state and position != 0.0:
            pnl[t] = position * returns[t]
            current_trade_pnl += pnl[t]
            trades.append(current_trade_pnl)
            position = 0.0
            in_trade = False
            continue

        # Accumulate PnL while in position
        if position != 0.0:
            pnl[t] = position * returns[t]
            current_trade_pnl += pnl[t]

    # Close any open trade at the end
    if in_trade and position != 0.0:
        trades.append(current_trade_pnl)

    cumulative_pnl = np.cumsum(pnl)
    n_trades = len(trades)
    trades_arr = np.array(trades) if trades else np.array([0.0])

    # Sharpe ratio
    if np.std(pnl) > 0:
        sharpe = (np.mean(pnl) / np.std(pnl)) * np.sqrt(annualization_factor)
    else:
        sharpe = 0.0

    # Max drawdown
    peak = np.maximum.accumulate(cumulative_pnl)
    drawdown = peak - cumulative_pnl
    max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

    # Hit rate
    if n_trades > 0:
        hit_rate = float(np.mean(trades_arr > 0))
    else:
        hit_rate = 0.0

    # Profit per trade
    ppt = float(np.mean(trades_arr)) if n_trades > 0 else 0.0

    return BacktestResult(
        pnl=pnl,
        cumulative_pnl=cumulative_pnl,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        hit_rate=hit_rate,
        profit_per_trade=ppt,
        n_trades=n_trades,
        total_pnl=float(cumulative_pnl[-1]) if len(cumulative_pnl) > 0 else 0.0,
    )
