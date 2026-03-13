"""Generate realistic synthetic DataFrames for dashboard development.

Produces mock data matching the schemas expected by the LOB Regime Scanner
pipeline so that the dashboard can be developed independently of Phases 2-3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_LEVELS = 10

_RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Helper: regime sequence via Markov chain
# ---------------------------------------------------------------------------


def _generate_regime_sequence(n: int, transition_matrix: np.ndarray) -> np.ndarray:
    """Sample a regime path from a 3-state Markov chain."""
    states = np.empty(n, dtype=np.int32)
    states[0] = 0
    for i in range(1, n):
        states[i] = _RNG.choice(3, p=transition_matrix[states[i - 1]])
    return states


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_transition_matrix() -> np.ndarray:
    """Return a realistic 3x3 HMM transition matrix."""
    tm = np.array(
        [
            [0.92, 0.05, 0.03],  # Quiet  -> persistent but transitions regularly
            [0.07, 0.87, 0.06],  # Trending -> moderate persistence
            [0.06, 0.08, 0.86],  # Toxic -> high self-persistence
        ]
    )
    return tm


def generate_snapshots(
    n_timestamps: int = 3600,
    start: str = "2025-01-15 09:00:00",
    freq: str = "1s",
) -> pd.DataFrame:
    """Generate a snapshot DataFrame matching ``book_reconstructor`` output.

    Columns: timestamp, mid_price, spread, bid_price_1..10, bid_qty_1..10,
    ask_price_1..10, ask_qty_1..10, last_trade_price, last_trade_qty,
    last_trade_side.
    """
    timestamps = pd.date_range(start, periods=n_timestamps, freq=freq)

    # Simulate a mid-price random walk with drift and volatility shifts
    returns = _RNG.normal(0, 0.0001, size=n_timestamps)
    # Add a trend in the middle third
    trend_start = n_timestamps // 3
    trend_end = 2 * n_timestamps // 3
    returns[trend_start:trend_end] += 0.00005
    mid_price = 42000.0 * np.exp(np.cumsum(returns))

    # Spread widens during volatile periods
    kernel_len = min(60, n_timestamps)
    kernel = np.ones(kernel_len) / kernel_len
    smoothed = np.convolve(returns, kernel, mode="same")[:n_timestamps]
    base_spread = 0.50 + 0.30 * np.abs(smoothed) / 0.0001
    spread = np.clip(base_spread + _RNG.exponential(0.1, n_timestamps), 0.10, 5.0)

    data: dict = {
        "timestamp": timestamps,
        "mid_price": mid_price,
        "spread": spread,
    }

    # Build bid/ask levels
    for i in range(1, N_LEVELS + 1):
        offset = spread / 2 + (i - 1) * 0.50
        data[f"bid_price_{i}"] = mid_price - offset
        data[f"ask_price_{i}"] = mid_price + offset

        # Volume decays with distance from mid; add noise
        base_vol = 5.0 / i
        data[f"bid_qty_{i}"] = np.clip(
            base_vol + _RNG.normal(0, base_vol * 0.3, n_timestamps), 0.01, 50.0
        )
        data[f"ask_qty_{i}"] = np.clip(
            base_vol + _RNG.normal(0, base_vol * 0.3, n_timestamps), 0.01, 50.0
        )

    # Last trade info
    data["last_trade_price"] = mid_price + _RNG.normal(0, 0.2, n_timestamps)
    data["last_trade_qty"] = np.abs(_RNG.exponential(0.5, n_timestamps))
    data["last_trade_side"] = _RNG.choice(["buy", "sell"], n_timestamps)

    return pd.DataFrame(data)


def generate_features(
    n_timestamps: int = 3600,
    start: str = "2025-01-15 09:00:00",
    freq: str = "1s",
    regimes: np.ndarray | None = None,
) -> pd.DataFrame:
    """Generate a feature DataFrame matching ``features.compute_features``.

    Features: OFI_1, OFI_5, OFI_10, OFI_velocity, VPIN, book_imbalance,
    weighted_mid, spread_bps, kyle_lambda, trade_aggression, cancel_ratio,
    realized_vol_1s, realized_vol_10s, realized_vol_60s, realized_vol_300s.
    """
    timestamps = pd.date_range(start, periods=n_timestamps, freq=freq)

    if regimes is None:
        tm = generate_transition_matrix()
        regimes = _generate_regime_sequence(n_timestamps, tm)

    # Regime-conditional feature distributions
    # Quiet=0: low vol, balanced. Trending=1: directional. Toxic=2: extreme.
    ofi_mean = np.where(regimes == 0, 0.0, np.where(regimes == 1, 0.5, -0.3))
    ofi_std = np.where(regimes == 0, 0.3, np.where(regimes == 1, 0.6, 1.2))

    vpin_base = np.where(regimes == 0, 0.25, np.where(regimes == 1, 0.40, 0.70))
    spread_base = np.where(regimes == 0, 1.5, np.where(regimes == 1, 2.5, 5.0))

    data = {
        "timestamp": timestamps,
        "OFI_1": _RNG.normal(ofi_mean, ofi_std),
        "OFI_5": _RNG.normal(ofi_mean * 0.8, ofi_std * 0.9),
        "OFI_10": _RNG.normal(ofi_mean * 0.6, ofi_std * 0.8),
        "OFI_velocity": _RNG.normal(0, 0.1, n_timestamps),
        "VPIN": np.clip(vpin_base + _RNG.normal(0, 0.08, n_timestamps), 0, 1),
        "book_imbalance": np.clip(_RNG.normal(ofi_mean * 0.3, 0.2, n_timestamps), -1, 1),
        "weighted_mid": 42000.0 + _RNG.normal(0, 10, n_timestamps),
        "spread_bps": np.clip(spread_base + _RNG.normal(0, 0.5, n_timestamps), 0.5, 15.0),
        "kyle_lambda": np.clip(
            np.where(regimes == 0, 0.01, np.where(regimes == 1, 0.02, 0.05))
            + _RNG.normal(0, 0.005, n_timestamps),
            0.001,
            0.2,
        ),
        "trade_aggression": np.clip(
            _RNG.beta(2, 5, n_timestamps) + np.where(regimes == 2, 0.2, 0.0),
            0,
            1,
        ),
        "cancel_ratio": np.clip(_RNG.beta(3, 7, n_timestamps), 0, 1),
        "realized_vol_1s": np.abs(
            _RNG.normal(0.0001, 0.00005, n_timestamps) * np.where(regimes == 2, 3, 1)
        ),
        "realized_vol_10s": np.abs(
            _RNG.normal(0.0003, 0.0001, n_timestamps) * np.where(regimes == 2, 3, 1)
        ),
        "realized_vol_60s": np.abs(
            _RNG.normal(0.0008, 0.0003, n_timestamps) * np.where(regimes == 2, 3, 1)
        ),
        "realized_vol_300s": np.abs(
            _RNG.normal(0.002, 0.0007, n_timestamps) * np.where(regimes == 2, 3, 1)
        ),
    }

    return pd.DataFrame(data)


def generate_hmm_output(
    n_timestamps: int = 3600,
) -> dict:
    """Generate HMM decode output: states, probabilities, transition matrix.

    Returns
    -------
    dict with keys:
        states : np.ndarray of shape (n_timestamps,)
        state_probs : np.ndarray of shape (n_timestamps, 3)
        transition_matrix : np.ndarray of shape (3, 3)
    """
    tm = generate_transition_matrix()
    states = _generate_regime_sequence(n_timestamps, tm)

    # Build soft probabilities: mostly confident, with some blurring
    state_probs = np.zeros((n_timestamps, 3))
    for i in range(n_timestamps):
        probs = _RNG.dirichlet([0.5, 0.5, 0.5])
        # Sharpen toward the true state
        probs[states[i]] += 3.0
        probs /= probs.sum()
        state_probs[i] = probs

    # Smooth probabilities with a small moving average for realism
    kernel = np.ones(10) / 10
    for col in range(3):
        state_probs[:, col] = np.convolve(state_probs[:, col], kernel, mode="same")
    # Re-normalize rows
    row_sums = state_probs.sum(axis=1, keepdims=True)
    state_probs = state_probs / row_sums

    return {
        "states": states,
        "state_probs": state_probs,
        "transition_matrix": tm,
    }


def generate_cumulative_pnl(
    n_timestamps: int = 3600,
    regimes: np.ndarray | None = None,
) -> np.ndarray:
    """Simulate cumulative PnL for the regime-conditional strategy."""
    if regimes is None:
        tm = generate_transition_matrix()
        regimes = _generate_regime_sequence(n_timestamps, tm)

    pnl_per_step = np.where(
        regimes == 0,
        _RNG.normal(0.001, 0.01, n_timestamps),
        np.where(
            regimes == 1,
            _RNG.normal(0.005, 0.02, n_timestamps),
            _RNG.normal(-0.002, 0.03, n_timestamps),
        ),
    )
    return np.cumsum(pnl_per_step)


def generate_backtest_stats(cumulative_pnl: np.ndarray) -> dict:
    """Compute summary statistics from cumulative PnL for the statistics bar."""
    pnl_diff = np.diff(cumulative_pnl, prepend=0)
    sharpe = float(np.mean(pnl_diff) / max(np.std(pnl_diff), 1e-10) * np.sqrt(252 * 86400))
    peak = np.maximum.accumulate(cumulative_pnl)
    drawdown = peak - cumulative_pnl
    max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0
    n_trades = int(np.sum(np.abs(np.diff(np.sign(pnl_diff))) > 0))
    hit_rate = float(np.mean(pnl_diff[pnl_diff != 0] > 0)) if np.any(pnl_diff != 0) else 0.0
    return {
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown": round(max_dd, 4),
        "n_trades": max(n_trades, 1),
        "hit_rate": round(hit_rate, 3),
        "total_pnl": round(float(cumulative_pnl[-1]), 4),
    }


def generate_all(
    n_timestamps: int = 3600,
    start: str = "2025-01-15 09:00:00",
    freq: str = "1s",
) -> dict:
    """Generate all mock data in one call.

    Returns dict with keys: snapshots, features, hmm, cumulative_pnl, backtest_stats.
    """
    hmm = generate_hmm_output(n_timestamps)
    snapshots = generate_snapshots(n_timestamps, start, freq)
    features = generate_features(n_timestamps, start, freq, hmm["states"])
    cum_pnl = generate_cumulative_pnl(n_timestamps, hmm["states"])

    return {
        "snapshots": snapshots,
        "features": features,
        "hmm": hmm,
        "cumulative_pnl": cum_pnl,
        "backtest_stats": generate_backtest_stats(cum_pnl),
    }
