"""Full data pipeline: load > features > HMM > backtest.

Wires the src modules together and produces DataFrames in the schema
expected by the dashboard components.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import run_backtest
from src.data_loader import load_snapshots_directory
from src.features import HMM_FEATURE_COLS, build_feature_matrix, estimate_vpin_bucket_volume
from src.hmm_model import REGIME_LABELS, RegimeDetector

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def _resolve_split(n_rows: int, train_frac: float | None, n_states: int) -> int | None:
    """Index splitting train from held-out data, or None to fit in-sample.

    The training slice must be large enough to actually fit the HMM (k-means
    init needs at least ``n_states`` rows, and a usable transition matrix
    needs many more), so tiny inputs fall back to the in-sample path rather
    than raising from inside hmmlearn.
    """
    if train_frac is None:
        return None
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")

    split_idx = int(n_rows * train_frac)
    min_train = max(n_states * 4, 20)
    if split_idx < min_train or n_rows - split_idx < 2:
        logger.warning(
            "Too few samples (%d) for a %.0f%% walk-forward split "
            "(train slice would be %d, need >= %d); fitting in-sample.",
            n_rows,
            train_frac * 100,
            split_idx,
            min_train,
        )
        return None
    return split_idx


class NoDataError(Exception):
    """Raised when no local data files are found."""


def _find_data_files(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
) -> Path:
    """Verify that data files exist for the given symbol and return the data dir.

    Raises NoDataError with a helpful message if nothing is found.
    """
    if not DATA_DIR.exists():
        raise NoDataError(
            f"Data directory {DATA_DIR} does not exist.\n"
            "Run `python data/download.py` to fetch order book snapshots first."
        )

    patterns = ["*.csv.gz", "*.csv"]
    files = []
    for pat in patterns:
        files.extend(DATA_DIR.glob(pat))

    files = [f for f in files if symbol.upper() in f.name.upper()]

    if not files:
        raise NoDataError(
            f"No data files for {symbol} found in {DATA_DIR}.\n"
            f"Run `python data/download.py --symbol {symbol}` to download data first."
        )

    return DATA_DIR


def run_pipeline(
    symbol: str = "BTCUSDT",
    start: str | None = None,
    end: str | None = None,
    sample_interval_us: int = 100_000,
    hmm_n_states: int = 3,
    train_frac: float | None = 0.7,
    fee_bps: float = 5.5,
    slippage_bps: float = 0.5,
) -> dict:
    """Execute the full LOB analysis pipeline.

    Parameters
    ----------
    symbol : str
        Trading pair symbol (e.g. ``BTCUSDT``).
    start / end : str or None
        ISO date strings to filter the data range (e.g. ``2025-01-01``).
    sample_interval_us : int
        Snapshot subsampling interval in microseconds (default 100ms).
    hmm_n_states : int
        Number of HMM states (default 3).
    train_frac : float or None
        Walk-forward split fraction. The HMM (including its feature scaler
        and VPIN bucket sizing) is fit on the first ``train_frac`` of the
        data only, then decodes the full series causally, and backtest
        statistics are reported separately for the train (in-sample) and
        held-out (out-of-sample) segments. ``None`` fits on the full series
        (fully in-sample, legacy behavior). Inputs too small to support a
        split fall back to the in-sample path with a warning.
    fee_bps / slippage_bps : float
        Per-side transaction cost assumptions passed to the backtest.

    Returns
    -------
    dict compatible with dashboard components::

        {
            "snapshots": pd.DataFrame,   # snapshots schema + datetime timestamp col
            "features": pd.DataFrame,    # columns expected by diagnostics panel
            "hmm": {
                "states": np.ndarray,
                "state_probs": np.ndarray,
                "transition_matrix": np.ndarray,
            },
            "cumulative_pnl": np.ndarray,
        }

    Raises
    ------
    NoDataError
        If no local data files are found.
    """
    data_dir = _find_data_files(symbol, start, end)

    # ── Step 1: Load snapshots directly from Tardis CSV ──────────────────
    logger.info("Loading snapshots for %s from %s ...", symbol, data_dir)
    snap_df = load_snapshots_directory(
        data_dir,
        symbol=symbol,
        sample_interval_us=sample_interval_us,
        start=start,
        end=end,
    )
    if snap_df.empty:
        raise NoDataError(f"Loaded 0 snapshots for {symbol}. Check your data files.")

    # Filter by date range if specified
    if start is not None:
        start_us = int(pd.Timestamp(start).timestamp() * 1e6)
        snap_df = snap_df[snap_df["timestamp"] >= start_us]
    if end is not None:
        end_ts = pd.Timestamp(end)
        if end_ts == end_ts.normalize():
            end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        end_us = int(end_ts.timestamp() * 1e6)
        snap_df = snap_df[snap_df["timestamp"] <= end_us]

    if snap_df.empty:
        raise NoDataError(f"No snapshots remain after date filtering ({start} – {end}).")

    snap_df = snap_df.reset_index(drop=True)

    # Add placeholder trade fields if missing
    if "last_trade_price" not in snap_df.columns:
        snap_df["last_trade_price"] = np.nan
        snap_df["last_trade_qty"] = np.nan
        snap_df["last_trade_side"] = ""

    logger.info("Loaded %d snapshots", len(snap_df))

    # ── Step 2: Compute features ─────────────────────────────────────────
    # Resolve the walk-forward split first: VPIN's volume-bucket size is a
    # full-sample statistic of whatever frame it sees, so it must be
    # estimated from the training segment alone or held-out volume leaks
    # into every VPIN value.
    split_idx = _resolve_split(len(snap_df), train_frac, hmm_n_states)
    train_slice = snap_df.iloc[:split_idx] if split_idx is not None else snap_df
    vpin_bucket_volume = estimate_vpin_bucket_volume(train_slice)

    # Build raw (un-z-scored) features.  The HMM's internal StandardScaler
    # handles normalisation; rolling z-score would remove the very
    # heteroscedasticity the HMM needs to distinguish regimes.
    logger.info("Computing features ...")
    feature_matrix = build_feature_matrix(
        snap_df,
        standardize=False,
        vpin_bucket_volume=vpin_bucket_volume,
    )

    # Select a curated subset for HMM to avoid curse of dimensionality
    # (30+ features with full covariance → ~1400 params for 3 states).
    hmm_cols = [c for c in HMM_FEATURE_COLS if c in feature_matrix.columns]
    hmm_features = feature_matrix[hmm_cols]

    # ── Step 3: Fit HMM and decode regimes ───────────────────────────────
    n_samples = len(hmm_features)
    fit_features = hmm_features.iloc[:split_idx] if split_idx is not None else hmm_features
    logger.info(
        "Fitting HMM with %d states on %d features (%d of %d samples) ...",
        hmm_n_states,
        len(hmm_cols),
        len(fit_features),
        n_samples,
    )
    detector = RegimeDetector(
        n_states=hmm_n_states,
        covariance_type="diag",
        labels=REGIME_LABELS,
    )
    # Fitting on the train slice only keeps the scaler causal for the
    # held-out segment: no full-sample statistics leak into it.
    detector.fit(fit_features, n_restarts=10)

    # Causal (forward-filtered) states drive the backtest: the label at bar
    # t uses no observation after t, so it is a signal a live system could
    # actually produce. The Viterbi path is a smoother — its label at t
    # depends on the whole series — so it is kept only for visualization.
    states = detector.predict_filtered(hmm_features)
    state_probs = detector.filtered_proba(hmm_features)
    states_smoothed = detector.predict(hmm_features)
    trans_mat = detector.transition_matrix()

    logger.info(
        "HMM converged=%s, log-likelihood=%.2f",
        detector.diagnostics.converged,
        detector.diagnostics.log_likelihood,
    )

    # ── Step 4: Run backtest ─────────────────────────────────────────────
    logger.info("Running backtest ...")
    mid = snap_df["mid_price"].values
    returns = np.diff(np.log(mid), prepend=np.log(mid[0]))

    # Annualize from the actual bar interval (24/7 crypto market)
    bars_per_year = 365.0 * 24 * 3600 * 1e6 / sample_interval_us

    # Directional signal: canonical CKS OFI at depth 1, falling back to the
    # simple volume-delta proxy.
    ofi_col = next(
        (c for c in ("ofi_cks_1", "ofi_1") if c in feature_matrix.columns),
        None,
    )
    if ofi_col is None:
        ofi_col = next(
            (
                c
                for c in feature_matrix.columns
                if c.startswith("ofi_") and "_velocity" not in c and "_zscore" not in c
            ),
            None,
        )
    ofi = feature_matrix[ofi_col].values if ofi_col else np.zeros(len(states))

    bt_kwargs = {
        "bars_per_year": bars_per_year,
        "fee_bps": fee_bps,
        "slippage_bps": slippage_bps,
    }

    def _stats(bt) -> dict:
        return {
            "sharpe_ratio": round(bt.sharpe_ratio, 2),
            "sharpe_ratio_gross": round(bt.sharpe_ratio_gross, 2),
            "max_drawdown": round(bt.max_drawdown, 4),
            "n_trades": bt.n_trades,
            "hit_rate": round(bt.hit_rate, 3),
            "total_pnl": round(bt.total_pnl, 4),
            "total_costs": round(bt.total_costs, 4),
        }

    if split_idx is not None:
        bt_is = run_backtest(states[:split_idx], returns[:split_idx], ofi[:split_idx], **bt_kwargs)
        bt_oos = run_backtest(states[split_idx:], returns[split_idx:], ofi[split_idx:], **bt_kwargs)
        bt_pnl = np.concatenate(
            [
                bt_is.cumulative_pnl,
                bt_is.cumulative_pnl[-1] + bt_oos.cumulative_pnl,
            ]
        )
        # Headline stats are the held-out segment, net of costs.
        backtest_stats = _stats(bt_oos)
        backtest_stats["in_sample"] = _stats(bt_is)
        backtest_stats["out_of_sample"] = _stats(bt_oos)
        backtest_stats["split_index"] = split_idx
        backtest_stats["train_frac"] = train_frac
        logger.info(
            "Backtest OOS: Sharpe=%.2f (gross %.2f), MaxDD=%.2f%%, Trades=%d, Costs=%.4f",
            bt_oos.sharpe_ratio,
            bt_oos.sharpe_ratio_gross,
            bt_oos.max_drawdown * 100,
            bt_oos.n_trades,
            bt_oos.total_costs,
        )
    else:
        bt = run_backtest(states, returns, ofi, **bt_kwargs)
        bt_pnl = bt.cumulative_pnl
        backtest_stats = _stats(bt)
        backtest_stats["in_sample"] = _stats(bt)
        logger.info(
            "Backtest (in-sample): Sharpe=%.2f, MaxDD=%.2f%%, Trades=%d",
            bt.sharpe_ratio,
            bt.max_drawdown * 100,
            bt.n_trades,
        )

    # ── Step 5: Prepare dashboard-compatible output ──────────────────────

    # Convert microsecond timestamps to datetime for dashboard display
    snap_out = snap_df.copy()
    snap_out["timestamp"] = pd.to_datetime(snap_out["timestamp"], unit="us")

    # Build features DataFrame with the column names the dashboard expects.
    # Features are raw (not z-scored) so values have meaningful units.
    feat_out = pd.DataFrame({"timestamp": snap_out["timestamp"].values})

    col_map = {
        "ofi_1": "OFI_1",
        "ofi_5": "OFI_5",
        "ofi_10": "OFI_10",
        "ofi_1_velocity": "OFI_velocity",
        "vpin": "VPIN",
        "book_imbalance": "book_imbalance",
        "weighted_mid": "weighted_mid",
        "spread_bps": "spread_bps",
        "trade_aggression": "trade_aggression",
        "cancellation_ratio": "cancel_ratio",
        "rvol_1s": "realized_vol_1s",
        "rvol_10s": "realized_vol_10s",
        "rvol_60s": "realized_vol_60s",
        "rvol_300s": "realized_vol_300s",
        "kyles_lambda": "kyle_lambda",
    }
    for src_col, dst_col in col_map.items():
        if src_col in feature_matrix.columns:
            feat_out[dst_col] = feature_matrix[src_col].values
        else:
            feat_out[dst_col] = 0.0

    return {
        "snapshots": snap_out,
        "features": feat_out,
        "hmm": {
            "states": states,
            "state_probs": state_probs,
            "states_smoothed": states_smoothed,
            "transition_matrix": trans_mat,
        },
        "cumulative_pnl": bt_pnl,
        "backtest_stats": backtest_stats,
    }
