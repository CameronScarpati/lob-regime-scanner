"""Full data pipeline: load → reconstruct → features → HMM → backtest.

Wires the src modules together and produces DataFrames in the schema
expected by the dashboard components.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import load_directory
from src.book_reconstructor import (
    reconstruct,
    snapshots_to_dataframe,
    resample_snapshots,
)
from src.features import build_feature_matrix
from src.hmm_model import RegimeDetector, REGIME_LABELS
from src.backtest import run_backtest

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


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

    patterns = ["*.jsonl.gz", "*.jsonl", "*.csv.gz", "*.csv"]
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
    resample_interval_us: int = 1_000_000,
    hmm_n_states: int = 3,
) -> dict:
    """Execute the full LOB analysis pipeline.

    Parameters
    ----------
    symbol : str
        Trading pair symbol (e.g. ``BTCUSDT``).
    start / end : str or None
        ISO date strings to filter the data range (e.g. ``2025-01-01``).
    resample_interval_us : int
        Resampling interval in microseconds (default 1s).
    hmm_n_states : int
        Number of HMM states (default 3).

    Returns
    -------
    dict compatible with dashboard components::

        {
            "snapshots": pd.DataFrame,   # book_reconstructor schema + datetime timestamp col
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

    # ── Step 1: Load raw events ──────────────────────────────────────────
    logger.info("Loading raw events for %s from %s ...", symbol, data_dir)
    events = load_directory(data_dir, symbol=symbol)
    if events.empty:
        raise NoDataError(f"Loaded 0 events for {symbol}. Check your data files.")

    # Filter by date range if specified
    if start is not None:
        start_us = int(pd.Timestamp(start).timestamp() * 1e6)
        events = events[events["timestamp_us"] >= start_us]
    if end is not None:
        end_us = int(pd.Timestamp(end).timestamp() * 1e6)
        events = events[events["timestamp_us"] <= end_us]

    if events.empty:
        raise NoDataError(
            f"No events remain after date filtering ({start} – {end})."
        )

    logger.info("Loaded %d raw events", len(events))

    # ── Step 2: Reconstruct order book snapshots ─────────────────────────
    logger.info("Reconstructing order book ...")
    raw_snapshots = reconstruct(events)
    snap_df = snapshots_to_dataframe(raw_snapshots)
    if snap_df.empty:
        raise NoDataError("Order book reconstruction produced 0 snapshots.")

    snap_df = resample_snapshots(snap_df, interval_us=resample_interval_us)

    # Add placeholder trade fields if missing
    if "last_trade_price" not in snap_df.columns:
        snap_df["last_trade_price"] = np.nan
        snap_df["last_trade_qty"] = np.nan
        snap_df["last_trade_side"] = ""

    logger.info("Resampled to %d snapshots", len(snap_df))

    # ── Step 3: Compute features ─────────────────────────────────────────
    logger.info("Computing features ...")
    feature_matrix = build_feature_matrix(snap_df)

    # ── Step 4: Fit HMM and decode regimes ───────────────────────────────
    logger.info("Fitting HMM with %d states ...", hmm_n_states)
    detector = RegimeDetector(
        n_states=hmm_n_states,
        covariance_type="full",
        labels=REGIME_LABELS,
    )
    detector.fit(feature_matrix)
    states = detector.predict(feature_matrix)
    state_probs = detector.predict_proba(feature_matrix)
    trans_mat = detector.transition_matrix()

    logger.info(
        "HMM converged=%s, log-likelihood=%.2f",
        detector.diagnostics.converged,
        detector.diagnostics.log_likelihood,
    )

    # ── Step 5: Run backtest ─────────────────────────────────────────────
    logger.info("Running backtest ...")
    mid = snap_df["mid_price"].values
    returns = np.diff(np.log(mid), prepend=np.log(mid[0]))

    # Use the first OFI column available
    ofi_col = next(
        (c for c in feature_matrix.columns if c.startswith("ofi_") and "_zscore" not in c and "_velocity" not in c),
        None,
    )
    ofi = feature_matrix[ofi_col].values if ofi_col else np.zeros(len(states))

    bt = run_backtest(states, returns, ofi)
    logger.info(
        "Backtest: Sharpe=%.2f, MaxDD=%.4f, Trades=%d",
        bt.sharpe_ratio,
        bt.max_drawdown,
        bt.n_trades,
    )

    # ── Step 6: Prepare dashboard-compatible output ──────────────────────
    # Convert microsecond timestamps to datetime for dashboard display
    snap_out = snap_df.copy()
    snap_out["timestamp"] = pd.to_datetime(snap_out["timestamp"], unit="us")

    # Build features DataFrame with the column names the dashboard expects
    feat_out = pd.DataFrame({"timestamp": snap_out["timestamp"].values})

    # Map real feature column names → dashboard expected names
    col_map = {
        "ofi_1": "OFI_1",
        "ofi_5": "OFI_5",
        "ofi_10": "OFI_10",
        "ofi_1_velocity": "OFI_velocity",
        "vpin": "VPIN",
        "book_imbalance": "book_imbalance",
        "weighted_mid": "weighted_mid",
        "spread_bps": "spread_bps",
        "kyles_lambda": "kyle_lambda",
        "trade_aggression": "trade_aggression",
        "cancellation_ratio": "cancel_ratio",
        "rvol_1s": "realized_vol_1s",
        "rvol_10s": "realized_vol_10s",
        "rvol_60s": "realized_vol_60s",
        "rvol_300s": "realized_vol_300s",
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
            "transition_matrix": trans_mat,
        },
        "cumulative_pnl": bt.cumulative_pnl,
    }
