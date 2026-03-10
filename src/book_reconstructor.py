"""Order book snapshot reconstructor.

Maintains full bid/ask book as sorted price-level dictionaries,
applies snapshot and delta updates from parsed Bybit data, resamples
to uniform time intervals, and outputs Parquet snapshots.

Output schema (per the project spec):
    timestamp: int64 (microseconds since epoch)
    mid_price: float64
    spread: float64
    bid_price_1..N: float64
    bid_qty_1..N: float64
    ask_price_1..N: float64
    ask_qty_1..N: float64
    last_trade_price: float64
    last_trade_qty: float64
    last_trade_side: str
"""

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

N_LEVELS = 10  # Number of book levels to include in output snapshots

# Try importing the C++ backend; fall back to pure Python if unavailable.
try:
    from src.cpp._lob_cpp import LOBEngine as _CppLOBEngine
    from src.cpp._lob_cpp import batch_reconstruct as _cpp_batch_reconstruct

    _CPP_AVAILABLE = True
    logger.info("C++ LOB engine loaded")
except ImportError:
    _CPP_AVAILABLE = False
    logger.debug("C++ LOB engine not available, using pure Python")


class OrderBook:
    """In-memory order book maintaining bid and ask price levels.

    Bids are stored highest-price-first; asks lowest-price-first.
    A quantity of zero removes the level.
    """

    __slots__ = ("bids", "asks", "last_update_ts")

    def __init__(self) -> None:
        self.bids: dict[float, float] = {}  # price -> qty
        self.asks: dict[float, float] = {}  # price -> qty
        self.last_update_ts: int = 0

    def update(self, side: str, price: float, qty: float) -> None:
        """Apply a single price-level update."""
        book = self.bids if side == "bid" else self.asks
        if qty <= 0:
            book.pop(price, None)
        else:
            book[price] = qty

    def apply_snapshot(self, side: str, levels: list[tuple[float, float]]) -> None:
        """Replace one side of the book with a full snapshot."""
        book = {}
        for price, qty in levels:
            if qty > 0:
                book[price] = qty
        if side == "bid":
            self.bids = book
        else:
            self.asks = book

    def best_bid(self) -> tuple[float, float] | None:
        """Return (price, qty) of the best bid, or None."""
        if not self.bids:
            return None
        price = max(self.bids)
        return (price, self.bids[price])

    def best_ask(self) -> tuple[float, float] | None:
        """Return (price, qty) of the best ask, or None."""
        if not self.asks:
            return None
        price = min(self.asks)
        return (price, self.asks[price])

    def mid_price(self) -> float | None:
        """Return the mid-price, or None if book is empty."""
        bb = self.best_bid()
        ba = self.best_ask()
        if bb is None or ba is None:
            return None
        return (bb[0] + ba[0]) / 2.0

    def spread(self) -> float | None:
        """Return the bid-ask spread, or None if book is empty."""
        bb = self.best_bid()
        ba = self.best_ask()
        if bb is None or ba is None:
            return None
        return ba[0] - bb[0]

    def top_n(self, side: str, n: int = N_LEVELS) -> list[tuple[float, float]]:
        """Return the top N levels for a given side.

        Bids: sorted descending by price (best bid first).
        Asks: sorted ascending by price (best ask first).
        Pads with (NaN, NaN) if fewer than N levels exist.
        """
        if side == "bid":
            levels = sorted(self.bids.items(), key=lambda x: -x[0])
        else:
            levels = sorted(self.asks.items(), key=lambda x: x[0])

        levels = levels[:n]
        # Pad to exactly n levels
        while len(levels) < n:
            levels.append((np.nan, np.nan))
        return levels

    def snapshot_dict(self, timestamp_us: int, n_levels: int = N_LEVELS) -> dict:
        """Produce a flat dictionary snapshot of the book state.

        Returns a dict matching the project spec schema.
        """
        mid = self.mid_price()
        sprd = self.spread()

        row: dict = {
            "timestamp": timestamp_us,
            "mid_price": mid if mid is not None else np.nan,
            "spread": sprd if sprd is not None else np.nan,
        }

        bid_levels = self.top_n("bid", n_levels)
        ask_levels = self.top_n("ask", n_levels)

        for i, (p, q) in enumerate(bid_levels, start=1):
            row[f"bid_price_{i}"] = p
            row[f"bid_qty_{i}"] = q

        for i, (p, q) in enumerate(ask_levels, start=1):
            row[f"ask_price_{i}"] = p
            row[f"ask_qty_{i}"] = q

        return row


def _reconstruct_python(
    events: pd.DataFrame,
    n_levels: int = N_LEVELS,
) -> list[dict]:
    """Pure-Python order book reconstruction (fallback)."""
    book = OrderBook()
    snapshots: list[dict] = []
    current_ts: int = -1

    grouped = events.sort_values(["timestamp_us", "seq"]).groupby(
        ["timestamp_us", "type", "update_id"], sort=True
    )

    for (ts_us, rec_type, _uid), group in grouped:
        ts_us = int(ts_us)

        if rec_type == "snapshot":
            for side in ("bid", "ask"):
                side_rows = group[group["side"] == side]
                if not side_rows.empty:
                    levels = list(
                        zip(side_rows["price"].values, side_rows["qty"].values)
                    )
                    book.apply_snapshot(side, levels)
        else:
            for _, row in group.iterrows():
                book.update(row["side"], row["price"], row["qty"])

        book.last_update_ts = ts_us

        if ts_us != current_ts:
            if current_ts >= 0:
                snapshots.append(book.snapshot_dict(current_ts, n_levels))
            current_ts = ts_us

    if current_ts >= 0:
        snapshots.append(book.snapshot_dict(current_ts, n_levels))

    logger.info("Reconstructed %d snapshots (Python)", len(snapshots))
    return snapshots


def _reconstruct_cpp(
    events: pd.DataFrame,
    n_levels: int = N_LEVELS,
) -> list[dict]:
    """C++ accelerated order book reconstruction."""
    events = events.sort_values(["timestamp_us", "seq"]).reset_index(drop=True)

    # Encode string columns to int arrays for C++
    type_map = {"snapshot": 0, "delta": 1}
    side_map = {"bid": 0, "ask": 1}

    timestamps = events["timestamp_us"].values.astype(np.int64)
    types = events["type"].map(type_map).values.astype(np.int32)
    sides = events["side"].map(side_map).values.astype(np.int32)
    prices = events["price"].values.astype(np.float64)
    qtys = events["qty"].values.astype(np.float64)
    update_ids = events["update_id"].values.astype(np.int64)

    result_dict = _cpp_batch_reconstruct(
        timestamps, types, sides, prices, qtys, update_ids, n_levels
    )

    # Convert the dict of arrays to a list of dicts (matching Python API)
    if not result_dict or "timestamp" not in result_dict:
        return []

    n_rows = len(result_dict["timestamp"])
    keys = list(result_dict.keys())
    snapshots = []
    for i in range(n_rows):
        row = {}
        for k in keys:
            val = result_dict[k][i]
            if k == "timestamp":
                row[k] = int(val)
            else:
                row[k] = float(val)
        snapshots.append(row)

    logger.info("Reconstructed %d snapshots (C++)", len(snapshots))
    return snapshots


def reconstruct(
    events: pd.DataFrame,
    n_levels: int = N_LEVELS,
    use_cpp: bool | None = None,
) -> list[dict]:
    """Reconstruct order book snapshots from a stream of events.

    Processes events in timestamp order. Snapshot-type events replace
    the entire side; delta-type events update individual levels.

    Args:
        events: DataFrame from data_loader with columns:
            timestamp_us, type, side, price, qty, update_id, seq
        n_levels: Number of price levels per side in output.
        use_cpp: Force C++ (True), force Python (False), or auto-detect (None).

    Returns:
        List of snapshot dicts, one per unique timestamp.
    """
    if events.empty:
        return []

    if use_cpp is None:
        use_cpp = _CPP_AVAILABLE

    if use_cpp and not _CPP_AVAILABLE:
        raise RuntimeError(
            "C++ LOB engine requested but not available. "
            "Rebuild with: pip install -e '.[dev]'"
        )

    if use_cpp:
        return _reconstruct_cpp(events, n_levels)
    return _reconstruct_python(events, n_levels)


def snapshots_to_dataframe(snapshots: list[dict]) -> pd.DataFrame:
    """Convert a list of snapshot dicts to a DataFrame."""
    if not snapshots:
        return pd.DataFrame()
    return pd.DataFrame(snapshots)


def resample_snapshots(
    df: pd.DataFrame,
    interval_us: int = 1_000_000,
    method: Literal["ffill", "nearest"] = "ffill",
) -> pd.DataFrame:
    """Resample snapshot DataFrame to uniform time intervals.

    Args:
        df: DataFrame from snapshots_to_dataframe with 'timestamp' in μs.
        interval_us: Resampling interval in microseconds.
            1_000_000 = 1 second, 100_000 = 100ms.
        method: Resampling method. 'ffill' carries forward the last
            known state; 'nearest' picks the closest snapshot.

    Returns:
        DataFrame resampled at uniform intervals.
    """
    if df.empty:
        return df

    df = df.sort_values("timestamp").reset_index(drop=True)

    ts_min = df["timestamp"].min()
    ts_max = df["timestamp"].max()

    # Create uniform timestamp grid
    grid = np.arange(ts_min, ts_max + 1, interval_us)

    if method == "ffill":
        # For each grid point, find the last snapshot at or before it
        idx = np.searchsorted(df["timestamp"].values, grid, side="right") - 1
        idx = np.clip(idx, 0, len(df) - 1)
        result = df.iloc[idx].copy()
        result["timestamp"] = grid[: len(result)]
        result = result.reset_index(drop=True)
    else:
        # Nearest snapshot
        idx = np.searchsorted(df["timestamp"].values, grid, side="left")
        idx = np.clip(idx, 0, len(df) - 1)
        result = df.iloc[idx].copy()
        result["timestamp"] = grid[: len(result)]
        result = result.reset_index(drop=True)

    logger.info(
        "Resampled %d snapshots to %d at %d μs intervals",
        len(df),
        len(result),
        interval_us,
    )
    return result


def save_parquet(df: pd.DataFrame, path: Path | str) -> Path:
    """Save a snapshot DataFrame to Parquet format.

    Args:
        df: Snapshot DataFrame.
        path: Output file path (should end in .parquet).

    Returns:
        Path to the saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=False)
    logger.info("Saved %d rows to %s", len(df), path)
    return path


def load_parquet(path: Path | str) -> pd.DataFrame:
    """Load a snapshot DataFrame from Parquet format."""
    return pd.read_parquet(path, engine="pyarrow")


def process_events_to_parquet(
    events: pd.DataFrame,
    output_path: Path | str,
    n_levels: int = N_LEVELS,
    interval_us: int = 1_000_000,
) -> pd.DataFrame:
    """Full pipeline: events -> reconstruct -> resample -> save Parquet.

    Args:
        events: Raw events DataFrame from data_loader.
        output_path: Path for the output Parquet file.
        n_levels: Number of book levels per side.
        interval_us: Resampling interval in microseconds.

    Returns:
        The resampled snapshot DataFrame.
    """
    snapshots = reconstruct(events, n_levels=n_levels)
    df = snapshots_to_dataframe(snapshots)
    if df.empty:
        logger.warning("No snapshots produced")
        return df

    df = resample_snapshots(df, interval_us=interval_us)

    # Add placeholder trade fields (populated downstream if trade data available)
    if "last_trade_price" not in df.columns:
        df["last_trade_price"] = np.nan
        df["last_trade_qty"] = np.nan
        df["last_trade_side"] = ""

    save_parquet(df, output_path)
    return df
