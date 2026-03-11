"""Parse Tardis book_snapshot CSV data into pandas DataFrames.

Handles Tardis book_snapshot CSV (.csv.gz) — Pre-reconstructed snapshots
from tardis.dev with columns: timestamp, local_timestamp,
asks[0..N].price, asks[0..N].amount, bids[0..N].price, bids[0..N].amount.

The primary output is a DataFrame of order book events that can be
fed into the BookReconstructor.
"""

import gzip
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load(path: Path | str, max_rows: int | None = None) -> pd.DataFrame:
    """Load a Tardis book_snapshot CSV and convert to the event format.

    Tardis book_snapshot_25 files have columns:
        timestamp, local_timestamp,
        asks[0].price, asks[0].amount, ..., asks[24].price, asks[24].amount,
        bids[0].price, bids[0].amount, ..., bids[24].price, bids[24].amount

    Each row is a full snapshot of the top N levels. We expand each row
    into individual price-level events compatible with the rest of the
    pipeline.

    Args:
        path: Path to the Tardis CSV (.csv.gz or .csv).
        max_rows: If set, only read this many snapshot rows.

    Returns:
        DataFrame with columns:
            timestamp_us: int64 (microseconds since epoch)
            type: str (always 'snapshot')
            side: str ('bid' or 'ask')
            price: float64
            qty: float64
            update_id: int64 (always 0)
            seq: int64 (always 0)
    """
    path = Path(path)

    read_kwargs = {}
    if max_rows is not None:
        read_kwargs["nrows"] = max_rows

    df = pd.read_csv(path, **read_kwargs)

    # Detect the number of levels per side
    ask_price_cols = sorted(
        [c for c in df.columns if c.startswith("asks[") and c.endswith("].price")],
        key=lambda c: int(c.split("[")[1].split("]")[0]),
    )
    bid_price_cols = sorted(
        [c for c in df.columns if c.startswith("bids[") and c.endswith("].price")],
        key=lambda c: int(c.split("[")[1].split("]")[0]),
    )

    n_ask_levels = len(ask_price_cols)
    n_bid_levels = len(bid_price_cols)

    # Timestamp: Tardis provides microseconds since epoch
    ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    timestamps = df[ts_col].values.astype(np.int64)

    # If timestamps look like milliseconds, convert to microseconds
    if len(timestamps) > 0 and timestamps.max() < 1e15:
        timestamps = timestamps * 1000

    n_rows = len(df)
    frames = []

    # Vectorized extraction — process all rows at once per level
    for side, price_cols in [("ask", ask_price_cols), ("bid", bid_price_cols)]:
        for i, pcol in enumerate(price_cols):
            acol = pcol.replace(".price", ".amount")
            prices = df[pcol].values.astype(np.float64)
            qtys = df[acol].values.astype(np.float64)

            # Filter out NaN and zero-quantity levels
            mask = np.isfinite(prices) & np.isfinite(qtys) & (qtys > 0)
            n_valid = mask.sum()
            if n_valid == 0:
                continue

            level_df = pd.DataFrame({
                "timestamp_us": timestamps[mask],
                "type": "snapshot",
                "side": side,
                "price": prices[mask],
                "qty": qtys[mask],
                "update_id": np.int64(0),
                "seq": np.int64(0),
            })
            frames.append(level_df)

    if not frames:
        return pd.DataFrame(
            columns=["timestamp_us", "type", "side", "price", "qty",
                      "update_id", "seq"]
        )

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values("timestamp_us").reset_index(drop=True)

    logger.info(
        "Loaded %d rows from Tardis snapshot %s (%d snapshots, %d+%d levels/side, "
        "ts range %.3fs)",
        len(result),
        path.name,
        n_rows,
        n_bid_levels,
        n_ask_levels,
        (result["timestamp_us"].max() - result["timestamp_us"].min()) / 1e6,
    )
    return result


# Keep old name as alias for backwards compatibility in imports
load_tardis_snapshot = load


def load_directory(
    directory: Path | str,
    symbol: str | None = None,
    sort: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Load all data files from a directory into a single DataFrame.

    Args:
        directory: Path to directory containing data files.
        symbol: If set, only load files matching this symbol.
        sort: Sort by timestamp after concatenation.
        **kwargs: Passed to the underlying loader.

    Returns:
        Concatenated DataFrame from all matching files.
    """
    directory = Path(directory)
    frames = []

    patterns = ["*.csv.gz", "*.csv"]
    files = []
    for pattern in patterns:
        files.extend(directory.glob(pattern))

    if symbol:
        files = [f for f in files if symbol.upper() in f.name.upper()]

    files = sorted(set(files))

    if not files:
        logger.warning("No data files found in %s", directory)
        return pd.DataFrame(
            columns=["timestamp_us", "type", "side", "price", "qty",
                      "update_id", "seq"]
        )

    for f in files:
        logger.info("Loading %s ...", f.name)
        frames.append(load(f, **kwargs))

    df = pd.concat(frames, ignore_index=True)
    if sort:
        df = df.sort_values("timestamp_us").reset_index(drop=True)

    logger.info("Loaded %d total rows from %d files", len(df), len(files))
    return df
