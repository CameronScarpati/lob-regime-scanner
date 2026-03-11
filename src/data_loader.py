"""Parse Tardis book_snapshot CSV data into pandas DataFrames.

Handles Tardis book_snapshot CSV (.csv.gz) — Pre-reconstructed snapshots
from tardis.dev with columns: timestamp, local_timestamp,
asks[0..N].price, asks[0..N].amount, bids[0..N].price, bids[0..N].amount.

Provides two loading paths:
  - load(): Expands into per-level events for BookReconstructor (slow, legacy)
  - load_snapshots(): Converts directly to the snapshots DataFrame format
    used by the features pipeline — much faster, skips reconstruct step.
"""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default: keep one snapshot per 100ms — fast enough for microstructure
# signals (OFI, spread dynamics) while keeping data tractable (~864k/day).
DEFAULT_SAMPLE_INTERVAL_US = 100_000


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
        for _i, pcol in enumerate(price_cols):
            acol = pcol.replace(".price", ".amount")
            prices = df[pcol].values.astype(np.float64)
            qtys = df[acol].values.astype(np.float64)

            # Filter out NaN and zero-quantity levels
            mask = np.isfinite(prices) & np.isfinite(qtys) & (qtys > 0)
            n_valid = mask.sum()
            if n_valid == 0:
                continue

            level_df = pd.DataFrame(
                {
                    "timestamp_us": timestamps[mask],
                    "type": "snapshot",
                    "side": side,
                    "price": prices[mask],
                    "qty": qtys[mask],
                    "update_id": np.int64(0),
                    "seq": np.int64(0),
                }
            )
            frames.append(level_df)

    if not frames:
        return pd.DataFrame(
            columns=["timestamp_us", "type", "side", "price", "qty", "update_id", "seq"]
        )

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values("timestamp_us").reset_index(drop=True)

    logger.info(
        "Loaded %d rows from Tardis snapshot %s (%d snapshots, %d+%d levels/side, ts range %.3fs)",
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
            columns=["timestamp_us", "type", "side", "price", "qty", "update_id", "seq"]
        )

    for f in files:
        logger.info("Loading %s ...", f.name)
        frames.append(load(f, **kwargs))

    df = pd.concat(frames, ignore_index=True)
    if sort:
        df = df.sort_values("timestamp_us").reset_index(drop=True)

    logger.info("Loaded %d total rows from %d files", len(df), len(files))
    return df


# ---------------------------------------------------------------------------
# Direct snapshots loading (skips expand→reconstruct round-trip)
# ---------------------------------------------------------------------------


def _detect_levels(columns: list[str]) -> tuple[list[str], list[str]]:
    """Detect ask and bid price columns from CSV headers."""
    ask_cols = sorted(
        [c for c in columns if c.startswith("asks[") and c.endswith("].price")],
        key=lambda c: int(c.split("[")[1].split("]")[0]),
    )
    bid_cols = sorted(
        [c for c in columns if c.startswith("bids[") and c.endswith("].price")],
        key=lambda c: int(c.split("[")[1].split("]")[0]),
    )
    return ask_cols, bid_cols


def load_snapshots(
    path: Path | str,
    n_levels: int = 10,
    sample_interval_us: int = DEFAULT_SAMPLE_INTERVAL_US,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Load Tardis book_snapshot CSV directly into snapshots DataFrame format.

    Converts directly to the schema used by the features pipeline, without
    the intermediate expand→reconstruct step. Much faster for large files.

    Args:
        path: Path to the Tardis CSV (.csv.gz or .csv).
        n_levels: Number of book levels per side in output (default 10).
        sample_interval_us: Keep one snapshot per this many microseconds
            (default 1_000_000 = 1s). Set to 0 to keep all rows.
        max_rows: If set, only read this many CSV rows before subsampling.

    Returns:
        DataFrame with columns:
            timestamp: int64 (microseconds since epoch)
            mid_price: float64
            spread: float64
            bid_price_1..N, bid_qty_1..N: float64
            ask_price_1..N, ask_qty_1..N: float64
    """
    path = Path(path)

    read_kwargs = {}
    if max_rows is not None:
        read_kwargs["nrows"] = max_rows

    logger.info("Reading %s ...", path.name)
    df = pd.read_csv(path, **read_kwargs)

    if df.empty:
        return pd.DataFrame()

    # Timestamps
    ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    timestamps = df[ts_col].values.astype(np.int64)
    if timestamps.max() < 1e15:
        timestamps = timestamps * 1000

    # Subsample: keep one row per interval
    if sample_interval_us > 0 and len(timestamps) > 1:
        keep = np.zeros(len(timestamps), dtype=bool)
        keep[0] = True
        last_kept = timestamps[0]
        for i in range(1, len(timestamps)):
            if timestamps[i] - last_kept >= sample_interval_us:
                keep[i] = True
                last_kept = timestamps[i]
        df = df.loc[keep].reset_index(drop=True)
        timestamps = timestamps[keep]
        logger.info(
            "Subsampled to %d snapshots (interval=%d μs)",
            len(df),
            sample_interval_us,
        )

    # Detect available levels
    ask_price_cols, bid_price_cols = _detect_levels(df.columns.tolist())
    n_avail_ask = len(ask_price_cols)
    n_avail_bid = len(bid_price_cols)
    n_out = min(n_levels, n_avail_ask, n_avail_bid)

    # Build output DataFrame
    result = pd.DataFrame({"timestamp": timestamps})

    # Extract price/qty arrays for top N levels
    for i in range(n_out):
        ask_p = df[f"asks[{i}].price"].values.astype(np.float64)
        ask_q = df[f"asks[{i}].amount"].values.astype(np.float64)
        bid_p = df[f"bids[{i}].price"].values.astype(np.float64)
        bid_q = df[f"bids[{i}].amount"].values.astype(np.float64)

        result[f"ask_price_{i + 1}"] = ask_p
        result[f"ask_qty_{i + 1}"] = ask_q
        result[f"bid_price_{i + 1}"] = bid_p
        result[f"bid_qty_{i + 1}"] = bid_q

    # Compute mid_price and spread from best bid/ask
    result["mid_price"] = (result["bid_price_1"] + result["ask_price_1"]) / 2.0
    result["spread"] = result["ask_price_1"] - result["bid_price_1"]

    logger.info(
        "Loaded %d snapshots from %s (%d levels/side, ts range %.1fs)",
        len(result),
        path.name,
        n_out,
        (timestamps[-1] - timestamps[0]) / 1e6 if len(timestamps) > 1 else 0,
    )
    return result


_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _parse_file_date(path: Path) -> str | None:
    """Extract the YYYY-MM-DD date string embedded in a filename, or None."""
    m = _DATE_RE.search(path.stem)
    return m.group(1) if m else None


def _filter_files_by_date(
    files: list[Path],
    start: str | None,
    end: str | None,
) -> list[Path]:
    """Keep only files whose embedded date falls within [start, end].

    Files without a parseable date are always included (conservative).
    """
    if start is None and end is None:
        return files

    filtered = []
    for f in files:
        file_date = _parse_file_date(f)
        if file_date is None:
            filtered.append(f)
            continue
        if start is not None and file_date < start:
            continue
        if end is not None and file_date > end:
            continue
        filtered.append(f)
    return filtered


def load_snapshots_directory(
    directory: Path | str,
    symbol: str | None = None,
    n_levels: int = 10,
    sample_interval_us: int = DEFAULT_SAMPLE_INTERVAL_US,
    start: str | None = None,
    end: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Load all Tardis CSV files from a directory into a single snapshots DataFrame.

    Args:
        directory: Path to directory containing CSV files.
        symbol: If set, only load files matching this symbol.
        n_levels: Number of book levels per side.
        sample_interval_us: Subsampling interval in microseconds.
        start: ISO date string (e.g. '2024-01-01') — skip files before this date.
        end: ISO date string — skip files after this date.
        **kwargs: Passed to load_snapshots().

    Returns:
        Concatenated snapshots DataFrame sorted by timestamp.
    """
    directory = Path(directory)
    files = []
    for pattern in ["*.csv.gz", "*.csv"]:
        files.extend(directory.glob(pattern))

    if symbol:
        files = [f for f in files if symbol.upper() in f.name.upper()]

    files = _filter_files_by_date(sorted(set(files)), start, end)

    if not files:
        logger.warning("No data files found in %s", directory)
        return pd.DataFrame()

    frames = []
    for f in files:
        frames.append(
            load_snapshots(f, n_levels=n_levels, sample_interval_us=sample_interval_us, **kwargs)
        )

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info("Loaded %d total snapshots from %d files", len(df), len(files))
    return df
