"""Parse Bybit L2 order book data into pandas DataFrames.

Handles two formats:
  1. JSONL (.jsonl.gz) — Line-delimited JSON with snapshot/delta records
     from quote-saver.bycsi.com. Each record has 'type' (snapshot/delta),
     'ts' (ms timestamp), and 'data' with 'b' (bids) and 'a' (asks).
  2. Legacy CSV (.csv.gz) — Older public.bybit.com format with columns:
     timestamp, side, price, qty.

The primary output is a DataFrame of raw order book events that can be
fed into the BookReconstructor.
"""

import gzip
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_jsonl(path: Path | str, max_records: int | None = None) -> pd.DataFrame:
    """Load Bybit JSONL order book data into a DataFrame.

    Each JSON record is flattened into rows — one row per price level
    per update. Snapshot records provide the full book state; delta
    records provide incremental changes.

    Args:
        path: Path to the .jsonl.gz or .jsonl file.
        max_records: If set, stop after this many JSON records (for testing).

    Returns:
        DataFrame with columns:
            timestamp_us: int64 (microseconds since epoch)
            type: str ('snapshot' or 'delta')
            side: str ('bid' or 'ask')
            price: float64
            qty: float64
            update_id: int64
            seq: int64
    """
    path = Path(path)
    records = []
    opener = gzip.open if path.suffix == ".gz" else open

    with opener(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_records is not None and i >= max_records:
                break
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON at line %d", i)
                continue

            ts_ms = msg.get("ts", 0)
            ts_us = int(ts_ms * 1000)  # ms -> μs
            rec_type = msg.get("type", "unknown")
            data = msg.get("data", {})
            update_id = data.get("u", 0)
            seq = data.get("seq", 0)

            for price_str, qty_str in data.get("b", []):
                records.append(
                    (ts_us, rec_type, "bid", float(price_str), float(qty_str),
                     update_id, seq)
                )
            for price_str, qty_str in data.get("a", []):
                records.append(
                    (ts_us, rec_type, "ask", float(price_str), float(qty_str),
                     update_id, seq)
                )

    if not records:
        return pd.DataFrame(
            columns=["timestamp_us", "type", "side", "price", "qty",
                      "update_id", "seq"]
        )

    df = pd.DataFrame(
        records,
        columns=["timestamp_us", "type", "side", "price", "qty",
                  "update_id", "seq"],
    )
    df["price"] = df["price"].astype(np.float64)
    df["qty"] = df["qty"].astype(np.float64)
    df["update_id"] = df["update_id"].astype(np.int64)
    df["seq"] = df["seq"].astype(np.int64)

    logger.info(
        "Loaded %d rows from %s (%d records, ts range %.3fs)",
        len(df),
        path.name,
        max_records or i + 1,
        (df["timestamp_us"].max() - df["timestamp_us"].min()) / 1e6,
    )
    return df


def load_csv(path: Path | str) -> pd.DataFrame:
    """Load legacy CSV format order book data.

    Expected columns: timestamp, side, price, qty
    Timestamp may be in seconds (float) or microseconds (int).

    Args:
        path: Path to the .csv.gz or .csv file.

    Returns:
        DataFrame with columns:
            timestamp_us: int64 (microseconds since epoch)
            type: str (always 'delta' for CSV format)
            side: str ('bid' or 'ask')
            price: float64
            qty: float64
            update_id: int64 (always 0)
            seq: int64 (always 0)
    """
    path = Path(path)
    df = pd.read_csv(path)

    # Normalize column names (handle various Bybit CSV header formats)
    col_map = {}
    for col in df.columns:
        lower = col.lower().strip()
        if "time" in lower:
            col_map[col] = "timestamp"
        elif lower in ("side",):
            col_map[col] = "side"
        elif lower in ("price",):
            col_map[col] = "price"
        elif lower in ("qty", "quantity", "size", "amount"):
            col_map[col] = "qty"
    df = df.rename(columns=col_map)

    # Convert timestamp to microseconds
    if "timestamp" in df.columns:
        ts = df["timestamp"].astype(np.float64)
        if ts.max() < 1e12:  # seconds
            df["timestamp_us"] = (ts * 1e6).astype(np.int64)
        elif ts.max() < 1e15:  # milliseconds
            df["timestamp_us"] = (ts * 1e3).astype(np.int64)
        else:  # already microseconds
            df["timestamp_us"] = ts.astype(np.int64)
    else:
        raise ValueError(f"No timestamp column found in {path}")

    # Normalize side
    df["side"] = df["side"].str.lower().str.strip()
    df["side"] = df["side"].replace({"buy": "bid", "sell": "ask", "b": "bid", "a": "ask"})

    df["type"] = "delta"
    df["update_id"] = np.int64(0)
    df["seq"] = np.int64(0)

    result = df[["timestamp_us", "type", "side", "price", "qty",
                  "update_id", "seq"]].copy()
    result["price"] = result["price"].astype(np.float64)
    result["qty"] = result["qty"].astype(np.float64)

    logger.info("Loaded %d rows from CSV %s", len(result), path.name)
    return result


def load(path: Path | str, **kwargs) -> pd.DataFrame:
    """Auto-detect format and load order book data.

    Args:
        path: Path to data file (.jsonl.gz, .jsonl, .csv.gz, or .csv).
        **kwargs: Passed to the underlying loader.

    Returns:
        Unified DataFrame with consistent schema.
    """
    path = Path(path)
    name = path.name.lower()

    if ".jsonl" in name:
        return load_jsonl(path, **kwargs)
    elif ".csv" in name:
        return load_csv(path, **kwargs)
    else:
        # Try JSONL first, fall back to CSV
        try:
            return load_jsonl(path, **kwargs)
        except (json.JSONDecodeError, KeyError):
            return load_csv(path, **kwargs)


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

    patterns = ["*.jsonl.gz", "*.jsonl", "*.csv.gz", "*.csv"]
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
