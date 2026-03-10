"""Bybit historical L2 order book data downloader.

Downloads compressed data files from Bybit's public data servers
for a given symbol and date range. Supports both linear perpetual
contracts and spot markets.

Data source: https://quote-saver.bycsi.com/orderbook/linear/{symbol}/
File format: ZIP archives containing line-delimited JSON with
snapshot and delta order book updates (200 levels).
"""

import gzip
import io
import logging
import os
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

DEFAULT_RAW_DIR = Path(__file__).parent / "raw"

BASE_URL = "https://quote-saver.bycsi.com/orderbook"

# Bybit changed data hosting over time; fall back if needed
LEGACY_BASE_URL = "https://public.bybit.com/orderbook"


def _build_url(symbol: str, dt: date, market: str = "linear") -> str:
    """Build the download URL for a given symbol and date.

    Args:
        symbol: Trading pair (e.g. 'BTCUSDT').
        dt: Date to download.
        market: Market type ('linear', 'inverse', 'spot').

    Returns:
        Full URL to the data file.
    """
    date_str = dt.strftime("%Y-%m-%d")
    filename = f"{date_str}_{symbol}_ob200.data.zip"
    return f"{BASE_URL}/{market}/{symbol}/{filename}"


def _build_legacy_url(symbol: str, dt: date) -> str:
    """Build legacy download URL (public.bybit.com format)."""
    date_str = dt.strftime("%Y-%m-%d")
    return f"{LEGACY_BASE_URL}/{symbol}/{date_str}.csv.gz"


def download_day(
    symbol: str,
    dt: date,
    output_dir: Path | None = None,
    market: str = "linear",
    timeout: int = 120,
) -> Path | None:
    """Download order book data for a single day.

    Attempts the primary URL first, then falls back to legacy format.

    Args:
        symbol: Trading pair (e.g. 'BTCUSDT').
        dt: Date to download.
        output_dir: Directory to save the file. Defaults to data/raw/.
        market: Market type for URL construction.
        timeout: Request timeout in seconds.

    Returns:
        Path to the downloaded file, or None if download failed.
    """
    if output_dir is None:
        output_dir = DEFAULT_RAW_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    date_str = dt.strftime("%Y-%m-%d")
    out_path = output_dir / f"{symbol}_{date_str}.jsonl.gz"

    if out_path.exists():
        logger.info("Already downloaded: %s", out_path.name)
        return out_path

    # Try primary URL (ZIP with JSON)
    url = _build_url(symbol, dt, market)
    logger.info("Downloading %s ...", url)

    try:
        resp = requests.get(url, timeout=timeout, stream=True)
        if resp.status_code == 200:
            return _save_zip_as_jsonl_gz(resp.content, out_path)
        logger.warning("Primary URL returned %d, trying legacy ...", resp.status_code)
    except requests.RequestException as e:
        logger.warning("Primary URL failed: %s, trying legacy ...", e)

    # Try legacy URL (csv.gz)
    legacy_url = _build_legacy_url(symbol, dt)
    legacy_out = output_dir / f"{symbol}_{date_str}.csv.gz"
    try:
        resp = requests.get(legacy_url, timeout=timeout, stream=True)
        if resp.status_code == 200:
            legacy_out.write_bytes(resp.content)
            logger.info("Saved (legacy): %s", legacy_out.name)
            return legacy_out
        logger.error("Legacy URL also returned %d", resp.status_code)
    except requests.RequestException as e:
        logger.error("Legacy URL also failed: %s", e)

    return None


def _save_zip_as_jsonl_gz(zip_bytes: bytes, out_path: Path) -> Path:
    """Extract JSON lines from a ZIP archive and save as gzipped JSONL.

    Bybit packages orderbook data as a ZIP containing one or more data
    files with line-delimited JSON records.
    """
    buf = io.BytesIO(zip_bytes)
    with zipfile.ZipFile(buf) as zf:
        lines = []
        for name in zf.namelist():
            with zf.open(name) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        lines.append(line)

    with gzip.open(out_path, "wb") as f:
        for line in lines:
            if isinstance(line, str):
                line = line.encode("utf-8")
            f.write(line + b"\n")

    logger.info("Saved: %s (%d records)", out_path.name, len(lines))
    return out_path


def download_range(
    symbol: str,
    start: date | str,
    end: date | str,
    output_dir: Path | None = None,
    market: str = "linear",
) -> list[Path]:
    """Download order book data for a date range.

    Args:
        symbol: Trading pair (e.g. 'BTCUSDT').
        start: Start date (inclusive). String 'YYYY-MM-DD' or date object.
        end: End date (inclusive). String 'YYYY-MM-DD' or date object.
        output_dir: Directory to save files.
        market: Market type.

    Returns:
        List of paths to successfully downloaded files.
    """
    if isinstance(start, str):
        start = datetime.strptime(start, "%Y-%m-%d").date()
    if isinstance(end, str):
        end = datetime.strptime(end, "%Y-%m-%d").date()

    if end < start:
        raise ValueError(f"end ({end}) must be >= start ({start})")

    paths = []
    current = start
    while current <= end:
        path = download_day(symbol, current, output_dir, market)
        if path is not None:
            paths.append(path)
        current += timedelta(days=1)

    logger.info(
        "Downloaded %d/%d days for %s",
        len(paths),
        (end - start).days + 1,
        symbol,
    )
    return paths


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Download Bybit L2 orderbook data")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument(
        "--market", default="linear", choices=["linear", "inverse", "spot"]
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else None
    download_range(args.symbol, args.start, args.end, output_dir, args.market)
