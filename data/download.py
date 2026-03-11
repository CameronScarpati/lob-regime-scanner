"""Historical L2 order book data downloader.

Supports multiple data sources:
  1. Bybit (primary): Downloads from quote-saver.bycsi.com and
     public.bybit.com. Note: Bybit orderbook archives are only
     available for recent dates (roughly May 2025 onward).
  2. Tardis.dev: Professional-grade tick-level data for 40+ crypto
     exchanges. Free sample data (1st of each month) without API key.
     Full access requires a paid API key from https://tardis.dev.
"""

import asyncio
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

# ---------------------------------------------------------------------------
# Bybit URLs
# ---------------------------------------------------------------------------
BASE_URL = "https://quote-saver.bycsi.com/orderbook"
LEGACY_BASE_URL = "https://public.bybit.com/orderbook"


def _build_url(symbol: str, dt: date, market: str = "linear") -> str:
    """Build the primary Bybit download URL."""
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
    """Download Bybit order book data for a single day.

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
    """Extract JSON lines from a ZIP archive and save as gzipped JSONL."""
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


# ---------------------------------------------------------------------------
# Tardis.dev downloader
# ---------------------------------------------------------------------------

# Map common symbol names to Tardis exchange identifiers.
# Tardis uses exchange-specific naming; Bybit linear perps use the
# same symbol format as the user would pass (e.g. "BTCUSDT").
TARDIS_EXCHANGE_MAP = {
    "bybit": "bybit",
    "binance": "binance-futures",
    "binance-spot": "binance",
    "okx": "okex-swap",
    "deribit": "deribit",
}


def _tardis_download_sync(
    exchange: str,
    data_types: list[str],
    symbols: list[str],
    from_date: str,
    to_date: str,
    download_dir: str,
    api_key: str = "",
) -> None:
    """Synchronous wrapper around the async tardis-dev download."""
    from tardis_dev import datasets

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(
            datasets.download(
                exchange=exchange,
                data_types=data_types,
                symbols=symbols,
                from_date=from_date,
                to_date=to_date,
                download_dir=download_dir,
                api_key=api_key,
            )
        )
    finally:
        loop.close()


def download_tardis(
    symbol: str,
    start: date | str,
    end: date | str,
    output_dir: Path | None = None,
    exchange: str = "bybit",
    data_type: str = "book_snapshot_25",
    api_key: str = "",
) -> list[Path]:
    """Download order book data via Tardis.dev.

    Tardis.dev provides pre-reconstructed L2 order book snapshots as
    compressed CSV files. Free sample data (1st of each month) is
    available without an API key.

    Args:
        symbol: Trading pair (e.g. 'BTCUSDT').
        start: Start date (inclusive).
        end: End date (inclusive).
        output_dir: Directory to save files. Defaults to data/raw/.
        exchange: Exchange name (bybit, binance, okx, deribit).
        data_type: Tardis data type. Recommended:
            'book_snapshot_25' — top 25 levels per side, snapshot on every change
            'book_snapshot_5'  — top 5 levels (smaller files)
            'incremental_book_L2' — raw incremental updates
        api_key: Tardis API key. Empty string uses free sample data
            (only 1st of each month available).

    Returns:
        List of paths to downloaded CSV files.
    """
    try:
        from tardis_dev import datasets  # noqa: F401
    except ImportError:
        raise ImportError(
            "tardis-dev is required for Tardis downloads. Install with:\n"
            "  pip install tardis-dev"
        )

    if isinstance(start, str):
        start = datetime.strptime(start, "%Y-%m-%d").date()
    if isinstance(end, str):
        end = datetime.strptime(end, "%Y-%m-%d").date()

    if end < start:
        raise ValueError(f"end ({end}) must be >= start ({start})")

    if output_dir is None:
        output_dir = DEFAULT_RAW_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tardis_exchange = TARDIS_EXCHANGE_MAP.get(exchange, exchange)

    from_str = start.strftime("%Y-%m-%d")
    # Tardis to_date is exclusive, so add 1 day
    to_str = (end + timedelta(days=1)).strftime("%Y-%m-%d")

    logger.info(
        "Downloading %s %s from Tardis.dev (%s, %s to %s) ...",
        tardis_exchange,
        symbol,
        data_type,
        from_str,
        end.strftime("%Y-%m-%d"),
    )
    if not api_key:
        logger.info(
            "No API key provided — only free sample data (1st of each month) "
            "will be available. Get an API key at https://tardis.dev"
        )

    # Tardis downloads files into a specific directory structure
    download_dir = str(output_dir / "tardis_cache")

    _tardis_download_sync(
        exchange=tardis_exchange,
        data_types=[data_type],
        symbols=[symbol],
        from_date=from_str,
        to_date=to_str,
        download_dir=download_dir,
        api_key=api_key,
    )

    # Collect downloaded files and copy/link to output_dir with standard naming
    cache_dir = Path(download_dir)
    downloaded = sorted(cache_dir.rglob("*.csv.gz"))
    paths = []

    for src in downloaded:
        # Tardis names files like: bybit_book_snapshot_25_2024-01-01_BTCUSDT.csv.gz
        dest = output_dir / src.name
        if not dest.exists():
            import shutil
            shutil.copy2(src, dest)
        paths.append(dest)
        logger.info("Saved: %s", dest.name)

    logger.info(
        "Downloaded %d file(s) for %s via Tardis.dev",
        len(paths),
        symbol,
    )
    return paths


# ---------------------------------------------------------------------------
# Bybit range downloader
# ---------------------------------------------------------------------------

def download_range(
    symbol: str,
    start: date | str,
    end: date | str,
    output_dir: Path | None = None,
    market: str = "linear",
) -> list[Path]:
    """Download Bybit order book data for a date range.

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Download L2 orderbook data",
        epilog=(
            "Examples:\n"
            "  # Tardis.dev free sample (1st of each month, no key needed):\n"
            "  python data/download.py --source tardis --symbol BTCUSDT "
            "--start 2024-01-01 --end 2024-01-01\n\n"
            "  # Tardis.dev with API key (any date):\n"
            "  python data/download.py --source tardis --symbol BTCUSDT "
            "--start 2024-06-15 --end 2024-06-21 "
            "--tardis-api-key YOUR_KEY\n\n"
            "  # Bybit direct (recent dates only, ~May 2025+):\n"
            "  python data/download.py --source bybit --symbol BTCUSDT "
            "--start 2025-06-01 --end 2025-06-07\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        default="tardis",
        choices=["tardis", "bybit"],
        help="Data source (default: tardis)",
    )
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output-dir", default=None, help="Output directory")

    # Bybit-specific options
    parser.add_argument(
        "--market", default="linear", choices=["linear", "inverse", "spot"],
        help="Bybit market type (default: linear)",
    )

    # Tardis-specific options
    parser.add_argument(
        "--exchange", default="bybit",
        help="Exchange for Tardis source (default: bybit)",
    )
    parser.add_argument(
        "--data-type",
        default="book_snapshot_25",
        choices=["book_snapshot_25", "book_snapshot_5", "incremental_book_L2"],
        help="Tardis data type (default: book_snapshot_25)",
    )
    parser.add_argument(
        "--tardis-api-key",
        default="",
        help="Tardis.dev API key (omit for free sample data: 1st of each month)",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else None

    if args.source == "tardis":
        api_key = args.tardis_api_key or os.environ.get("TARDIS_API_KEY", "")
        download_tardis(
            args.symbol,
            args.start,
            args.end,
            output_dir,
            exchange=args.exchange,
            data_type=args.data_type,
            api_key=api_key,
        )
    else:
        download_range(args.symbol, args.start, args.end, output_dir, args.market)
