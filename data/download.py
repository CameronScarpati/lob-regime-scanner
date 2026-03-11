"""Historical L2 order book data downloader.

Downloads from Tardis.dev — professional-grade tick-level data for 40+
crypto exchanges. Free sample data (1st of each month) without API key
via direct HTTP download. Full access requires a paid API key from
https://tardis.dev.
"""

import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

DEFAULT_RAW_DIR = Path(__file__).parent / "raw"

# ---------------------------------------------------------------------------
# Tardis.dev downloader (direct HTTP — no SDK required)
# ---------------------------------------------------------------------------

TARDIS_DATASETS_URL = "https://datasets.tardis.dev/v1"

# Map common exchange names to Tardis exchange identifiers.
TARDIS_EXCHANGE_MAP = {
    "bybit": "bybit",
    "binance": "binance-futures",
    "binance-spot": "binance",
    "okx": "okex-swap",
    "deribit": "deribit",
}


def _build_tardis_url(
    exchange: str, data_type: str, dt: date, symbol: str
) -> str:
    """Build a direct Tardis.dev datasets download URL.

    URL format: https://datasets.tardis.dev/v1/{exchange}/{data_type}/{YYYY}/{MM}/{DD}/{SYMBOL}.csv.gz
    """
    return (
        f"{TARDIS_DATASETS_URL}/{exchange}/{data_type}"
        f"/{dt.year:04d}/{dt.month:02d}/{dt.day:02d}/{symbol}.csv.gz"
    )


def _download_tardis_day(
    exchange: str,
    data_type: str,
    dt: date,
    symbol: str,
    output_dir: Path,
    api_key: str = "",
    timeout: int = 300,
) -> Path | None:
    """Download a single day of Tardis data via direct HTTP.

    Args:
        exchange: Tardis exchange identifier (e.g. 'bybit').
        data_type: Tardis data type (e.g. 'book_snapshot_25').
        dt: Date to download.
        symbol: Trading pair (e.g. 'BTCUSDT').
        output_dir: Directory to save the file.
        api_key: Tardis API key (empty for free 1st-of-month data).
        timeout: Request timeout in seconds.

    Returns:
        Path to the downloaded file, or None if download failed.
    """
    # Standard output filename: {exchange}_{data_type}_{date}_{symbol}.csv.gz
    date_str = dt.strftime("%Y-%m-%d")
    out_name = f"{exchange}_{data_type}_{date_str}_{symbol}.csv.gz"
    out_path = output_dir / out_name

    if out_path.exists():
        logger.info("Already downloaded: %s", out_path.name)
        return out_path

    url = _build_tardis_url(exchange, data_type, dt, symbol)
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    logger.info("Downloading %s ...", url)

    try:
        resp = requests.get(url, headers=headers, timeout=timeout, stream=True)
        if resp.status_code == 200:
            out_path.write_bytes(resp.content)
            size_mb = len(resp.content) / (1024 * 1024)
            logger.info("Saved: %s (%.1f MB)", out_path.name, size_mb)
            return out_path

        if resp.status_code == 401:
            logger.error(
                "Tardis returned 401 Unauthorized for %s. "
                "Free data is only available for the 1st of each month. "
                "For other dates, provide an API key via --tardis-api-key "
                "or the TARDIS_API_KEY environment variable.",
                date_str,
            )
        elif resp.status_code == 404:
            logger.error(
                "Tardis returned 404 for %s %s on %s. "
                "The symbol or exchange may not be available for this date.",
                exchange,
                symbol,
                date_str,
            )
        else:
            logger.error(
                "Tardis returned HTTP %d for %s",
                resp.status_code,
                date_str,
            )
    except requests.RequestException as e:
        logger.error("Tardis download failed for %s: %s", date_str, e)

    return None


def download(
    symbol: str,
    start: date | str,
    end: date | str,
    output_dir: Path | None = None,
    exchange: str = "bybit",
    data_type: str = "book_snapshot_25",
    api_key: str = "",
) -> list[Path]:
    """Download order book data via Tardis.dev using direct HTTP.

    Downloads pre-reconstructed L2 order book snapshots as compressed
    CSV files. No SDK installation required — uses the Tardis datasets
    HTTP API directly.

    Free sample data (1st of each month) is available without an API key.
    For other dates, provide an API key from https://tardis.dev.

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

    logger.info(
        "Downloading %s %s from Tardis.dev (%s, %s to %s) ...",
        tardis_exchange,
        symbol,
        data_type,
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
    )
    if not api_key:
        # Check if any requested dates are NOT 1st of month
        non_free = []
        current = start
        while current <= end:
            if current.day != 1:
                non_free.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
        if non_free:
            logger.warning(
                "No API key provided — only the 1st of each month is free. "
                "These dates will likely fail: %s. "
                "Get an API key at https://tardis.dev",
                ", ".join(non_free[:5]) + ("..." if len(non_free) > 5 else ""),
            )
        else:
            logger.info(
                "No API key — using free sample data (1st of each month)."
            )

    paths = []
    current = start
    while current <= end:
        path = _download_tardis_day(
            exchange=tardis_exchange,
            data_type=data_type,
            dt=current,
            symbol=symbol,
            output_dir=output_dir,
            api_key=api_key,
        )
        if path is not None:
            paths.append(path)
        current += timedelta(days=1)

    logger.info(
        "Downloaded %d/%d day(s) for %s via Tardis.dev",
        len(paths),
        (end - start).days + 1,
        symbol,
    )
    return paths


# Backwards-compatible alias
download_tardis = download


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
        description="Download L2 orderbook data from Tardis.dev",
        epilog=(
            "Examples:\n"
            "  # Free sample (1st of each month, no key needed):\n"
            "  python data/download.py --symbol BTCUSDT "
            "--start 2024-01-01 --end 2024-01-01\n\n"
            "  # With API key (any date):\n"
            "  python data/download.py --symbol BTCUSDT "
            "--start 2024-06-15 --end 2024-06-21 "
            "--tardis-api-key YOUR_KEY\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument(
        "--exchange", default="bybit",
        help="Exchange name (default: bybit)",
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
    api_key = args.tardis_api_key or os.environ.get("TARDIS_API_KEY", "")

    download(
        args.symbol,
        args.start,
        args.end,
        output_dir,
        exchange=args.exchange,
        data_type=args.data_type,
        api_key=api_key,
    )
