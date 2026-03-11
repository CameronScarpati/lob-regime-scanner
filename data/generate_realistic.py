"""Generate realistic synthetic Tardis-format L2 data for pipeline validation.

Produces CSV.gz files in the Tardis book_snapshot_25 format, simulating a
3-day BTCUSDT window with a liquidation cascade on day 2 for compelling
visuals and pipeline stress testing.

Each CSV row is a full book snapshot with columns:
  timestamp, local_timestamp,
  asks[0].price, asks[0].amount, ..., asks[24].price, asks[24].amount,
  bids[0].price, bids[0].amount, ..., bids[24].price, bids[24].amount
"""

import gzip
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parent / "raw"

# Simulation parameters
N_LEVELS = 25  # Tardis book_snapshot_25 provides 25 levels per side
TICK_SIZE = 0.10  # BTCUSDT tick size
BASE_PRICE = 97_500.0  # BTC price around Feb 2025
SNAPSHOT_INTERVAL_S = 5  # Emit a snapshot every 5 seconds


def _generate_book_levels(
    mid: float, n_levels: int, rng: np.random.Generator
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Generate bid/ask levels around a mid price.

    Returns (bids, asks) as lists of (price, amount) tuples.
    """
    bids = []
    asks = []
    half_spread = TICK_SIZE * rng.uniform(0.5, 2.0)

    for i in range(n_levels):
        bid_price = mid - half_spread - i * TICK_SIZE
        ask_price = mid + half_spread + i * TICK_SIZE

        # Volume decays exponentially with distance, plus noise
        base_vol = max(0.001, rng.exponential(2.0 / (1 + i * 0.05)))
        bid_vol = round(base_vol * rng.uniform(0.5, 2.0), 3)
        ask_vol = round(base_vol * rng.uniform(0.5, 2.0), 3)

        bids.append((round(bid_price, 2), bid_vol))
        asks.append((round(ask_price, 2), ask_vol))

    return bids, asks


def _build_csv_header(n_levels: int) -> str:
    """Build the Tardis book_snapshot CSV header."""
    cols = ["timestamp", "local_timestamp"]
    for i in range(n_levels):
        cols.append(f"asks[{i}].price")
        cols.append(f"asks[{i}].amount")
    for i in range(n_levels):
        cols.append(f"bids[{i}].price")
        cols.append(f"bids[{i}].amount")
    return ",".join(cols)


def _format_snapshot_row(
    ts_us: int,
    local_ts_us: int,
    asks: list[tuple[float, float]],
    bids: list[tuple[float, float]],
) -> str:
    """Format a single snapshot as a CSV row."""
    parts = [str(ts_us), str(local_ts_us)]
    for price, amount in asks:
        parts.append(f"{price:.2f}")
        parts.append(f"{amount:.3f}")
    for price, amount in bids:
        parts.append(f"{price:.2f}")
        parts.append(f"{amount:.3f}")
    return ",".join(parts)


def _simulate_day(
    base_date: datetime,
    start_price: float,
    rng: np.random.Generator,
    volatility: float = 0.0001,
    drift: float = 0.0,
    cascade_start_hour: int | None = None,
    cascade_duration_min: int = 30,
) -> tuple[list[str], float]:
    """Simulate one day of L2 order book snapshots.

    Returns (list_of_csv_rows, closing_price).
    """
    rows = []
    mid = start_price

    # Generate snapshots for 24 hours at SNAPSHOT_INTERVAL_S resolution
    total_seconds = 24 * 3600
    n_steps = total_seconds // SNAPSHOT_INTERVAL_S

    for step in range(n_steps):
        elapsed_s = step * SNAPSHOT_INTERVAL_S
        ts = base_date + timedelta(seconds=elapsed_s)
        ts_us = int(ts.timestamp() * 1_000_000)
        local_ts_us = ts_us + rng.integers(100, 5000)  # small local delay

        # Adjust volatility during cascade
        vol = volatility
        dr = drift
        if cascade_start_hour is not None:
            cascade_start_s = cascade_start_hour * 3600
            cascade_end_s = cascade_start_s + cascade_duration_min * 60
            if cascade_start_s <= elapsed_s < cascade_end_s:
                progress = (elapsed_s - cascade_start_s) / (cascade_end_s - cascade_start_s)
                if progress < 0.6:
                    vol = volatility * 5
                    dr = -0.00003 * SNAPSHOT_INTERVAL_S
                else:
                    vol = volatility * 3
                    dr = 0.00001 * SNAPSHOT_INTERVAL_S

        # Random walk for mid-price (scale vol by interval)
        ret = rng.normal(dr, vol * np.sqrt(SNAPSHOT_INTERVAL_S))
        mid = mid * (1 + ret)

        bids, asks = _generate_book_levels(mid, N_LEVELS, rng)
        rows.append(_format_snapshot_row(ts_us, local_ts_us, asks, bids))

    logger.info(
        "Simulated day %s: %d snapshots, price %.2f -> %.2f",
        base_date.strftime("%Y-%m-%d"),
        len(rows),
        start_price,
        mid,
    )
    return rows, mid


def generate_realistic_data(
    symbol: str = "BTCUSDT",
    start_date: str = "2025-02-01",
    n_days: int = 3,
    output_dir: Path | None = None,
    seed: int = 42,
) -> list[Path]:
    """Generate realistic L2 order book data files in Tardis CSV format.

    Day 1: Normal trading (quiet/trending regimes)
    Day 2: Liquidation cascade mid-day (toxic regime)
    Day 3: Recovery with elevated volatility

    Args:
        symbol: Trading pair.
        start_date: Start date string.
        n_days: Number of days to generate.
        output_dir: Output directory (default: data/raw/).
        seed: Random seed.

    Returns:
        List of generated file paths.
    """
    if output_dir is None:
        output_dir = RAW_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    base = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC)
    price = BASE_PRICE
    paths = []

    header = _build_csv_header(N_LEVELS)

    day_configs = [
        {"volatility": 0.000015, "drift": 0.0, "cascade_start_hour": None},
        {"volatility": 0.00002, "drift": 0.0, "cascade_start_hour": 14, "cascade_duration_min": 45},
        {"volatility": 0.000025, "drift": 0.0, "cascade_start_hour": None},
    ]

    for day_idx in range(min(n_days, len(day_configs))):
        day_date = base + timedelta(days=day_idx)
        date_str = day_date.strftime("%Y-%m-%d")
        out_path = output_dir / f"bybit_book_snapshot_25_{date_str}_{symbol}.csv.gz"

        if out_path.exists():
            logger.info("Already exists: %s", out_path.name)
            paths.append(out_path)
            price = price * (1 + rng.normal(0, 0.01))
            continue

        config = day_configs[day_idx]
        rows, price = _simulate_day(day_date, price, rng, **config)

        with gzip.open(out_path, "wt", encoding="utf-8") as f:
            f.write(header + "\n")
            for row in rows:
                f.write(row + "\n")

        logger.info("Wrote %s (%d snapshots)", out_path.name, len(rows))
        paths.append(out_path)

    return paths


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    paths = generate_realistic_data()
    print(f"\nGenerated {len(paths)} files:")
    for p in paths:
        size_mb = p.stat().st_size / 1024 / 1024
        print(f"  {p.name}  ({size_mb:.1f} MB)")
