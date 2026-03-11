"""Generate realistic synthetic Bybit L2 data for pipeline validation.

Produces JSONL.gz files in the exact format that Bybit's quote-saver API
returns, simulating a 3-day BTCUSDT window with a liquidation cascade
on day 2 for compelling visuals and pipeline stress testing.

Each JSON record has the Bybit schema:
  {"type": "snapshot"|"delta", "ts": <ms>, "data": {"b": [...], "a": [...], "u": <id>, "seq": <n>}}
"""

import gzip
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parent / "raw"

# Simulation parameters
N_LEVELS = 200  # Bybit ob200 provides 200 levels
TICK_SIZE = 0.10  # BTCUSDT tick size
BASE_PRICE = 97_500.0  # BTC price around Feb 2025
SNAPSHOT_INTERVAL_S = 60  # Full snapshot every 60 seconds
DELTA_INTERVAL_MS = 100  # Delta updates every 100ms


def _generate_book_levels(
    mid: float, n_levels: int, rng: np.random.Generator
) -> tuple[list[list[str]], list[list[str]]]:
    """Generate bid/ask levels around a mid price."""
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

        bids.append([f"{bid_price:.2f}", f"{bid_vol:.3f}"])
        asks.append([f"{ask_price:.2f}", f"{ask_vol:.3f}"])

    return bids, asks


def _simulate_day(
    base_date: datetime,
    start_price: float,
    rng: np.random.Generator,
    volatility: float = 0.0001,
    drift: float = 0.0,
    cascade_start_hour: int | None = None,
    cascade_duration_min: int = 30,
) -> tuple[list[dict], float]:
    """Simulate one day of L2 order book updates.

    Returns (list_of_records, closing_price).
    """
    records = []
    mid = start_price
    update_id = 1
    seq = 1

    # Generate updates for 24 hours
    total_seconds = 24 * 3600
    # Use 1-second resolution for deltas (not 100ms — keeps file size manageable)
    n_steps = total_seconds

    for step in range(n_steps):
        ts_ms = int((base_date + timedelta(seconds=step)).timestamp() * 1000)
        current_hour = step / 3600

        # Adjust volatility during cascade
        vol = volatility
        dr = drift
        if cascade_start_hour is not None:
            cascade_start_s = cascade_start_hour * 3600
            cascade_end_s = cascade_start_s + cascade_duration_min * 60
            if cascade_start_s <= step < cascade_end_s:
                progress = (step - cascade_start_s) / (cascade_end_s - cascade_start_s)
                # Sharp sell-off then partial recovery
                if progress < 0.6:
                    vol = volatility * 5
                    dr = -0.00003  # strong downward drift per second
                else:
                    vol = volatility * 3
                    dr = 0.00001  # partial recovery

        # Random walk for mid-price
        ret = rng.normal(dr, vol)
        mid = mid * (1 + ret)

        # Emit snapshot at every step (ensures consistent book state)
        # Use fewer levels for delta-style records to keep file size down
        if step % SNAPSHOT_INTERVAL_S == 0:
            n_lvl = N_LEVELS
        else:
            n_lvl = 20  # top-of-book only for non-snapshot updates

        bids, asks = _generate_book_levels(mid, n_lvl, rng)
        # All records are snapshots to ensure consistent book state
        rec_type = "snapshot"

        record = {
            "type": rec_type,
            "ts": ts_ms,
            "data": {
                "b": bids,
                "a": asks,
                "u": update_id,
                "seq": seq,
            },
        }

        records.append(record)
        update_id += 1
        seq += 1

        # Only emit every 5th second to keep file sizes reasonable
        # (still gives us ~17k records per day)
        if step % 5 != 0:
            continue

    # Filter to only the records we want (every 5th)
    # Actually, let's just thin the records we already built
    records_thinned = records[::5]

    logger.info(
        "Simulated day %s: %d records, price %.2f -> %.2f",
        base_date.strftime("%Y-%m-%d"),
        len(records_thinned),
        start_price,
        mid,
    )
    return records_thinned, mid


def generate_realistic_data(
    symbol: str = "BTCUSDT",
    start_date: str = "2025-02-01",
    n_days: int = 3,
    output_dir: Path | None = None,
    seed: int = 42,
) -> list[Path]:
    """Generate realistic L2 order book data files.

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
    base = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    price = BASE_PRICE
    paths = []

    day_configs = [
        {"volatility": 0.000015, "drift": 0.0, "cascade_start_hour": None},
        {"volatility": 0.00002, "drift": 0.0, "cascade_start_hour": 14,
         "cascade_duration_min": 45},
        {"volatility": 0.000025, "drift": 0.0, "cascade_start_hour": None},
    ]

    for day_idx in range(min(n_days, len(day_configs))):
        day_date = base + timedelta(days=day_idx)
        date_str = day_date.strftime("%Y-%m-%d")
        out_path = output_dir / f"{symbol}_{date_str}.jsonl.gz"

        if out_path.exists():
            logger.info("Already exists: %s", out_path.name)
            paths.append(out_path)
            price = price * (1 + rng.normal(0, 0.01))  # approximate
            continue

        config = day_configs[day_idx]
        records, price = _simulate_day(
            day_date, price, rng, **config
        )

        with gzip.open(out_path, "wt", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

        logger.info("Wrote %s (%d records)", out_path.name, len(records))
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
