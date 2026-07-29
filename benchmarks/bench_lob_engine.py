"""Benchmark the C++ LOB reconstruction engine.

Measures order book update throughput through two paths:

1. ``LOBEngine.update()`` called per event from Python — includes pybind11
   call overhead, so this is the floor.
2. ``batch_reconstruct()`` — the batch path the pipeline actually uses,
   where the whole event array is processed in C++ and snapshots are
   emitted once per unique timestamp.

Run with ``make bench`` (the extension is built by ``make install-dev``).
Numbers are hardware-dependent single-threaded measurements on synthetic
events; treat them as indicative, not as a validated performance claim.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cpp import CPP_AVAILABLE

N_EVENTS = 2_000_000
EVENTS_PER_TIMESTAMP = 20
N_LEVELS = 10
SEED = 42


def generate_events(
    n_events: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic incremental (delta) events around a fixed mid price.

    Prices sit on a 0.5-tick grid within ~50 ticks of mid on each side;
    roughly 20% of events have qty 0 (level deletions).
    """
    mid = 50_000.0
    tick = 0.5

    sides = rng.integers(0, 2, n_events).astype(np.int32)  # 0=bid, 1=ask
    offsets = rng.integers(1, 51, n_events).astype(np.float64)
    prices = np.where(
        sides == 0,
        mid - offsets * tick,
        mid + offsets * tick,
    )
    qtys = rng.uniform(0.0, 5.0, n_events)
    qtys[rng.random(n_events) < 0.2] = 0.0  # deletions

    timestamps = (np.arange(n_events, dtype=np.int64) // EVENTS_PER_TIMESTAMP) * 1000
    types = np.ones(n_events, dtype=np.int32)  # all incremental updates
    update_ids = np.zeros(n_events, dtype=np.int64)
    return timestamps, types, sides, prices, qtys, update_ids


def bench_update_loop(n_events: int, rng: np.random.Generator) -> float:
    """Per-event update() calls from Python. Returns events/sec."""
    from src.cpp import LOBEngine

    _, _, sides, prices, qtys, _ = generate_events(n_events, rng)
    side_strs = np.where(sides == 0, "bid", "ask").tolist()
    prices_list = prices.tolist()
    qtys_list = qtys.tolist()

    engine = LOBEngine()
    start = time.perf_counter()
    for side, price, qty in zip(side_strs, prices_list, qtys_list, strict=True):
        engine.update(side, price, qty)
    elapsed = time.perf_counter() - start
    return n_events / elapsed


def bench_batch_reconstruct(n_events: int, rng: np.random.Generator) -> float:
    """Batch path: full event array processed in C++. Returns events/sec."""
    from src.cpp import batch_reconstruct

    timestamps, types, sides, prices, qtys, update_ids = generate_events(n_events, rng)
    start = time.perf_counter()
    batch_reconstruct(timestamps, types, sides, prices, qtys, update_ids, N_LEVELS)
    elapsed = time.perf_counter() - start
    return n_events / elapsed


def main() -> int:
    if not CPP_AVAILABLE:
        print(
            "C++ extension not built. Run `make install-dev` first "
            "(it compiles src/cpp via pybind11).",
            file=sys.stderr,
        )
        return 1

    rng = np.random.default_rng(SEED)
    print(
        f"LOB engine benchmark — {N_EVENTS:,} synthetic delta events, "
        f"{EVENTS_PER_TIMESTAMP} events per timestamp, {N_LEVELS} snapshot levels.\n"
    )

    # Warm-up run so first-call overhead is not measured
    bench_batch_reconstruct(10_000, rng)

    per_call = bench_update_loop(min(N_EVENTS, 500_000), rng)
    print(f"update() per-call from Python : {per_call:>12,.0f} updates/sec")

    batch = bench_batch_reconstruct(N_EVENTS, rng)
    print(f"batch_reconstruct() in C++    : {batch:>12,.0f} updates/sec")

    print(
        "\nSingle-threaded, synthetic events, hardware-dependent. "
        "Indicative only — not a validated performance claim."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
