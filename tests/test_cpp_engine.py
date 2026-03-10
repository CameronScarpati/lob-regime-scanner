"""Tests for the C++ LOB engine (pybind11 extension)."""

import math

import numpy as np
import pandas as pd
import pytest

from src.cpp import CPP_AVAILABLE

if not CPP_AVAILABLE:
    pytest.skip("C++ LOB engine not built", allow_module_level=True)

from src.cpp._lob_cpp import LOBEngine, batch_reconstruct
from src.book_reconstructor import reconstruct


class TestLOBEngine:
    def test_empty_book(self):
        e = LOBEngine()
        bp, bq = e.best_bid()
        ap, aq = e.best_ask()
        assert math.isnan(bp) and math.isnan(bq)
        assert math.isnan(ap) and math.isnan(aq)
        assert math.isnan(e.mid_price())
        assert math.isnan(e.spread())

    def test_update_and_best(self):
        e = LOBEngine()
        e.update("bid", 50000.0, 1.5)
        e.update("bid", 49999.0, 2.0)
        e.update("ask", 50001.0, 1.0)
        e.update("ask", 50002.0, 3.0)

        assert e.best_bid() == (50000.0, 1.5)
        assert e.best_ask() == (50001.0, 1.0)
        assert e.mid_price() == 50000.5
        assert e.spread() == 1.0

    def test_update_removes_zero_qty(self):
        e = LOBEngine()
        e.update("bid", 50000.0, 1.5)
        e.update("bid", 50000.0, 0.0)
        assert math.isnan(e.best_bid()[0])

    def test_apply_snapshot_replaces(self):
        e = LOBEngine()
        e.update("bid", 50000.0, 1.0)
        e.update("bid", 49999.0, 2.0)
        e.apply_snapshot("bid", [(49998.0, 3.0), (49997.0, 4.0)])

        assert e.best_bid() == (49998.0, 3.0)

    def test_top_n_bids_descending(self):
        e = LOBEngine()
        e.update("bid", 50000.0, 1.0)
        e.update("bid", 49999.0, 2.0)
        e.update("bid", 49998.0, 3.0)

        levels = e.top_n("bid", 3)
        prices = [p for p, _ in levels]
        assert prices == [50000.0, 49999.0, 49998.0]

    def test_top_n_asks_ascending(self):
        e = LOBEngine()
        e.update("ask", 50001.0, 1.0)
        e.update("ask", 50002.0, 2.0)
        e.update("ask", 50003.0, 3.0)

        levels = e.top_n("ask", 3)
        prices = [p for p, _ in levels]
        assert prices == [50001.0, 50002.0, 50003.0]

    def test_top_n_pads_with_nan(self):
        e = LOBEngine()
        e.update("bid", 50000.0, 1.0)
        levels = e.top_n("bid", 3)
        assert len(levels) == 3
        assert levels[0] == (50000.0, 1.0)
        assert math.isnan(levels[1][0])
        assert math.isnan(levels[2][0])

    def test_snapshot_filters_zero_qty(self):
        e = LOBEngine()
        e.apply_snapshot("ask", [(100.0, 0.0), (101.0, 1.0)])
        assert e.best_ask() == (101.0, 1.0)
        levels = e.top_n("ask", 2)
        assert levels[0] == (101.0, 1.0)
        assert math.isnan(levels[1][0])

    def test_last_update_ts(self):
        e = LOBEngine()
        assert e.last_update_ts == 0
        e.last_update_ts = 123456
        assert e.last_update_ts == 123456


def _make_events(
    snapshot_bids=None,
    snapshot_asks=None,
    delta_updates=None,
    base_ts: int = 1_000_000_000_000,
) -> pd.DataFrame:
    """Helper to construct an events DataFrame for testing."""
    rows = []
    if snapshot_bids or snapshot_asks:
        for price, qty in (snapshot_bids or []):
            rows.append((base_ts, "snapshot", "bid", price, qty, 1, 1))
        for price, qty in (snapshot_asks or []):
            rows.append((base_ts, "snapshot", "ask", price, qty, 1, 1))
    if delta_updates:
        for i, (ts_offset, side, price, qty) in enumerate(delta_updates):
            rows.append(
                (base_ts + ts_offset, "delta", side, price, qty, 2 + i, 2 + i)
            )
    return pd.DataFrame(
        rows,
        columns=["timestamp_us", "type", "side", "price", "qty",
                 "update_id", "seq"],
    )


class TestBatchReconstruct:
    def test_snapshot_only(self):
        events = _make_events(
            snapshot_bids=[(50000.0, 1.0), (49999.0, 2.0)],
            snapshot_asks=[(50001.0, 1.5), (50002.0, 3.0)],
        )
        snapshots = reconstruct(events, n_levels=2, use_cpp=True)

        assert len(snapshots) == 1
        snap = snapshots[0]
        assert snap["mid_price"] == 50000.5
        assert snap["bid_price_1"] == 50000.0
        assert snap["ask_price_1"] == 50001.0

    def test_snapshot_then_delta(self):
        events = _make_events(
            snapshot_bids=[(50000.0, 1.0)],
            snapshot_asks=[(50001.0, 1.5)],
            delta_updates=[
                (1000, "bid", 50000.0, 2.0),
            ],
        )
        snapshots = reconstruct(events, n_levels=2, use_cpp=True)
        assert len(snapshots) == 2
        assert snapshots[1]["bid_qty_1"] == 2.0

    def test_delta_removes_level(self):
        events = _make_events(
            snapshot_bids=[(50000.0, 1.0), (49999.0, 2.0)],
            snapshot_asks=[(50001.0, 1.5)],
            delta_updates=[
                (1000, "bid", 50000.0, 0.0),
            ],
        )
        snapshots = reconstruct(events, n_levels=2, use_cpp=True)
        assert len(snapshots) == 2
        assert snapshots[1]["bid_price_1"] == 49999.0

    def test_empty_events(self):
        events = pd.DataFrame(
            columns=["timestamp_us", "type", "side", "price", "qty",
                     "update_id", "seq"]
        )
        snapshots = reconstruct(events, n_levels=10, use_cpp=True)
        assert len(snapshots) == 0

    def test_cpp_matches_python(self):
        """Verify C++ and Python produce identical results."""
        events = _make_events(
            snapshot_bids=[(50000.0, 1.0), (49999.0, 2.0), (49998.0, 0.5)],
            snapshot_asks=[(50001.0, 1.5), (50002.0, 3.0), (50003.0, 0.8)],
            delta_updates=[
                (1_000, "bid", 50000.0, 2.0),
                (2_000, "ask", 50001.0, 0.0),
                (3_000, "bid", 49997.0, 5.0),
                (3_000, "ask", 50004.0, 1.0),
                (4_000, "bid", 49999.0, 0.0),
            ],
        )

        py_snaps = reconstruct(events, n_levels=3, use_cpp=False)
        cpp_snaps = reconstruct(events, n_levels=3, use_cpp=True)

        assert len(py_snaps) == len(cpp_snaps)
        for py_row, cpp_row in zip(py_snaps, cpp_snaps):
            assert py_row.keys() == cpp_row.keys()
            for k in py_row:
                py_val = py_row[k]
                cpp_val = cpp_row[k]
                if isinstance(py_val, float) and math.isnan(py_val):
                    assert math.isnan(cpp_val), f"Mismatch on {k}: py=NaN, cpp={cpp_val}"
                else:
                    assert py_val == cpp_val, f"Mismatch on {k}: py={py_val}, cpp={cpp_val}"


class TestBatchReconstructDirect:
    """Test the raw batch_reconstruct C function directly."""

    def test_basic(self):
        ts = np.array([1000, 1000, 1000, 1000], dtype=np.int64)
        types = np.array([0, 0, 0, 0], dtype=np.int32)  # all snapshot
        sides = np.array([0, 0, 1, 1], dtype=np.int32)  # bid, bid, ask, ask
        prices = np.array([100.0, 99.0, 101.0, 102.0])
        qtys = np.array([1.0, 2.0, 1.5, 3.0])
        uids = np.array([1, 1, 1, 1], dtype=np.int64)

        result = batch_reconstruct(ts, types, sides, prices, qtys, uids, 2)

        assert "timestamp" in result
        assert len(result["timestamp"]) == 1
        assert result["mid_price"][0] == 100.5
        assert result["spread"][0] == 1.0
        assert result["bid_price_1"][0] == 100.0
        assert result["ask_price_1"][0] == 101.0
