"""Tests for order book snapshot reconstructor."""

import numpy as np
import pandas as pd

from src.book_reconstructor import (
    OrderBook,
    load_parquet,
    process_events_to_parquet,
    reconstruct,
    resample_snapshots,
    save_parquet,
)


class TestOrderBook:
    def test_empty_book(self):
        book = OrderBook()
        assert book.best_bid() is None
        assert book.best_ask() is None
        assert book.mid_price() is None
        assert book.spread() is None

    def test_update_adds_levels(self):
        book = OrderBook()
        book.update("bid", 50000.0, 1.5)
        book.update("bid", 49999.0, 2.0)
        book.update("ask", 50001.0, 1.0)

        assert book.best_bid() == (50000.0, 1.5)
        assert book.best_ask() == (50001.0, 1.0)

    def test_update_removes_zero_qty(self):
        book = OrderBook()
        book.update("bid", 50000.0, 1.5)
        book.update("bid", 50000.0, 0.0)

        assert book.best_bid() is None

    def test_mid_price(self):
        book = OrderBook()
        book.update("bid", 50000.0, 1.0)
        book.update("ask", 50002.0, 1.0)

        assert book.mid_price() == 50001.0

    def test_spread(self):
        book = OrderBook()
        book.update("bid", 50000.0, 1.0)
        book.update("ask", 50002.0, 1.0)

        assert book.spread() == 2.0

    def test_apply_snapshot_replaces_book(self):
        book = OrderBook()
        book.update("bid", 50000.0, 1.0)
        book.update("bid", 49999.0, 2.0)

        # Snapshot replaces everything
        book.apply_snapshot("bid", [(49998.0, 3.0), (49997.0, 4.0)])

        assert book.best_bid() == (49998.0, 3.0)
        assert 50000.0 not in book.bids

    def test_top_n_bids_sorted_descending(self):
        book = OrderBook()
        book.update("bid", 50000.0, 1.0)
        book.update("bid", 49999.0, 2.0)
        book.update("bid", 49998.0, 3.0)

        levels = book.top_n("bid", 3)
        prices = [p for p, _ in levels]
        assert prices == [50000.0, 49999.0, 49998.0]

    def test_top_n_asks_sorted_ascending(self):
        book = OrderBook()
        book.update("ask", 50001.0, 1.0)
        book.update("ask", 50002.0, 2.0)
        book.update("ask", 50003.0, 3.0)

        levels = book.top_n("ask", 3)
        prices = [p for p, _ in levels]
        assert prices == [50001.0, 50002.0, 50003.0]

    def test_top_n_pads_with_nan(self):
        book = OrderBook()
        book.update("bid", 50000.0, 1.0)

        levels = book.top_n("bid", 3)
        assert len(levels) == 3
        assert levels[0] == (50000.0, 1.0)
        assert np.isnan(levels[1][0])
        assert np.isnan(levels[2][0])

    def test_snapshot_dict_schema(self):
        book = OrderBook()
        book.update("bid", 50000.0, 1.5)
        book.update("ask", 50001.0, 1.0)

        snap = book.snapshot_dict(1000000, n_levels=2)

        assert snap["timestamp"] == 1000000
        assert snap["mid_price"] == 50000.5
        assert snap["spread"] == 1.0
        assert snap["bid_price_1"] == 50000.0
        assert snap["bid_qty_1"] == 1.5
        assert snap["ask_price_1"] == 50001.0
        assert snap["ask_qty_1"] == 1.0
        assert np.isnan(snap["bid_price_2"])
        assert np.isnan(snap["ask_price_2"])


def _make_events(
    snapshot_bids=None,
    snapshot_asks=None,
    delta_updates=None,
    base_ts: int = 1_000_000_000_000,
) -> pd.DataFrame:
    """Helper to construct an events DataFrame for testing."""
    rows = []

    if snapshot_bids or snapshot_asks:
        for price, qty in snapshot_bids or []:
            rows.append((base_ts, "snapshot", "bid", price, qty, 1, 1))
        for price, qty in snapshot_asks or []:
            rows.append((base_ts, "snapshot", "ask", price, qty, 1, 1))

    if delta_updates:
        for i, (ts_offset, side, price, qty) in enumerate(delta_updates):
            rows.append((base_ts + ts_offset, "delta", side, price, qty, 2 + i, 2 + i))

    return pd.DataFrame(
        rows,
        columns=["timestamp_us", "type", "side", "price", "qty", "update_id", "seq"],
    )


class TestReconstruct:
    def test_snapshot_only(self):
        events = _make_events(
            snapshot_bids=[(50000.0, 1.0), (49999.0, 2.0)],
            snapshot_asks=[(50001.0, 1.5), (50002.0, 3.0)],
        )
        snapshots = reconstruct(events, n_levels=2)

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
                (1000, "bid", 50000.0, 2.0),  # Update bid qty
            ],
        )
        snapshots = reconstruct(events, n_levels=2)

        # Should have 2 snapshots (one at snapshot ts, one at delta ts)
        assert len(snapshots) == 2
        # After delta, bid qty should be updated
        assert snapshots[1]["bid_qty_1"] == 2.0

    def test_delta_removes_level(self):
        events = _make_events(
            snapshot_bids=[(50000.0, 1.0), (49999.0, 2.0)],
            snapshot_asks=[(50001.0, 1.5)],
            delta_updates=[
                (1000, "bid", 50000.0, 0.0),  # Remove best bid
            ],
        )
        snapshots = reconstruct(events, n_levels=2)

        assert len(snapshots) == 2
        # After removing best bid, next bid becomes best
        assert snapshots[1]["bid_price_1"] == 49999.0

    def test_empty_events(self):
        events = pd.DataFrame(
            columns=["timestamp_us", "type", "side", "price", "qty", "update_id", "seq"]
        )
        snapshots = reconstruct(events)
        assert len(snapshots) == 0


class TestResampleSnapshots:
    def test_basic_resampling(self):
        # Create snapshots at irregular intervals
        data = {
            "timestamp": [1_000_000, 1_500_000, 3_200_000, 5_000_000],
            "mid_price": [50000.0, 50001.0, 50002.0, 50003.0],
            "spread": [1.0, 1.0, 1.5, 1.0],
        }
        df = pd.DataFrame(data)

        result = resample_snapshots(df, interval_us=1_000_000)

        # Grid: 1M, 2M, 3M, 4M, 5M = 5 points
        assert len(result) == 5
        # Timestamps should be uniform
        diffs = np.diff(result["timestamp"].values)
        assert all(d == 1_000_000 for d in diffs)

    def test_ffill_carries_forward(self):
        data = {
            "timestamp": [1_000_000, 5_000_000],
            "mid_price": [50000.0, 50005.0],
            "spread": [1.0, 2.0],
        }
        df = pd.DataFrame(data)

        result = resample_snapshots(df, interval_us=1_000_000)

        # Between 1M and 5M, mid-points should carry forward the first value
        assert result["mid_price"].iloc[0] == 50000.0
        assert result["mid_price"].iloc[1] == 50000.0  # ffill from 1M
        assert result["mid_price"].iloc[2] == 50000.0  # ffill

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = resample_snapshots(df)
        assert result.empty


class TestParquetIO:
    def test_save_and_load(self, tmp_path):
        data = {
            "timestamp": [1000000, 2000000],
            "mid_price": [50000.0, 50001.0],
            "spread": [1.0, 1.5],
            "bid_price_1": [49999.5, 50000.0],
            "bid_qty_1": [1.0, 2.0],
            "ask_price_1": [50000.5, 50001.0],
            "ask_qty_1": [1.5, 1.0],
        }
        df = pd.DataFrame(data)

        path = tmp_path / "test.parquet"
        save_parquet(df, path)

        loaded = load_parquet(path)
        pd.testing.assert_frame_equal(df, loaded)


class TestProcessEventsToPipeline:
    def test_end_to_end(self, tmp_path):
        events = _make_events(
            snapshot_bids=[
                (50000.0, 1.0),
                (49999.0, 2.0),
            ],
            snapshot_asks=[
                (50001.0, 1.5),
                (50002.0, 3.0),
            ],
            delta_updates=[
                (1_000_000, "bid", 50000.0, 2.0),
                (2_000_000, "ask", 50001.0, 0.5),
                (3_000_000, "bid", 49998.0, 5.0),
            ],
        )

        output_path = tmp_path / "output.parquet"
        df = process_events_to_parquet(events, output_path, n_levels=3, interval_us=1_000_000)

        assert output_path.exists()
        assert not df.empty
        assert "mid_price" in df.columns
        assert "bid_price_1" in df.columns
        assert "ask_price_1" in df.columns
        assert "last_trade_price" in df.columns

        # Verify uniform timestamps
        if len(df) > 1:
            diffs = np.diff(df["timestamp"].values)
            assert all(d == 1_000_000 for d in diffs)

    def test_empty_events_produces_empty_output(self, tmp_path):
        events = pd.DataFrame(
            columns=["timestamp_us", "type", "side", "price", "qty", "update_id", "seq"]
        )
        output_path = tmp_path / "empty.parquet"
        df = process_events_to_parquet(events, output_path)
        assert df.empty
