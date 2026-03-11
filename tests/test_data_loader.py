"""Tests for Tardis book_snapshot data loader."""

import gzip
from pathlib import Path

import numpy as np
import pytest

from src.data_loader import load, load_directory, load_snapshots, load_snapshots_directory


def _make_tardis_csv_gz(
    path: Path,
    n_levels: int = 3,
    n_rows: int = 2,
    base_ts_us: int = 1700000000000000,
    base_price: float = 50000.0,
) -> Path:
    """Helper: write a Tardis book_snapshot CSV file."""
    cols = ["timestamp", "local_timestamp"]
    for i in range(n_levels):
        cols.append(f"asks[{i}].price")
        cols.append(f"asks[{i}].amount")
    for i in range(n_levels):
        cols.append(f"bids[{i}].price")
        cols.append(f"bids[{i}].amount")

    rows = []
    for r in range(n_rows):
        ts = base_ts_us + r * 1_000_000  # 1s apart
        local_ts = ts + 500
        parts = [str(ts), str(local_ts)]
        for i in range(n_levels):
            parts.append(f"{base_price + 0.5 + i * 0.5:.2f}")  # ask prices
            parts.append(f"{1.0 + i * 0.5:.3f}")  # ask amounts
        for i in range(n_levels):
            parts.append(f"{base_price - 0.5 - i * 0.5:.2f}")  # bid prices
            parts.append(f"{1.5 - i * 0.3:.3f}")  # bid amounts
        rows.append(",".join(parts))

    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for row in rows:
            f.write(row + "\n")

    return path


class TestLoad:
    def test_loads_snapshot(self, tmp_path):
        path = tmp_path / "book_snapshot_25_test.csv.gz"
        _make_tardis_csv_gz(path, n_levels=3, n_rows=2)

        df = load(path)

        # 2 rows x (3 ask + 3 bid) = 12 events
        assert len(df) == 12
        assert set(df["side"].unique()) == {"bid", "ask"}
        assert (df["type"] == "snapshot").all()
        assert df["price"].dtype == np.float64
        assert df["qty"].dtype == np.float64

    def test_max_rows_limits_parsing(self, tmp_path):
        path = tmp_path / "book_snapshot_25_test.csv.gz"
        _make_tardis_csv_gz(path, n_levels=3, n_rows=5)

        df = load(path, max_rows=2)

        # Only 2 snapshot rows x 6 levels = 12
        assert len(df) == 12

    def test_empty_file_returns_empty_df(self, tmp_path):
        path = tmp_path / "book_snapshot_25_empty.csv.gz"
        _make_tardis_csv_gz(path, n_levels=3, n_rows=0)

        df = load(path)
        assert len(df) == 0
        assert "timestamp_us" in df.columns

    def test_timestamp_preserved_as_microseconds(self, tmp_path):
        path = tmp_path / "book_snapshot_25_test.csv.gz"
        ts_us = 1700000000000000
        _make_tardis_csv_gz(path, n_levels=1, n_rows=1, base_ts_us=ts_us)

        df = load(path)
        assert (df["timestamp_us"] == ts_us).all()

    def test_millisecond_timestamps_converted(self, tmp_path):
        path = tmp_path / "book_snapshot_25_test.csv.gz"
        # Use a timestamp that looks like milliseconds (< 1e15)
        ts_ms = 1700000000000  # 13 digits = milliseconds
        _make_tardis_csv_gz(path, n_levels=1, n_rows=1, base_ts_us=ts_ms)

        df = load(path)
        expected_us = ts_ms * 1000
        assert (df["timestamp_us"] == expected_us).all()

    def test_prices_parsed_correctly(self, tmp_path):
        path = tmp_path / "book_snapshot_25_test.csv.gz"
        _make_tardis_csv_gz(path, n_levels=3, n_rows=1, base_price=50000.0)

        df = load(path)
        ask_prices = sorted(df[df["side"] == "ask"]["price"].tolist())
        assert ask_prices == [50000.50, 50001.00, 50001.50]

        bid_prices = sorted(df[df["side"] == "bid"]["price"].tolist(), reverse=True)
        assert bid_prices == [49999.50, 49999.00, 49998.50]

    def test_zero_qty_levels_excluded(self, tmp_path):
        """Levels with zero quantity should be filtered out."""
        path = tmp_path / "book_snapshot_25_test.csv.gz"
        # Write a custom file with a zero-qty level
        cols = [
            "timestamp",
            "local_timestamp",
            "asks[0].price",
            "asks[0].amount",
            "bids[0].price",
            "bids[0].amount",
        ]
        with gzip.open(path, "wt") as f:
            f.write(",".join(cols) + "\n")
            f.write("1700000000000000,1700000000000500,50001.00,1.5,50000.00,0.0\n")

        df = load(path)
        # bid has qty=0, should be excluded
        assert len(df) == 1
        assert df.iloc[0]["side"] == "ask"


class TestLoadDirectory:
    def test_loads_multiple_files(self, tmp_path):
        _make_tardis_csv_gz(
            tmp_path / "bybit_book_snapshot_25_2024-01-01_BTCUSDT.csv.gz",
            n_levels=3,
            n_rows=2,
        )
        _make_tardis_csv_gz(
            tmp_path / "bybit_book_snapshot_25_2024-01-02_BTCUSDT.csv.gz",
            n_levels=3,
            n_rows=1,
            base_ts_us=1700100000000000,
        )

        df = load_directory(tmp_path)
        assert len(df) == 18  # (2+1) rows x 6 levels

    def test_filters_by_symbol(self, tmp_path):
        _make_tardis_csv_gz(
            tmp_path / "bybit_book_snapshot_25_2024-01-01_BTCUSDT.csv.gz",
            n_levels=3,
            n_rows=2,
        )
        _make_tardis_csv_gz(
            tmp_path / "bybit_book_snapshot_25_2024-01-01_ETHUSDT.csv.gz",
            n_levels=3,
            n_rows=1,
        )

        df = load_directory(tmp_path, symbol="BTCUSDT")
        assert len(df) == 12  # Only BTC: 2 rows x 6 levels

    def test_empty_directory(self, tmp_path):
        df = load_directory(tmp_path)
        assert len(df) == 0


# ---------------------------------------------------------------------------
# load_snapshots (direct Tardis → snapshots DataFrame)
# ---------------------------------------------------------------------------


class TestLoadSnapshots:
    def test_produces_snapshots_schema(self, tmp_path):
        path = tmp_path / "book_snapshot_25_test.csv.gz"
        _make_tardis_csv_gz(path, n_levels=5, n_rows=3)

        df = load_snapshots(path, n_levels=3, sample_interval_us=0)

        assert len(df) == 3
        assert "timestamp" in df.columns
        assert "mid_price" in df.columns
        assert "spread" in df.columns
        assert "bid_price_1" in df.columns
        assert "ask_price_1" in df.columns
        assert "bid_qty_3" in df.columns
        assert "ask_qty_3" in df.columns

    def test_mid_price_and_spread(self, tmp_path):
        path = tmp_path / "book_snapshot_25_test.csv.gz"
        _make_tardis_csv_gz(path, n_levels=1, n_rows=1, base_price=50000.0)

        df = load_snapshots(path, n_levels=1, sample_interval_us=0)

        # ask = 50000.50, bid = 49999.50
        assert df["mid_price"].iloc[0] == pytest.approx(50000.0)
        assert df["spread"].iloc[0] == pytest.approx(1.0)

    def test_subsampling(self, tmp_path):
        path = tmp_path / "book_snapshot_25_test.csv.gz"
        # 10 rows, 1μs apart — with 5μs interval should keep ~2-3
        _make_tardis_csv_gz(path, n_levels=1, n_rows=10, base_ts_us=1700000000000000)

        # Rows are 1s apart (1_000_000 μs), interval=3s → ~4 rows
        df = load_snapshots(path, n_levels=1, sample_interval_us=3_000_000)
        assert len(df) == 4  # t=0, t=3, t=6, t=9

    def test_no_subsampling(self, tmp_path):
        path = tmp_path / "book_snapshot_25_test.csv.gz"
        _make_tardis_csv_gz(path, n_levels=1, n_rows=5)

        df = load_snapshots(path, n_levels=1, sample_interval_us=0)
        assert len(df) == 5

    def test_n_levels_capped_to_available(self, tmp_path):
        path = tmp_path / "book_snapshot_25_test.csv.gz"
        _make_tardis_csv_gz(path, n_levels=3, n_rows=1)

        # Request 10 levels, but only 3 available
        df = load_snapshots(path, n_levels=10, sample_interval_us=0)
        assert "bid_price_3" in df.columns
        assert "bid_price_4" not in df.columns

    def test_empty_file(self, tmp_path):
        path = tmp_path / "book_snapshot_25_test.csv.gz"
        _make_tardis_csv_gz(path, n_levels=3, n_rows=0)

        df = load_snapshots(path, sample_interval_us=0)
        assert df.empty

    def test_max_rows(self, tmp_path):
        path = tmp_path / "book_snapshot_25_test.csv.gz"
        _make_tardis_csv_gz(path, n_levels=1, n_rows=10)

        df = load_snapshots(path, n_levels=1, max_rows=3, sample_interval_us=0)
        assert len(df) == 3


class TestLoadSnapshotsDirectory:
    def test_loads_multiple_files(self, tmp_path):
        _make_tardis_csv_gz(
            tmp_path / "bybit_book_snapshot_25_2024-01-01_BTCUSDT.csv.gz",
            n_levels=3,
            n_rows=2,
        )
        _make_tardis_csv_gz(
            tmp_path / "bybit_book_snapshot_25_2024-01-02_BTCUSDT.csv.gz",
            n_levels=3,
            n_rows=3,
            base_ts_us=1700100000000000,
        )

        df = load_snapshots_directory(tmp_path, n_levels=3, sample_interval_us=0)
        assert len(df) == 5

    def test_filters_by_symbol(self, tmp_path):
        _make_tardis_csv_gz(
            tmp_path / "bybit_book_snapshot_25_2024-01-01_BTCUSDT.csv.gz",
            n_levels=3,
            n_rows=2,
        )
        _make_tardis_csv_gz(
            tmp_path / "bybit_book_snapshot_25_2024-01-01_ETHUSDT.csv.gz",
            n_levels=3,
            n_rows=3,
        )

        df = load_snapshots_directory(tmp_path, symbol="BTCUSDT", n_levels=3, sample_interval_us=0)
        assert len(df) == 2

    def test_empty_directory(self, tmp_path):
        df = load_snapshots_directory(tmp_path)
        assert df.empty
