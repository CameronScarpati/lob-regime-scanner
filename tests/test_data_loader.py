"""Tests for Bybit L2 data loader."""

import gzip
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_csv, load_jsonl, load, load_directory


def _make_jsonl_gz(records: list[dict], path: Path) -> Path:
    """Helper: write JSON records to a gzipped JSONL file."""
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return path


def _sample_snapshot(ts_ms: int = 1700000000000) -> dict:
    """Create a sample Bybit orderbook snapshot record."""
    return {
        "topic": "orderbook.200.BTCUSDT",
        "type": "snapshot",
        "ts": ts_ms,
        "data": {
            "s": "BTCUSDT",
            "b": [["50000.00", "1.5"], ["49999.50", "2.0"], ["49999.00", "3.0"]],
            "a": [["50000.50", "1.0"], ["50001.00", "2.5"], ["50001.50", "0.5"]],
            "u": 100,
            "seq": 1000,
        },
        "cts": ts_ms - 5,
    }


def _sample_delta(ts_ms: int = 1700000001000) -> dict:
    """Create a sample Bybit orderbook delta record."""
    return {
        "topic": "orderbook.200.BTCUSDT",
        "type": "delta",
        "ts": ts_ms,
        "data": {
            "s": "BTCUSDT",
            "b": [["50000.00", "2.0"]],
            "a": [["50000.50", "0"]],
            "u": 101,
            "seq": 1001,
        },
        "cts": ts_ms - 5,
    }


class TestLoadJsonl:
    def test_loads_snapshot(self, tmp_path):
        path = tmp_path / "test.jsonl.gz"
        _make_jsonl_gz([_sample_snapshot()], path)

        df = load_jsonl(path)

        assert len(df) == 6  # 3 bids + 3 asks
        assert set(df["side"].unique()) == {"bid", "ask"}
        assert df["type"].iloc[0] == "snapshot"
        assert df["price"].dtype == np.float64
        assert df["qty"].dtype == np.float64

    def test_loads_snapshot_and_delta(self, tmp_path):
        path = tmp_path / "test.jsonl.gz"
        _make_jsonl_gz([_sample_snapshot(), _sample_delta()], path)

        df = load_jsonl(path)

        # 6 from snapshot + 2 from delta
        assert len(df) == 8
        assert (df["type"] == "snapshot").sum() == 6
        assert (df["type"] == "delta").sum() == 2

    def test_max_records_limits_parsing(self, tmp_path):
        path = tmp_path / "test.jsonl.gz"
        records = [_sample_snapshot(), _sample_delta(), _sample_delta()]
        _make_jsonl_gz(records, path)

        df = load_jsonl(path, max_records=1)

        # Only the first record (snapshot with 6 levels)
        assert len(df) == 6

    def test_empty_file_returns_empty_df(self, tmp_path):
        path = tmp_path / "empty.jsonl.gz"
        _make_jsonl_gz([], path)

        df = load_jsonl(path)
        assert len(df) == 0
        assert "timestamp_us" in df.columns

    def test_timestamp_converted_to_microseconds(self, tmp_path):
        path = tmp_path / "test.jsonl.gz"
        ts_ms = 1700000000000
        _make_jsonl_gz([_sample_snapshot(ts_ms)], path)

        df = load_jsonl(path)
        expected_us = ts_ms * 1000
        assert (df["timestamp_us"] == expected_us).all()

    def test_prices_parsed_correctly(self, tmp_path):
        path = tmp_path / "test.jsonl.gz"
        _make_jsonl_gz([_sample_snapshot()], path)

        df = load_jsonl(path)
        bid_prices = df[df["side"] == "bid"]["price"].sort_values(ascending=False).tolist()
        assert bid_prices == [50000.00, 49999.50, 49999.00]


class TestLoadCsv:
    def test_loads_basic_csv(self, tmp_path):
        path = tmp_path / "test.csv.gz"
        data = pd.DataFrame({
            "timestamp": [1700000000.0, 1700000000.0, 1700000001.0],
            "side": ["Buy", "Sell", "Buy"],
            "price": [50000.0, 50001.0, 50000.5],
            "qty": [1.0, 2.0, 0.5],
        })
        data.to_csv(path, index=False)

        df = load_csv(path)

        assert len(df) == 3
        assert set(df.columns) == {
            "timestamp_us", "type", "side", "price", "qty", "update_id", "seq"
        }
        assert set(df["side"].unique()) == {"bid", "ask"}

    def test_timestamp_seconds_to_microseconds(self, tmp_path):
        path = tmp_path / "test.csv"
        data = pd.DataFrame({
            "timestamp": [1700000000.5],
            "side": ["Buy"],
            "price": [50000.0],
            "qty": [1.0],
        })
        data.to_csv(path, index=False)

        df = load_csv(path)
        assert df["timestamp_us"].iloc[0] == 1700000000500000

    def test_side_normalization(self, tmp_path):
        path = tmp_path / "test.csv"
        data = pd.DataFrame({
            "timestamp": [1.0, 2.0, 3.0, 4.0],
            "side": ["Buy", "Sell", "buy", "sell"],
            "price": [100.0, 101.0, 100.0, 101.0],
            "qty": [1.0, 1.0, 1.0, 1.0],
        })
        data.to_csv(path, index=False)

        df = load_csv(path)
        assert set(df["side"].unique()) == {"bid", "ask"}


class TestLoadAutoDetect:
    def test_detects_jsonl(self, tmp_path):
        path = tmp_path / "data.jsonl.gz"
        _make_jsonl_gz([_sample_snapshot()], path)

        df = load(path)
        assert len(df) == 6

    def test_detects_csv(self, tmp_path):
        path = tmp_path / "data.csv"
        data = pd.DataFrame({
            "timestamp": [1.0],
            "side": ["Buy"],
            "price": [50000.0],
            "qty": [1.0],
        })
        data.to_csv(path, index=False)

        df = load(path)
        assert len(df) == 1


class TestLoadDirectory:
    def test_loads_multiple_files(self, tmp_path):
        _make_jsonl_gz([_sample_snapshot()], tmp_path / "BTCUSDT_2024-01-01.jsonl.gz")
        _make_jsonl_gz([_sample_delta()], tmp_path / "BTCUSDT_2024-01-02.jsonl.gz")

        df = load_directory(tmp_path)
        assert len(df) == 8  # 6 + 2

    def test_filters_by_symbol(self, tmp_path):
        _make_jsonl_gz([_sample_snapshot()], tmp_path / "BTCUSDT_2024-01-01.jsonl.gz")
        _make_jsonl_gz([_sample_delta()], tmp_path / "ETHUSDT_2024-01-01.jsonl.gz")

        df = load_directory(tmp_path, symbol="BTCUSDT")
        assert len(df) == 6  # Only BTC

    def test_empty_directory(self, tmp_path):
        df = load_directory(tmp_path)
        assert len(df) == 0
