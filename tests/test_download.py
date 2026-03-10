"""Tests for Bybit data download module."""

import gzip
import io
import json
import zipfile
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from data.download import (
    _build_url,
    _build_legacy_url,
    _save_zip_as_jsonl_gz,
    download_day,
    download_range,
)


class TestBuildUrl:
    def test_linear_url(self):
        url = _build_url("BTCUSDT", date(2025, 6, 15), "linear")
        assert url == (
            "https://quote-saver.bycsi.com/orderbook/linear/BTCUSDT/"
            "2025-06-15_BTCUSDT_ob200.data.zip"
        )

    def test_spot_url(self):
        url = _build_url("BTCUSDT", date(2025, 6, 15), "spot")
        assert "spot/BTCUSDT" in url

    def test_legacy_url(self):
        url = _build_legacy_url("BTCUSDT", date(2025, 6, 15))
        assert url == (
            "https://public.bybit.com/orderbook/BTCUSDT/2025-06-15.csv.gz"
        )


class TestSaveZipAsJsonlGz:
    def test_extracts_json_lines(self, tmp_path):
        # Create a ZIP in memory with JSON lines
        records = [
            json.dumps({"type": "snapshot", "ts": 1000, "data": {"b": [], "a": []}})
            + "\n",
            json.dumps({"type": "delta", "ts": 1001, "data": {"b": [], "a": []}})
            + "\n",
        ]

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("data.jsonl", "".join(records))
        zip_bytes = buf.getvalue()

        out_path = tmp_path / "out.jsonl.gz"
        result = _save_zip_as_jsonl_gz(zip_bytes, out_path)

        assert result == out_path
        assert out_path.exists()

        # Read back and verify
        with gzip.open(out_path, "rt") as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 2
        assert json.loads(lines[0])["type"] == "snapshot"


class TestDownloadDay:
    @patch("data.download.requests.get")
    def test_skips_existing_file(self, mock_get, tmp_path):
        existing = tmp_path / "BTCUSDT_2025-06-15.jsonl.gz"
        existing.write_bytes(b"existing data")

        result = download_day("BTCUSDT", date(2025, 6, 15), tmp_path)

        assert result == existing
        mock_get.assert_not_called()

    @patch("data.download.requests.get")
    def test_downloads_and_saves_zip(self, mock_get, tmp_path):
        # Create mock ZIP response
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("data.jsonl", '{"type":"snapshot","ts":1,"data":{"b":[],"a":[]}}\n')
        zip_bytes = buf.getvalue()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = zip_bytes
        mock_get.return_value = mock_resp

        result = download_day("BTCUSDT", date(2025, 6, 15), tmp_path)

        assert result is not None
        assert result.exists()
        assert result.suffix == ".gz"

    @patch("data.download.requests.get")
    def test_returns_none_on_failure(self, mock_get, tmp_path):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp

        result = download_day("BTCUSDT", date(2025, 6, 15), tmp_path)
        assert result is None


class TestDownloadRange:
    @patch("data.download.download_day")
    def test_downloads_range(self, mock_dl, tmp_path):
        mock_dl.side_effect = [
            tmp_path / "f1.jsonl.gz",
            tmp_path / "f2.jsonl.gz",
            None,  # One failed day
        ]

        result = download_range("BTCUSDT", "2025-06-01", "2025-06-03", tmp_path)
        assert len(result) == 2
        assert mock_dl.call_count == 3

    def test_invalid_range_raises(self, tmp_path):
        with pytest.raises(ValueError, match="end .* must be >= start"):
            download_range("BTCUSDT", "2025-06-15", "2025-06-10", tmp_path)
