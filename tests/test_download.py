"""Tests for data download module (Tardis.dev)."""

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from data.download import (
    _build_tardis_url,
    _download_tardis_day,
    download,
    download_tardis,
)


# ---------------------------------------------------------------------------
# Tardis URL building
# ---------------------------------------------------------------------------

class TestBuildTardisUrl:
    def test_bybit_snapshot_url(self):
        url = _build_tardis_url("bybit", "book_snapshot_25", date(2024, 1, 1), "BTCUSDT")
        assert url == (
            "https://datasets.tardis.dev/v1/bybit/book_snapshot_25"
            "/2024/01/01/BTCUSDT.csv.gz"
        )

    def test_binance_futures_url(self):
        url = _build_tardis_url("binance-futures", "incremental_book_L2", date(2024, 3, 1), "ETHUSDT")
        assert url == (
            "https://datasets.tardis.dev/v1/binance-futures/incremental_book_L2"
            "/2024/03/01/ETHUSDT.csv.gz"
        )

    def test_pads_month_and_day(self):
        url = _build_tardis_url("bybit", "book_snapshot_5", date(2024, 2, 5), "BTCUSDT")
        assert "/2024/02/05/" in url


# ---------------------------------------------------------------------------
# Tardis _download_tardis_day (direct HTTP)
# ---------------------------------------------------------------------------

class TestDownloadTardisDay:
    @patch("data.download.requests.get")
    def test_downloads_and_saves(self, mock_get, tmp_path):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"fake,csv,data\n1,2,3\n"
        mock_get.return_value = mock_resp

        result = _download_tardis_day(
            "bybit", "book_snapshot_25", date(2024, 1, 1),
            "BTCUSDT", tmp_path,
        )

        assert result is not None
        assert result.exists()
        assert result.name == "bybit_book_snapshot_25_2024-01-01_BTCUSDT.csv.gz"
        assert result.read_bytes() == b"fake,csv,data\n1,2,3\n"

    @patch("data.download.requests.get")
    def test_sends_auth_header(self, mock_get, tmp_path):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"data"
        mock_get.return_value = mock_resp

        _download_tardis_day(
            "bybit", "book_snapshot_25", date(2024, 1, 1),
            "BTCUSDT", tmp_path, api_key="test-key",
        )

        _, kwargs = mock_get.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer test-key"

    @patch("data.download.requests.get")
    def test_no_auth_header_without_key(self, mock_get, tmp_path):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"data"
        mock_get.return_value = mock_resp

        _download_tardis_day(
            "bybit", "book_snapshot_25", date(2024, 1, 1),
            "BTCUSDT", tmp_path,
        )

        _, kwargs = mock_get.call_args
        assert "Authorization" not in kwargs["headers"]

    @patch("data.download.requests.get")
    def test_skips_existing(self, mock_get, tmp_path):
        existing = tmp_path / "bybit_book_snapshot_25_2024-01-01_BTCUSDT.csv.gz"
        existing.write_bytes(b"cached")

        result = _download_tardis_day(
            "bybit", "book_snapshot_25", date(2024, 1, 1),
            "BTCUSDT", tmp_path,
        )

        assert result == existing
        mock_get.assert_not_called()

    @patch("data.download.requests.get")
    def test_returns_none_on_401(self, mock_get, tmp_path):
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_get.return_value = mock_resp

        result = _download_tardis_day(
            "bybit", "book_snapshot_25", date(2024, 6, 15),
            "BTCUSDT", tmp_path,
        )
        assert result is None

    @patch("data.download.requests.get")
    def test_returns_none_on_404(self, mock_get, tmp_path):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp

        result = _download_tardis_day(
            "bybit", "book_snapshot_25", date(2024, 1, 1),
            "INVALID", tmp_path,
        )
        assert result is None


# ---------------------------------------------------------------------------
# download (multi-day)
# ---------------------------------------------------------------------------

class TestDownload:
    @patch("data.download._download_tardis_day")
    def test_downloads_date_range(self, mock_dl, tmp_path):
        mock_dl.side_effect = [
            tmp_path / "f1.csv.gz",
            tmp_path / "f2.csv.gz",
        ]

        result = download(
            "BTCUSDT", "2024-01-01", "2024-01-02",
            output_dir=tmp_path,
        )
        assert len(result) == 2
        assert mock_dl.call_count == 2

    @patch("data.download._download_tardis_day")
    def test_maps_exchange_name(self, mock_dl, tmp_path):
        mock_dl.return_value = tmp_path / "f.csv.gz"

        download(
            "BTCUSDT", "2024-01-01", "2024-01-01",
            output_dir=tmp_path, exchange="binance",
        )

        # Should map "binance" -> "binance-futures"
        _, kwargs = mock_dl.call_args
        assert kwargs["exchange"] == "binance-futures"

    @patch("data.download._download_tardis_day")
    def test_passes_api_key(self, mock_dl, tmp_path):
        mock_dl.return_value = tmp_path / "f.csv.gz"

        download(
            "BTCUSDT", "2024-01-01", "2024-01-01",
            output_dir=tmp_path, api_key="my-key",
        )

        _, kwargs = mock_dl.call_args
        assert kwargs["api_key"] == "my-key"

    def test_invalid_range_raises(self, tmp_path):
        with pytest.raises(ValueError, match="end .* must be >= start"):
            download("BTCUSDT", "2024-02-01", "2024-01-01", tmp_path)

    def test_download_tardis_alias(self):
        assert download_tardis is download
