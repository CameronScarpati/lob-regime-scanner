"""Tests for microstructure feature computations."""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    AUTOCORR_LAGS,
    N_LEVELS,
    OFI_DEPTHS,
    RVOL_HORIZONS,
    ZSCORE_WINDOW,
    _rolling_zscore,
    build_feature_matrix,
    compute_book_imbalance,
    compute_cancellation_ratio,
    compute_kyles_lambda,
    compute_ofi,
    compute_realized_volatility,
    compute_return_autocorrelation,
    compute_spread_bps,
    compute_trade_flow_aggression,
    compute_vpin,
    compute_weighted_mid,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_snapshot_df(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Create a realistic-looking snapshot DataFrame for testing.

    Generates *n* rows with 10 bid/ask price+qty levels, mid_price,
    spread, and optional trade fields.
    """
    rng = np.random.RandomState(seed)

    base_price = 30_000.0
    # Random walk for mid-price
    returns = rng.normal(0, 0.0001, size=n)
    mid_prices = base_price * np.exp(np.cumsum(returns))

    rows: dict[str, np.ndarray] = {
        "timestamp": np.arange(n) * 1_000_000,  # 1-second intervals in μs
        "mid_price": mid_prices,
    }

    for i in range(1, N_LEVELS + 1):
        offset = i * 0.5
        rows[f"bid_price_{i}"] = mid_prices - offset
        rows[f"ask_price_{i}"] = mid_prices + offset
        rows[f"bid_qty_{i}"] = rng.exponential(10, size=n).round(4)
        rows[f"ask_qty_{i}"] = rng.exponential(10, size=n).round(4)

    rows["spread"] = rows["ask_price_1"] - rows["bid_price_1"]

    # Trade fields
    rows["last_trade_price"] = mid_prices + rng.choice([-0.5, 0.5], size=n)
    rows["last_trade_qty"] = rng.exponential(1.0, size=n).round(4)
    sides = rng.choice(["buy", "sell"], size=n)
    df = pd.DataFrame(rows)
    df["last_trade_side"] = sides
    return df


@pytest.fixture
def snapshot_df() -> pd.DataFrame:
    return _make_snapshot_df()


@pytest.fixture
def small_df() -> pd.DataFrame:
    """Minimal dataframe (20 rows) for fast tests."""
    return _make_snapshot_df(n=20, seed=99)


# ---------------------------------------------------------------------------
# Rolling z-score
# ---------------------------------------------------------------------------

class TestRollingZscore:
    def test_output_shape(self, snapshot_df: pd.DataFrame) -> None:
        s = snapshot_df["mid_price"]
        z = _rolling_zscore(s, window=50)
        assert len(z) == len(s)

    def test_mean_near_zero(self) -> None:
        """After warm-up the z-scores should centre near zero."""
        s = pd.Series(np.random.randn(1000))
        z = _rolling_zscore(s, window=100)
        # Tail should be roughly zero-mean
        assert abs(z.iloc[200:].mean()) < 0.5

    def test_constant_series(self) -> None:
        """Constant input → NaN z-scores (std = 0)."""
        s = pd.Series([5.0] * 100)
        z = _rolling_zscore(s, window=10)
        assert z.isna().all()


# ---------------------------------------------------------------------------
# OFI
# ---------------------------------------------------------------------------

class TestOFI:
    def test_columns_present(self, snapshot_df: pd.DataFrame) -> None:
        ofi = compute_ofi(snapshot_df)
        for d in OFI_DEPTHS:
            assert f"ofi_{d}" in ofi.columns
            assert f"ofi_{d}_zscore" in ofi.columns
            assert f"ofi_{d}_velocity" in ofi.columns

    def test_output_length(self, snapshot_df: pd.DataFrame) -> None:
        ofi = compute_ofi(snapshot_df)
        assert len(ofi) == len(snapshot_df)

    def test_first_row_nan(self, snapshot_df: pd.DataFrame) -> None:
        """OFI uses diff(), so first row should be NaN."""
        ofi = compute_ofi(snapshot_df)
        assert np.isnan(ofi["ofi_1"].iloc[0])

    def test_custom_depths(self, snapshot_df: pd.DataFrame) -> None:
        ofi = compute_ofi(snapshot_df, depths=[1, 3])
        assert "ofi_1" in ofi.columns
        assert "ofi_3" in ofi.columns
        assert "ofi_5" not in ofi.columns

    def test_known_values(self) -> None:
        """Verify OFI computation with hand-crafted data."""
        df = pd.DataFrame({
            "bid_qty_1": [10.0, 12.0, 8.0],
            "ask_qty_1": [10.0, 10.0, 14.0],
        })
        ofi = compute_ofi(df, depths=[1])
        # row 1: Δbid=2, Δask=0 → OFI=2
        assert ofi["ofi_1"].iloc[1] == pytest.approx(2.0)
        # row 2: Δbid=-4, Δask=4 → OFI=-8
        assert ofi["ofi_1"].iloc[2] == pytest.approx(-8.0)


# ---------------------------------------------------------------------------
# VPIN
# ---------------------------------------------------------------------------

class TestVPIN:
    def test_output_length(self, snapshot_df: pd.DataFrame) -> None:
        vpin = compute_vpin(snapshot_df)
        assert len(vpin) == len(snapshot_df)

    def test_range(self, snapshot_df: pd.DataFrame) -> None:
        """VPIN values should be in [0, 1]."""
        vpin = compute_vpin(snapshot_df)
        assert vpin.min() >= -0.01  # small tolerance
        assert vpin.max() <= 1.01

    def test_series_name(self, snapshot_df: pd.DataFrame) -> None:
        vpin = compute_vpin(snapshot_df)
        assert vpin.name == "vpin"


# ---------------------------------------------------------------------------
# Book imbalance
# ---------------------------------------------------------------------------

class TestBookImbalance:
    def test_range(self, snapshot_df: pd.DataFrame) -> None:
        bi = compute_book_imbalance(snapshot_df)
        assert bi.min() >= -1.0 - 1e-9
        assert bi.max() <= 1.0 + 1e-9

    def test_balanced_book(self) -> None:
        df = pd.DataFrame({
            f"bid_qty_{i}": [10.0] for i in range(1, N_LEVELS + 1)
        } | {
            f"ask_qty_{i}": [10.0] for i in range(1, N_LEVELS + 1)
        })
        bi = compute_book_imbalance(df)
        assert bi.iloc[0] == pytest.approx(0.0)

    def test_all_bids(self) -> None:
        df = pd.DataFrame({
            f"bid_qty_{i}": [10.0] for i in range(1, N_LEVELS + 1)
        } | {
            f"ask_qty_{i}": [0.0] for i in range(1, N_LEVELS + 1)
        })
        bi = compute_book_imbalance(df)
        # Denom is 0 for ask side → should handle gracefully
        # With all-zero asks, denom = bid only, so (bid - 0) / bid = 1
        # Actually denom = bid + 0 = bid, numerator = bid => 1.0
        assert bi.iloc[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Weighted mid-price
# ---------------------------------------------------------------------------

class TestWeightedMid:
    def test_symmetric_book(self) -> None:
        """When bid_qty == ask_qty, weighted mid == arithmetic mid."""
        df = pd.DataFrame({
            "bid_price_1": [100.0],
            "ask_price_1": [102.0],
            "bid_qty_1": [5.0],
            "ask_qty_1": [5.0],
        })
        wm = compute_weighted_mid(df)
        assert wm.iloc[0] == pytest.approx(101.0)

    def test_asymmetric_book(self) -> None:
        """Weighted mid skews toward the side with more quantity."""
        df = pd.DataFrame({
            "bid_price_1": [100.0],
            "ask_price_1": [102.0],
            "bid_qty_1": [1.0],
            "ask_qty_1": [9.0],
        })
        wm = compute_weighted_mid(df)
        # (102*1 + 100*9) / (1 + 9) = (102 + 900) / 10 = 100.2
        assert wm.iloc[0] == pytest.approx(100.2)


# ---------------------------------------------------------------------------
# Spread (bps)
# ---------------------------------------------------------------------------

class TestSpreadBps:
    def test_positive(self, snapshot_df: pd.DataFrame) -> None:
        spread = compute_spread_bps(snapshot_df)
        assert (spread >= 0).all()

    def test_known_value(self) -> None:
        df = pd.DataFrame({
            "bid_price_1": [99.99],
            "ask_price_1": [100.01],
            "mid_price": [100.0],
        })
        spread = compute_spread_bps(df)
        # (0.02 / 100) * 10000 = 2.0 bps
        assert spread.iloc[0] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Kyle's lambda
# ---------------------------------------------------------------------------

class TestKylesLambda:
    def test_output_length(self, snapshot_df: pd.DataFrame) -> None:
        kl = compute_kyles_lambda(snapshot_df, window=50)
        assert len(kl) == len(snapshot_df)

    def test_name(self, snapshot_df: pd.DataFrame) -> None:
        kl = compute_kyles_lambda(snapshot_df, window=50)
        assert kl.name == "kyles_lambda"


# ---------------------------------------------------------------------------
# Trade flow aggression
# ---------------------------------------------------------------------------

class TestTradeFlowAggression:
    def test_with_trade_data(self, snapshot_df: pd.DataFrame) -> None:
        agg = compute_trade_flow_aggression(snapshot_df)
        assert len(agg) == len(snapshot_df)
        assert agg.name == "trade_aggression"

    def test_no_trade_data(self) -> None:
        """Should return NaN series when no trade data available."""
        df = _make_snapshot_df(n=20)
        df["last_trade_price"] = np.nan
        agg = compute_trade_flow_aggression(df)
        assert agg.isna().all()


# ---------------------------------------------------------------------------
# Cancellation ratio
# ---------------------------------------------------------------------------

class TestCancellationRatio:
    def test_non_negative(self, snapshot_df: pd.DataFrame) -> None:
        cr = compute_cancellation_ratio(snapshot_df)
        # Ratio should be >= 0 (or NaN)
        assert (cr.dropna() >= -1e-9).all()

    def test_output_length(self, snapshot_df: pd.DataFrame) -> None:
        cr = compute_cancellation_ratio(snapshot_df)
        assert len(cr) == len(snapshot_df)


# ---------------------------------------------------------------------------
# Realized volatility
# ---------------------------------------------------------------------------

class TestRealizedVolatility:
    def test_columns(self, snapshot_df: pd.DataFrame) -> None:
        rv = compute_realized_volatility(snapshot_df)
        for h in RVOL_HORIZONS:
            assert f"rvol_{h}s" in rv.columns

    def test_non_negative(self, snapshot_df: pd.DataFrame) -> None:
        rv = compute_realized_volatility(snapshot_df)
        for col in rv.columns:
            assert (rv[col].dropna() >= 0).all()

    def test_monotone_horizons(self, snapshot_df: pd.DataFrame) -> None:
        """Longer horizons should have higher (or equal) rvol on average."""
        rv = compute_realized_volatility(snapshot_df)
        means = [rv[f"rvol_{h}s"].mean() for h in RVOL_HORIZONS]
        # Each should be >= the previous (on average)
        for i in range(1, len(means)):
            assert means[i] >= means[i - 1] - 1e-9


# ---------------------------------------------------------------------------
# Return autocorrelation
# ---------------------------------------------------------------------------

class TestReturnAutocorrelation:
    def test_columns(self, snapshot_df: pd.DataFrame) -> None:
        ac = compute_return_autocorrelation(snapshot_df, window=50)
        for k in AUTOCORR_LAGS:
            assert f"ret_autocorr_{k}" in ac.columns

    def test_range(self, snapshot_df: pd.DataFrame) -> None:
        ac = compute_return_autocorrelation(snapshot_df, window=50)
        for col in ac.columns:
            vals = ac[col].dropna()
            if len(vals) > 0:
                assert vals.min() >= -1.0 - 1e-9
                assert vals.max() <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Feature matrix assembly
# ---------------------------------------------------------------------------

class TestBuildFeatureMatrix:
    def test_shape(self, snapshot_df: pd.DataFrame) -> None:
        fm = build_feature_matrix(snapshot_df, include_vpin=False)
        assert fm.shape[0] == len(snapshot_df)
        assert fm.shape[1] > 0

    def test_no_nan(self, snapshot_df: pd.DataFrame) -> None:
        """After assembly, no NaN or inf values should remain."""
        fm = build_feature_matrix(snapshot_df, include_vpin=False)
        assert not fm.isna().any().any()
        assert not np.isinf(fm.values).any()

    def test_with_vpin(self, snapshot_df: pd.DataFrame) -> None:
        fm = build_feature_matrix(snapshot_df, include_vpin=True)
        assert "vpin" in fm.columns

    def test_expected_column_count(self, snapshot_df: pd.DataFrame) -> None:
        fm = build_feature_matrix(snapshot_df, include_vpin=False)
        # OFI: 3 depths × 3 (raw, zscore, velocity) = 9
        # book_imbalance, weighted_mid, spread_bps, kyles_lambda,
        # trade_aggression, cancellation_ratio = 6
        # rvol: 4 horizons = 4
        # ret_autocorr: 10 lags = 10
        expected = 9 + 6 + 4 + 10
        assert fm.shape[1] == expected

    def test_all_columns_finite(self, snapshot_df: pd.DataFrame) -> None:
        fm = build_feature_matrix(snapshot_df, include_vpin=False)
        assert np.all(np.isfinite(fm.values))

    def test_small_input(self, small_df: pd.DataFrame) -> None:
        """Should not crash on small inputs."""
        fm = build_feature_matrix(small_df, include_vpin=False)
        assert fm.shape[0] == len(small_df)
        assert not fm.isna().any().any()
