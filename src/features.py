"""Microstructure feature computations.

Computes OFI (multi-level), VPIN, book imbalance, weighted mid-price,
spread, Kyle's lambda, trade flow aggression, cancellation ratio,
realized volatility, and return autocorrelation.
"""

import logging
import math

import numpy as np
import pandas as pd

# flowrisk uses np.math (removed in NumPy 2.0); shim it back in.
if not hasattr(np, "math"):
    np.math = math  # type: ignore[attr-defined]

from flowrisk.toxicity.vpin import BulkVPIN, BulkVPINConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_LEVELS = 10
OFI_DEPTHS = [1, 5, 10]
ZSCORE_WINDOW = 300  # 5 minutes at 1-second resolution
RVOL_HORIZONS = [1, 10, 60, 300]
AUTOCORR_LAGS = list(range(1, 11))


# ---------------------------------------------------------------------------
# Helper: rolling z-score
# ---------------------------------------------------------------------------


def _rolling_zscore(series: pd.Series, window: int = ZSCORE_WINDOW) -> pd.Series:
    """Z-score normalise *series* using a trailing rolling window."""
    mu = series.rolling(window, min_periods=1).mean()
    sigma = series.rolling(window, min_periods=1).std()
    sigma = sigma.replace(0, np.nan)
    return (series - mu) / sigma


# ---------------------------------------------------------------------------
# 2.1  Order Flow Imbalance (OFI)
# ---------------------------------------------------------------------------


def compute_ofi(df: pd.DataFrame, depths: list[int] | None = None) -> pd.DataFrame:
    """Compute OFI at multiple depths with z-score normalisation and velocity.

    OFI_t = Σ_{i=1}^{depth} [ΔV^{bid}_{t,i} − ΔV^{ask}_{t,i}]

    Parameters
    ----------
    df : DataFrame with columns bid_qty_1..N, ask_qty_1..N.
    depths : list of depth levels (default [1, 5, 10]).

    Returns
    -------
    DataFrame indexed like *df* with columns:
        ofi_{d}, ofi_{d}_zscore, ofi_{d}_velocity  for each depth d.
    """
    if depths is None:
        depths = OFI_DEPTHS

    result = pd.DataFrame(index=df.index)
    for d in depths:
        bid_cols = [f"bid_qty_{i}" for i in range(1, d + 1)]
        ask_cols = [f"ask_qty_{i}" for i in range(1, d + 1)]

        bid_sum = df[bid_cols].sum(axis=1)
        ask_sum = df[ask_cols].sum(axis=1)

        delta_bid = bid_sum.diff()
        delta_ask = ask_sum.diff()

        ofi = delta_bid - delta_ask
        result[f"ofi_{d}"] = ofi
        result[f"ofi_{d}_zscore"] = _rolling_zscore(ofi)
        result[f"ofi_{d}_velocity"] = ofi.diff()

    return result


# ---------------------------------------------------------------------------
# 2.2  VPIN (flowrisk)
# ---------------------------------------------------------------------------


def compute_vpin(
    df: pd.DataFrame,
    bucket_volume: float | None = None,
    n_buckets: int = 20,
) -> pd.Series:
    """Compute VPIN using the flowrisk library.

    Trades are classified with the tick rule applied to mid_price changes.
    Volume buckets are sized from *bucket_volume* (defaults to
    median total volume / 50).

    Returns a Series aligned to df.index.
    """
    prices = df["mid_price"].values.astype(float)
    # Use last_trade_qty if available, else proxy from top-of-book
    if "last_trade_qty" in df.columns and df["last_trade_qty"].notna().any():
        volumes = df["last_trade_qty"].fillna(0).values.astype(float)
    else:
        volumes = (df["bid_qty_1"] + df["ask_qty_1"]).values.astype(float) / 2.0

    # Clean NaN/inf — flowrisk cannot handle them
    prices = pd.Series(prices).ffill().bfill().values.astype(float)
    volumes = np.nan_to_num(volumes, nan=0.0, posinf=0.0, neginf=0.0)
    volumes = np.clip(volumes, 0.0, None)

    if np.sum(np.isfinite(prices)) < 10:
        logger.warning("Too few valid prices for VPIN")
        return pd.Series(np.nan, index=df.index, name="vpin")

    # Break identical consecutive prices with sub-tick noise to avoid
    # NaN in flowrisk's PnL volatility estimate (division by zero when
    # consecutive prices are equal from forward-fill resampling).
    tick = np.median(np.abs(np.diff(prices[prices > 0])))
    if tick == 0 or not np.isfinite(tick):
        tick = 0.01
    noise_scale = tick * 1e-6  # negligible relative to price
    prices = prices + np.random.default_rng(42).normal(0, noise_scale, len(prices))

    if bucket_volume is None:
        total_vol = np.nansum(volumes)
        bucket_volume = max(total_vol / 50.0, 1.0)

    # Build the time-bar DataFrame expected by flowrisk
    time_bars = pd.DataFrame(
        {
            "time": range(len(prices)),
            "price": prices,
            "volume": volumes,
        }
    )

    cfg = BulkVPINConfig()
    cfg.BUCKET_MAX_VOLUME = float(bucket_volume)
    cfg.N_BUCKET_OR_BUCKET_DECAY = n_buckets
    cfg.TIME_BAR_TIME_STAMP_COL_NAME = "time"
    cfg.TIME_BAR_PRICE_COL_NAME = "price"
    cfg.TIME_BAR_VOLUME_COL_NAME = "volume"
    cfg.N_TIME_BAR_FOR_INITIALIZATION = min(2, max(len(time_bars) - 2, 1))

    estimator = BulkVPIN(cfg)
    vpin_df = estimator.estimate(time_bars)

    # flowrisk may return fewer rows than input; align to original index
    vpin_vals = vpin_df["vpin"].values
    if len(vpin_vals) < len(df):
        padded = np.full(len(df), np.nan)
        padded[: len(vpin_vals)] = vpin_vals
        vpin_vals = padded
    elif len(vpin_vals) > len(df):
        vpin_vals = vpin_vals[: len(df)]

    vpin_series = pd.Series(vpin_vals, index=df.index, name="vpin")
    return vpin_series


# ---------------------------------------------------------------------------
# 2.3  Additional microstructure features
# ---------------------------------------------------------------------------


def compute_book_imbalance(df: pd.DataFrame, depth: int = N_LEVELS) -> pd.Series:
    """(V_bid − V_ask) / (V_bid + V_ask) at top *depth* levels."""
    bid_cols = [f"bid_qty_{i}" for i in range(1, depth + 1)]
    ask_cols = [f"ask_qty_{i}" for i in range(1, depth + 1)]
    v_bid = df[bid_cols].sum(axis=1)
    v_ask = df[ask_cols].sum(axis=1)
    denom = v_bid + v_ask
    denom = denom.replace(0, np.nan)
    return ((v_bid - v_ask) / denom).rename("book_imbalance")


def compute_weighted_mid(df: pd.DataFrame) -> pd.Series:
    """ask_1 × bid_qty_1 + bid_1 × ask_qty_1) / (bid_qty_1 + ask_qty_1)."""
    num = df["ask_price_1"] * df["bid_qty_1"] + df["bid_price_1"] * df["ask_qty_1"]
    denom = df["bid_qty_1"] + df["ask_qty_1"]
    denom = denom.replace(0, np.nan)
    return (num / denom).rename("weighted_mid")


def compute_spread_bps(df: pd.DataFrame) -> pd.Series:
    """Spread in basis points: (ask_1 − bid_1) / mid × 10000."""
    mid = df["mid_price"].replace(0, np.nan)
    return (((df["ask_price_1"] - df["bid_price_1"]) / mid) * 10_000).rename("spread_bps")


def compute_kyles_lambda(
    df: pd.DataFrame,
    window: int = ZSCORE_WINDOW,
) -> pd.Series:
    """Rolling regression slope of ΔP on signed √volume (Kyle's λ).

    Uses last_trade_qty and last_trade_side when available, else proxies
    from mid-price direction and top-of-book volume.
    """
    delta_p = df["mid_price"].diff()

    # Trade sign: +1 for buys, −1 for sells; fall back to tick rule
    sign = pd.Series(0.0, index=df.index)
    if "last_trade_side" in df.columns:
        mapped = df["last_trade_side"].map({"buy": 1.0, "sell": -1.0})
        if mapped.notna().sum() > len(df) * 0.1:
            sign = mapped.fillna(0.0)
    # Tick rule fallback when trade-side data is missing or sparse
    if (sign == 0).all():
        sign = np.sign(delta_p).fillna(0.0)

    if "last_trade_qty" in df.columns and df["last_trade_qty"].notna().any():
        vol = df["last_trade_qty"].fillna(0)
    else:
        vol = (df["bid_qty_1"] + df["ask_qty_1"]) / 2.0

    signed_sqrt_vol = sign * np.sqrt(vol.abs() if hasattr(vol, "abs") else np.abs(vol))

    # Rolling OLS slope: cov(x, y) / var(x)
    x = signed_sqrt_vol
    y = delta_p

    xy_mean = (x * y).rolling(window, min_periods=max(window // 2, 2)).mean()
    x_mean = x.rolling(window, min_periods=max(window // 2, 2)).mean()
    y_mean = y.rolling(window, min_periods=max(window // 2, 2)).mean()
    x2_mean = (x**2).rolling(window, min_periods=max(window // 2, 2)).mean()

    cov_xy = xy_mean - x_mean * y_mean
    var_x = x2_mean - x_mean**2
    var_x = var_x.replace(0, np.nan)

    return (cov_xy / var_x).rename("kyles_lambda")


def compute_trade_flow_aggression(df: pd.DataFrame) -> pd.Series:
    """Fraction of trades at or beyond the opposite quote (rolling 5-min).

    A trade is aggressive if a buy occurs at or above ask_1 (or a sell
    at or below bid_1).  Without per-trade data we proxy from snapshots.
    """
    if "last_trade_price" not in df.columns or df["last_trade_price"].isna().all():
        return pd.Series(np.nan, index=df.index, name="trade_aggression")

    price = df["last_trade_price"]
    side = df.get("last_trade_side", pd.Series("", index=df.index))

    is_aggressive = pd.Series(0.0, index=df.index)
    buy_mask = side == "buy"
    sell_mask = side == "sell"
    is_aggressive.loc[buy_mask] = (price[buy_mask] >= df.loc[buy_mask, "ask_price_1"]).astype(float)
    is_aggressive.loc[sell_mask] = (price[sell_mask] <= df.loc[sell_mask, "bid_price_1"]).astype(
        float
    )

    return is_aggressive.rolling(ZSCORE_WINDOW, min_periods=1).mean().rename("trade_aggression")


def compute_cancellation_ratio(df: pd.DataFrame) -> pd.Series:
    """Proxy cancellation ratio from rolling volume decrease.

    True cancellation data requires order-level feed.  We approximate by
    measuring the fraction of disappeared volume across all levels over
    a rolling window.
    """
    bid_cols = [f"bid_qty_{i}" for i in range(1, N_LEVELS + 1)]
    ask_cols = [f"ask_qty_{i}" for i in range(1, N_LEVELS + 1)]
    total_vol = df[bid_cols + ask_cols].sum(axis=1)
    vol_decrease = (-total_vol.diff()).clip(lower=0)
    total_abs = total_vol.rolling(ZSCORE_WINDOW, min_periods=1).sum()
    total_abs = total_abs.replace(0, np.nan)
    cancel = vol_decrease.rolling(ZSCORE_WINDOW, min_periods=1).sum() / total_abs
    return cancel.rename("cancellation_ratio")


def compute_realized_volatility(
    df: pd.DataFrame,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """√(Σ r²_i) at multiple horizons (in rows, assuming 1-second bars)."""
    if horizons is None:
        horizons = RVOL_HORIZONS
    log_ret = np.log(df["mid_price"] / df["mid_price"].shift(1))
    result = pd.DataFrame(index=df.index)
    for h in horizons:
        r2_sum = (log_ret**2).rolling(h, min_periods=1).sum()
        result[f"rvol_{h}s"] = np.sqrt(r2_sum)
    return result


def compute_return_autocorrelation(
    df: pd.DataFrame,
    lags: list[int] | None = None,
    window: int = ZSCORE_WINDOW,
) -> pd.DataFrame:
    """Rolling autocorrelation of 1-second log returns at lags 1..10."""
    if lags is None:
        lags = AUTOCORR_LAGS
    log_ret = np.log(df["mid_price"] / df["mid_price"].shift(1))
    result = pd.DataFrame(index=df.index)
    for k in lags:
        shifted = log_ret.shift(k)
        result[f"ret_autocorr_{k}"] = log_ret.rolling(
            window, min_periods=max(window // 2, k + 2)
        ).corr(shifted)
    return result


# ---------------------------------------------------------------------------
# 2.4  Feature matrix assembly
# ---------------------------------------------------------------------------


def build_feature_matrix(
    df: pd.DataFrame,
    zscore_window: int = ZSCORE_WINDOW,
    include_vpin: bool = True,
    standardize: bool = True,
) -> pd.DataFrame:
    """Assemble all features into a (T, F) matrix.

    Parameters
    ----------
    df : Snapshot DataFrame from book_reconstructor
         (columns: timestamp, mid_price, spread, bid/ask_price/qty_1..10).
    zscore_window : rolling window for z-score standardisation.
    include_vpin : if True, compute VPIN (can be slow on large data).
    standardize : if True, apply rolling z-score to all features.  Set to
        False when the consumer (e.g. HMM) applies its own standardisation;
        double-normalising removes the heteroscedasticity that the HMM needs
        to distinguish regimes.

    Returns
    -------
    DataFrame of shape (T, F) with all computed features.
    """
    parts: list[pd.DataFrame | pd.Series] = []

    # OFI (multi-depth, already z-scored internally)
    ofi = compute_ofi(df)
    parts.append(ofi)

    # VPIN
    if include_vpin:
        try:
            vpin = compute_vpin(df)
            parts.append(vpin)
        except Exception:
            logger.warning("VPIN computation failed; skipping", exc_info=True)

    # Book imbalance
    parts.append(compute_book_imbalance(df))

    # Weighted mid-price
    parts.append(compute_weighted_mid(df))

    # Spread (bps)
    parts.append(compute_spread_bps(df))

    # Kyle's lambda
    parts.append(compute_kyles_lambda(df, window=zscore_window))

    # Trade flow aggression
    parts.append(compute_trade_flow_aggression(df))

    # Cancellation ratio
    parts.append(compute_cancellation_ratio(df))

    # Realized volatility (multi-horizon)
    parts.append(compute_realized_volatility(df))

    # Return autocorrelation (k=1..10)
    parts.append(compute_return_autocorrelation(df, window=zscore_window))

    # Concatenate
    features = pd.concat(parts, axis=1)

    # ---- Optional rolling z-score standardisation -------------------------
    if standardize:
        ofi_zscore_cols = [c for c in features.columns if c.endswith("_zscore")]
        cols_to_standardise = [c for c in features.columns if c not in ofi_zscore_cols]

        for col in cols_to_standardise:
            features[col] = _rolling_zscore(features[col], window=zscore_window)

    # ---- NaN / inf handling -----------------------------------------------
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.ffill(inplace=True)
    features.bfill(inplace=True)
    features.fillna(0.0, inplace=True)

    logger.info(
        "Feature matrix assembled: shape %s, columns: %s",
        features.shape,
        list(features.columns),
    )
    return features


# Curated subset of features for HMM regime detection.
# Keeping this small avoids the curse of dimensionality with full-covariance HMMs.
HMM_FEATURE_COLS = [
    "ofi_1",
    "vpin",
    "book_imbalance",
    "spread_bps",
    "kyles_lambda",
    "rvol_1s",
    "rvol_60s",
    "ret_autocorr_1",
]
