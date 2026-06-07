# LOB Regime Scanner — Codebase Audit

Date: 2026-06-07
Branch: `claude/audit-codebase-analysis-TTEL9`
Auditor: Claude (Opus 4.7), code-only review, no execution of the pipeline against real data.

This document is a **reality check**, not a design review. It describes what the code actually does, the choices baked in, and the gaps between the documentation/marketing and the implementation. No improvement suggestions.

> **Resume bullet mapping is blocked.** Your prompt contained the literal placeholder `(paste bullet)` with no resume bullet text attached. Section 3 lists every claim made by the project's own README/docs and maps it to code; you can use that as scaffolding to evaluate any specific resume bullet you send next.

---

## 1. What the code does, function-by-function

### 1.1 `src/data_loader.py` — Tardis CSV ingestion

Parses Tardis `book_snapshot_25` CSVs. Two parallel paths:

- `load(path, max_rows)` (`data_loader.py:28`) — reads the CSV, expands each row × side × level into one event row (`timestamp_us, type='snapshot', side, price, qty, update_id=0, seq=0`). All update_ids/seqs are hardcoded to 0 (`:106-107`). Output is a long-format event DataFrame consumed by the legacy reconstruction path.
- `load_snapshots(path, n_levels=10, sample_interval_us=1_000_000, max_rows)` (`:202`) — reads CSV and emits the wide-format snapshots schema (`timestamp, mid_price, spread, bid_price_1..N, bid_qty_1..N, ask_price_1..N, ask_qty_1..N`) directly. This is the path the dashboard actually uses.
- Subsampling is a **manual Python loop** (`:247-254`), not vectorized. For a million-row file it does a million iterations in Python.
- Timestamp unit detection: if `timestamps.max() < 1e15` it multiplies by 1000 to convert ms→μs (`:243-244`, `:80-81`). Threshold is a magic constant.
- `mid_price` and `spread` (`:285-286`) are recomputed as `(bid_1 + ask_1)/2` and `ask_1 − bid_1` — they overwrite anything Tardis might have provided.
- `load_directory` / `load_snapshots_directory` (`:136`, `:333`) glob `*.csv.gz` and `*.csv`, filter by `symbol` (case-insensitive substring match on filename) and by start/end date via a regex on the filename stem `_DATE_RE = r"(\d{4}-\d{2}-\d{2})"` (`:298`). Files without a parseable date are unconditionally included (`:322-324`) — comment calls this "conservative."
- `load_tardis_snapshot = load` (`:133`) is a backwards-compat alias.

### 1.2 `src/book_reconstructor.py` — Order book replay

- `OrderBook` (`:42`) maintains `bids: dict[price→qty]` and `asks: dict[price→qty]`. `best_bid` does `max(self.bids)` (`:79`), `best_ask` does `min(self.asks)` (`:86`) — i.e. linear scan of dict keys per call.
- `top_n(side, n=10)` (`:105`) sorts the dict items, returns the top N, pads missing slots with `(NaN, NaN)`.
- `_reconstruct_python(events, n_levels)` (`:151`) groups events by `(timestamp_us, type, update_id)` (`:160-162`), applies snapshot or delta updates, and emits one snapshot per **unique timestamp**. Emission logic: snapshot is emitted **for the previous timestamp** when ts changes, plus a final emission for the last ts (`:181-187`). This means the first ts is silently dropped if more than one ts exists.
- `_reconstruct_cpp(events, n_levels)` (`:193`) maps `type`→{0,1}, `side`→{0,1} via `.map()` (`:206-207`), passes int/float arrays to `_lob_cpp.batch_reconstruct`, then iterates row-by-row in Python (`:222-230`) reconstructing dicts — defeating most of the point of returning column-major arrays from C++.
- `reconstruct(events, n_levels, use_cpp=None)` (`:236`) auto-detects C++ via the module-level `_CPP_AVAILABLE` flag (`:33-39`).
- `resample_snapshots(df, interval_us=1_000_000, method='ffill')` (`:278`) builds `np.arange(ts_min, ts_max+1, interval_us)`, uses `np.searchsorted` for ffill or nearest. Memory blows up if `interval_us` is small and the time range is large — no guard.
- `process_events_to_parquet(...)` (`:352`) is the convenience wrapper. **Adds NaN/empty `last_trade_price/qty/side` columns** (`:378-381`) — there is no trade-data path anywhere in the codebase that fills these.

### 1.3 `src/cpp/lob_engine.{hpp,cpp}` and `src/cpp/bindings.cpp`

- `LOBEngine` uses `std::map<double, double, std::greater<>>` for bids and a default-ordered `std::map<double, double>` for asks (`lob_engine.hpp:71-72`). Red-black trees, log n insert/erase/lookup.
- `update(side, price, qty)` erases when `qty<=0`, otherwise inserts (`lob_engine.cpp:14-26`). String-comparison `side == "bid"` per call.
- `best_bid` / `best_ask` use `bids_.begin()` and `asks_.begin()` (`lob_engine.cpp:48-58`) — O(1), good. NaN when empty.
- `top_n(side, n)` (`lob_engine.cpp:70-96`) iterates the map up to n entries, pads with NaN.
- `batch_reconstruct(...)` (`lob_engine.cpp:143`):
  - First pass counts unique adjacent timestamps to pre-allocate (`:153-162`). This means **non-adjacent equal timestamps get double-allocated**; output is trimmed at the end (`:271-275`) via `data.resize`.
  - Second pass uses a hand-written `while`-loop that groups consecutive events sharing `(ts, type, update_id)` (`:219-251`). Same emission logic as Python: write snapshot for the previous ts when ts changes.
  - Output is **column-major** in a flat `std::vector<double>` (`:165-166`); `write_snapshot` lambda (`:184-207`) hardcodes the column ordering.
- `bindings.cpp:py_batch_reconstruct` (`:15-66`) calls into `batch_reconstruct` and then **converts the column-major flat buffer to a dict of per-column numpy arrays by element-wise Python-side copy** (`:42-63`). That defeats most of the speed benefit — the inner copy loop is `for r in range(n_rows)` with `arr_buf(r) = data[r + c*n_rows]`.
- pybind11 module name: `_lob_cpp` (`bindings.cpp:68`).

### 1.4 `src/features.py` — Microstructure feature engine

- Module-level shim `np.math = math` (`:15-16`) — required because flowrisk imports `np.math.gamma` etc., removed in NumPy 2.0.
- Constants: `N_LEVELS=10, OFI_DEPTHS=[1,5,10], ZSCORE_WINDOW=300, RVOL_HORIZONS=[1,10,60,300], AUTOCORR_LAGS=range(1,11)` (`:25-29`).
- `_rolling_zscore(series, window=300)` (`:37`) — uses `min_periods=1`, replaces 0 stddev with NaN.
- `compute_ofi(df, depths)` (`:50`): for each depth d, sums `bid_qty_1..d`, `ask_qty_1..d`, takes `.diff()` to get `Δbid_sum − Δask_sum`. **Not Cont/Kukanov/Stoikov OFI** — that formula tracks per-level changes that are gated on whether the price level moved up/down/stayed. This computes the difference of summed volumes only, which collapses level-up vs. level-down behavior. It also adds `ofi_{d}_zscore` (rolling) and `ofi_{d}_velocity` (`.diff()` of OFI).
- `compute_vpin(df, bucket_volume=None, n_buckets=20)` (`:92`):
  - Pulls `mid_price` from snapshots (`:105`).
  - **Volumes**: if `last_trade_qty` exists and has any non-null, uses it; otherwise proxies with `(bid_qty_1 + ask_qty_1)/2` (`:107-110`). The proxy is a *liquidity* proxy, not a *trade volume* proxy — VPIN computed on it is not meaningful.
  - Injects **synthetic noise into prices** with `np.random.default_rng(42).normal(0, tick*1e-6, ...)` to dodge a flowrisk division-by-zero (`:124-128`). Same seed every call.
  - Default `bucket_volume = max(total_vol/50, 1.0)` (`:131-132`). Same-data lookahead inside the bucket sizing — fine for descriptive use, not for a streaming claim.
  - Delegates to `flowrisk.toxicity.vpin.BulkVPIN` (`:151-152`). Output rows can mismatch input length; aligned by left-pad with NaN (`:156-159`). No documented reason flowrisk returns fewer rows than input.
- `compute_book_imbalance(df, depth=10)` (`:172`) — `(V_bid − V_ask)/(V_bid + V_ask)` across `bid_qty_1..depth`.
- `compute_weighted_mid(df)` (`:183`) — micro-price = `(ask_1·bid_qty_1 + bid_1·ask_qty_1)/(bid_qty_1+ask_qty_1)`.
- `compute_spread_bps(df)` (`:191`) — `(ask_1 − bid_1)/mid · 1e4`.
- `compute_kyles_lambda(df, window=300)` (`:197`):
  - Trade sign from `last_trade_side` if >10% non-null mapped buy/sell (`:210-213`); else `sign(Δmid)` tick rule (`:215-216`).
  - Volume from `last_trade_qty` if available, else `(bid_qty_1+ask_qty_1)/2` (`:218-221`).
  - Rolling OLS slope via `cov(x,y)/var(x)` manually (`:229-238`). `min_periods=max(window//2, 2)`.
  - For real Tardis snapshots (no trade fields), this reduces to: λ ≈ Cov(Δmid, sign(Δmid)·√TOB_vol) / Var(sign(Δmid)·√TOB_vol). Since `sign(Δmid)·Δmid = |Δmid|`, this is heavily correlated with realized volatility. The label "Kyle's λ" is only nominal in that path.
- `compute_trade_flow_aggression(df)` (`:241`) — returns NaN if `last_trade_price` is missing (`:247`). With Tardis snapshot-only data, this is always NaN.
- `compute_cancellation_ratio(df)` (`:264`) — proxy: rolling sum of negative `Δtotal_vol`, divided by rolling sum of `total_vol`. The comment admits "True cancellation data requires order-level feed" (`:266-269`). This conflates resting-volume drops with cancellations and ignores executions.
- `compute_realized_volatility(df, horizons)` (`:281`) — `sqrt(rolling_sum(log_ret²))` at `[1,10,60,300]` rows. Horizon labels are "1s, 10s, 60s, 300s" but **the unit is rows, not seconds** — only matches seconds when bars are exactly 1s.
- `compute_return_autocorrelation(df, lags, window=300)` (`:296`) — rolling `corr(log_ret, log_ret.shift(k))` for k=1..10. `min_periods=max(window//2, k+2)`.
- `build_feature_matrix(df, zscore_window=300, include_vpin=True, standardize=True)` (`:319`):
  - Concatenates everything above. With `standardize=True` applies rolling z-score to all non-`_zscore` columns (`:384-389`). With `standardize=False` it does not.
  - Final NaN/inf cleanup: `replace(±inf→NaN) → ffill → bfill → fillna(0.0)` (`:392-395`). The `fillna(0.0)` after ffill+bfill only fires if a feature is all-NaN, which forces it to all-zero silently.
- `HMM_FEATURE_COLS = ["ofi_1","vpin","book_imbalance","spread_bps","kyles_lambda","rvol_1s","rvol_60s","ret_autocorr_1"]` (`:407-416`). **Note**: `ofi_1` is the raw (non-z-scored) column; the rolling z-scored OFI columns (`ofi_1_zscore`, etc.) exist but are NOT in this list.

### 1.5 `src/hmm_model.py` — Gaussian HMM wrapper

- `REGIME_LABELS = {0:"Quiet", 1:"Trending", 2:"Toxic"}` (`:20`).
- `RegimeDetector(n_states=3, covariance_type="full", n_iter=200, random_state=42, labels=None)` (`:74`).
- `fit(X, n_restarts=1)` (`:127`):
  - Fits a single `StandardScaler` on the **entire input** (`:148-155`) — global standardization on whatever is passed. There is no train/test boundary in this method.
  - Runs `GaussianHMM.fit` `n_restarts` times with seeds `random_state, random_state+1, ...` (`:160-174`), keeps best by `model.score(arr)`.
  - `_regularize_covars` adds `1e-3·I` to each component's covariance and symmetrizes (`:112-125`). For `diag` floors at `1e-3`. For `spherical` floors at `1e-3`. `tied` is silently ignored.
  - `_sort_states_by_volatility(arr)` (`:191`) reorders states by covariance-trace (`full`/`diag`) or value (`spherical`). For `tied` it returns early without reordering. State 0 becomes lowest-trace, last becomes highest.
- `predict(X)` (`:229`), `predict_proba(X)` (`:242`), `score(X)` (`:254`) — all run `_to_array(X, scale=True)` which applies the *fitted* scaler (`:100-110`).
- `bic` / `aic` (`:260`, `:268`): `ll = model.score(arr) * n_samples`. **hmmlearn's `score` is the total log-likelihood, not per-sample** (per its docs). The code's comment "hmmlearn returns per-sample" is wrong, so BIC/AIC are inflated by a factor of `n_samples`. Model selection still works as long as the term is the same in every comparison, but the reported values are not standard BIC/AIC.
- `_count_params(n_features)` (`:276`) — correct parameter counts for the four covariance types.
- `regime_stats(X, returns=None)` (`:300`) — decodes states, computes mean/std/skew/kurtosis per regime over either supplied returns or the standardized features. Durations via `_compute_durations` (`:419`).
- `compare_threshold_regimes(X, threshold_states)` (`:342`) — confusion matrix between HMM states and externally supplied threshold-rule states.
- `select_model(X, state_range=range(2,6), covariance_type="full", n_iter=200, random_state=42)` (`:374`) — refits a `RegimeDetector` for each K, records BIC/AIC/log-lik, picks argmin per criterion.

### 1.6 `src/backtest.py` — Regime-conditional strategy

- `run_backtest(states, returns, ofi, quiet_state=0, trending_state=1, toxic_state=2, annualization_factor=365*24*3600, ofi_smooth_window=120, cooldown_bars=30, stop_loss=0.002)` (`:28`).
- Smooths OFI with an EMA (`alpha = 2/(window+1)`, `:88-92`) using a **plain Python `for` loop**. `ofi_ema[0] = ofi[0]` — if the first OFI is NaN, the whole EMA is NaN.
- Loop logic (`:94-143`):
  - Entry: state transition into Trending, position currently flat, previous state was not Toxic, cooldown elapsed. Direction = `sign(ofi_ema[t])` (default long if 0).
  - Exit on Toxic, on return to Quiet, or on `current_trade_pnl < −stop_loss`.
  - Uses `returns[t]` (same-bar) for PnL while in position (`:114, :124, :134`). No lag between signal and PnL realization → look-ahead by one bar.
- Sharpe: `mean(pnl)/std(pnl) · sqrt(annualization_factor)` (`:152-156`), computed over **all bars including zeros**, which deflates std for sparse-position regimes.
- Max drawdown: `peak − cum_pnl` then `max(...)` — absolute, not percentage, despite the dashboard formatting it as `{max_dd:.2%}` (`dashboard/app.py:406`).
- Hit rate: fraction of trades with `>0` PnL.

### 1.7 `data/download.py` — Tardis HTTP downloader

- Builds URL `https://datasets.tardis.dev/v1/{exchange}/{data_type}/{Y}/{M}/{D}/{symbol}.csv.gz` (`:36-44`).
- `_download_tardis_day(...)` (`:47`) — single-shot `requests.get` with `stream=True` but reads `resp.content` (`:89`) which materializes the whole body before writing — `stream=True` is wasted.
- Skips file if it already exists (`:75-77`).
- Maps friendly exchange names: `binance→binance-futures`, `binance-spot→binance`, `okx→okex-swap`, etc. (`:27-33`).
- `download(...)` (`:122`) iterates day-by-day, warns if any non-1st-of-month date was requested without an API key.

### 1.8 `data/generate_realistic.py` — Synthetic Tardis file generator

- Constants: `N_LEVELS=25, TICK_SIZE=0.10, BASE_PRICE=97_500, SNAPSHOT_INTERVAL_S=5` (`:25-28`).
- `_generate_book_levels(mid, n_levels, rng)` (`:31`) — half-spread random in `[0.5·tick, 2·tick]`, volume decays as `exp(2/(1+i·0.05))`, multiplied by `Uniform(0.5, 2)`. Bids and asks have **independent random qty**; no symmetry, no order-of-magnitude calibration to real data.
- `_simulate_day(...)` (`:86`) — GBM-style mid with `vol=0.0001` baseline, optional "liquidation cascade" window (vol×5, drift `−0.00003`/snapshot) lasting `cascade_duration_min` (default 30; for day 2 it's 45 in the `day_configs` block).
- `generate_realistic_data(symbol, start_date, n_days=3, output_dir, seed=42)` (`:144`) — writes 3 days of CSV.gz files following Tardis schema. Day 2 includes a cascade. This is **not** a microstructure-faithful simulator; volumes and spreads do not respond to flow, and there are no informed-trader dynamics that VPIN/Kyle's λ would meaningfully pick up.

### 1.9 `dashboard/pipeline.py` — End-to-end run

- `DATA_DIR = .../data/raw` (`:22`).
- `run_pipeline(symbol="BTCUSDT", start, end, sample_interval_us=100_000, hmm_n_states=3)` (`:60`):
  - Loads via `load_snapshots_directory` (`:104`), then **applies the same `start`/`end` date filter a second time** via `pd.Timestamp` arithmetic (`:114-123`). Tardis files already filtered upstream.
  - Inserts placeholder `last_trade_price`, `last_trade_qty=NaN`, `last_trade_side=""` (`:131-134`) **always** — i.e. the Tardis snapshot path always lacks trade data.
  - `build_feature_matrix(snap_df, standardize=False)` (`:143`). Comment claims rolling z-score would erase regime variance (`:139-141`).
  - Selects `HMM_FEATURE_COLS` (`:147`) — 8 columns nominally.
  - Fits `RegimeDetector(n_states=3, covariance_type="diag", labels=REGIME_LABELS)` with `n_restarts=10` (`:152-157`). **Covariance type silently overridden from the `RegimeDetector` default of "full" to "diag"**, contradicting the README/dashboard footer/docs which advertise full-covariance HMM (`README:332`, `app.py:332`, `app.py:596`, `methodology.md:96`).
  - **No train/test split.** Same dataset is used for `fit`, `predict`, and `predict_proba` (`:157-159`).
  - Computes returns as `np.diff(np.log(mid), prepend=np.log(mid[0]))` (`:171`). Same-bar return (no shift) — when paired with backtest using `returns[t]` for `position·returns[t]`, the strategy realizes the return that drives the regime classification itself.
  - OFI signal column: prefers `ofi_1`; otherwise scans columns (`:174-184`).
  - Renames feature columns for the dashboard via `col_map` (`:205-221`); converts μs timestamp to `pd.to_datetime(unit='us')` (`:199`).

### 1.10 `dashboard/app.py` — Layout, CLI, factory

- CLI flags (`:100-149`): `--symbol BTCUSDT`, `--start`, `--end`, `--demo`, `--host 0.0.0.0`, `--port 8050`, **`--sample-interval 1000`** (ms), `--debug`. Pipeline gets `args.sample_interval * 1000` μs (`:177`).
- `load_data(args)` (`:157`) — uses `_mock_data.generate_all(n_timestamps=3600)` in demo, else `run_pipeline`. On `NoDataError` it prints a message to stderr and `sys.exit(1)`.
- `create_app(args=None)` (`:194`) — defaults to demo mode if no args (used by `app = create_app()` at module top so importing for tests doesn't need data).
- Builds the layout with `dash_mantine_components`. Header badges hardcode the model description `"GaussianHMM (K=3, full cov)"` (`:332`); footer hardcodes `"3-state full-covariance HMM"` (`:596`). Both contradict pipeline.py which fits **diag** covariance.
- `_compute_regime_durations(states)` (`:77`) — duplicates the logic in `_compute_durations` from `hmm_model.py:419`.

### 1.11 `dashboard/callbacks.py`

Single callback (`:34`): the time-range slider value triggers rebuilding **all four figures from scratch** on every drag tick. There is no figure caching or pre-computation; for large data this will block the worker.

### 1.12 `dashboard/components/heatmap.py`

- `_build_volume_matrix(snapshots, n_levels=10)` (`:23`): bins price into 80 fixed-width bins from 1st to 99th percentile of all prices (`:44-46`). Iterates each timestep and each level via **nested Python loops** (`:51-60`).
- Subsamples to ≤2000 timesteps (`:77-81`). Regime array is sliced with the same `[::step]` stride.
- Trade markers are emitted from `last_trade_qty/last_trade_price/last_trade_side`, which are NaN/empty on the real pipeline path. So the "large trade" diamonds only show in demo mode.

### 1.13 `dashboard/components/regime_probs.py`

Two-panel: stacked-area posterior probabilities + 3×3 transition matrix heatmap. Straightforward.

### 1.14 `dashboard/components/depth_surface.py`

- `_build_depth_grid(snapshots, n_levels=10, n_time_samples=200)` (`:17`): subsamples to 200 time steps, builds a `(T, 2·n_levels)` volume grid with **price offsets `[-n_levels..-1, 1..n_levels]`** — i.e. uses **level index, not price**, as the x-axis. The axis title labels it as "← Bids | Asks →" (`:116`), but the values are integer levels with a gap between −1 and +1 spanning the spread.
- `gaussian_filter(volume_grid, sigma=[sigma_t, 0.4])` smooths along time and level (`:54-55`). Hardcoded `sigma=0.4` across price levels.
- `surfacecolor=side_grid` with a hard-stepped two-tone colorscale (`:70-76`). The "regimes" parameter is accepted but **not used** in the figure (`:63`). Misleading signature.

### 1.15 `dashboard/components/diagnostics.py`

- `_add_regime_backgrounds(...)` (`:22`) draws `add_vrect` per regime segment, with a `max_vrects=100` early-exit (`:37-38`) — if there are >100 transitions, regime shading is silently dropped.
- Five subplots (VPIN / OFI / Kyle's λ / Spread / PnL) when `kyle_lambda` exists, else four.
- VPIN "Alert threshold" line hardcoded at `y=0.5` (`:121`). This is documented in some literature but is a heuristic, not data-driven.

### 1.16 `dashboard/_mock_data.py`

- `_RNG = np.random.default_rng(42)` at module scope (`:17`) — synthetic data is fully deterministic per import.
- Mock features use **regime-conditional means and variances drawn from `np.where(regimes==k, ...)`** (`:128-167`). The HMM is then "fit" on data whose regimes were directly generated from the labels it has to recover. In demo mode the regime probabilities and PnL bypass the HMM entirely (`generate_all` returns precomputed `hmm["states"]` and `cumulative_pnl` — see `app.py:165-167`).
- `generate_backtest_stats(cum_pnl)` (`:233`) uses `sqrt(252·86400)` for Sharpe annualization (`:236`) — that's a stock-market day count applied to second bars, **different from `src/backtest.py`'s `365·24·3600` (`:35`)**. Two different conventions in the same project.
- `n_trades` defined as count of sign-flips in the diff (`:240`) — has nothing to do with the backtest module's trade counting.

### 1.17 Tests

`tests/test_*.py` — 160 test functions total (README badge says **158**; off by two). Coverage:

- `test_book_reconstructor.py` (20): OrderBook ops, reconstruct semantics, resample, parquet I/O.
- `test_cpp_engine.py` (15): Mirrors the Python tests against the C++ engine; includes a `test_cpp_matches_python` parity test.
- `test_data_loader.py` (20): Tardis CSV parsing, subsampling, directory loading.
- `test_download.py` (14): URL building, HTTP mocked via `unittest.mock.patch`.
- `test_features.py` (35): One test per feature plus build_feature_matrix shape/no-NaN checks.
- `test_hmm.py` (28): Fit/predict, sorting, BIC/AIC, durations, backtest, integration. **No test of pipeline.py end-to-end with real or generated CSV data.**
- `test_dashboard.py` (28): Figure construction, mock-data schemas, app/CLI args. Pipeline test only verifies `NoDataError` is raised on missing files (`:280-289`) — never actually runs the pipeline.

Notable gaps: no test exercises VPIN with `flowrisk` short-circuit code, no test verifies that `ofi_1` is the column that flows into the HMM, no test for `pipeline.run_pipeline` happy-path, no benchmark/perf test backing the "1M+ updates/sec" claim, no test for the diag-vs-full covariance mismatch between pipeline and docs.

---

## 2. Every algorithmic / modeling choice with file:line

### 2.1 Data ingestion

| Choice | Where |
|---|---|
| Default subsample to 1 snapshot / 1 second | `src/data_loader.py:25` (`DEFAULT_SAMPLE_INTERVAL_US = 1_000_000`) |
| ms→μs threshold = `1e15` | `src/data_loader.py:80`, `:243` |
| Output capped at 10 book levels by default | `src/data_loader.py:204` (`n_levels=10`) |
| `mid_price = (bid_1 + ask_1)/2` | `src/data_loader.py:285` |
| `spread = ask_1 − bid_1` | `src/data_loader.py:286` |
| Date filter regex `(\d{4}-\d{2}-\d{2})` | `src/data_loader.py:298` |
| Files without a parseable date kept regardless | `src/data_loader.py:322-324` |
| All ingested rows have `update_id=0, seq=0` | `src/data_loader.py:106-107` |

### 2.2 Reconstruction

| Choice | Where |
|---|---|
| `N_LEVELS = 10` (output levels) | `src/book_reconstructor.py:29` |
| Emit one snapshot per unique timestamp; first ts dropped | `src/book_reconstructor.py:181-187`, `lob_engine.cpp:255-263` |
| Bids sorted descending via `std::map<..., std::greater<double>>` | `src/cpp/lob_engine.hpp:71` |
| Resample method default = `ffill` | `src/book_reconstructor.py:281` |
| Resample interval default = 1s | `src/book_reconstructor.py:280` |
| `qty <= 0` removes a level | `src/book_reconstructor.py:59`, `lob_engine.cpp:16-23` |

### 2.3 Features

| Choice | Where |
|---|---|
| OFI = `Δ(sum(bid_qty_1..d)) − Δ(sum(ask_qty_1..d))` | `src/features.py:73-80` |
| OFI depths = `[1, 5, 10]` | `src/features.py:26` |
| z-score window = 300 rows | `src/features.py:27` |
| VPIN volumes from `last_trade_qty` else `(bid_qty_1+ask_qty_1)/2` proxy | `src/features.py:107-110` |
| VPIN price noise `tick·1e-6` w/ rng seed 42 | `src/features.py:124-128` |
| VPIN bucket vol default = `max(total_vol/50, 1)` | `src/features.py:131-132` |
| VPIN buckets per window = 20 | `src/features.py:95` (`n_buckets=20`) |
| Book imbalance depth = 10 | `src/features.py:172` |
| Weighted mid = micro-price (cross-quantity weighted) | `src/features.py:185` |
| Spread bps = `(ask−bid)/mid · 1e4` | `src/features.py:194` |
| Kyle's λ = rolling `cov(Δmid, sign·√vol)/var(sign·√vol)` | `src/features.py:229-238` |
| Kyle's λ sign rule: use trade-side if >10% present, else tick | `src/features.py:212` |
| Kyle's λ window = `ZSCORE_WINDOW = 300` | `src/features.py:199` |
| Trade aggression returns NaN if no `last_trade_price` | `src/features.py:247-248` |
| Cancellation ratio = proxy from rolling Δvolume | `src/features.py:264-278` |
| Realized vol horizons = `[1,10,60,300]` | `src/features.py:28` |
| Autocorrelation lags = `1..10` | `src/features.py:29` |
| NaN handling: `inf→NaN → ffill → bfill → 0` | `src/features.py:392-395` |
| HMM feature columns hardcoded subset of 8 | `src/features.py:407-416` |
| HMM uses raw `ofi_1`, not `ofi_1_zscore` | `src/features.py:408` |

### 2.4 HMM

| Choice | Where |
|---|---|
| Default `n_states=3` | `src/hmm_model.py:76` |
| Default `covariance_type="full"` | `src/hmm_model.py:77` |
| Default `n_iter=200` | `src/hmm_model.py:78` |
| Default `random_state=42` | `src/hmm_model.py:79` |
| Single global `StandardScaler.fit_transform` on training input | `src/hmm_model.py:148-155` |
| Restart seeds = `random_state + i` | `src/hmm_model.py:161` |
| Best-of-restarts by `model.score(arr)` | `src/hmm_model.py:171-174` |
| Covariance floor = `1e-3` | `src/hmm_model.py:112` |
| State sort key = `np.trace(Σ_k)` (full) or `sum(diag)` | `src/hmm_model.py:191-210` |
| `tied` covariance silently skips reordering | `src/hmm_model.py:207-208` |
| BIC/AIC use `model.score · n_samples` as total LL | `src/hmm_model.py:265, 273` (comment claims hmmlearn returns per-sample — see §4.2) |
| Default model-selection range = `range(2, 6)` | `src/hmm_model.py:395` |
| Pipeline override: `covariance_type="diag"` | `dashboard/pipeline.py:154` |
| Pipeline override: `n_restarts=10` | `dashboard/pipeline.py:157` |
| Regime label dict `{0:"Quiet", 1:"Trending", 2:"Toxic"}` | `src/hmm_model.py:20`, `dashboard/_constants.py:3` |

### 2.5 Backtest

| Choice | Where |
|---|---|
| Annualization factor = `365·24·3600` (crypto 24/7, seconds) | `src/backtest.py:35` |
| OFI EMA smoothing window = 120 | `src/backtest.py:36` |
| Cooldown bars = 30 | `src/backtest.py:37` |
| Stop loss = 0.002 (units of cumulative PnL, not %) | `src/backtest.py:38` |
| Entry only on transition into Trending, position flat, prev ≠ Toxic | `src/backtest.py:101-110` |
| Long if `ofi_ema[t]>0` else short | `src/backtest.py:108` |
| PnL uses same-bar `returns[t]` (no signal lag) | `src/backtest.py:114, 124, 134` |
| Sharpe computed over all bars (incl. flat) | `src/backtest.py:152-156` |
| Dashboard re-annualizes via `sqrt(252·86400)` in mock stats | `dashboard/_mock_data.py:236` |

### 2.6 Dashboard

| Choice | Where |
|---|---|
| `REGIME_NAMES`, `REGIME_COLORS` hardcoded for 3 regimes | `dashboard/_constants.py:3-11` |
| Default sample interval CLI = 1000 ms | `dashboard/app.py:139` |
| Pipeline default sample interval = 100 ms | `dashboard/pipeline.py:64` |
| Heatmap max time steps = 2000 | `dashboard/components/heatmap.py:77` |
| Heatmap price bins = 80, `1%`–`99%` quantiles | `dashboard/components/heatmap.py:44-46` |
| Large-trade percentile = 90th | `dashboard/components/heatmap.py:172` |
| Depth surface time samples = 200 | `dashboard/components/depth_surface.py:20` |
| Depth surface gaussian σ = `[max(0.6, n_t/200), 0.4]` | `dashboard/components/depth_surface.py:54-55` |
| Diagnostics max regime-rectangles = 100 | `dashboard/components/diagnostics.py:28, 37` |
| VPIN alert line = 0.5 | `dashboard/components/diagnostics.py:121` |

---

## 3. Claim → code mapping (README + docs/methodology + docs/results)

> Your resume bullet was not pasted. The table below covers every public-facing claim the project itself makes. If a resume bullet repeats one of these, the same verdict applies.

| Claim | Stated where | Supported by code? |
|---|---|---|
| "1M+ updates/sec" C++ engine | `README:130`, `:143`, `cpp/lob_engine.hpp:3`, `bindings.cpp:69-70`, `PROJECT_SPEC:78` | **Unsupported.** No benchmark, no perf test, no measured number anywhere. C++ batch path additionally pays a Python-side per-element copy in `bindings.cpp:42-63` that bottlenecks throughput. |
| "30+ microstructure features" | `README:40`, `:130`, `methodology.md:75` | **Verifiable.** With `include_vpin=True, OFI_DEPTHS=[1,5,10]` build_feature_matrix yields 9 (OFI) + 1 (VPIN) + 6 (book_imbal, w-mid, spread_bps, kyles_λ, trade_agg, cancel_ratio) + 4 (rvol) + 10 (autocorr) = **30 columns** (test asserts the same number minus VPIN at `tests/test_features.py:378`). |
| "Multi-level OFI (Cont, Kukanov & Stoikov, 2014)" | `README:41`, `:287`, `app.py:589`, `methodology.md:11-19` | **Misattribution.** The formula in CKS 2014 conditions level-by-level on whether the quote moved, and aggregates per-level signed events; what `features.py:73-80` computes is the difference of `Δ(sum bid_qty) − Δ(sum ask_qty)` across the top-d levels. Same name, simpler quantity. |
| "VPIN via flowrisk" | `README:42`, `:144`, `methodology.md:43` | **True at the API level.** `flowrisk.toxicity.vpin.BulkVPIN` is called in `features.py:151-152`. However: the volumes fed in are a **TOB-liquidity proxy**, not trade volumes, whenever `last_trade_qty` is absent — i.e. on every real Tardis path (`features.py:107-110`, `pipeline.py:131-134`). Prices are also jittered with deterministic noise (`features.py:124-128`). |
| "Kyle's λ via rolling OLS" | `README:41`, `:287`, `app.py:590`, `methodology.md:47-58` | **Mechanically present** (`features.py:229-238`). For real data, λ devolves to `cov(Δmid, sign(Δmid)·√TOB_vol)/var(...)`, which correlates with realized vol because `sign(Δmid)·Δmid = |Δmid|`. The "price impact coefficient" label is questionable in the no-trade-data path. |
| "Trailing rolling-window z-score, no lookahead" | `README:287`, `methodology.md:19-23`, `methodology.md:77` | **Partially false.** `build_feature_matrix` has a rolling-z-score branch, but the pipeline calls it with `standardize=False` (`pipeline.py:143`) and instead feeds raw features to `RegimeDetector.fit`, which applies a **global, in-sample `StandardScaler.fit_transform`** (`hmm_model.py:148-155`). That is a global, lookahead-using standardization on the data the HMM is then evaluated on. |
| "Walk-forward 70/30 train/test" | `methodology.md:170-176`, `README:291` | **Unsupported.** No split exists. `pipeline.py:157-159` fits and predicts on the same array. Grep for `train`, `test`, `split`, `walk` in `src/` and `dashboard/pipeline.py` returns nothing relevant. |
| "3-state full-covariance Gaussian HMM" | `README:13`, `:332` (`"GaussianHMM (K=3, full cov)"`), `:596`, `app.py:332`, `app.py:596`, `methodology.md:96`, `:122` | **Contradicted by the pipeline.** `RegimeDetector.__init__` defaults to `"full"`, but `dashboard/pipeline.py:154` explicitly sets `covariance_type="diag"`. The dashboard label and footer both still claim full covariance. |
| "Baum-Welch EM up to 200 iterations" | `README:124`, `methodology.md:122` | **True.** `n_iter=200` (`hmm_model.py:78`); `GaussianHMM` uses Baum-Welch internally. |
| "BIC/AIC across K∈{2,3,4,5} selects K=3" | `README:289`, `docs/results.md:7-18` | **Code exists** (`hmm_model.py:374-416`), but the specific tabulated results in `docs/results.md` are reported numerically without any committed run script or data to reproduce them. No experiment record in the repo. The BIC/AIC formula uses `score·n_samples` despite a comment saying `score` is per-sample (see §4.2). |
| "States auto-sorted by covariance trace (volatility proxy)" | `README:289`, `methodology.md:135` | **True for full/diag/spherical** (`hmm_model.py:191-210`). Silently skipped for `tied`. Note: sorting happens after the *standardized* features are scaled to unit-variance globally; the trace ordering post-scaling can differ from the natural-units ordering. |
| "Viterbi decoding" | `README:42`, `methodology.md:125-132` | **True** — `model.predict(X)` in hmmlearn is Viterbi (`hmm_model.py:240`). |
| "Synchronized crosshairs / regime filter buttons / play-pause animation" | `PROJECT_SPEC:254-257` | **Unsupported.** Only a time-range RangeSlider is wired up (`app.py:500`, `callbacks.py:42`). No crosshair sync (Plotly's default hover is the only thing), no regime toggle, no animation. |
| Sharpe 1.8–2.5, MDD 0.3–0.8%, Hit 55–62%, "2.1× over threshold" | `README:99-106`, `docs/results.md:91-100`, `:107-114` | **Unsupported.** The backtest implementation has no transaction costs, uses same-bar PnL, computes Sharpe over all bars including flat, and is annualized at `365·24·3600`. No saved results, no reproducible run, no comparison-against-threshold script in the repo. |
| "VPIN spikes precede regime transitions by 30–120s" | `README:73`, `docs/results.md:41-49` | **Unsupported.** No event-study or lead-lag analysis exists in the code. The claim is asserted only in markdown. |
| "Kyle's λ 2–3× higher in Toxic" | `README:74`, `docs/results.md:55-63` | **Unsupported.** No regime-conditional λ computation script in the repo. `regime_stats` (`hmm_model.py:300`) can compute it but is not wired into any committed analysis. |
| Specific transition matrix numbers (`Quiet→Quiet=0.96`, etc.) | `README:91-95`, `docs/results.md:71-77` | **Unsupported.** No saved run. The same numbers appear in two places with identical formatting, suggesting hand-written rather than measured. |
| "158 tests passing" badge | `README:10`, `:147`, `:263` | **Off by 2.** Repo contains 160 test functions (see §1.17). Whether they all pass requires running pytest; the test code paths for `pipeline.run_pipeline` end-to-end with real data are not present. |
| "Bybit historical L2 data, free, no API key" | `PROJECT_SPEC:43-50` | **Partially obsolete.** `data/download.py` actually hits Tardis.dev (`download.py:24, 36-44`); Tardis's free tier is "1st of each month only" (`download.py:99-101, 138`). PROJECT_SPEC.md still describes Bybit public URLs that the code does not use. |
| "Default 100ms sampling captures microstructure" | `README:229` | **Inconsistent.** README says default 100; CLI defaults to **1000 ms** (`dashboard/app.py:139`). `pipeline.run_pipeline` standalone defaults to **100_000 μs = 100 ms** (`pipeline.py:64`), but CLI overrides it. |

---

## 4. Hardcoded parameters

Below: every numeric/string constant that affects behavior, with the file:line where it lives.

### 4.1 Magic numbers

```
src/data_loader.py:25            DEFAULT_SAMPLE_INTERVAL_US = 1_000_000      (1 s)
src/data_loader.py:80, :243      ms-vs-μs threshold = 1e15
src/book_reconstructor.py:29     N_LEVELS = 10
src/book_reconstructor.py:280    resample interval default = 1_000_000 μs
src/features.py:25               N_LEVELS = 10
src/features.py:26               OFI_DEPTHS = [1, 5, 10]
src/features.py:27               ZSCORE_WINDOW = 300
src/features.py:28               RVOL_HORIZONS = [1, 10, 60, 300]
src/features.py:29               AUTOCORR_LAGS = list(range(1, 11))
src/features.py:95               VPIN n_buckets = 20
src/features.py:124-128          VPIN noise = tick * 1e-6, rng seed = 42
src/features.py:132              VPIN bucket_volume = total_vol / 50
src/features.py:149              N_TIME_BAR_FOR_INITIALIZATION = min(2, ...)
src/features.py:212              Trade-side mapping threshold = 10% of rows
src/features.py:407-416          HMM_FEATURE_COLS = 8 hardcoded names
src/hmm_model.py:20              REGIME_LABELS dict (3 entries)
src/hmm_model.py:76-79           n_states=3, cov="full", n_iter=200, seed=42
src/hmm_model.py:112             Covariance floor = 1e-3
src/hmm_model.py:395             select_model state_range default = range(2,6)
src/backtest.py:35               annualization_factor = 365·24·3600
src/backtest.py:36-38            ofi_smooth_window=120, cooldown_bars=30, stop_loss=0.002
data/generate_realistic.py:25-28 N_LEVELS=25, TICK_SIZE=0.10, BASE_PRICE=97_500, SNAPSHOT_INTERVAL_S=5
data/generate_realistic.py:179-183 day_configs: vol/drift/cascade per day
dashboard/_mock_data.py:17       Mock RNG seed = 42 (module-scope)
dashboard/_mock_data.py:42-47    Mock transition matrix
dashboard/_mock_data.py:128-167  Mock per-regime feature parameters
dashboard/_mock_data.py:236      Mock Sharpe annualization = sqrt(252·86400)  <-- conflicts with src/backtest.py:35
dashboard/_constants.py:3        REGIME_NAMES dict
dashboard/_constants.py:7-11     REGIME_COLORS dict
dashboard/app.py:107             default symbol = "BTCUSDT"
dashboard/app.py:127             default host = "0.0.0.0"
dashboard/app.py:133             default port = 8050
dashboard/app.py:139             default sample-interval = 1000 ms       <-- conflicts with pipeline.py:64 = 100 ms
dashboard/app.py:167             demo n_timestamps = 3600
dashboard/app.py:238             n_marks (slider ticks) = 6
dashboard/app.py:285             container width = "1860px"
dashboard/app.py:332, :596       Hardcoded "GaussianHMM (K=3, full cov)"  <-- contradicts pipeline.py:154 = "diag"
dashboard/components/heatmap.py:44-46  Price bins = 80, percentile [1, 99]
dashboard/components/heatmap.py:77     max_time_steps = 2000
dashboard/components/heatmap.py:172    Large-trade percentile = 90
dashboard/components/depth_surface.py:20  n_time_samples = 200
dashboard/components/depth_surface.py:54-55  Gaussian sigma vectors
dashboard/components/diagnostics.py:28, :37  max_vrects = 100
dashboard/components/diagnostics.py:121  VPIN alert threshold = 0.5
```

### 4.2 Suspicious / contradictory sections

1. **Covariance type contradiction.** `RegimeDetector` defaults to `covariance_type="full"` (`hmm_model.py:77`), README/dashboard footer/header insist on "K=3, full cov" (`README:332`, `app.py:332`, `app.py:596`), but `dashboard/pipeline.py:154` overrides to `"diag"`. Anyone reading the dashboard label will be wrong about what's actually fit.

2. **Standardization story is double-spoken.**
   - `methodology.md:19-23, 77, 174` claims rolling z-score with no lookahead.
   - `pipeline.py:139-143` deliberately sets `standardize=False` and comments that rolling z-score would erase regime structure.
   - `hmm_model.py:148-155` then applies a global `StandardScaler.fit_transform` on the full input array.
   - Net effect: features fed to the HMM are globally standardized on the same data they are decoded on. Acceptable for *describing* regimes; inconsistent with the documented "lookahead-safe" framing and incompatible with the claimed 70/30 walk-forward (which doesn't exist anyway).

3. **No train/test split.** `pipeline.run_pipeline` calls `detector.fit(hmm_features, n_restarts=10)` then `detector.predict(hmm_features)` on the same array (`pipeline.py:157-158`). `methodology.md:170-176` describes a 70/30 split that is not implemented anywhere in the repo.

4. **VPIN volumes proxy is invalid.** `features.py:107-110` substitutes `(bid_qty_1 + ask_qty_1)/2` (a *liquidity* snapshot) for trade volume whenever `last_trade_qty` is empty. The Tardis snapshot pipeline always sets these fields to NaN (`pipeline.py:131-134`), so all real-data VPIN values are computed against TOB liquidity, not flow. The VPIN literature requires actual trade volume; the resulting number isn't VPIN.

5. **Synthetic noise injected into prices before VPIN.** `features.py:124-128` jitters prices with `tick·1e-6` Gaussian noise to dodge a `flowrisk` divide-by-zero. Fine for not crashing; quietly mutates the input series.

6. **`BIC/AIC` use `score·n_samples`.** `hmm_model.py:265, 273` multiply `model.score(arr)` by `n_samples`, with the comment "hmmlearn returns per-sample." Per hmmlearn docs (and behavior), `score()` returns the **total log-likelihood**, not per-sample. The BIC/AIC values returned are therefore `n_samples`× too large. Argmin is invariant under a positive scalar, so model selection is unaffected — but any reported absolute BIC/AIC number is wrong, and the comment is misleading.

7. **Backtest is same-bar.** `backtest.py:114, 124, 134` apply `position * returns[t]` where `returns[t]` is the same bar that decoded `states[t]`. With `pipeline.py:171` computing returns as `np.diff(np.log(mid), prepend=...)`, this is the price move *into* `t`, not after. Look-ahead by one bar in a strategy that triggers on same-bar regime detection.

8. **Sharpe denominator includes flat bars.** `backtest.py:153` computes `std(pnl)` over the full PnL vector (mostly zeros when flat), inflating apparent Sharpe.

9. **Two annualization factors in the same repo.** `src/backtest.py:35` uses `365·24·3600` (crypto). `dashboard/_mock_data.py:236` uses `252·86400` (equity day count × seconds-per-day, which is itself meaningless for crypto). The dashboard's mock stats and the real backtest's Sharpe are not comparable.

10. **Max-drawdown is absolute, displayed as percent.** `backtest.py:159-161` returns `peak − cum_pnl` in PnL units. The dashboard format string is `f"{max_dd:.2%}"` (`app.py:406`), which multiplies by 100 and adds a `%` — wrong units, wrong scale.

11. **Heatmap large-trade markers depend on data that doesn't exist.** `dashboard/components/heatmap.py:169-199` reads `last_trade_qty/price/side`. These are NaN/empty on the pipeline path (`pipeline.py:131-134`). So the "Large Trades" diamonds only appear in demo mode.

12. **`compute_trade_flow_aggression` always returns NaN on Tardis data.** `features.py:247-248` — returns NaN if `last_trade_price` is missing. On the real pipeline it's always missing. The feature exists, the column ships, but it conveys nothing.

13. **Cancellation ratio is a renamed delta of liquidity.** `features.py:264-278`: rolling drop in `total_vol` divided by rolling `total_vol`. Comment admits the proxy is wrong (`:266-269`). It conflates cancels with fills with renewals.

14. **`OFI velocity` is just `OFI.diff()`.** `features.py:82` — second derivative of book volume. Real-data interpretation is unclear; it's used as a column in the diagnostic dashboard via `OFI_velocity` mapping (`pipeline.py:209`).

15. **Realized vol horizons are labeled in seconds but indexed in rows.** `compute_realized_volatility` rolling window is in DataFrame index units (`features.py:291`); the column names `rvol_1s, rvol_10s, ...` assume the bar interval is exactly 1 second. With CLI default `--sample-interval 1000` ms this happens to match. With `--sample-interval 100` (advertised default in README) it does **not**: `rvol_60s` is then a 6-second window.

16. **`OFI_velocity` column in dashboard maps from `ofi_1_velocity`.** `pipeline.py:209` — only the depth-1 velocity is exposed even though depths 5 and 10 also exist.

17. **`pipeline.py` filters by date twice.** `data_loader.load_snapshots_directory` filters by filename (`data_loader.py:307-330`); `pipeline.py:114-123` then filters by epoch-μs comparison after loading. The second pass is redundant on properly-named files but is the only guard for files without a date in the filename.

18. **`_lob_cpp` returned dict copied element-wise.** `bindings.cpp:42-63` rebuilds per-column numpy arrays with a Python-driven nested loop. Negates much of the C++ speed advantage. Then `_reconstruct_cpp` (`book_reconstructor.py:222-230`) iterates row-by-row in Python to build per-snapshot dicts. Two layers of un-vectorization on top of the C++ engine.

19. **`OrderBook.update("bid", ...)` does string comparison per call** (`book_reconstructor.py:58`, C++ `lob_engine.cpp:14`). At "1M+ updates/sec" target the string compare in Python is the bottleneck; in C++ less so but still O(string-length).

20. **`generate_realistic.py` Day 2 cascade has `cascade_duration_min=45` for day index 1 only.** `data/generate_realistic.py:181`. Function signature defaults to 30 (`:93`). Easy to miss when reading.

21. **Demo-mode HMM is not actually fit.** `app.py:163-167` calls `_mock_data.generate_all(...)`, which already includes precomputed `hmm.states`, `state_probs`, `transition_matrix` (`_mock_data.py:173-209`). `RegimeDetector` is never invoked in demo mode. The dashboard footer still claims "Gaussian HMM (Baum-Welch EM)" — true only on the non-demo path.

22. **Module-scope demo app.** `dashboard/app.py:619` runs `app = create_app()` at import time, which constructs the entire layout, runs `generate_all(3600)`, and builds four Plotly figures. Importing `dashboard.app` for *any* reason — including from a test — does all this work. Tests rely on this behavior (`test_dashboard.py:254-271`).

23. **`Diagnostics` panel silently drops regime shading when there are >100 transitions.** `dashboard/components/diagnostics.py:37-38`. No log message, no fallback rendering.

24. **`depth_surface.create_depth_surface_figure(snapshots, regimes)` ignores `regimes`.** `dashboard/components/depth_surface.py:61-65`: `regimes` is in the signature, never used in the body. The doc text in `_constants.py:36-41` describes the surface as side-colored, which it is — but a future caller passing real regime info will see no effect.

25. **Regime-overlay band in `heatmap.py` uses a stacked-area trick.** `dashboard/components/heatmap.py:94-110`. Three separate fill-to-zero scatter traces with binary masks; on transition points the fills can produce visual artifacts because all three traces share the same x-axis without explicit per-segment shapes.

26. **CLAUDE.md rules vs git log.** The instructions in `CLAUDE.md` say "NEVER add Claude as a co-author" and require committing as the repo owner. Past commits in `git log` are authored by GitHub PR bots (`claude/...` branch names, "Add files via upload" merges), which technically complies but obscures who wrote what.

27. **PROJECT_SPEC.md describes a Bybit downloader.** `PROJECT_SPEC.md:43-72` defines the data layer around `public.bybit.com/orderbook/`. The actual `data/download.py` uses Tardis.dev. The spec is stale.

28. **Mock data RNG is module-scope** (`dashboard/_mock_data.py:17`). Importing `_mock_data` and calling generators multiple times advances the same global RNG; results are deterministic-but-stateful, which can confuse test isolation.

29. **`_compute_durations` exists in two places.** `src/hmm_model.py:419` and `dashboard/app.py:77` (`_compute_regime_durations`). Same algorithm, two implementations.

30. **Tests do not exercise the real-data pipeline.** `tests/test_dashboard.py:279-289` only verifies that `_find_data_files` raises `NoDataError` when files are missing. No test ingests a CSV (generated or real), runs `pipeline.run_pipeline`, and verifies any output.

---

## 5. Summary of unsupported / weak claims

The strongest claims in the marketing materials are not backed by code:

- "1M+ updates/sec" — never measured.
- "Walk-forward 70/30 train/test" — not implemented.
- "Full-covariance HMM" — pipeline runs diag.
- "VPIN" — computed on TOB liquidity, not trade flow, on every real-data path.
- "OFI (Cont, Kukanov & Stoikov 2014)" — formula is a simplified diff-of-sums, not CKS.
- "Sharpe 1.8–2.5" / "Kyle's λ 2–3× in Toxic" / "VPIN leads transitions by 30–120s" — no experiment, no saved results, no reproducible script.
- "158 tests" — 160 tests exist; whether they pass requires running.
- "Trailing rolling z-score, no lookahead" — bypassed in the pipeline in favor of global StandardScaler.
- "Synchronized crosshairs / regime toggles / animation" — not implemented; only a time-range slider.
- Default sample-interval claim conflicts between README (100 ms) and CLI default (1000 ms).

What *is* solid:
- Tardis CSV ingestion and order-book reconstruction (both Python and C++ paths, with parity test at `tests/test_cpp_engine.py:165-191`).
- Resampling, parquet I/O.
- Per-feature unit tests covering shape, sign, range, and a handful of hand-computed values.
- HMM wrapping of `hmmlearn.GaussianHMM` with sensible state-sorting and restart logic.
- Dashboard renders correctly in demo mode; layout is clean.
