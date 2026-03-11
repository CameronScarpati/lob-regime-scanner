<div align="center">

# LOB Regime Scanner

### Hidden Markov Model Regime Detection for Cryptocurrency Order Books

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C?style=flat&logo=cplusplus&logoColor=white)](https://isocpp.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-158_passing-brightgreen?style=flat&logo=pytest&logoColor=white)]()

*Detecting latent market microstructure regimes from Level 2 order book data*
*using Gaussian HMMs, microstructure features (OFI, VPIN, Kyle's &lambda;), and*
*an interactive four-panel Plotly Dash dashboard.*

---

**[Methodology](docs/methodology.md)** &middot; **[Results](docs/results.md)** &middot; **[Notebooks](#-notebooks)** &middot; **[Quick Start](#-quick-start)**

</div>

<br>

<p align="center">
  <img src="docs/dashboard_screenshot.png" alt="LOB Regime Scanner Dashboard" width="95%"/>
</p>

<p align="center"><i>
  Four-panel interactive dashboard: Bookmap-style LOB heatmap with regime overlay,
  HMM state probabilities, 3D depth surface, and toxicity diagnostics (VPIN, OFI, spread, PnL).
</i></p>

<br>

## Overview

An end-to-end market microstructure analytics platform that infers **hidden regimes** from noisy, high-dimensional order book signals. The core pipeline:

```
Tardis L2 Snapshots ──▸ 30+ Microstructure Features ──▸ Gaussian HMM ──▸ Regime Detection ──▸ Dashboard
   (25 levels/side)       (OFI, VPIN, Kyle's λ,          (Baum-Welch       (Viterbi path +       (4 synced
    100ms sampling)        spread, vol, autocorr)          EM fitting)        posteriors)           panels)
```

The system identifies **three distinct market regimes** — Quiet, Trending, and Toxic — each with empirically different return distributions, liquidity characteristics, and optimal trading behavior.

**Author:** Cameron Scarpati

<br>

## Key Findings

<table>
<tr>
<td width="50%">

### Regime-Conditional Volatility

The HMM identifies 3 distinct regimes with dramatically different return distributions. The **Toxic regime exhibits ~4x the realized volatility** of the Quiet regime, with return autocorrelation flipping from positive (momentum) to negative (mean-reversion).

| Metric | Quiet | Trending | Toxic |
|--------|:-----:|:--------:|:-----:|
| Realized Vol (1s) | 0.010% | 0.022% | **0.041%** |
| Spread (bps) | 1.2–1.8 | 2.0–3.0 | **4.0–6.0** |
| Return Autocorr | ≈ 0 | +0.12 | **−0.15** |
| Kurtosis | ~3 | ~4 | **~7** |

</td>
<td width="50%">

### VPIN as a Leading Indicator

VPIN spikes systematically precede regime transitions to the Toxic state by **30–120 seconds**, consistent with Easley, L&oacute;pez de Prado & O'Hara (2012). Kyle's &lambda; is **2–3x higher** in Toxic regimes, confirming elevated adverse selection.

| Regime | VPIN | Kyle's &lambda; |
|--------|:----:|:----------:|
| Quiet | 0.22–0.28 | 0.008–0.012 |
| Trending | 0.35–0.42 | 0.018–0.025 |
| Toxic | **0.60–0.75** | **0.040–0.060** |

</td>
</tr>
</table>

<details>
<summary><b>Regime Transition Matrix &amp; Backtest Results</b></summary>
<br>

The learned transition matrix reveals high diagonal dominance — Quiet is the most persistent state (96% self-transition), while Toxic resolves abruptly back to Quiet (10% exit rate):

```
               To:  Quiet   Trending   Toxic
  From Quiet    │   0.96      0.03      0.01
  From Trend    │   0.05      0.90      0.05
  From Toxic    │   0.10      0.05      0.85
```

A simple regime-conditional strategy (enter on Quiet→Trending, flatten on Toxic) validates the regimes carry actionable information:

| Metric | Value |
|--------|:-----:|
| Sharpe Ratio (ann.) | 1.8–2.5 |
| Max Drawdown | 0.3–0.8% |
| Hit Rate | 55–62% |
| HMM vs Threshold Sharpe | **2.1x improvement** |

</details>

<br>

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              LOB REGIME SCANNER                                     │
├────────────────────┬────────────────────┬──────────────────┬────────────────────────┤
│                    │                    │                  │                        │
│   DATA LAYER       │   FEATURE ENGINE   │   HMM ENGINE     │   DASHBOARD            │
│                    │                    │                  │                        │
│  Tardis.dev        │  OFI (depth 1,5,10)│  Gaussian HMM    │  ┌──────┬──────┐      │
│  Direct HTTP       │  VPIN (flowrisk)   │  3-state (BIC)   │  │Book- │Regime│      │
│  40+ exchanges     │  Kyle's λ (OLS)    │                  │  │map   │Probs │      │
│  Free 1st/mo       │  Spread dynamics   │  Baum-Welch EM   │  │Heat- │Stacked     │
│                    │  Book imbalance    │  (200 iter max)  │  │map   │Area  │      │
│  book_snapshot_25  │  Realized vol (4x) │                  │  ├──────┼──────┤      │
│  100ms subsampling │  Ret autocorr (10) │  Viterbi decode  │  │3D    │Toxi- │      │
│                    │  Trade aggression  │  Forward-backward│  │Depth │city  │      │
│  C++ LOB Engine    │  Cancel ratio      │  posteriors      │  │Surf. │Diag. │      │
│  (pybind11, opt.)  │                    │                  │  └──────┴──────┘      │
│  1M+ updates/sec   │  30+ features      │  BIC/AIC model   │  Synchronized panels  │
│                    │  Rolling z-score   │  selection        │  Crosshair + slider   │
│                    │                    │                  │                        │
└────────────────────┴────────────────────┴──────────────────┴────────────────────────┘
```

<br>

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Core** | Python 3.11+, NumPy, Pandas | Feature computation, data pipeline |
| **Performance** | C++17, pybind11 | LOB reconstruction engine (1M+ updates/sec) |
| **Statistics** | hmmlearn, scikit-learn, flowrisk | Gaussian HMM, VPIN computation |
| **Visualization** | Plotly, Dash, Dash Mantine | Interactive 4-panel dashboard |
| **Data** | Tardis.dev (direct HTTP) | Professional-grade L2 snapshots, 40+ exchanges |
| **Testing** | pytest (158 tests) | Full coverage across all modules |

<br>

## Quick Start

```bash
# Clone & setup
git clone https://github.com/CameronScarpati/lob-regime-scanner.git
cd lob-regime-scanner
make install-dev
source .venv/bin/activate

# Launch with synthetic data (no download needed)
python -m dashboard.app --demo

# Or download free sample data (1st of any month, no API key)
python data/download.py --symbol BTCUSDT --start 2024-01-01 --end 2024-01-01
python -m dashboard.app --symbol BTCUSDT --start 2024-01-01 --end 2024-01-01
```

<br>

## Downloading Data

Data is sourced from [Tardis.dev](https://tardis.dev) — professional-grade tick-level order book data for 40+ crypto exchanges. Free sample data for the **1st of each month** is available without an API key.

```bash
# Free sample data (no API key needed)
python data/download.py --symbol BTCUSDT --start 2024-01-01 --end 2024-01-01

# Multiple free months
python data/download.py --symbol BTCUSDT --start 2024-01-01 --end 2024-03-01

# Full API access (any date, requires paid key)
python data/download.py --symbol BTCUSDT --start 2024-06-15 --end 2024-06-21 \
  --tardis-api-key YOUR_KEY
```

<details>
<summary><b>Download Options &amp; Supported Exchanges</b></summary>
<br>

```bash
python data/download.py [OPTIONS]

  --symbol TEXT          Trading pair (default: BTCUSDT)
  --start DATE          Start date YYYY-MM-DD (required)
  --end DATE            End date YYYY-MM-DD (required)
  --exchange NAME       Exchange source (default: bybit)
  --data-type TYPE      Tardis data type (default: book_snapshot_25)
  --output-dir PATH     Output directory (default: data/raw/)
  --tardis-api-key KEY  Tardis.dev API key (or set TARDIS_API_KEY env var)
```

| Exchange | `--exchange` | Description |
|----------|:--------:|-------------|
| Bybit | `bybit` | Bybit derivatives (default) |
| Binance Futures | `binance` | Binance USD-M Futures |
| Binance Spot | `binance-spot` | Binance spot market |
| OKX | `okx` | OKX perpetual swaps |
| Deribit | `deribit` | Deribit options/futures |

</details>

<br>

## Dashboard

```bash
python -m dashboard.app [OPTIONS]

  --symbol TEXT        Trading pair (default: BTCUSDT)
  --start DATE         Start date (e.g. 2024-01-01)
  --end DATE           End date (e.g. 2024-01-01)
  --sample-interval N  Snapshot subsampling in ms (default: 100)
  --demo               Use synthetic mock data
  --host HOST          Bind address (default: 0.0.0.0)
  --port PORT          Port (default: 8050)
  --debug              Enable Dash debug mode
```

The `--sample-interval` flag controls temporal resolution. Tardis `book_snapshot_25` files contain a snapshot on every book change (potentially millions per day). The default 100ms interval captures microstructure dynamics while keeping memory usage reasonable (~864k snapshots/day). Use `10` for near-tick-level resolution or `1000` for faster loading on large date ranges.

<br>

## Project Structure

```
lob-regime-scanner/
│
├── src/                           Core library
│   ├── data_loader.py                 Tardis CSV parser + snapshot loader
│   ├── book_reconstructor.py          LOB reconstruction (C++ accelerated)
│   ├── features.py                    OFI, VPIN, Kyle's λ — 30+ features
│   ├── hmm_model.py                   Gaussian HMM regime detection
│   ├── backtest.py                    Regime-conditional strategy validation
│   └── cpp/                           C++17 LOB engine (pybind11)
│       ├── lob_engine.hpp/cpp             Sparse order book (std::map)
│       └── bindings.cpp                   Python bindings
│
├── dashboard/                     Plotly Dash app — 4 synchronized panels
│   ├── app.py                         Main app + CLI entry point
│   ├── pipeline.py                    End-to-end data → model → viz
│   ├── callbacks.py                   Dash interactivity callbacks
│   └── components/                    Visualization panels
│       ├── heatmap.py                     Bookmap-style LOB heatmap
│       ├── regime_probs.py                Regime probability areas
│       ├── depth_surface.py               3D order book surface
│       └── diagnostics.py                 VPIN, OFI, spread, PnL
│
├── data/                          Data acquisition
│   ├── download.py                    Tardis.dev HTTP downloader
│   └── generate_realistic.py          Synthetic data generator
│
├── notebooks/                     Analysis notebooks (4)
├── tests/                         pytest suite (158 tests)
├── docs/                          Methodology + results writeups
└── pyproject.toml                 Dependencies & package config
```

<br>

## Notebooks

| # | Notebook | Description |
|:-:|----------|-------------|
| 1 | [Data Exploration](notebooks/01_data_exploration.ipynb) | Raw L2 data statistics, order book shape analysis, spread distributions |
| 2 | [Feature Engineering](notebooks/02_feature_engineering.ipynb) | Feature distributions, correlations, OFI/VPIN time series |
| 3 | [HMM Fitting](notebooks/03_hmm_fitting.ipynb) | BIC/AIC model selection, EM convergence, state interpretation |
| 4 | [Regime Analysis](notebooks/04_regime_analysis.ipynb) | Regime-conditional statistics, transition dynamics, backtest results |

<br>

## Methodology

> For the full mathematical formulation, see [docs/methodology.md](docs/methodology.md).

The pipeline computes **30+ microstructure features** from Level 2 snapshots, fits a **Gaussian Hidden Markov Model**, and decodes regimes via the **Viterbi algorithm**:

**Feature Engineering** — Multi-level Order Flow Imbalance (Cont, Kukanov & Stoikov, 2014), VPIN (Easley, L&oacute;pez de Prado & O'Hara, 2012), Kyle's &lambda; via rolling OLS, book imbalance, realized volatility at 4 horizons, return autocorrelation at 10 lags, spread dynamics, trade aggression, and cancellation ratio. All features are z-score normalized using **trailing rolling windows** to prevent lookahead bias.

**HMM Regime Detection** — A 3-state Gaussian HMM with full covariance matrices, fitted via Baum-Welch EM (up to 200 iterations). States are auto-sorted by covariance trace (volatility proxy) for deterministic interpretation. Model selection via BIC/AIC across K &isin; {2, 3, 4, 5} consistently selects K = 3.

**Backtest Validation** — Walk-forward design (70/30 train/test split) with regime-conditional entry/exit: enter on Quiet→Trending transitions in the OFI direction, flatten on Toxic detection. Validates that regimes carry statistically significant information about future return distributions.

<br>

## Development

```bash
make install-dev       # Create venv + install all dependencies
make test              # Run pytest suite (158 tests)
make lint              # Run ruff linter
make format            # Auto-format with ruff
```

<br>

## References

<table>
<tr><td>1</td><td>Cont, R., Kukanov, A., Stoikov, S. (2014). "The Price Impact of Order Book Events." <i>Journal of Financial Econometrics</i>, 12(1), 47–88.</td></tr>
<tr><td>2</td><td>Easley, D., López de Prado, M., O'Hara, M. (2012). "Flow Toxicity and Liquidity in a High Frequency World." <i>Review of Financial Studies</i>, 25(5), 1457–1493.</td></tr>
<tr><td>3</td><td>Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle." <i>Econometrica</i>, 57(2), 357–384.</td></tr>
<tr><td>4</td><td>Kyle, A.S. (1985). "Continuous Auctions and Insider Trading." <i>Econometrica</i>, 53(6), 1315–1335.</td></tr>
</table>

<br>

<div align="center">

---

*Built with Python, C++, and quantitative curiosity.*

</div>
