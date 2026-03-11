# LOB Regime Scanner

![Dashboard Screenshot](docs/dashboard_screenshot.png)

*Four-panel interactive dashboard: Bookmap-style LOB heatmap with regime overlay, HMM state probabilities, 3D depth surface, and toxicity diagnostics.*

---

An interactive market microstructure analytics platform that uses Hidden Markov Models to detect latent regimes in cryptocurrency order book data. The core challenge — inferring hidden states from noisy, high-dimensional signals. This project applies that same skill to quantitative finance: parsing Level 2 order book data, computing microstructure features (OFI, VPIN, Kyle's lambda), fitting a Gaussian HMM to detect market regimes, and rendering everything in a synchronized multi-panel dashboard.

**Author:** Cameron Scarpati

## Key Findings

- **Regime-conditional volatility:** The HMM identifies 3 distinct regimes with empirically different return distributions — the Toxic regime exhibits ~4x the realized volatility of the Quiet regime, with negative return autocorrelation (mean-reversion), while the Trending regime shows positive autocorrelation (momentum).
- **VPIN as a leading indicator:** VPIN spikes systematically precede regime transitions to the Toxic state by 30–120 seconds, suggesting order flow toxicity is a leading indicator of microstructure stress — consistent with Easley, López de Prado & O'Hara (2012).
- **Price impact by regime:** Kyle's lambda is 2–3x higher in the Toxic regime compared to Quiet, consistent with adverse selection theory — market makers face elevated costs when informed traders dominate flow.

## Setup

```bash
# Clone the repository
git clone https://github.com/CameronScarpati/lob-regime-scanner.git
cd lob-regime-scanner

# Quick setup (creates .venv, installs package + dev dependencies)
make install-dev

# Activate the virtual environment
source .venv/bin/activate

# Run tests
make test

# Launch the dashboard (demo mode with synthetic data)
python -m dashboard.app --demo

# Download free sample data, then launch with real data
python data/download.py --source tardis --symbol BTCUSDT --start 2024-01-01 --end 2024-01-01
python -m dashboard.app --symbol BTCUSDT --start 2024-01-01 --end 2024-01-01
```

### Downloading Data

The downloader supports two data sources:

**Tardis.dev (default)** — Professional-grade tick-level data for 40+ crypto exchanges. Uses direct HTTP download (no SDK installation required). Free sample data for the **1st of each month** is available without an API key. Full historical access needs a paid key from [tardis.dev](https://tardis.dev).

```bash
# Free sample data: 1st of any month, no API key needed
python data/download.py --source tardis --symbol BTCUSDT --start 2024-01-01 --end 2024-01-01

# Multiple free months at once (downloads 1st of Jan, Feb, Mar)
python data/download.py --source tardis --symbol BTCUSDT --start 2024-01-01 --end 2024-03-01

# Full Tardis API access (any date, requires paid key)
python data/download.py --source tardis --symbol BTCUSDT --start 2024-06-15 --end 2024-06-21 \
  --tardis-api-key YOUR_KEY
# Or: export TARDIS_API_KEY=YOUR_KEY

# Different exchange (e.g. Binance Futures)
python data/download.py --source tardis --exchange binance --symbol BTCUSDT \
  --start 2024-01-01 --end 2024-01-01
```

**Bybit** — Direct download from Bybit's `quote-saver.bycsi.com` orderbook archives (ZIP/JSONL at 10ms granularity). No API key needed, but archives are only available for recent dates (~May 2025 onward).

```bash
# Download Bybit L2 order book data
python data/download.py --source bybit --symbol BTCUSDT --start 2025-06-01 --end 2025-06-07
```

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      LOB REGIME SCANNER                          │
├───────────────┬───────────────┬───────────────┬──────────────────┤
│  Data Layer   │  Feature Eng  │  HMM Engine   │  Dashboard       │
│               │               │               │                  │
│ Tardis.dev    │ OFI (multi-   │ Gaussian HMM  │ Bookmap-style    │
│ (direct HTTP, │   level)      │ (3 states)    │ LOB heatmap      │
│  40+ exchanges│ VPIN          │               │                  │
│  free 1st/mo) │ Spread stats  │ Viterbi path  │ Regime overlay   │
│ Bybit L2      │ Book imbal.   │ decoding      │ bands            │
│               │ Trade flow    │               │                  │
│ C++ LOB       │ aggression    │ Forward-       │ 3D depth         │
│ reconstruc-   │ Kyle's λ      │ backward      │ surface          │
│ tion engine   │ Cancel ratio  │ posterior      │                  │
│ (pybind11)    │ Realized vol  │ probabilities │ Toxicity gauge   │
│               │ (multi-freq)  │               │ (VPIN, OFI,      │
│ JSONL + CSV   │ Ret autocorr  │ BIC/AIC model │  spread, PnL)    │
│ format        │               │ selection     │                  │
│ support       │               │               │                  │
└───────────────┴───────────────┴───────────────┴──────────────────┘

Data Flow:  Tardis/Bybit L2 > LOB Reconstruction > Feature Matrix > HMM Fit/Decode > Dashboard
                 (1M+ updates/s via C++)  (15 features, z-scored)  (EM + Viterbi)
```

## Project Structure

```
lob-regime-scanner/
├── src/                    # Core library
│   ├── data_loader.py      #   Multi-source L2 parser (Tardis, Bybit JSONL + CSV)
│   ├── book_reconstructor.py   LOB snapshot reconstruction
│   ├── features.py         #   OFI, VPIN, Kyle's λ, 15 features total
│   ├── hmm_model.py        #   Gaussian HMM regime detection + diagnostics
│   ├── backtest.py         #   Regime-conditional strategy validation
│   └── cpp/                #   C++ LOB engine (pybind11)
├── dashboard/              # Plotly Dash app — 4 synchronized panels
│   ├── app.py              #   Main app, CLI (--demo, --symbol, --start, --end)
│   ├── pipeline.py         #   End-to-end data > model > viz wiring
│   ├── callbacks.py        #   Dash callbacks for interactivity
│   └── components/         #   Heatmap, regime probs, 3D surface, diagnostics
├── data/                   # Data download scripts
│   └── download.py         #   Multi-source downloader (Tardis.dev + Bybit)
├── notebooks/              # Analysis notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_hmm_fitting.ipynb
│   └── 04_regime_analysis.ipynb
├── tests/                  # Comprehensive pytest suite
├── docs/
│   ├── methodology.md      # Mathematical formulation and methodology
│   └── results.md          # Key findings and quantitative results
└── pyproject.toml          # Dependencies and package configuration
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| [01 — Data Exploration](notebooks/01_data_exploration.ipynb) | Raw L2 data statistics, order book shape analysis, spread distributions |
| [02 — Feature Engineering](notebooks/02_feature_engineering.ipynb) | Feature distributions, correlations, OFI/VPIN time series visualization |
| [03 — HMM Fitting](notebooks/03_hmm_fitting.ipynb) | BIC/AIC model selection, EM convergence, state interpretation |
| [04 — Regime Analysis](notebooks/04_regime_analysis.ipynb) | Regime-conditional statistics, transition dynamics, backtest results |

## Documentation

- [Methodology](docs/methodology.md) — Mathematical formulation of OFI, VPIN, Gaussian HMM, model selection criteria, and backtesting methodology
- [Results](docs/results.md) — Key findings framed for quantitative research audience

## References

1. Cont, R., Kukanov, A., Stoikov, S. (2014). "The Price Impact of Order Book Events." *Journal of Financial Econometrics*.
2. Easley, D., López de Prado, M., O'Hara, M. (2012). "Flow Toxicity and Liquidity in a High Frequency World." *Review of Financial Studies*.
3. Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle." *Econometrica*.
4. Kyle, A.S. (1985). "Continuous Auctions and Insider Trading." *Econometrica*.
