# LOB Regime Scanner

An interactive market microstructure analytics platform that uses Hidden Markov Models to detect latent regimes in cryptocurrency order book data. Built as a bridge between undergraduate research in hidden-state inference (DevStats, CRA Award) and quantitative finance.

**Author:** Cameron Scarpati (incoming CMU MSCF, former Morgan Stanley Speedway team)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LOB REGIME SCANNER                        │
├──────────────┬──────────────┬──────────────┬────────────────┤
│  Data Layer  │ Feature Eng  │  HMM Engine  │  Dashboard     │
│              │              │              │                │
│ Bybit L2     │ OFI (multi-  │ Gaussian HMM │ Bookmap-style  │
│ historical   │   level)     │ with asymm.  │ LOB heatmap    │
│ data loader  │ VPIN         │ transitions  │                │
│              │ Spread stats │              │ Regime overlay │
│ C++ LOB      │ Book imbal.  │ Viterbi path │ bands          │
│ reconstruc-  │ Trade flow   │ decoding     │                │
│ tion engine  │ aggression   │              │ 3D depth       │
│              │ Kyle lambda  │ Online Baum- │ surface        │
│              │ Cancel ratio │ Welch for    │                │
│              │ Realized vol │ streaming    │ Toxicity gauge │
│              │ (multi-freq) │ updates      │ (VPIN)         │
└──────────────┴──────────────┴──────────────┴────────────────┘
```

## Stack

- **Python** — primary language for data processing, feature engineering, HMM, and dashboard
- **C++** — optional performance-critical LOB reconstruction via pybind11
- **Plotly Dash** — interactive multi-panel dashboard

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
```

Or manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Project Structure

```
lob-regime-scanner/
├── data/               # Data download scripts and raw storage
├── src/                # Core modules (data loading, features, HMM, backtest)
│   └── cpp/            # Optional C++ LOB engine with pybind11
├── dashboard/          # Plotly Dash app with 4 synchronized panels
├── notebooks/          # Exploratory analysis and model fitting
├── tests/              # Unit tests
└── docs/               # Methodology and results write-ups
```

## Dashboard Panels

1. **Order Book Heatmap** — Bookmap-style visualization with regime overlay bands
2. **Regime State Probabilities** — HMM posterior probabilities + transition matrix
3. **3D Depth Surface** — Order book depth evolution as an interactive 3D landscape
4. **Toxicity & Diagnostics** — VPIN, OFI, spread time series with regime conditioning

## References

- Cont, Kukanov, Stoikov (2014). "The Price Impact of Order Book Events"
- Easley, López de Prado, O'Hara (2012). "Flow Toxicity and Liquidity in a High Frequency World"
- Hamilton (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series"
