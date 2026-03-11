# LOB Regime Scanner: Hidden-State Inference for Market Microstructure

## Project Brief

Build an end-to-end market microstructure analytics platform that ingests limit order book (LOB) data, computes order flow features, detects hidden market regimes using a Hidden Markov Model, and renders everything in an interactive multi-panel dashboard. The project demonstrates the same core skill — inferring hidden states from noisy signals — that I used in my undergraduate research (DevStats, a CRA award-winning academic integrity system), applied to the domain of quantitative finance.

**Target audience:** Quant research recruiters at Two Sigma, Millennium, DE Shaw, and similar firms.
**Author:** Cameron Scarpati (incoming CMU MSCF, former Morgan Stanley Speedway team)
**Stack:** Python (primary), C++ (performance-critical LOB reconstruction), Plotly Dash (dashboard)

---

## Architecture Overview

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
│ Real-time    │ Book imbal.  │ Viterbi path │ bands          │
│ WebSocket    │ Trade flow   │ decoding     │                │
│ feed (opt.)  │ aggression   │              │ 3D depth       │
│              │ Kyle lambda  │ Online Baum- │ surface        │
│ C++ LOB      │ Cancel ratio │ Welch for    │                │
│ reconstruc-  │ Realized vol │ streaming    │ Toxicity gauge │
│ tion engine  │ (multi-freq) │ updates      │ (VPIN)         │
│              │              │              │                │
│ FI-2010      │ Rolling      │ Regime-cond. │ Price impact   │
│ benchmark    │ z-scores     │ return stats │ curves         │
│ (optional)   │              │              │                │
└──────────────┴──────────────┴──────────────┴────────────────┘
```

---

## Phase 1: Data Pipeline (Week 1-2)

### 1.1 Bybit Historical L2 Data Loader

Download and parse Bybit historical order book data (free, no API key needed).

**Data source:** https://public.bybit.com/orderbook/ — organized as `/{symbol}/{date}.csv.gz`
- Focus on `BTCUSDT` perpetual futures
- Each row: timestamp (μs), side, price, quantity for 500 levels
- Download 2-3 weeks of data covering at least one significant price move (e.g., a liquidation cascade)

**Tasks:**
1. Write a download script that fetches date ranges for a given symbol
2. Parse compressed CSV files into a pandas DataFrame with columns: `timestamp`, `side`, `price`, `qty`, `level`
3. Build an order book snapshot reconstructor: at each timestamp, maintain the full bid/ask book as sorted price-level arrays
4. Resample to uniform time intervals (100ms or 1s) for feature computation
5. Store processed data in Parquet format for fast reload

**Data schema for processed snapshots:**
```
timestamp: int64 (microseconds since epoch)
mid_price: float64
spread: float64
bid_price_1..10: float64
bid_qty_1..10: float64
ask_price_1..10: float64
ask_qty_1..10: float64
last_trade_price: float64
last_trade_qty: float64
last_trade_side: str ("buy" | "sell")
```

### 1.2 C++ LOB Reconstruction Engine (Optional Performance Layer)

For the performance-critical path, build a C++ order book reconstructor:
- Use a `std::map<double, double>` or flat sorted array for each side
- Expose via pybind11 so Python can call `book.update(side, price, qty)` and `book.snapshot()`
- Target: process 1M+ updates/second (demonstrate Speedway-level systems thinking)
- This is optional but impressive — start with pure Python, add C++ if time allows

---

## Phase 2: Feature Engineering (Week 3-4)

### 2.1 Order Flow Imbalance (OFI)

The primary signal. Computed as the net change in resting volume at the top N price levels:

```
OFI_t = Σ_{i=1}^{N} [ΔV^{bid}_{t,i} - ΔV^{ask}_{t,i}]
```

Where `ΔV^{bid}_{t,i}` is the change in bid volume at level `i` from time `t-1` to `t`.

**Implementation:**
- Compute OFI at multiple depths: top 1, top 5, top 10 levels
- Normalize by rolling standard deviation (z-score) over a 5-minute window
- Also compute the "OFI velocity" (rate of change of OFI)

**Reference:** Cont, Kukanov, Stoikov (2014) "The Price Impact of Order Book Events"

### 2.2 VPIN (Volume-Synchronized Probability of Informed Trading)

VPIN estimates the probability that order flow is dominated by informed traders.

**Implementation:**
1. Classify each trade as buy or sell using the tick rule (if no trade-side data) or use Bybit's taker-side field
2. Partition trades into volume buckets of fixed size V (e.g., V = median daily volume / 50)
3. Within each bucket, compute `VPIN = |V_buy - V_sell| / V`
4. Use the `flowrisk` Python library (pip install flowrisk) as a reference implementation

**Reference:** Easley, López de Prado, O'Hara (2012) "Flow Toxicity and Liquidity in a High Frequency World"

### 2.3 Additional Microstructure Features

Compute the following at each resampled timestamp:

| Feature | Formula | Interpretation |
|---------|---------|----------------|
| Book imbalance ratio | `(V_bid - V_ask) / (V_bid + V_ask)` at top N levels | Directional pressure |
| Weighted mid-price | `(ask_1 × bid_qty_1 + bid_1 × ask_qty_1) / (bid_qty_1 + ask_qty_1)` | Fairer mid estimate |
| Spread (bps) | `(ask_1 - bid_1) / mid × 10000` | Liquidity measure |
| Kyle's lambda | Rolling regression slope of `ΔP` on signed `√volume` | Price impact |
| Trade flow aggression | Fraction of trades at or beyond the opposite quote | Urgency signal |
| Cancellation ratio | Cancelled volume / total order volume (rolling window) | HFT activity proxy |
| Realized volatility | `√(Σ r²_i)` at 1s, 10s, 60s, 300s horizons | Multi-scale vol |
| Return autocorrelation | `corr(r_t, r_{t-k})` for k=1..10 at 1s resolution | Mean-rev vs momentum |

### 2.4 Feature Matrix Assembly

Stack all features into a matrix `X` of shape `(T, F)` where T = number of timestamps and F = number of features. Standardize each feature to zero mean, unit variance using a **rolling window** (not global, to avoid lookahead bias). Handle NaN/inf values from early-window periods.

---

## Phase 3: Hidden Markov Model Regime Detection (Week 5-6)

### 3.1 Model Specification

Use a Gaussian Hidden Markov Model with **3 states** representing:
- **State 0: "Quiet"** — low volatility, balanced order flow, tight spreads
- **State 1: "Trending"** — directional OFI, rising volatility, momentum
- **State 2: "Toxic/Stressed"** — extreme OFI, wide spreads, high VPIN, mean-reversion

The number of states (3) is a starting point. Use BIC/AIC to evaluate 2, 3, 4, and 5 states.

**Library:** `hmmlearn.GaussianHMM`

```python
from hmmlearn import GaussianHMM

model = GaussianHMM(
    n_components=3,         # number of hidden states
    covariance_type="full",  # full covariance between features
    n_iter=200,
    random_state=42,
    verbose=True
)

# Fit on training data (first 70% of time series)
model.fit(X_train)

# Decode most likely state sequence
states = model.predict(X_test)

# Get state probabilities at each timestamp
state_probs = model.predict_proba(X_test)
```

### 3.2 Regime-Conditional Analysis

After decoding regimes, compute:
1. **Regime-conditional return distributions:** mean, std, skewness, kurtosis of forward 1s/10s/60s returns in each regime
2. **Regime duration statistics:** how long does each regime persist? What are the transition probabilities?
3. **Regime-conditional spread and VPIN:** are toxic regimes associated with wider spreads and higher VPIN? (Validation that the model learns meaningful states)
4. **Regime-conditional price impact curves:** do Kyle's lambda estimates differ by regime? (They should — impact should be higher in toxic regimes)

### 3.3 Simple Regime-Conditional Trading Signal

Backtest a minimal strategy to demonstrate the regime detection has alpha:
- When the model detects a transition from Quiet-to-Trending, enter a position in the direction of OFI
- When the model detects Toxic/Stressed, flatten all positions
- Compute Sharpe ratio, max drawdown, hit rate, and profit per trade
- **Important:** this is NOT the point of the project — the regime detection and visualization are. The backtest is a validation that the detected regimes contain useful information. Keep it simple.

### 3.4 Model Diagnostics

- Plot log-likelihood convergence during EM fitting
- Compare BIC/AIC for 2-5 state models
- Run on 2-3 different date ranges to check regime stability
- Compare HMM regime labels against simple threshold-based regimes (e.g., high/low volatility) to show the HMM captures more structure

---

## Phase 4: Interactive Dashboard (Week 7-8)

### 4.1 Technology Stack

Use **Plotly Dash** for the dashboard framework. It's Python-native (good for learning), produces professional interactive visualizations, and deploys easily.

```
pip install dash plotly pandas numpy hmmlearn flowrisk pybind11
```

### 4.2 Dashboard Layout — Four Synchronized Panels

The dashboard should have a header with the instrument name, date range, and model metadata, followed by four main panels arranged in a 2x2 grid:

#### Panel 1: Order Book Heatmap with Regime Overlay (Top Left — LARGEST PANEL)

This is the centerpiece. Render a Bookmap-style heatmap:
- **X-axis:** time
- **Y-axis:** price level
- **Color intensity:** resting volume at that price level (use a sequential colormap like `Viridis` or `Inferno`)
- **Overlay:** color-coded horizontal bands at the top of the panel showing the detected regime (green = Quiet, blue = Trending, red = Toxic)
- **Mid-price line:** white/yellow line showing the mid-price evolution
- **Trade markers:** small circles on the heatmap where large trades occurred, sized by volume

Use `plotly.graph_objects.Heatmap` with `zsmooth='best'` for smooth rendering. For the regime overlay, use `plotly.graph_objects.Scatter` with `fill='tozeroy'` in a subplot with shared x-axis.

**Reference visualization:** Bookmap (bookmap.com) — your free version of this is impressive.

#### Panel 2: Regime State Probabilities (Top Right)

A stacked area chart showing the HMM's posterior probability of each state over time:
- Three colored areas stacked to 1.0 at each timestamp
- Sharp transitions = high-confidence regime changes
- Gradual transitions = ambiguous periods (interesting for analysis)
- Use `plotly.graph_objects.Scatter` with `stackgroup='one'`

Below this, show a small **transition matrix heatmap** — the 3×3 matrix of regime transition probabilities learned by the HMM.

#### Panel 3: 3D Order Book Depth Surface (Bottom Left)

A 3D surface plot showing order book depth evolution:
- **X-axis:** time (last N minutes)
- **Y-axis:** price level (centered on mid-price)
- **Z-axis:** volume
- **Color:** regime state
- Use `plotly.graph_objects.Surface`
- Allow rotation/zoom interaction

This is the "wow factor" visualization — seeing the order book as a landscape that shifts and morphs over time is visually stunning and immediately intuitive.

#### Panel 4: Toxicity & Diagnostics (Bottom Right)

A multi-row subplot with:
1. **VPIN time series** with threshold line (e.g., 0.5) and color-coded background by regime
2. **OFI (normalized)** with regime-conditional mean lines
3. **Spread (bps)** — should widen during Toxic regimes
4. **Cumulative PnL** of the simple regime-conditional strategy (if implemented)

### 4.3 Interactivity

- **Synchronized crosshairs:** hovering on any panel highlights the same timestamp on all panels
- **Date range slider:** allow selecting subsets of the data
- **Regime filter buttons:** toggle individual regimes on/off to isolate their characteristics
- **Play/pause animation:** ability to "replay" the order book evolution at adjustable speed

### 4.4 Dashboard Code Structure

```
lob-regime-scanner/
├── README.md                   # Project overview, setup, screenshots
├── requirements.txt
├── data/
│   ├── download.py             # Bybit data downloader
│   └── raw/                    # Raw downloaded files
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Parse Bybit L2 data > DataFrames
│   ├── book_reconstructor.py   # Build LOB snapshots from updates
│   ├── features.py             # All feature computations (OFI, VPIN, etc.)
│   ├── hmm_model.py            # HMM fitting, decoding, diagnostics
│   ├── backtest.py             # Simple regime-conditional strategy
│   └── cpp/                    # Optional C++ LOB engine
│       ├── lob_engine.cpp
│       ├── lob_engine.hpp
│       └── bindings.cpp        # pybind11 bindings
├── dashboard/
│   ├── app.py                  # Main Dash app
│   ├── components/
│   │   ├── heatmap.py          # Panel 1: LOB heatmap
│   │   ├── regime_probs.py     # Panel 2: state probabilities
│   │   ├── depth_surface.py    # Panel 3: 3D surface
│   │   └── diagnostics.py      # Panel 4: VPIN, OFI, spread
│   ├── callbacks.py            # Dash callbacks for interactivity
│   └── assets/
│       └── styles.css          # Dashboard styling
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_hmm_fitting.ipynb
│   └── 04_regime_analysis.ipynb
├── tests/
│   ├── test_features.py
│   ├── test_hmm.py
│   └── test_book_reconstructor.py
└── docs/
    ├── methodology.md          # Mathematical details
    └── results.md              # Key findings and visualizations
```

---

## Phase 5: Write-Up and Presentation (Final Polish)

### 5.1 README.md

The README is the first thing a recruiter sees. It should include:
1. A single compelling screenshot of the dashboard at the top
2. One-paragraph summary: "An interactive market microstructure analytics platform that uses Hidden Markov Models to detect latent regimes in cryptocurrency order book data. Built as a bridge between my undergraduate research in hidden-state inference (DevStats, CRA Award) and quantitative finance."
3. Key findings (2-3 bullets)
4. Setup instructions
5. Architecture diagram
6. Links to notebooks with detailed analysis

### 5.2 Key Findings to Highlight

Frame results around questions a Two Sigma researcher would care about:
1. "The HMM identifies 3 distinct regimes that correspond to empirically different return distributions — the toxic regime has 4x the volatility of the quiet regime and negative return autocorrelation (mean-reversion), while the trending regime shows positive autocorrelation (momentum)"
2. "VPIN spikes systematically precede regime transitions to the toxic state by 30-120 seconds, suggesting order flow toxicity is a leading indicator of microstructure stress"
3. "Price impact (Kyle's lambda) is 2-3x higher in the toxic regime, consistent with adverse selection theory — market makers face higher costs when informed traders dominate flow"

### 5.3 Methodology Document

Write a 2-3 page methodology document covering:
- Mathematical formulation of OFI, VPIN, and the Gaussian HMM
- Model selection (BIC/AIC comparison)
- Backtesting methodology (walk-forward, no lookahead)
- Limitations and future work

---

## Technical Notes

### Python Dependencies
```
numpy>=1.24
pandas>=2.0
plotly>=5.18
dash>=2.14
hmmlearn>=0.3
scikit-learn>=1.3
flowrisk>=0.2
pyarrow>=14.0       # for Parquet
requests>=2.31      # for data download
gzip                # stdlib, for .csv.gz
pybind11>=2.11      # optional, for C++ bindings
pytest>=7.4         # for tests
```

### Performance Targets
- Data loading: 1 day of L2 data in < 30 seconds
- Feature computation: < 5 seconds for 1 day at 100ms resolution
- HMM fitting: < 60 seconds for 1 week of data (3 states, ~10 features)
- Dashboard rendering: < 2 seconds for initial load, < 200ms for interactions

### Stretch Goals (If Time Permits)
1. **Real-time mode:** Connect to Bybit WebSocket feed and run the HMM regime detector live, updating the dashboard every second
2. **Multi-asset comparison:** Run the same regime detector on BTC, ETH, and SOL simultaneously and visualize regime synchronization across assets (when do all three enter "toxic" mode at once?)
3. **C++ Viterbi decoder:** Implement the Viterbi algorithm in C++ with pybind11 bindings — demonstrate that regime decoding runs in < 1μs per timestamp (Speedway-level performance on a finance problem)
4. **Regime-conditional optimal execution:** Given a parent order to execute, simulate how a TWAP vs. regime-aware execution strategy would perform — enter passively during Quiet, more aggressively during Trending, pause during Toxic

---

## References

1. Cont, R., Kukanov, A., Stoikov, S. (2014). "The Price Impact of Order Book Events." Journal of Financial Econometrics.
2. Easley, D., López de Prado, M., O'Hara, M. (2012). "Flow Toxicity and Liquidity in a High Frequency World." Review of Financial Studies.
3. Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle." Econometrica.
4. Salvi, J. (2025). "Asymmetric Hidden Markov Models for Intraday Alpha." SSRN: 5315733.
5. Backhouse, T., et al. (2025). "Painting the Market: Generative Diffusion Models for LOB Simulation." arXiv: 2509.05107.
6. Zhang, Z., et al. (2019). "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books." IEEE TSP.
7. Berti, L., et al. (2025). "TLOB: Transformer with Dual Attention for LOB Price Prediction." arXiv: 2502.15757.

---

## How to Use This Document with Claude Code

Run `claude` in the project root directory and paste:

```
Read PROJECT_SPEC.md and help me build this project phase by phase.
Start with Phase 1: set up the project structure, create requirements.txt,
and build the Bybit L2 data downloader. Let's begin.
```

Then iterate phase by phase. Each phase should end with working, tested code before moving to the next.
