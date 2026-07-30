# Methodology

This document describes the mathematical formulation of the microstructure features, the Hidden Markov Model regime detection framework, model selection criteria, backtesting methodology, and known limitations. This is a learning project: where the implemented pipeline differs from the textbook formulation or from a production-grade design (diagonal covariance, a single 70/30 walk-forward split, stylized fills and costs), the difference is called out explicitly rather than glossed over. See Section 5 for the consolidated limitations.

---

## 1. Microstructure Features

### 1.1 Order Flow Imbalance (OFI)

OFI measures the net change in resting liquidity across the top levels of the limit order book. Two formulations are implemented:

**Simplified multi-level proxy** (`compute_ofi`) at time *t* and depth *d*:

```
OFI_t^(d) = Σ_{i=1}^{d} [ ΔV^{bid}_{t,i} − ΔV^{ask}_{t,i} ]
```

where `ΔV^{bid}_{t,i} = V^{bid}_{t,i} − V^{bid}_{t−1,i}` is the change in bid volume at price level *i* from the previous snapshot. A positive OFI indicates net buying pressure (bid liquidity increasing relative to ask liquidity); a negative OFI indicates selling pressure. This version captures net order-book pressure without price-move conditioning, and is kept for comparison and dashboard display.

**Canonical Cont-Kukanov-Stoikov OFI** (`compute_ofi_cks`), which conditions each level's contribution on the direction of the price move at that level (Cont, Kukanov & Stoikov, 2014):

```
e^{bid}_{t,i} = V^{bid}_{t,i} · 1{P^{bid}_{t,i} ≥ P^{bid}_{t−1,i}} − V^{bid}_{t−1,i} · 1{P^{bid}_{t,i} ≤ P^{bid}_{t−1,i}}
e^{ask}_{t,i} = V^{ask}_{t,i} · 1{P^{ask}_{t,i} ≤ P^{ask}_{t−1,i}} − V^{ask}_{t−1,i} · 1{P^{ask}_{t,i} ≥ P^{ask}_{t−1,i}}

OFI_t^(d) = Σ_{i=1}^{d} [ e^{bid}_{t,i} − e^{ask}_{t,i} ]
```

A bid price improvement counts the entire new queue as buying pressure and a bid price drop counts the vanished queue as selling pressure, while an unchanged price contributes only the net queue change. The paper defines this at the best level; applying it per level and summing is the standard multi-level extension. **The HMM feature set uses the canonical formulation** (`ofi_cks_1`).

We compute OFI at depths *d* ∈ {1, 5, 10} to capture both top-of-book and deeper liquidity dynamics. Each OFI series is z-score normalized using a rolling 300-second (5-minute) trailing window to remove level effects and produce a stationary signal suitable for HMM ingestion:

```
z_t = (OFI_t − μ̂_t) / σ̂_t
```

where μ̂_t and σ̂_t are the rolling mean and standard deviation. We also compute OFI velocity (first difference of OFI) to capture the acceleration of order flow changes.

### 1.2 VPIN (Volume-Synchronized Probability of Informed Trading)

VPIN estimates the probability that order flow is dominated by informed traders, following Easley, López de Prado & O'Hara (2012). The construction proceeds in three steps:

1. **Trade classification:** Each snapshot is classified as a buy or sell using the tick rule applied to mid-price changes. If trade-side data is available from the exchange, it is used directly.

2. **Volume bucketing:** Trades are partitioned into buckets of fixed total volume *V*. We set *V* = (total session volume) / 50, following the original authors' recommendation.

3. **VPIN computation:** Within each volume bucket *n*:

```
VPIN_n = |V^{buy}_n − V^{sell}_n| / V
```

The final VPIN estimate is the rolling average over the most recent *N* = 20 buckets. VPIN values near 0 indicate balanced (uninformed) flow; values approaching 1 indicate extreme order flow imbalance from potentially informed participants.

We use the `flowrisk` library's `BulkVPIN` estimator for computation.

### 1.3 Kyle's Lambda (Price Impact Coefficient)

Kyle's lambda (Kyle, 1985) measures the price impact of order flow — the amount the price moves per unit of signed volume. We estimate it via rolling OLS regression:

```
ΔP_t = α + λ · sign(trade_t) · √|volume_t| + ε_t
```

where the slope coefficient λ is:

```
λ̂ = Cov(ΔP, signed_√vol) / Var(signed_√vol)
```

computed over a trailing 300-second window. Higher λ indicates greater adverse selection cost, typically observed during periods of informed trading activity.

### 1.4 Additional Features

| Feature | Formula | Range | Interpretation |
|---------|---------|-------|----------------|
| Book imbalance | `(V_bid − V_ask) / (V_bid + V_ask)` at top 10 levels | [−1, 1] | Directional pressure from resting orders |
| Weighted mid-price | `(ask₁ × bid_qty₁ + bid₁ × ask_qty₁) / (bid_qty₁ + ask_qty₁)` | ℝ | Volume-weighted fair price estimate |
| Spread (bps) | `(ask₁ − bid₁) / mid × 10,000` | [0, ∞) | Liquidity/transaction cost measure |
| Trade flow aggression | Fraction of trades at or beyond opposite quote (rolling) | [0, 1] | Urgency proxy |
| Cancellation ratio | Disappeared volume / total volume (rolling) | [0, 1] | HFT activity proxy |
| Realized volatility | `√(Σ r²_i)` at horizons 1s, 10s, 60s, 300s | [0, ∞) | Multi-scale volatility |
| Return autocorrelation | `corr(r_t, r_{t−k})` for *k* = 1, …, 10 (rolling) | [−1, 1] | Mean-reversion vs. momentum signature |

### 1.5 Feature Matrix Assembly

All features are assembled into a matrix **X** of shape (*T* × *F*), where *T* is the number of timestamps and *F* ≈ 36 features (9 simple OFI columns at 3 depths × 3 metrics, 6 canonical CKS OFI columns at 3 depths × 2 metrics, VPIN, 7 additional features, 4 realized volatility horizons, 10 autocorrelation lags).

The feature module supports z-score standardization using a **trailing rolling window** (`build_feature_matrix(standardize=True)`). The default dashboard pipeline calls `build_feature_matrix(standardize=False)` and feeds the raw features to the HMM, whose `StandardScaler` is **fit on the walk-forward training segment only** (Section 4.1). Held-out rows are therefore scaled with train-segment statistics, never with future data. In the legacy fully in-sample mode (`train_frac=None`) the scaler still sees the whole series, which is lookahead; that mode exists for comparison, not for evaluation. NaN and ±∞ values from early-window periods are forward-filled and any remaining leading NaNs are set to zero — there is deliberately **no backward fill**, which would leak future values into warm-up rows.

---

## 2. Hidden Markov Model

### 2.1 Model Specification

We model the observed feature vector **x**_t as being emitted from one of *K* = 3 hidden states (regimes) governed by a first-order Markov chain. The generative process is:

```
z_t | z_{t−1} ~ Categorical(A_{z_{t−1}, :})
x_t | z_t = k  ~ N(μ_k, Σ_k)
```

where:
- *z_t* ∈ {0, 1, 2} is the hidden state at time *t*
- **A** is the *K* × *K* transition probability matrix, with *A_{ij}* = P(*z_t* = *j* | *z_{t−1}* = *i*)
- **μ**_k ∈ ℝ^F is the emission mean for state *k*
- **Σ**_k is the covariance matrix for state *k* (diagonal in the default pipeline; see Section 2.2)
- **π** = P(*z*₁) is the initial state distribution

The three states are interpreted (post-hoc, after fitting) as:
- **State 0 — Quiet:** Low volatility, balanced OFI, tight spreads, low VPIN
- **State 1 — Trending:** Directional OFI, elevated volatility, positive return autocorrelation
- **State 2 — Toxic/Stressed:** Extreme OFI, wide spreads, high VPIN, mean-reverting returns

### 2.2 Parameter Estimation

Parameters θ = {**π**, **A**, {**μ**_k, **Σ**_k}} are estimated via the Expectation-Maximization (EM) algorithm, specifically the Baum-Welch algorithm for HMMs:

1. **E-step:** Compute the forward-backward probabilities α_t(k) and β_t(k), then derive the posterior state probabilities γ_t(k) = P(z_t = k | **x**_{1:T}) and pairwise marginals ξ_t(i, j) = P(z_t = i, z_{t+1} = j | **x**_{1:T}).

2. **M-step:** Update parameters using sufficient statistics:

```
π̂_k = γ₁(k)

Â_{ij} = Σ_t ξ_t(i, j) / Σ_t γ_t(i)

μ̂_k = Σ_t γ_t(k) · x_t / Σ_t γ_t(k)

Σ̂_k = Σ_t γ_t(k) · (x_t − μ̂_k)(x_t − μ̂_k)ᵀ / Σ_t γ_t(k)
```

We run EM for up to 200 iterations with convergence tolerance on the log-likelihood. The `RegimeDetector` supports any `hmmlearn` covariance type, but the **default dashboard pipeline uses `covariance_type="diag"`**. With a curated 8-feature subset, a full covariance HMM would carry on the order of 100+ free parameters across three states, so diagonal covariance is used to keep the parameter count manageable and the fit stable. Full covariance (Σ_k ∈ ℝ^{F×F}, as written above) is available but is not the default.

### 2.3 State Decoding: Smoothing vs Filtering

There are two distinct decodes here, and the difference matters more than it first appears.

**Smoothed (Viterbi / forward-backward).** The Viterbi algorithm finds the single most likely *path*:

```
ẑ*_{1:T} = argmax_{z_{1:T}} P(z_{1:T} | x_{1:T}, θ)
```

via dynamic programming in O(*T* × *K*²) time, and the forward-backward algorithm gives the smoothed marginals P(z_t | x_{1:T}). Both condition on the **entire** series, including observations *after* t. That makes them the best retrospective estimate, and it also makes them **unusable as a trading signal**: a volatility burst beginning after *t* can retroactively relabel bar *t*, so a backtest driven by them "knows" about moves that have not happened yet. `RegimeDetector.predict` / `.predict_proba` implement these, and the pipeline keeps the Viterbi path only for visualization (returned as `states_smoothed`).

**Filtered (forward algorithm only).** The causal counterpart conditions only on the past:

```
alpha_t(k) ∝ P(x_t | z_t = k) · Σ_j A_{jk} · alpha_{t-1}(j),     alpha_1(k) ∝ pi_k · P(x_1 | z_1 = k)
ẑ_t = argmax_k P(z_t = k | x_{1:t}, θ)
```

computed in log space with per-step normalization for numerical stability (`RegimeDetector.filtered_proba` / `.predict_filtered`). Because the recursion never looks forward, the estimate at *t* is **prefix-invariant**: appending later data cannot change it. That property is asserted directly in `tests/test_hmm.py::TestCausalDecode`, along with a companion test demonstrating that the Viterbi path is *not* prefix-invariant.

**The pipeline trades the filtered decode.** On overlapping-regime data the two disagree materially (roughly a quarter of held-out bars in the test fixture; under 1% on the well-separated synthetic sample), and the disagreement is systematically flattering to the strategy, since retroactive Toxic labels let it "flatten before the crash." Fitting the model and scaler on the training segment alone (Section 4.1) fixes parameter and scaling leakage but does nothing about this, so the decode had to be made causal separately.

### 2.4 State Ordering

After fitting, states are reordered by the trace of their covariance matrix (a proxy for total volatility), ensuring State 0 always corresponds to the lowest-variance (Quiet) regime and State 2 to the highest-variance (Toxic) regime. This provides consistent labeling across different data subsets and random seeds.

---

## 3. Model Selection

### 3.1 Information Criteria

We evaluate models with *K* ∈ {2, 3, 4, 5} states using both the Bayesian Information Criterion (BIC) and Akaike Information Criterion (AIC):

```
BIC = p · ln(T) − 2 · ℓ(θ̂)
AIC = 2p − 2 · ℓ(θ̂)
```

where ℓ(θ̂) is the maximized log-likelihood, *T* is the number of observations, and *p* is the number of free parameters:

```
p = (K − 1) + K(K − 1) + K·F + K · F(F + 1)/2
     ↑ start     ↑ trans    ↑ means  ↑ covariances (full)
```

BIC penalizes model complexity more heavily than AIC (via the ln(*T*) term vs. the constant 2), making it more conservative for large sample sizes typical in high-frequency data.

### 3.2 Selection Procedure

For each candidate *K*, we fit the HMM, compute BIC and AIC, and select the *K* that minimizes each criterion. The sweep is implemented in `select_model`. The default pipeline uses *K* = 3, chosen for interpretability rather than asserted as provably optimal. On the synthetic and sample data, BIC tends to favor three states, but that is a property of this data and is not claimed as a general result for real markets.

---

## 4. Backtesting Methodology

### 4.1 Walk-Forward Design

The backtest exists to visualize that detected regimes move with returns in a structured way, not as a production trading strategy. The default pipeline now implements the standard walk-forward hygiene:

1. **Train/test split.** The HMM (including its feature scaler) is fit on the first 70% of the series (`train_frac=0.7`), then decodes the full series. Headline backtest statistics come from the held-out 30%; the in-sample segment's statistics are reported alongside for comparison.
2. **Causal standardization.** Because the scaler is fit on the training segment only, no held-out observation is scaled using future statistics (Section 1.5). VPIN's volume-bucket size, also a full-sample statistic, is estimated from the training segment for the same reason.
3. **Causal decoding.** The states the backtest trades on come from the filtered (forward-only) decode, not the smoothed Viterbi path (Section 2.3), so a regime label never reflects an observation that had not happened yet.
4. **Next-bar execution.** The position decided from information at bar *t* is held over bar *t + 1*; a signal never earns the bar it fires on.
5. **Transaction costs.** Each unit of turnover is charged taker fees plus slippage (defaults: 5.5 + 0.5 bps per side). All headline metrics are net of costs; gross variants are reported alongside.

What this still is not: a robust out-of-sample validation. The sample is a single instrument over a short window, fills are modeled naively (a full fill at the decision bar's mid with returns accruing from the next bar, no queue position, no partial fills, no market impact), and one 70/30 split is not a rolling re-estimation. Passing `train_frac=None` reproduces the legacy fully in-sample behavior for comparison. Inputs too small to support a split (fewer than roughly 30 rows) fall back to the in-sample path with a warning rather than failing inside the HMM fit.

### 4.2 Strategy Logic

The signal exploits regime transitions as entry/exit triggers:

- **Entry:** When the model detects a Quiet-to-Trending transition, enter a position in the direction of OFI (long if OFI > 0, short if OFI < 0).
- **Exit (risk):** When the model detects a Toxic regime, flatten all positions immediately.
- **Exit (mean-reversion):** When the model detects a return to Quiet, close the position.

### 4.3 Performance Metrics

| Metric | Formula |
|--------|---------|
| Sharpe ratio | `(mean(pnl) / std(pnl)) × √(bars_per_year)`, on net-of-cost PnL (gross also reported) |
| Max drawdown | `max(1 − equity_t / peak_equity_t)` with `equity = exp(cumsum(pnl))` anchored at 1.0 before the first bar — a fraction of peak equity, so a series that loses from bar one is measured against starting capital |
| Hit rate | Fraction of completed trades with positive net PnL |
| Profit per trade | Mean net PnL across all completed trades |

`bars_per_year` is computed from the actual bar interval of the input series (the pipeline passes `365 × 24 × 3600 × 10⁶ / sample_interval_us` for a 24/7 crypto market), so a 100ms-sampled run is annualized consistently rather than assuming 1-second bars. Annualizing a sub-second Sharpe over a continuous year still extrapolates heavily, so the figure should be treated as a diagnostic, not a performance estimate.

---

## 5. Limitations and Future Work

### 5.1 Known Limitations

- **One split, short sample:** The walk-forward design (Section 4.1) makes the headline metrics out-of-sample, but a single 70/30 split on a single instrument over a short window is not a robust validation. Rolling re-estimation over longer, more varied samples would be needed before believing any number.
- **Naive fills:** The backtest assumes a full fill at the decision bar's mid (returns accrue from the next bar) plus a flat slippage assumption. Queue position, partial fills, and market impact are not modeled.
- **Filtered decode is noisier than the smoothed path:** Causal decoding (Section 2.3) is the correct choice for a signal, but it flips states more readily than Viterbi, which raises turnover and therefore costs. That is a real property of trading on live estimates, not an artifact to tune away.
- **Curated feature subset, diagonal covariance:** Roughly 30 features are computed, but the HMM uses a curated subset of 8, fit with diagonal covariance (Section 2.2). Off-diagonal feature correlations are not modeled in the default pipeline.
- **Coarse VPIN / Kyle's lambda inputs:** Without a trade-level feed, both are computed from snapshot proxies rather than true signed trade flow, so they are noisy estimates.
- **Stationarity assumption:** The Gaussian HMM assumes stationary emission distributions within each regime. In practice, the parameters of each regime may drift over multi-day horizons (e.g., baseline spread levels change with market conditions). Periodic model re-fitting would be needed for production use.

- **Fixed number of states:** While BIC selects *K* = 3, the optimal number of regimes may vary across different market conditions, asset classes, or time horizons. An infinite HMM (Bayesian nonparametric approach) could adaptively determine *K*.

- **Gaussian emissions:** Financial features often exhibit heavy tails and skewness. A Student-*t* HMM or mixture-of-Gaussians emission model could better capture tail behavior within each regime.

- **Trade data approximation:** Without a full order-level feed, cancellation ratio and trade flow aggression are proxied from snapshot data. These proxies introduce measurement noise relative to the true quantities.

- **Single-asset analysis:** This study focuses on BTCUSDT perpetual futures. Cross-asset regime synchronization (e.g., BTC, ETH, SOL entering Toxic simultaneously) would provide richer market structure insights.

- **Stylized transaction costs:** The backtest charges a flat per-side fee plus slippage assumption on every unit of turnover. Real costs vary with spread, order size, and venue tier, so the strategy remains a validation tool for regime detection quality, not a P&L forecast.

### 5.2 Future Directions

- **Online Baum-Welch:** Implement streaming regime detection for real-time dashboard updates via the recursive forward algorithm.
- **Asymmetric transitions:** Allow different transition dynamics for up-moves vs. down-moves, following Salvi (2025).
- **Regime-conditional optimal execution:** Simulate passive (Quiet), aggressive (Trending), and paused (Toxic) execution strategies.
- **C++ Viterbi decoder:** Achieve sub-microsecond per-timestamp decoding for latency-sensitive applications.

---

## References

1. Cont, R., Kukanov, A., Stoikov, S. (2014). "The Price Impact of Order Book Events." *Journal of Financial Econometrics*, 12(1), 47–88.
2. Easley, D., López de Prado, M., O'Hara, M. (2012). "Flow Toxicity and Liquidity in a High Frequency World." *Review of Financial Studies*, 25(5), 1457–1493.
3. Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle." *Econometrica*, 57(2), 357–384.
4. Kyle, A.S. (1985). "Continuous Auctions and Insider Trading." *Econometrica*, 53(6), 1315–1335.
5. Salvi, J. (2025). "Asymmetric Hidden Markov Models for Intraday Alpha." SSRN: 5315733.
