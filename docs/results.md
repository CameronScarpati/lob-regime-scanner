# Results

This document summarizes what the LOB Regime Scanner produces when run on the
synthetic and free sample data included with the project. Everything here is
qualitative and illustrative. The default pipeline is walk-forward (fit on the
first 70%, headline statistics on the held-out 30%, net of costs), but the
sample is a single instrument over a short window, so nothing here has been
validated at scale on real market data. Read the **Scope and Limitations**
section of the README before reading anything quantitative into these notes.

---

## 1. Model Selection

A BIC/AIC sweep over *K* ∈ {2, 3, 4, 5} states is implemented (`select_model` in
`src/hmm_model.py`). The default pipeline uses **K = 3**, chosen for
interpretability: three states map naturally onto a calm / directional /
stressed reading of the order book, and adding states tends to split existing
regimes rather than reveal new structure. K = 3 is a modeling choice, not a
claim that three is provably optimal on any particular dataset.

The 3-state model typically converges well within the 200-iteration EM cap, and
multiple random restarts land on consistent state decompositions on the sample
data.

---

## 2. Regime Characteristics

After fitting, the three states are sorted by covariance trace (a variance
proxy) and labeled Quiet, Trending, and Toxic. On the synthetic and sample data
they separate along intuitive lines:

- **Quiet** — lowest variance: tighter spreads, balanced order flow, near-zero
  return autocorrelation.
- **Trending** — moderate variance: directional order flow imbalance and
  positive short-horizon return autocorrelation, a momentum signature.
- **Toxic** — highest variance: wider spreads, elevated VPIN, and negative
  return autocorrelation, a mean-reversion signature.

The *direction* of these differences is the point, not any specific magnitude.
The exact values depend entirely on the data fed in, and the synthetic generator
is tuned to produce regime-like behavior, so the separation should be read as
"the model recovers the structure built into the synthetic data," not as a
measured property of real markets.

---

## 3. VPIN and Kyle's Lambda by Regime

VPIN (volume-synchronized probability of informed trading) and Kyle's lambda (a
rolling price-impact estimate) are computed as additional microstructure
features. On the sample data, both tend to run higher in the highest-variance
(Toxic) state than in the Quiet state, which is the direction adverse-selection
theory would predict.

Two honest caveats:

- In the default pipeline, trade-side data is usually absent, so VPIN and Kyle's
  lambda fall back to proxies derived from top-of-book quantities rather than
  true signed trade flow. They are coarse.
- Any statement that VPIN "leads" Toxic transitions by a specific number of
  seconds would be a property of the synthetic data, not a validated empirical
  finding, so no lead time is claimed here.

---

## 4. Regime Transition Dynamics

The learned transition matrix is strongly diagonal: each state tends to persist
rather than flip every step, which is the behavior the HMM's transition
structure is designed to capture, and the main reason an HMM is preferable to a
memoryless volatility threshold. Regime durations vary by state, with the
lowest-variance state generally the most persistent. Specific persistence
probabilities and durations depend on the data and are not reported as fixed
numbers.

---

## 5. Backtest

A deliberately simple regime-conditional rule is included: enter on a
Quiet-to-Trending transition in the order-flow direction, flatten on Toxic
detection. It is a visualization aid, not a strategy.

The backtest runs on causally decoded states (filtered, not smoothed — see
methodology Section 2.3), applies next-bar execution, charges taker fees plus
slippage on every unit of turnover, annualizes from the actual bar interval,
and reports its headline statistics on the held-out walk-forward segment (with
in-sample and gross-of-cost figures alongside). Even so, one 70/30 split on a
short single-instrument sample with naive fill assumptions is not a validation,
so no Sharpe ratio is reproduced here. The purpose of the backtest is to show
that the regime labels move with returns in a structured way, not to claim
alpha.

One qualitative result is worth stating plainly, because it is the most useful
thing the exercise produced: on the sample data the naive regime-flipping rule
looks positive **gross** of costs and turns **negative once fees and slippage
are charged**. A rule that re-enters on every Quiet-to-Trending transition
simply does not clear roughly 6 bps per side. That sign flip, not any Sharpe
number, is the honest takeaway.

---

## 6. HMM vs a Simple Threshold

The HMM is compared against a memoryless volatility-threshold classifier
(`compare_threshold_regimes`). The qualitative advantage of the HMM is
structural: it models the features jointly and builds persistence into the
transition matrix, so its regimes are more stable over time than a threshold
that flips on single-feature noise. This is a design argument, supported by the
comparison utility, rather than a tuned performance number.

---

## Summary

On synthetic and sample order book data, the Gaussian HMM recovers three
interpretable states (Quiet, Trending, Toxic) that differ in volatility, spread,
and return autocorrelation in the directions microstructure intuition would
predict. The project demonstrates the mechanics end to end: feature
construction, HMM fitting and Viterbi decoding, regime-conditional analysis, and
an interactive dashboard.

It is a learning project. The regimes have not been validated at scale on real
market data, the walk-forward backtest covers one short single-instrument
sample with stylized fills and costs, and the numbers it produces should not
be read as evidence of a tradeable signal.
