"""Hidden Markov Model regime detection engine.

Fits a Gaussian HMM with configurable states, decodes regime sequences
via Viterbi, and computes regime-conditional statistics.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy import stats as sp_stats

# Default regime labels
REGIME_LABELS = {0: "Quiet", 1: "Trending", 2: "Toxic"}


@dataclass
class RegimeStats:
    """Container for regime-conditional analysis results."""

    means: dict[int, dict[str, float]] = field(default_factory=dict)
    stds: dict[int, dict[str, float]] = field(default_factory=dict)
    skews: dict[int, dict[str, float]] = field(default_factory=dict)
    kurtoses: dict[int, dict[str, float]] = field(default_factory=dict)
    durations: dict[int, dict[str, float]] = field(default_factory=dict)
    transition_matrix: np.ndarray | None = None


@dataclass
class ModelSelection:
    """BIC/AIC model selection results."""

    n_states: list[int] = field(default_factory=list)
    bics: list[float] = field(default_factory=list)
    aics: list[float] = field(default_factory=list)
    log_likelihoods: list[float] = field(default_factory=list)
    best_n_bic: int = 0
    best_n_aic: int = 0


@dataclass
class Diagnostics:
    """Model diagnostics container."""

    log_likelihood: float = 0.0
    n_iter: int = 0
    converged: bool = False
    monitor_history: list[float] = field(default_factory=list)


class RegimeDetector:
    """Gaussian HMM wrapper for market regime detection.

    Parameters
    ----------
    n_states : int
        Number of hidden states (default 3: Quiet, Trending, Toxic).
    covariance_type : str
        Covariance parameterization ('full', 'diag', 'tied', 'spherical').
    n_iter : int
        Maximum EM iterations.
    random_state : int
        Random seed for reproducibility.
    labels : dict[int, str] | None
        Human-readable regime labels.
    """

    def __init__(
        self,
        n_states: int = 3,
        covariance_type: str = "full",
        n_iter: int = 200,
        random_state: int = 42,
        labels: dict[int, str] | None = None,
    ) -> None:
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self.labels = labels or {i: f"State_{i}" for i in range(n_states)}
        self.model: GaussianHMM | None = None
        self._fitted = False
        self._diagnostics = Diagnostics()

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def diagnostics(self) -> Diagnostics:
        return self._diagnostics

    def _to_array(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Convert input to 2D numpy array."""
        if isinstance(X, pd.DataFrame):
            return X.values.astype(np.float64)
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr

    def fit(self, X: pd.DataFrame | np.ndarray) -> RegimeDetector:
        """Fit the HMM on a feature matrix.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix (e.g., from features.py).

        Returns
        -------
        self
        """
        arr = self._to_array(X)
        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(arr)

        self._fitted = True
        self._diagnostics = Diagnostics(
            log_likelihood=self.model.score(arr),
            n_iter=self.model.monitor_.iter,
            converged=self.model.monitor_.converged,
            monitor_history=list(self.model.monitor_.history),
        )
        self._sort_states_by_volatility(arr)
        return self

    def _sort_states_by_volatility(self, X: np.ndarray) -> None:
        """Relabel states so that State 0 has lowest variance (Quiet)
        and the highest-variance state is last (Toxic)."""
        if self.model is None:
            return
        # Use trace of covariance as a volatility proxy
        # Use internal _covars_ for diag (compact shape n_components x n_features)
        if self.covariance_type == "diag":
            raw_covars = self.model._covars_  # shape (n_components, n_features)
            vol = np.array([np.sum(c) for c in raw_covars])
        elif self.covariance_type == "full":
            raw_covars = self.model.covars_
            vol = np.array([np.trace(c) for c in raw_covars])
        elif self.covariance_type == "spherical":
            raw_covars = self.model.covars_
            vol = np.array(raw_covars)
        else:  # tied
            return  # can't reorder with tied covariance

        order = np.argsort(vol)
        if np.array_equal(order, np.arange(self.n_states)):
            return  # already sorted

        self.model.means_ = self.model.means_[order]
        self.model.startprob_ = self.model.startprob_[order]
        self.model.transmat_ = self.model.transmat_[order][:, order]

        if self.covariance_type == "diag":
            self.model._covars_ = raw_covars[order]
        elif self.covariance_type == "full":
            reordered = raw_covars[order].copy()
            # Enforce symmetry to avoid floating-point validation errors
            for i in range(len(reordered)):
                reordered[i] = (reordered[i] + reordered[i].T) / 2.0
            self.model._covars_ = reordered
        elif self.covariance_type == "spherical":
            self.model.covars_ = raw_covars[order]

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Decode most likely state sequence (Viterbi).

        Returns
        -------
        states : ndarray of shape (n_samples,)
            Integer state labels at each timestep.
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        arr = self._to_array(X)
        return self.model.predict(arr)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Posterior state probabilities at each timestep.

        Returns
        -------
        probs : ndarray of shape (n_samples, n_states)
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        arr = self._to_array(X)
        return self.model.predict_proba(arr)

    def score(self, X: pd.DataFrame | np.ndarray) -> float:
        """Log-likelihood of the data under the fitted model."""
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self.model.score(self._to_array(X))

    def bic(self, X: pd.DataFrame | np.ndarray) -> float:
        """Bayesian Information Criterion."""
        arr = self._to_array(X)
        n_samples, n_features = arr.shape
        n_params = self._count_params(n_features)
        ll = self.model.score(arr) * n_samples  # hmmlearn returns per-sample
        return n_params * np.log(n_samples) - 2 * ll

    def aic(self, X: pd.DataFrame | np.ndarray) -> float:
        """Akaike Information Criterion."""
        arr = self._to_array(X)
        n_samples, n_features = arr.shape
        n_params = self._count_params(n_features)
        ll = self.model.score(arr) * n_samples
        return 2 * n_params - 2 * ll

    def _count_params(self, n_features: int) -> int:
        """Count free parameters in the model."""
        k = self.n_states
        # Start probabilities: k-1
        # Transition matrix: k*(k-1)
        # Means: k * n_features
        n_params = (k - 1) + k * (k - 1) + k * n_features
        # Covariance parameters
        if self.covariance_type == "full":
            n_params += k * n_features * (n_features + 1) // 2
        elif self.covariance_type == "diag":
            n_params += k * n_features
        elif self.covariance_type == "spherical":
            n_params += k
        elif self.covariance_type == "tied":
            n_params += n_features * (n_features + 1) // 2
        return n_params

    def transition_matrix(self) -> np.ndarray:
        """Return the learned transition probability matrix."""
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self.model.transmat_.copy()

    def regime_stats(
        self,
        X: pd.DataFrame | np.ndarray,
        returns: dict[str, np.ndarray] | None = None,
    ) -> RegimeStats:
        """Compute regime-conditional analysis.

        Parameters
        ----------
        X : array-like
            Feature matrix used for regime decoding.
        returns : dict mapping horizon label to return array, optional
            E.g., {"1s": ret_1s, "10s": ret_10s, "60s": ret_60s}.
            Each array should have the same length as X.

        Returns
        -------
        RegimeStats with means, stds, skews, kurtoses, durations,
        and transition_matrix.
        """
        states = self.predict(X)
        result = RegimeStats(transition_matrix=self.transition_matrix())

        # Use returns if provided, else use features for distribution stats
        data = returns or {}
        if not data:
            arr = self._to_array(X)
            data = {f"feature_{i}": arr[:, i] for i in range(arr.shape[1])}

        for state in range(self.n_states):
            mask = states == state
            if mask.sum() == 0:
                continue
            result.means[state] = {k: float(np.mean(v[mask])) for k, v in data.items()}
            result.stds[state] = {k: float(np.std(v[mask])) for k, v in data.items()}
            result.skews[state] = {k: float(sp_stats.skew(v[mask])) for k, v in data.items()}
            result.kurtoses[state] = {k: float(sp_stats.kurtosis(v[mask])) for k, v in data.items()}

        # Duration statistics
        result.durations = _compute_durations(states, self.n_states)
        return result

    def compare_threshold_regimes(
        self,
        X: pd.DataFrame | np.ndarray,
        threshold_states: np.ndarray,
    ) -> dict[str, Any]:
        """Compare HMM regimes against threshold-based regimes.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        threshold_states : ndarray
            Regime labels from a simple threshold method.

        Returns
        -------
        dict with agreement_rate and confusion_matrix.
        """
        hmm_states = self.predict(X)
        agreement = np.mean(hmm_states == threshold_states)

        n_states_max = max(self.n_states, int(threshold_states.max()) + 1)
        confusion = np.zeros((n_states_max, n_states_max), dtype=int)
        for h, t in zip(hmm_states, threshold_states, strict=False):
            confusion[h, t] += 1

        return {
            "agreement_rate": float(agreement),
            "confusion_matrix": confusion,
        }


def select_model(
    X: pd.DataFrame | np.ndarray,
    state_range: range | list[int] | None = None,
    covariance_type: str = "full",
    n_iter: int = 200,
    random_state: int = 42,
) -> ModelSelection:
    """BIC/AIC model selection over a range of state counts.

    Parameters
    ----------
    X : array-like
        Feature matrix.
    state_range : iterable of ints
        Number of states to try (default: 2-5).

    Returns
    -------
    ModelSelection with results and best model count.
    """
    if state_range is None:
        state_range = range(2, 6)

    result = ModelSelection()
    for n in state_range:
        det = RegimeDetector(
            n_states=n,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
        )
        det.fit(X)
        bic_val = det.bic(X)
        aic_val = det.aic(X)
        ll_val = det.score(X)
        result.n_states.append(n)
        result.bics.append(bic_val)
        result.aics.append(aic_val)
        result.log_likelihoods.append(ll_val)

    result.best_n_bic = result.n_states[int(np.argmin(result.bics))]
    result.best_n_aic = result.n_states[int(np.argmin(result.aics))]
    return result


def _compute_durations(states: np.ndarray, n_states: int) -> dict[int, dict[str, float]]:
    """Compute duration statistics per regime.

    Returns dict mapping state -> {mean, std, min, max, count}.
    """
    durations: dict[int, list[int]] = {s: [] for s in range(n_states)}
    if len(states) == 0:
        return {
            s: {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}
            for s in range(n_states)
        }

    current_state = states[0]
    current_len = 1
    for s in states[1:]:
        if s == current_state:
            current_len += 1
        else:
            durations[current_state].append(current_len)
            current_state = s
            current_len = 1
    durations[current_state].append(current_len)

    result = {}
    for s in range(n_states):
        d = durations[s]
        if d:
            result[s] = {
                "mean": float(np.mean(d)),
                "std": float(np.std(d)),
                "min": float(np.min(d)),
                "max": float(np.max(d)),
                "count": len(d),
            }
        else:
            result[s] = {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "count": 0,
            }
    return result
