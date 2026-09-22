"""Golden-value checks for the information criteria used in model selection.

hmmlearn's ``score`` returns the total log-likelihood of the sequence, so BIC and
AIC must use it directly. These tests pin the criteria to hand computations from
the fitted model's own log-likelihood and parameter count. The detector scales
inputs with its fitted scaler before scoring, so the hand computations score the
same scaled array.
"""

import numpy as np

from src.hmm_model import RegimeDetector


def _three_regime_data(n_per_regime: int = 150, n_features: int = 2, seed: int = 7) -> np.ndarray:
    rng = np.random.RandomState(seed)
    blocks = [
        rng.normal(0.0, 0.3, (n_per_regime, n_features)),
        rng.normal(2.0, 1.0, (n_per_regime, n_features)),
        rng.normal(5.0, 3.0, (n_per_regime, n_features)),
    ]
    return np.vstack(blocks)


def _fit(X: np.ndarray, n_states: int, covariance_type: str) -> RegimeDetector:
    det = RegimeDetector(
        n_states=n_states, covariance_type=covariance_type, n_iter=100, random_state=0
    )
    det.fit(X)
    return det


class TestInformationCriteria:
    def test_score_is_total_log_likelihood(self):
        X = _three_regime_data()
        det = _fit(X, 3, "diag")
        arr = det._to_array(X)
        total, _ = det.model.score_samples(arr)
        assert np.isclose(det.score(X), total)

    def test_bic_matches_hand_computation(self):
        X = _three_regime_data()
        n_samples, n_features = X.shape
        for cov in ("diag", "full"):
            det = _fit(X, 3, cov)
            ll = det.model.score(det._to_array(X))
            n_params = det._count_params(n_features)
            expected = n_params * np.log(n_samples) - 2.0 * ll
            assert np.isclose(det.bic(X), expected)

    def test_aic_matches_hand_computation(self):
        X = _three_regime_data()
        _, n_features = X.shape
        det = _fit(X, 3, "diag")
        ll = det.model.score(det._to_array(X))
        n_params = det._count_params(n_features)
        assert np.isclose(det.aic(X), 2.0 * n_params - 2.0 * ll)

    def test_parameter_count_diag_three_states_two_features(self):
        # (k-1) start + k(k-1) transitions + k*F means + k*F diagonal variances
        det = RegimeDetector(n_states=3, covariance_type="diag")
        assert det._count_params(2) == 2 + 6 + 6 + 6

    def test_criteria_scale_with_sample_size_not_squared(self):
        """Doubling the data roughly doubles the likelihood term; it must not quadruple it."""
        X = _three_regime_data(n_per_regime=100)
        X2 = np.vstack([X, X])
        det = _fit(X, 3, "diag")
        ll_ratio = det.model.score(det._to_array(X2)) / det.model.score(det._to_array(X))
        assert 1.5 < ll_ratio < 2.5
        bic_small = det.bic(X)
        bic_large = det.bic(X2)
        # With the criterion built from the total log-likelihood, the -2*ll term
        # roughly doubles; a per-sample scaling bug would make it roughly quadruple.
        assert bic_large / bic_small < 3.0
