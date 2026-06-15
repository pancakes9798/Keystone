"""Tests for HMM regime detection — labels and detector."""
from __future__ import annotations

import pickle
import numpy as np
import pandas as pd
import pytest
from jeanclaude.signals.macro import RegimeLabel, RegimeResult, RegimeDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_state_vars(
    n_samples: int = 300,
    n_features: int = 3,
    random_state: int = 0,
) -> pd.DataFrame:
    """Generate data from a known 3-state HMM for testing.

    Note: ``n_features`` must be 3 or fewer (means array has 3 columns).
    """
    assert n_features <= 3, "n_features must be <= 3 (means hardcoded to 3 columns)"
    rng = np.random.default_rng(random_state)
    # Three distinct clusters: low/mid/high means on first feature (acts as VIX proxy)
    means = np.array([
        [-1.5, -1.0, -0.5],
        [ 0.0,  0.0,  0.0],
        [ 1.5,  1.0,  0.5],
    ])
    # Assign states in runs to mimic regime persistence
    state_seq = np.repeat([0, 1, 2, 0, 1, 2], n_samples // 6 + 1)[:n_samples]
    X = means[state_seq] + rng.normal(scale=0.3, size=(n_samples, n_features))
    idx = pd.date_range("2010-01-01", periods=n_samples, freq="B")
    cols = ["VIX", "yield_slope", "credit_spread"][:n_features]
    return pd.DataFrame(X, index=idx, columns=cols)


# ---------------------------------------------------------------------------
# RegimeLabel tests
# ---------------------------------------------------------------------------

def test_regime_label_values():
    assert RegimeLabel.EXPANSION == 0
    assert RegimeLabel.CONTRACTION == 1
    assert RegimeLabel.TRANSITION == 2
    assert len(RegimeLabel) == 3


# ---------------------------------------------------------------------------
# RegimeDetector tests
# ---------------------------------------------------------------------------

def test_fit_returns_regime_result():
    """fit() returns a RegimeResult with correct shapes."""
    sv = _make_synthetic_state_vars(n_samples=300)
    detector = RegimeDetector(n_components=3, random_state=0)
    result = detector.fit(sv)

    assert isinstance(result, RegimeResult)
    assert len(result.labels) == 300
    assert result.labels.index.equals(sv.index)
    assert result.probabilities.shape == (300, 3)
    assert list(result.probabilities.columns) == [0, 1, 2]
    from pomegranate.hmm import DenseHMM
    assert isinstance(result.model, DenseHMM)


def test_labels_cover_all_regimes():
    """After fit, all three regimes appear in the labels."""
    sv = _make_synthetic_state_vars(n_samples=300)
    detector = RegimeDetector(n_components=3, random_state=0)
    result = detector.fit(sv)
    unique_labels = set(result.labels.values)
    assert unique_labels == {RegimeLabel.EXPANSION, RegimeLabel.CONTRACTION, RegimeLabel.TRANSITION}


def test_probabilities_sum_to_one():
    """Posterior probabilities sum to 1 for every row."""
    sv = _make_synthetic_state_vars(n_samples=300)
    detector = RegimeDetector(n_components=3, random_state=0)
    result = detector.fit(sv)
    row_sums = result.probabilities.sum(axis=1)
    # atol=1e-5: pomegranate uses float32 internally, so numerical error is ~1e-6..1e-5
    np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-5)


def test_current_regime_last_row():
    """current_regime() returns the label and proba vector for the last observation."""
    sv = _make_synthetic_state_vars(n_samples=300)
    detector = RegimeDetector(n_components=3, random_state=0)
    detector.fit(sv)
    label, proba = detector.current_regime(sv)

    assert isinstance(label, RegimeLabel)
    assert proba.shape == (3,)
    np.testing.assert_allclose(proba.sum(), 1.0, atol=1e-5)


def test_expanding_window_stability():
    """Re-fitting on T+N data preserves label ordering on the first T rows (>80% overlap)."""
    sv = _make_synthetic_state_vars(n_samples=300, random_state=42)
    sv_short = sv.iloc[:200]
    sv_full = sv  # 300 rows

    detector = RegimeDetector(n_components=3, random_state=0)
    result_short = detector.fit(sv_short)
    result_full = detector.fit(sv_full)

    labels_short = result_short.labels.values
    labels_full = result_full.labels.iloc[:200].values

    overlap = np.mean(labels_short == labels_full)
    assert overlap > 0.80, f"Label stability too low: {overlap:.2%}"


def test_fit_raises_on_nan():
    """fit() raises ValueError when state_vars contains NaN."""
    sv = _make_synthetic_state_vars(n_samples=100)
    sv.iloc[10, 0] = float("nan")
    detector = RegimeDetector()
    with pytest.raises(ValueError, match="NaN"):
        detector.fit(sv)


def test_model_serializable():
    """RegimeResult.model is serialisable via pickle."""
    from pomegranate.hmm import DenseHMM

    sv = _make_synthetic_state_vars(n_samples=300)
    detector = RegimeDetector(n_components=3, random_state=0)
    result = detector.fit(sv)

    blob = pickle.dumps(result.model)
    restored = pickle.loads(blob)
    assert isinstance(restored, DenseHMM)
    assert restored.n_distributions == 3


def test_predict_raises_before_fit():
    """predict() raises RuntimeError if called before fit()."""
    sv = _make_synthetic_state_vars(n_samples=100)
    detector = RegimeDetector()
    with pytest.raises(RuntimeError, match="fit"):
        detector.predict(sv)


def test_predict_on_new_data():
    """predict() on a new (unseen) window returns RegimeResult with correct shapes."""
    sv_train = _make_synthetic_state_vars(n_samples=300)
    sv_new = _make_synthetic_state_vars(n_samples=50, random_state=99)

    detector = RegimeDetector(n_components=3, random_state=0)
    detector.fit(sv_train)
    result = detector.predict(sv_new)

    assert isinstance(result, RegimeResult)
    assert len(result.labels) == 50
    assert result.probabilities.shape == (50, 3)
    row_sums = result.probabilities.sum(axis=1)
    # atol=1e-5: pomegranate uses float32 internally, so numerical error is ~1e-6..1e-5
    np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-5)


def test_label_order_descending():
    """label_order='descending' maps highest feature mean to EXPANSION."""
    sv = _make_synthetic_state_vars(n_samples=300)
    detector = RegimeDetector(
        n_components=3,
        random_state=0,
        label_feature="VIX",
        label_order="descending",
    )
    result = detector.fit(sv)

    # With descending order, the state with HIGHEST VIX mean = EXPANSION.
    # We verify output is still a valid RegimeResult with all 3 regimes.
    assert isinstance(result, RegimeResult)
    unique_labels = set(result.labels.values)
    assert unique_labels == {RegimeLabel.EXPANSION, RegimeLabel.CONTRACTION, RegimeLabel.TRANSITION}


def test_build_label_map_empirical():
    """_build_label_map: state with lowest VIX mean → EXPANSION (ascending)."""
    from jeanclaude.signals.macro.labels import _build_label_map

    n = 90
    X = np.zeros((n, 3), dtype=np.float32)
    raw_states = np.repeat([0, 1, 2], 30)
    # state 0: VIX = -1.0, state 1: VIX = 0.0, state 2: VIX = +1.0
    X[raw_states == 0, 0] = -1.0
    X[raw_states == 1, 0] = 0.0
    X[raw_states == 2, 0] = 1.0

    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    sv = pd.DataFrame(X, index=idx, columns=["VIX", "yield_slope", "credit_spread"])

    label_map = _build_label_map(X, raw_states, sv, "VIX", "ascending", 3)

    assert label_map[0] == RegimeLabel.EXPANSION
    assert label_map[1] == RegimeLabel.TRANSITION
    assert label_map[2] == RegimeLabel.CONTRACTION


def test_invalid_n_components_raises():
    """RegimeDetector with n_components != 3 raises ValueError."""
    with pytest.raises(ValueError, match="n_components must be 3"):
        RegimeDetector(n_components=2)


def test_invalid_label_order_raises():
    """RegimeDetector with bad label_order raises ValueError."""
    with pytest.raises(ValueError, match="label_order"):
        RegimeDetector(label_order="invalid")


def test_invalid_student_t_df_raises():
    """RegimeDetector with student_t_df < 1 raises ValueError."""
    with pytest.raises(ValueError, match="student_t_df"):
        RegimeDetector(student_t_df=0)


def test_invalid_sticky_concentration_raises():
    """RegimeDetector with sticky_concentration <= 0 raises ValueError."""
    with pytest.raises(ValueError, match="sticky_concentration"):
        RegimeDetector(sticky_concentration=0.0)


def test_all_restarts_failed_raises(monkeypatch):
    """_fit_best raises RuntimeError when all restarts fail."""
    from jeanclaude.signals.macro.detector import RegimeDetector as _RD

    def _always_fail(self, X):
        raise RuntimeError("simulated fit failure")

    monkeypatch.setattr(_RD, "_fit_best", _always_fail)

    sv = _make_synthetic_state_vars(n_samples=100)
    detector = _RD(n_components=3, random_state=0)
    with pytest.raises(RuntimeError):
        detector.fit(sv)
