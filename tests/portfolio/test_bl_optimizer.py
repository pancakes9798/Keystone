# tests/portfolio/test_bl_optimizer.py
"""Tests for BLOptimizer — Black-Litterman portfolio optimizer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jeanclaude.portfolio.optimizer.bl import BLOptimizer
from jeanclaude.signals.composite.types import CompositeSignal
from jeanclaude.signals.macro.labels import RegimeLabel

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TICKERS = ["SPY", "QQQ", "TLT", "GLD", "SHY"]
ASSET_CLASS_MAP = {
    "SPY": "equity",
    "QQQ": "equity",
    "TLT": "bonds",
    "GLD": "gold",
    "SHY": "cash",
}

_rng = np.random.default_rng(42)

N_EXP = 60
N_CONT = 20
N_TRANS = 20
T = N_EXP + N_CONT + N_TRANS
_DATES = pd.date_range("2020-01-01", periods=T, freq="B")


def _make_returns_and_labels() -> tuple[pd.DataFrame, pd.Series]:
    """Rendimenti sintetici con struttura di regime coerente.

    Expansion: equity positivo, bonds negativo.
    Contraction: equity negativo, bonds positivo.
    Transition: neutro per tutti.
    """
    r_exp = _rng.normal(
        [0.001, 0.001, -0.001, 0.0, 0.0], 0.005, (N_EXP, len(TICKERS))
    )
    r_cont = _rng.normal(
        [-0.001, -0.001, 0.001, 0.001, 0.0], 0.005, (N_CONT, len(TICKERS))
    )
    r_trans = _rng.normal(
        [0.0, 0.0, 0.0, 0.0, 0.0], 0.005, (N_TRANS, len(TICKERS))
    )

    data = np.vstack([r_exp, r_cont, r_trans])
    returns = pd.DataFrame(data, columns=TICKERS, index=_DATES)

    labels_arr = (
        [RegimeLabel.EXPANSION] * N_EXP
        + [RegimeLabel.CONTRACTION] * N_CONT
        + [RegimeLabel.TRANSITION] * N_TRANS
    )
    regime_labels = pd.Series(labels_arr, index=_DATES)

    return returns, regime_labels


@pytest.fixture
def returns_and_labels():
    return _make_returns_and_labels()


@pytest.fixture
def returns(returns_and_labels):
    return returns_and_labels[0]


@pytest.fixture
def regime_labels(returns_and_labels):
    return returns_and_labels[1]


@pytest.fixture
def optimizer():
    return BLOptimizer(asset_class_map=ASSET_CLASS_MAP, min_obs=5)


@pytest.fixture
def optimizer_fitted(optimizer, returns, regime_labels):
    optimizer.fit(returns, regime_labels)
    return optimizer


@pytest.fixture
def prior_weights():
    return pd.Series(1.0 / len(TICKERS), index=TICKERS)


def _make_signal(
    confidence: float,
    regime_proba: np.ndarray | None = None,
) -> CompositeSignal:
    if regime_proba is None:
        regime_proba = np.array([0.8, 0.1, 0.1])
    return CompositeSignal(
        scores=pd.Series({"equity": 0.8, "bonds": -0.3, "gold": 0.1, "cash": -0.2}),
        confidence=confidence,
        regime=RegimeLabel.EXPANSION,
        regime_proba=regime_proba,
        timestamp=pd.Timestamp("2023-01-01"),
    )


# ---------------------------------------------------------------------------
# fit() tests
# ---------------------------------------------------------------------------


def test_fit_stores_mu_regime(optimizer, returns, regime_labels):
    """_mu_regime contiene un'entry pd.Series per ogni RegimeLabel dopo fit()."""
    optimizer.fit(returns, regime_labels)
    assert optimizer._mu_regime is not None
    for label in RegimeLabel:
        assert label in optimizer._mu_regime
        ser = optimizer._mu_regime[label]
        assert isinstance(ser, pd.Series)
        assert set(TICKERS).issubset(ser.index)


def test_fit_returns_self(optimizer, returns, regime_labels):
    """fit() restituisce self per il chaining."""
    result = optimizer.fit(returns, regime_labels)
    assert result is optimizer


def test_fit_raises_insufficient_obs(returns, regime_labels):
    """Regime con < min_obs osservazioni → ValueError."""
    strict = BLOptimizer(asset_class_map=ASSET_CLASS_MAP, min_obs=9999)
    with pytest.raises(ValueError, match="min_obs"):
        strict.fit(returns, regime_labels)


def test_fit_raises_missing_ticker(returns, regime_labels):
    """Ticker in asset_class_map non presente in returns.columns → ValueError."""
    bad_map = {**ASSET_CLASS_MAP, "MISSING_TICKER": "equity"}
    bad_opt = BLOptimizer(asset_class_map=bad_map, min_obs=5)
    with pytest.raises(ValueError, match="MISSING_TICKER"):
        bad_opt.fit(returns, regime_labels)


# ---------------------------------------------------------------------------
# optimize() tests
# ---------------------------------------------------------------------------


def test_optimize_raises_before_fit(optimizer, returns, prior_weights):
    """optimize() senza fit() → RuntimeError."""
    signal = _make_signal(0.5)
    with pytest.raises(RuntimeError, match="fit"):
        optimizer.optimize(signal, prior_weights, returns)


def test_optimize_returns_series(optimizer_fitted, returns, prior_weights):
    """Output è pd.Series, index=ticker, sum≈1."""
    signal = _make_signal(0.5)
    w = optimizer_fitted.optimize(signal, prior_weights, returns)
    assert isinstance(w, pd.Series)
    assert set(w.index) == set(TICKERS)
    assert abs(w.sum() - 1.0) < 1e-4


def test_optimize_long_only(optimizer_fitted, returns, prior_weights):
    """Nessun peso negativo (vincolo w >= 0)."""
    signal = _make_signal(0.5)
    w = optimizer_fitted.optimize(signal, prior_weights, returns)
    assert (w >= -1e-6).all(), f"Pesi negativi trovati:\n{w}"


def test_optimize_high_confidence_closer_to_views(optimizer_fitted, returns, prior_weights):
    """Con regime_proba favorevole a equity (expansion), alta confidence → più equity.

    In expansion, μ_regime[EXPANSION][equity] > μ_regime[EXPANSION][bonds],
    quindi Q[equity] > Q[bonds]. Alta confidence → views dominano → più equity.
    """
    regime_proba = np.array([0.95, 0.025, 0.025])  # expansion dominante
    signal_high = _make_signal(0.99, regime_proba=regime_proba)
    signal_low = _make_signal(0.01, regime_proba=regime_proba)

    w_high = optimizer_fitted.optimize(signal_high, prior_weights, returns)
    w_low = optimizer_fitted.optimize(signal_low, prior_weights, returns)

    equity_tickers = [t for t, c in ASSET_CLASS_MAP.items() if c == "equity"]
    equity_high = w_high[equity_tickers].sum()
    equity_low = w_low[equity_tickers].sum()
    assert equity_high >= equity_low - 0.02, (
        f"Alta confidence dovrebbe spostare i pesi verso equity. "
        f"equity_high={equity_high:.4f}, equity_low={equity_low:.4f}"
    )


def test_optimize_zero_confidence_ignores_views(optimizer_fitted, returns, prior_weights):
    """Con confidence≈0, Omega→∞, le views vengono ignorate → μ̂≈π.

    Due segnali con regime_proba opposti ma confidence≈0 devono produrre
    pesi quasi identici, poiché entrambi degenerano verso il prior di equilibrio.
    """
    signal_exp = _make_signal(1e-7, regime_proba=np.array([0.98, 0.01, 0.01]))
    signal_cont = _make_signal(1e-7, regime_proba=np.array([0.01, 0.98, 0.01]))

    w_exp = optimizer_fitted.optimize(signal_exp, prior_weights, returns)
    w_cont = optimizer_fitted.optimize(signal_cont, prior_weights, returns)

    max_diff = np.abs(w_exp.values - w_cont.values).max()
    assert max_diff < 0.10, (
        f"Con confidence≈0, i pesi dovrebbero convergere indipendentemente da regime_proba. "
        f"max_diff={max_diff:.4f}"
    )


# ---------------------------------------------------------------------------
# P matrix tests
# ---------------------------------------------------------------------------


def test_p_matrix_rows_sum_to_one(optimizer_fitted):
    """Ogni riga di P somma a 1.0 (equal-weighted dentro ogni asset class)."""
    P, classes = optimizer_fitted._build_p_matrix()
    for i, c in enumerate(classes):
        row_sum = P[i].sum()
        assert abs(row_sum - 1.0) < 1e-10, (
            f"Riga {i} (classe '{c}') somma a {row_sum}, atteso 1.0"
        )


def test_p_matrix_shape(optimizer_fitted):
    """P ha forma (k, N) con k = num asset class, N = num ticker."""
    P, classes = optimizer_fitted._build_p_matrix()
    k = len(classes)
    N = len(TICKERS)
    assert P.shape == (k, N)


# ---------------------------------------------------------------------------
# Q vector tests
# ---------------------------------------------------------------------------


def test_q_vector_uses_full_proba(optimizer_fitted):
    """Q calcolato con regime_proba intero; segnali con proba opposti producono Q diversi."""
    _, classes = optimizer_fitted._build_p_matrix()

    signal_exp = _make_signal(0.5, regime_proba=np.array([0.9, 0.05, 0.05]))
    signal_cont = _make_signal(0.5, regime_proba=np.array([0.05, 0.9, 0.05]))

    Q_exp = optimizer_fitted._build_q_vector(signal_exp, classes)
    Q_cont = optimizer_fitted._build_q_vector(signal_cont, classes)

    assert not np.allclose(Q_exp, Q_cont), (
        "Q deve differire tra segnali con regime_proba opposti"
    )
