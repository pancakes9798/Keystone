"""Tests for VaR/CVaR/Component VaR metrics and RiskFilter."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jeanclaude.portfolio.risk import (
    var_historical,
    cvar_historical,
    component_var,
    var_decomposition,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_returns():
    """100 rendimenti: 5 a -0.02, 95 a +0.01 — VaR_95 = 0.02 esatto."""
    return pd.Series([-0.02] * 5 + [0.01] * 95)


@pytest.fixture
def portfolio_data():
    """Portafoglio 2 asset, 300 osservazioni, asset A molto più rischioso."""
    rng = np.random.default_rng(42)
    n = 300
    returns = pd.DataFrame({
        "A": rng.standard_normal(n) * 0.03,   # alta volatilità
        "B": rng.standard_normal(n) * 0.005,  # bassa volatilità
    }, index=pd.date_range("2020-01-01", periods=n, freq="B"))
    weights = pd.Series({"A": 0.5, "B": 0.5})
    return weights, returns


# ---------------------------------------------------------------------------
# var_historical
# ---------------------------------------------------------------------------

def test_var_historical_basic(simple_returns):
    """VaR al 95% su serie con 5 osservazioni a -0.02 = 0.02."""
    var = var_historical(simple_returns, confidence=0.95)
    assert abs(var - 0.02) < 1e-10


def test_var_is_nonnegative():
    """VaR è sempre >= 0 anche su rendimenti positivi."""
    all_positive = pd.Series([0.01] * 50)
    assert var_historical(all_positive) >= 0.0


# ---------------------------------------------------------------------------
# cvar_historical
# ---------------------------------------------------------------------------

def test_cvar_geq_var(simple_returns):
    """CVaR >= VaR per definizione."""
    var = var_historical(simple_returns, confidence=0.95)
    cvar = cvar_historical(simple_returns, confidence=0.95)
    assert cvar >= var - 1e-10


def test_cvar_equals_tail_mean(simple_returns):
    """CVaR al 95% su simple_returns = media dei 5 valori a -0.02 = 0.02."""
    cvar = cvar_historical(simple_returns, confidence=0.95)
    assert abs(cvar - 0.02) < 1e-10


# ---------------------------------------------------------------------------
# component_var
# ---------------------------------------------------------------------------

def test_component_var_index(portfolio_data):
    """component_var ha index = nomi asset."""
    weights, returns = portfolio_data
    cv = component_var(weights, returns, confidence=0.95)
    assert list(cv.index) == ["A", "B"]


def test_component_var_sum_equals_portfolio_cvar(portfolio_data):
    """sum(component_var) == CVaR del portafoglio (proprietà matematica)."""
    weights, returns = portfolio_data
    r_p = returns @ weights
    cvar_p = cvar_historical(r_p, confidence=0.95)
    cv = component_var(weights, returns, confidence=0.95)
    assert abs(cv.sum() - cvar_p) < 1e-10


def test_component_var_risky_asset_higher(portfolio_data):
    """Asset A (vol 6x rispetto a B) deve avere component VaR maggiore."""
    weights, returns = portfolio_data
    cv = component_var(weights, returns, confidence=0.95)
    assert cv["A"] > cv["B"]


# ---------------------------------------------------------------------------
# var_decomposition
# ---------------------------------------------------------------------------

def test_var_decomposition_columns(portfolio_data):
    """Output ha colonne: asset, weight, component_var, pct_contribution."""
    weights, returns = portfolio_data
    df = var_decomposition(weights, returns, confidence=0.95)
    assert list(df.columns) == ["asset", "weight", "component_var", "pct_contribution"]


def test_var_decomposition_pct_sums_to_one(portfolio_data):
    """pct_contribution somma a 1.0."""
    weights, returns = portfolio_data
    df = var_decomposition(weights, returns, confidence=0.95)
    assert abs(df["pct_contribution"].sum() - 1.0) < 1e-10


def test_var_decomposition_sorted(portfolio_data):
    """Output ordinato per component_var decrescente."""
    weights, returns = portfolio_data
    df = var_decomposition(weights, returns, confidence=0.95)
    assert df["component_var"].iloc[0] >= df["component_var"].iloc[1]


def test_var_decomposition_pct_sums_to_one_zero_cvar():
    """pct_contribution somma a 1.0 anche quando CVaR ≈ 0 (solo rendimenti positivi)."""
    weights = pd.Series({"A": 0.5, "B": 0.5})
    positive_returns = pd.DataFrame({
        "A": [0.01] * 100,
        "B": [0.005] * 100,
    })
    df = var_decomposition(weights, positive_returns, confidence=0.95)
    assert abs(df["pct_contribution"].sum() - 1.0) < 1e-10


def test_component_var_empty_returns():
    """component_var su returns vuoto restituisce zeros, non eccezione."""
    weights = pd.Series({"A": 0.5, "B": 0.5})
    empty_returns = pd.DataFrame({"A": [], "B": []})
    cv = component_var(weights, empty_returns, confidence=0.95)
    assert (cv == 0.0).all()


# ---------------------------------------------------------------------------
# RiskFilter
# ---------------------------------------------------------------------------

from jeanclaude.portfolio.risk import RiskFilter


@pytest.fixture
def risky_data():
    """Portafoglio 3 asset: A molto rischioso, B/C tranquilli. 252 osservazioni."""
    rng = np.random.default_rng(0)
    n = 252
    returns = pd.DataFrame({
        "A": rng.standard_normal(n) * 0.04 - 0.005,
        "B": rng.standard_normal(n) * 0.005,
        "C": rng.standard_normal(n) * 0.005,
    }, index=pd.date_range("2020-01-01", periods=n, freq="B"))
    weights = pd.Series({"A": 1/3, "B": 1/3, "C": 1/3})
    return weights, returns


def test_risk_filter_passthrough_safe(portfolio_data):
    """Con rendimenti molto piccoli, i pesi non vengono modificati."""
    weights, returns = portfolio_data
    tiny_returns = returns * 0.001
    rf = RiskFilter(cvar_limit=0.10, drawdown_limit=0.50, confidence=0.95)
    filtered = rf.apply(weights, tiny_returns)
    pd.testing.assert_series_equal(filtered, weights)


def test_risk_filter_truncates_worst(risky_data):
    """Con CVaR > limite, il peso dell'asset peggiore (A) viene ridotto."""
    weights, returns = risky_data
    rf = RiskFilter(cvar_limit=0.005, drawdown_limit=0.50, confidence=0.95)
    filtered = rf.apply(weights, returns)
    assert filtered["A"] < weights["A"]


def test_risk_filter_drawdown_scales_all(risky_data):
    """Con drawdown > limite, tutti i pesi vengono scalati (somma < 1)."""
    weights, returns = risky_data
    rf = RiskFilter(cvar_limit=0.50, drawdown_limit=0.001, confidence=0.95)
    filtered = rf.apply(weights, returns)
    assert filtered.sum() < 1.0 - 1e-6
