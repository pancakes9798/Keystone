"""Tests for jeanclaude.signals.formulaic.fields — synthetic only."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from jeanclaude.signals.formulaic import fields


def _rets(n, cols, vals):
    idx = pd.bdate_range("2010-01-01", periods=n)
    return pd.DataFrame({c: [v] * n for c, v in zip(cols, vals)}, index=idx)


def test_log_price_reconstructs_cumulative_return():
    r = _rets(10, ["A"], [0.01])
    lp = fields.log_price(r)
    assert lp["A"].iloc[-1] == pytest.approx(np.log(1.01) * 10)


def test_cumret_compound_over_window():
    r = _rets(300, ["A"], [0.001])
    c = fields.cumret(r, asof=r.index[-1], months_back=12, months_skip=1)
    # finestra = ultimi 252g escludendo gli ultimi 21 = 231 giorni a 0.1%/g
    expected = (1.001) ** (252 - 21) - 1.0
    assert c["A"] == pytest.approx(expected, rel=1e-9)


def test_cumret_insufficient_history_is_nan():
    r = _rets(100, ["A"], [0.001])  # < 12*21 = 252 righe
    c = fields.cumret(r, asof=r.index[-1], months_back=12, months_skip=1)
    assert np.isnan(c["A"])


def test_cumret_is_pit_safe():
    r = _rets(300, ["A"], [0.001])
    asof = r.index[260]
    c_a = fields.cumret(r, asof=asof, months_back=12, months_skip=1)
    r_mut = r.copy()
    r_mut.iloc[261:] = 99.0  # garbage nel futuro
    c_b = fields.cumret(r_mut, asof=asof, months_back=12, months_skip=1)
    assert c_a["A"] == pytest.approx(c_b["A"], rel=1e-12)


def test_market_return_is_cross_sectional_mean():
    r = _rets(5, ["A", "B"], [0.02, -0.01])
    mkt = fields.market_return(r)
    assert mkt.iloc[-1] == pytest.approx(0.005)


def test_resid_return_removes_market_beta():
    idx = pd.bdate_range("2010-01-01", periods=200)
    rng = np.random.default_rng(0)
    m = rng.normal(0, 0.01, 200)
    r = pd.DataFrame({"A": 2 * m, "B": m}, index=idx)
    resid = fields.resid_return(r, window=60)
    assert abs(resid["A"].dropna().iloc[-1]) < 1e-6
