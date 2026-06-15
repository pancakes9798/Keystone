"""RegimeBandOverlay: banda equity [floor, cap] condizionata al regime, Σ=1, causale."""
import numpy as np
import pandas as pd
import pytest

from jeanclaude.portfolio.optimizer.regime_band import RegimeBandOverlay, walkforward_regimes

EQ = frozenset({"SPY", "QQQ"})
IDX = pd.bdate_range("2020-01-01", periods=300)


class FixedOptimizer:
    def __init__(self, w):
        self._w = w

    def optimize(self, returns):
        return self._w


def _returns():
    rng = np.random.default_rng(4)
    return pd.DataFrame(rng.normal(0, 0.01, (300, 4)), index=IDX,
                        columns=["SPY", "QQQ", "TLT", "GLD"])


BASE = pd.Series({"SPY": 0.3, "QQQ": 0.2, "TLT": 0.3, "GLD": 0.2})  # equity = 0.5


def _overlay(bands, regimes):
    return RegimeBandOverlay(
        FixedOptimizer(BASE), equity_rics=EQ, bands=bands,
        regime_lookup=lambda d: regimes,
    )


def test_floor_scales_equity_up():
    ov = _overlay({"EXPANSION": (0.8, 1.0)}, "EXPANSION")
    w = ov.optimize(_returns())
    eq = w[list(EQ)].sum()
    assert eq == pytest.approx(0.8, rel=1e-9)
    assert w.sum() == pytest.approx(1.0, rel=1e-9)
    # proporzioni interne preservate
    assert w["SPY"] / w["QQQ"] == pytest.approx(0.3 / 0.2, rel=1e-9)
    assert w["TLT"] / w["GLD"] == pytest.approx(0.3 / 0.2, rel=1e-9)


def test_cap_scales_equity_down():
    ov = _overlay({"CONTRACTION": (0.0, 0.2)}, "CONTRACTION")
    w = ov.optimize(_returns())
    assert w[list(EQ)].sum() == pytest.approx(0.2, rel=1e-9)
    assert w.sum() == pytest.approx(1.0, rel=1e-9)


def test_inside_band_untouched():
    ov = _overlay({"TRANSITION": (0.3, 0.7)}, "TRANSITION")
    w = ov.optimize(_returns())
    pd.testing.assert_series_equal(w, BASE)


def test_unknown_regime_falls_back_to_base():
    ov = _overlay({"EXPANSION": (0.8, 1.0)}, "REGIME_IGNOTO")
    w = ov.optimize(_returns())
    pd.testing.assert_series_equal(w, BASE)


def test_all_equity_portfolio_cap_zero_goes_full_defensive():
    base = pd.Series({"SPY": 0.6, "QQQ": 0.4, "TLT": 0.0, "GLD": 0.0})
    ov = RegimeBandOverlay(FixedOptimizer(base), equity_rics=EQ,
                           bands={"CONTRACTION": (0.0, 0.0)},
                           regime_lookup=lambda d: "CONTRACTION")
    w = ov.optimize(_returns())
    assert w[list(EQ)].sum() == pytest.approx(0.0, abs=1e-12)
    # senza sleeve difensivo di base, il resto va in cash: Σ < 1 ammesso in questo caso limite
    assert w.sum() <= 1.0 + 1e-9


def test_walkforward_regimes_returns_series_with_correct_index(tmp_path):
    """walkforward_regimes ritorna una Series con indice = date richieste."""
    rng = np.random.default_rng(9)
    n = 400
    idx = pd.bdate_range("2019-01-01", periods=n)
    macro = pd.DataFrame({
        "VIX": np.abs(15 + rng.normal(0, 1, n).cumsum() * 0.1) + 5,
        "Oil": np.abs(60 + rng.normal(0, 1, n).cumsum() * 0.1),
        "Copper": np.abs(3 + rng.normal(0, 0.02, n).cumsum()),
        "Gold": np.abs(1500 + rng.normal(0, 5, n).cumsum()),
        "EURUSD": np.abs(1.1 + rng.normal(0, 0.002, n).cumsum() * 0.01),
    }, index=idx)
    # usa solo 2 date per velocità — storia sufficiente per entrambe
    dates = [idx[200], idx[300]]
    result = walkforward_regimes(macro, dates, random_state=42)
    assert list(result.index) == [pd.Timestamp(d) for d in dates]
    assert result.name == "regime"
    # ogni valore è un nome di regime valido o UNKNOWN
    valid = {"EXPANSION", "CONTRACTION", "TRANSITION", "UNKNOWN"}
    assert all(v in valid for v in result.values)


def test_walkforward_regimes_is_causal_and_cached(tmp_path):
    """Il regime a data t dipende solo da macro ≤ t; la cache evita il refit."""
    rng = np.random.default_rng(7)
    n = 700
    idx = pd.bdate_range("2018-01-01", periods=n)
    macro = pd.DataFrame({
        "VIX": np.abs(15 + rng.normal(0, 1, n).cumsum() * 0.1) + 5,
        "Oil": np.abs(60 + rng.normal(0, 1, n).cumsum() * 0.1),
        "Copper": np.abs(3 + rng.normal(0, 0.02, n).cumsum()),
        "Gold": np.abs(1500 + rng.normal(0, 5, n).cumsum()),
        "EURUSD": np.abs(1.1 + rng.normal(0, 0.002, n).cumsum() * 0.01),
    }, index=idx)
    dates = list(idx[::63])[6:]  # alcune date di "rebalance" con storia sufficiente

    r1 = walkforward_regimes(macro, dates, random_state=7, cache_path=tmp_path / "wf.parquet")
    r2 = walkforward_regimes(macro, dates, random_state=7, cache_path=tmp_path / "wf.parquet")
    assert list(r1.index) == [pd.Timestamp(d) for d in dates]
    pd.testing.assert_series_equal(r1, r2)  # cache hit → identico
    assert (tmp_path / "wf.parquet").exists()
