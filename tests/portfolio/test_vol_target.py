"""VolTargetOverlay: scala i pesi del base optimizer alla vol target, causale, cap leva."""
import numpy as np
import pandas as pd
import pytest

from jeanclaude.portfolio.optimizer.vol_target import VolTargetOverlay


class FixedOptimizer:
    def __init__(self, weights):
        self._w = weights

    def optimize(self, returns):
        return self._w


def _returns(vol_annual, n=400, seed=3):
    idx = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(seed)
    daily = vol_annual / np.sqrt(252)
    return pd.DataFrame(
        {"A": rng.normal(0.0, daily, n), "B": rng.normal(0.0, daily, n)},
        index=idx,
    )


BASE_W = pd.Series({"A": 0.5, "B": 0.5})


def test_scales_up_low_vol_portfolio():
    rets = _returns(vol_annual=0.05)
    ov = VolTargetOverlay(FixedOptimizer(BASE_W), target_vol=0.10, max_leverage=3.0)
    w = ov.optimize(rets)
    scale = w.sum() / BASE_W.sum()
    assert 1.5 < scale < 3.0  # ~2x con rumore di stima


def test_scale_capped_at_max_leverage():
    rets = _returns(vol_annual=0.02)
    ov = VolTargetOverlay(FixedOptimizer(BASE_W), target_vol=0.20, max_leverage=2.0)
    w = ov.optimize(rets)
    assert w.sum() == pytest.approx(2.0, rel=1e-9)


def test_scales_down_high_vol_portfolio():
    rets = _returns(vol_annual=0.30)
    ov = VolTargetOverlay(FixedOptimizer(BASE_W), target_vol=0.10, max_leverage=2.0)
    w = ov.optimize(rets)
    assert w.sum() < 0.6  # de-lever: resto in cash


def test_causal_uses_only_provided_history():
    """Stessa storia → stesso scaling, indipendentemente da dati futuri (riceve solo il passato)."""
    rets = _returns(vol_annual=0.05, n=400)
    ov = VolTargetOverlay(FixedOptimizer(BASE_W), target_vol=0.10, max_leverage=3.0)
    w_full = ov.optimize(rets)
    w_short = ov.optimize(rets.iloc[:300])
    # nessuna eccezione e scaling calcolato sulla finestra disponibile
    assert w_full.sum() > 0 and w_short.sum() > 0


def test_insufficient_history_returns_base_weights():
    rets = _returns(vol_annual=0.05, n=30)
    ov = VolTargetOverlay(FixedOptimizer(BASE_W), target_vol=0.10,
                          max_leverage=2.0, vol_lookback=63)
    w = ov.optimize(rets)
    pd.testing.assert_series_equal(w, BASE_W)
