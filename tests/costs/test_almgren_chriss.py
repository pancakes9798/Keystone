from __future__ import annotations

import pytest
import pandas as pd
import numpy as np

from jeanclaude.costs.model import AlmgrenChrissModel
from jeanclaude.costs.types import CostConfig, TradeImpact


# ── Reference inputs ──────────────────────────────────────────────────────
TICKERS = ["A", "B"]
W_CURRENT = pd.Series({"A": 0.30, "B": 0.30})
W_TARGET  = pd.Series({"A": 0.50, "B": 0.20})
ADV       = pd.Series({"A": 1_000_000.0, "B": 500_000.0})
SPREAD    = pd.Series({"A": 0.001,       "B": 0.002})
VOL       = pd.Series({"A": 0.01,        "B": 0.015})
AUM       = 100_000.0

EXPECTED_TOTAL_EUR = 20.714
EXPECTED_TOTAL_PCT = 0.00020714
EXPECTED_TURNOVER  = 0.15
EXPECTED_A_BPS     = (10.408 / AUM) * 10_000   # ≈ 1.0408
EXPECTED_B_BPS     = (10.306 / AUM) * 10_000   # ≈ 1.0306


def test_estimate_returns_trade_impact():
    model = AlmgrenChrissModel()
    impact = model.estimate(W_CURRENT, W_TARGET, ADV, SPREAD, VOL, AUM)
    assert isinstance(impact, TradeImpact)


def test_estimate_total_cost_eur():
    model = AlmgrenChrissModel()
    impact = model.estimate(W_CURRENT, W_TARGET, ADV, SPREAD, VOL, AUM)
    assert impact.total_cost_eur == pytest.approx(EXPECTED_TOTAL_EUR, rel=1e-3)


def test_estimate_total_cost_pct():
    model = AlmgrenChrissModel()
    impact = model.estimate(W_CURRENT, W_TARGET, ADV, SPREAD, VOL, AUM)
    assert impact.total_cost_pct == pytest.approx(EXPECTED_TOTAL_PCT, rel=1e-3)


def test_estimate_turnover():
    model = AlmgrenChrissModel()
    impact = model.estimate(W_CURRENT, W_TARGET, ADV, SPREAD, VOL, AUM)
    assert impact.turnover_pct == pytest.approx(EXPECTED_TURNOVER, rel=1e-6)


def test_estimate_per_asset_bps():
    model = AlmgrenChrissModel()
    impact = model.estimate(W_CURRENT, W_TARGET, ADV, SPREAD, VOL, AUM)
    assert impact.per_asset_bps["A"] == pytest.approx(EXPECTED_A_BPS, rel=1e-3)
    assert impact.per_asset_bps["B"] == pytest.approx(EXPECTED_B_BPS, rel=1e-3)


def test_zero_trade_zero_cost():
    """When weights don't change, cost must be zero."""
    model = AlmgrenChrissModel()
    impact = model.estimate(W_CURRENT, W_CURRENT, ADV, SPREAD, VOL, AUM)
    assert impact.total_cost_eur == pytest.approx(0.0, abs=1e-10)
    assert impact.turnover_pct == pytest.approx(0.0, abs=1e-10)


def test_cost_increases_with_trade_size():
    """Doubling AUM on same trade should increase cost."""
    model = AlmgrenChrissModel()
    impact_small = model.estimate(W_CURRENT, W_TARGET, ADV, SPREAD, VOL, 50_000.0)
    impact_large = model.estimate(W_CURRENT, W_TARGET, ADV, SPREAD, VOL, 200_000.0)
    assert impact_large.total_cost_eur > impact_small.total_cost_eur


def test_missing_current_weights_treated_as_zero():
    """Assets in w_target but not w_current default to 0 (new position)."""
    model = AlmgrenChrissModel()
    w_curr_partial = pd.Series({"A": 0.30})  # B missing
    impact = model.estimate(w_curr_partial, W_TARGET, ADV, SPREAD, VOL, AUM)
    # B trade_size = 0.20 * 100_000 = 20_000 EUR (not 10_000)
    assert impact.total_cost_eur > EXPECTED_TOTAL_EUR


def test_custom_config_params():
    cfg = CostConfig(eta=0.0, gamma=0.0)  # only spread cost
    model = AlmgrenChrissModel(config=cfg)
    impact = model.estimate(W_CURRENT, W_TARGET, ADV, SPREAD, VOL, AUM)
    # Only spread: 0.5*0.001*20000 + 0.5*0.002*10000 = 10 + 10 = 20.0
    assert impact.total_cost_eur == pytest.approx(20.0, rel=1e-6)
