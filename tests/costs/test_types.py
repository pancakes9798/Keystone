import pytest
from jeanclaude.costs.types import CostConfig, TradeImpact
import pandas as pd


def test_cost_config_defaults():
    cfg = CostConfig()
    assert cfg.eta == 0.1
    assert cfg.gamma == 0.1
    assert cfg.vol_window == 63
    assert cfg.adv_window == 20
    assert cfg.aum_source == "fixed"
    assert cfg.fixed_aum == 100_000.0


def test_cost_config_frozen():
    cfg = CostConfig()
    with pytest.raises((AttributeError, TypeError)):
        cfg.eta = 0.5  # type: ignore[misc]


def test_cost_config_custom():
    cfg = CostConfig(eta=0.2, gamma=0.05, aum_source="dynamic")
    assert cfg.eta == 0.2
    assert cfg.aum_source == "dynamic"


def test_trade_impact_fields():
    per_asset = pd.Series({"A": 5.0, "B": 3.0})
    impact = TradeImpact(
        per_asset_bps=per_asset,
        total_cost_pct=0.0002,
        total_cost_eur=20.0,
        turnover_pct=0.15,
    )
    assert impact.total_cost_eur == 20.0
    assert impact.turnover_pct == 0.15
    assert impact.per_asset_bps["A"] == 5.0


def test_public_exports():
    from jeanclaude.costs import AlmgrenChrissModel, CostConfig, TradeImpact
    assert AlmgrenChrissModel is not None
    assert CostConfig is not None
    assert TradeImpact is not None
