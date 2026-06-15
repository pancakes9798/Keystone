from jeanclaude.agents.config import AgentConfig, RebalanceConfig, ETF_UNIVERSE, MACRO_TICKERS


def test_default_rebalance_config():
    cfg = RebalanceConfig()
    assert cfg.min_cooldown_days == 10
    assert cfg.signal_drift_threshold == 0.05
    assert cfg.regime_change_trigger is True
    assert cfg.risk_breach_trigger is True
    assert cfg.cvar_limit == 0.02


def test_default_agent_config():
    cfg = AgentConfig()
    assert len(cfg.universe) == 15
    assert cfg.initial_nav == 100_000.0
    assert cfg.max_weight == 0.25
    assert isinstance(cfg.rebalance, RebalanceConfig)


def test_etf_universe_completeness():
    assert "XLK" in ETF_UNIVERSE
    assert "TLT" in ETF_UNIVERSE
    assert "GLD" in ETF_UNIVERSE
    assert "EEM" in ETF_UNIVERSE
    assert len(ETF_UNIVERSE) == 15


def test_custom_config():
    cfg = AgentConfig(
        universe=["SPY", "TLT"],
        initial_nav=50_000.0,
        rebalance=RebalanceConfig(min_cooldown_days=5),
    )
    assert cfg.universe == ["SPY", "TLT"]
    assert cfg.initial_nav == 50_000.0
    assert cfg.rebalance.min_cooldown_days == 5
