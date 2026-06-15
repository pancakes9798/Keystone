from __future__ import annotations

from dataclasses import dataclass, field

ETF_UNIVERSE: list[str] = [
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP",
    "TLT", "IEF", "HYG",
    "GLD", "SLV", "USO",
    "EEM", "EFA",
]

MACRO_TICKERS: dict[str, str] = {
    "VIX":    "^VIX",
    "Oil":    "CL=F",
    "Copper": "HG=F",
    "Gold":   "GC=F",
    "EURUSD": "EURUSD=X",
}


@dataclass
class RebalanceConfig:
    min_cooldown_days: int = 10
    regime_change_trigger: bool = True
    signal_drift_threshold: float = 0.05
    risk_breach_trigger: bool = True
    cvar_limit: float = 0.02


@dataclass
class AgentConfig:
    universe: list[str] = field(default_factory=lambda: ETF_UNIVERSE.copy())
    macro_tickers: dict[str, str] = field(default_factory=lambda: MACRO_TICKERS.copy())
    history_start: str = "2010-01-01"
    initial_nav: float = 100_000.0
    max_weight: float = 0.25
    delta: float = 2.5
    tau: float = 0.025
    turnover_penalty: float = 0.02
    rebalance: RebalanceConfig = field(default_factory=RebalanceConfig)
    report_recipients: list[str] = field(default_factory=list)
    data_dir: str = "data/agent"
