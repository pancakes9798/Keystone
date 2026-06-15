from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


@dataclass(frozen=True)
class CostConfig:
    """Configuration for Almgren-Chriss transaction cost model."""

    eta: float = 0.1
    gamma: float = 0.1
    vol_window: int = 63
    adv_window: int = 20
    aum_source: Literal["fixed", "dynamic"] = "fixed"
    fixed_aum: float = 100_000.0


@dataclass
class TradeImpact:
    """Result of a single Almgren-Chriss cost estimation."""

    per_asset_bps: pd.Series  # cost per asset in basis points
    total_cost_pct: float  # total cost as fraction of AUM
    total_cost_eur: float  # total cost in currency units
    turnover_pct: float  # one-way turnover = Σ|Δw| / 2
