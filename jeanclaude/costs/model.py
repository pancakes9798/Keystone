from __future__ import annotations

import pandas as pd

from .types import CostConfig, TradeImpact


class AlmgrenChrissModel:
    """Two-term quadratic transaction cost model inspired by Almgren & Chriss.

    Cost components per asset i:
        spread_cost_i      = 0.5 * spread_i * |trade_size_i|
        temporary_impact_i = eta   * vol_i * participation_i * |trade_size_i|
        permanent_impact_i = gamma * vol_i * participation_i^2 * |trade_size_i|

    Where participation_i = |trade_size_i| / ADV_i.
    All costs are in currency units (EUR/USD), summing to total_cost_eur.
    """

    def __init__(self, config: CostConfig | None = None) -> None:
        self.config = config or CostConfig()

    def estimate(
        self,
        w_current: pd.Series,
        w_target: pd.Series,
        adv: pd.Series,
        spread: pd.Series,
        vol: pd.Series,
        aum: float,
    ) -> TradeImpact:
        """Estimate transaction costs for a portfolio rebalance.

        Parameters
        ----------
        w_current : pd.Series
            Current portfolio weights (asset → float). Missing assets = 0.
        w_target : pd.Series
            Target portfolio weights (asset → float).
        adv : pd.Series
            Average daily volume per asset in currency units.
        spread : pd.Series
            Bid-ask spread per asset as a fraction of mid price.
        vol : pd.Series
            Daily return volatility per asset.
        aum : float
            Portfolio value in currency units.

        Returns
        -------
        TradeImpact
        """
        assets = w_target.index
        w_curr = w_current.reindex(assets, fill_value=0.0)
        delta_w = (w_target - w_curr).abs()

        trade_size = delta_w * aum  # EUR, always non-negative

        adv_a    = adv.reindex(assets, fill_value=1.0).clip(lower=1.0)
        spread_a = spread.reindex(assets, fill_value=0.001).clip(lower=0.0)
        vol_a    = vol.reindex(assets, fill_value=0.01).clip(lower=1e-6)

        participation = trade_size / adv_a  # fraction of ADV

        spread_cost      = 0.5 * spread_a * trade_size
        temporary_impact = self.config.eta   * vol_a * participation * trade_size
        permanent_impact = self.config.gamma * vol_a * (participation ** 2) * trade_size

        total_per_asset = spread_cost + temporary_impact + permanent_impact
        per_asset_bps = (total_per_asset / aum) * 10_000 if aum > 0 else total_per_asset * 0

        total_cost_eur = float(total_per_asset.sum())
        total_cost_pct = total_cost_eur / aum if aum > 0 else 0.0
        turnover_pct = float(delta_w.sum()) / 2.0

        return TradeImpact(
            per_asset_bps=per_asset_bps.copy(),  # copy to avoid aliasing
            total_cost_pct=total_cost_pct,
            total_cost_eur=total_cost_eur,
            turnover_pct=turnover_pct,
        )
