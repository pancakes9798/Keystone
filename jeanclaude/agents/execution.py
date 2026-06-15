from __future__ import annotations

import logging

import pandas as pd

from .base import BaseAgent
from .events import NoRebalanceEvent, PriceEvent, WeightsEvent

logger = logging.getLogger(__name__)


class ExecutionAgent(BaseAgent):
    def __init__(self, broker: object) -> None:
        super().__init__()
        self._broker = broker

    async def run(
        self,
        weights_result: WeightsEvent | NoRebalanceEvent,
        price_event: PriceEvent,
    ) -> None:
        prices = price_event.prices.iloc[-1]
        date = price_event.date

        nav_recorded = False
        try:
            daily_returns = (
                price_event.returns.iloc[-1]
                if not price_event.returns.empty
                else pd.Series(dtype=float)
            )
            self._broker.record_daily_nav(daily_returns=daily_returns, date=date)
            nav_recorded = True
        except Exception as exc:
            logger.error("ExecutionAgent: record_daily_nav fallito (%s)", exc)

        if isinstance(weights_result, WeightsEvent):
            if not nav_recorded:
                logger.error(
                    "ExecutionAgent: NAV non registrato per %s — rebalance saltato "
                    "per integrità contabile (il re-run del ciclo recupererà).", date
                )
                return
            try:
                result = self._broker.execute_rebalance(
                    new_weights=weights_result.weights,
                    prices=prices,
                    date=date,
                )
                if result.cost_eur > 0.0:
                    self._broker.apply_rebalance_cost(date=date, cost_eur=result.cost_eur)
                logger.info(
                    "ExecutionAgent: ribilanciamento eseguito | date=%s reason=%s cost=%.2f",
                    date, weights_result.reason, result.cost_eur,
                )
            except Exception as exc:
                logger.error("ExecutionAgent: execute_rebalance fallito (%s) — skip", exc)
