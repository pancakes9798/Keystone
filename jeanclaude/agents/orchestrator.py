from __future__ import annotations

import asyncio
import logging
from datetime import date as Date

import numpy as np
import pandas as pd

from jeanclaude.data.storage.parquet_store import ParquetStore
from jeanclaude.execution.paper.broker import PaperBroker
from jeanclaude.signals.macro.labels import RegimeLabel

from .config import AgentConfig
from .data import DataAgent
from .events import NoRebalanceEvent, PriceEvent, RegimeEvent, SignalEvent, WeightsEvent
from .execution import ExecutionAgent
from .portfolio import PortfolioAgent
from .regime import RegimeAgent
from .report import ReportAgent
from .signal import SignalAgent

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, config: AgentConfig) -> None:
        self._config = config
        store = ParquetStore(config.data_dir)
        broker = PaperBroker(initial_capital=config.initial_nav, store=store)
        self._data_agent = DataAgent(config)
        self._regime_agent = RegimeAgent()
        self._signal_agent = SignalAgent()
        self._portfolio_agent = PortfolioAgent(config, broker)
        self._execution_agent = ExecutionAgent(broker)
        self._report_agent = ReportAgent(config, broker)
        self._last_regime_event: RegimeEvent | None = None

    async def run_once(self, target_date: pd.Timestamp | None = None) -> None:
        date = target_date or pd.Timestamp.today().normalize()
        logger.info("Orchestrator: ciclo giornaliero | date=%s", date.date())

        # DataAgent — abort on failure
        try:
            price_event = await self._data_agent.run(date)
        except Exception as exc:
            logger.error("Orchestrator: DataAgent fallito (%s) — abort", exc)
            return

        # RegimeAgent — degraded mode: usa ultimo regime noto
        try:
            regime_event = await self._regime_agent.run(price_event)
            self._last_regime_event = regime_event
        except Exception as exc:
            logger.warning("Orchestrator: RegimeAgent fallito (%s) — uso ultimo regime", exc)
            if self._last_regime_event is not None:
                regime_event = RegimeEvent(
                    date=date,
                    label=self._last_regime_event.label,
                    probabilities=self._last_regime_event.probabilities,
                    labels_history=self._last_regime_event.labels_history,
                    changed=False,
                )
            else:
                regime_event = RegimeEvent(
                    date=date,
                    label=RegimeLabel.TRANSITION,
                    probabilities=np.array([1 / 3, 1 / 3, 1 / 3]),
                    labels_history=pd.Series(dtype="int64"),
                    changed=False,
                )

        # SignalAgent — degraded mode: views vuote
        try:
            signal_event = await self._signal_agent.run(regime_event, price_event)
        except Exception as exc:
            logger.warning("Orchestrator: SignalAgent fallito (%s) — views vuote", exc)
            tickers = list(price_event.returns.columns)
            signal_event = SignalEvent(
                date=date,
                views=pd.Series(0.0, index=tickers),
                confidence=pd.Series(
                    [1 / 3, 1 / 3, 1 / 3],
                    index=["EXPANSION", "CONTRACTION", "TRANSITION"],
                ),
            )

        # PortfolioAgent — degraded mode: NoRebalanceEvent
        try:
            weights_result = await self._portfolio_agent.run(signal_event, price_event, regime_event)
        except Exception as exc:
            logger.warning("Orchestrator: PortfolioAgent fallito (%s) — NoRebalanceEvent", exc)
            weights_result = NoRebalanceEvent(date=date, reason="no_trigger")

        # ExecutionAgent — soft
        try:
            await self._execution_agent.run(weights_result, price_event)
        except Exception as exc:
            logger.error("Orchestrator: ExecutionAgent fallito (%s)", exc)

        # ReportAgent — soft
        try:
            await self._report_agent.run(weights_result, price_event)
        except Exception as exc:
            logger.error("Orchestrator: ReportAgent fallito (%s)", exc)

    def run(self, target_date: pd.Timestamp | None = None) -> None:
        asyncio.run(self.run_once(target_date))
