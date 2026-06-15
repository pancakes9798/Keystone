from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

from .base import BaseAgent
from .config import AgentConfig
from .events import PriceEvent

logger = logging.getLogger(__name__)


class DataAgent(BaseAgent):
    def __init__(self, config: AgentConfig) -> None:
        super().__init__()
        self._config = config

    async def run(self, date: pd.Timestamp) -> PriceEvent:
        end_str = (date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        start_str = self._config.history_start

        prices = self._fetch_prices(start_str, end_str)
        macro = self._fetch_macro(start_str, end_str)
        # simple returns: il broker compone (1+r); anche optimizer/risk filter li usano coerentemente
        returns = prices.pct_change().dropna(how="all")

        logger.info(
            "DataAgent: %d price rows, %d macro rows, %d assets",
            len(prices), len(macro), len(prices.columns),
        )
        return PriceEvent(date=date, prices=prices, returns=returns, macro=macro)

    def _fetch_prices(self, start: str, end: str) -> pd.DataFrame:
        raw = yf.download(
            self._config.universe, start=start, end=end,
            progress=False, auto_adjust=True,
        )["Close"]
        raw.index = pd.to_datetime(raw.index).tz_localize(None)
        available = [t for t in self._config.universe if t in raw.columns]
        missing = [t for t in self._config.universe if t not in raw.columns]
        if missing:
            logger.warning("DataAgent: tickers non disponibili e scartati: %s", missing)
        return raw[available].ffill().dropna(how="all")

    def _fetch_macro(self, start: str, end: str) -> pd.DataFrame:
        frames: dict[str, pd.Series] = {}
        for name, ticker in self._config.macro_tickers.items():
            try:
                raw = yf.download(
                    ticker, start=start, end=end,
                    progress=False, auto_adjust=True,
                )["Close"].squeeze()
                raw.index = pd.to_datetime(raw.index).tz_localize(None)
                frames[name] = raw
            except Exception as exc:
                logger.warning("DataAgent: macro fetch failed for %s (%s): %s", name, ticker, exc)
        if not frames:
            logger.warning("DataAgent: tutti i ticker macro falliti, macro frame vuoto")
        return pd.DataFrame(frames).ffill()
