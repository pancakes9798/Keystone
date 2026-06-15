from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from jeanclaude.signals.macro.labels import RegimeLabel

from .base import BaseAgent
from .events import PriceEvent, RegimeEvent, SignalEvent

logger = logging.getLogger(__name__)

_MIN_REGIME_OBS = 20


class SignalAgent(BaseAgent):
    async def run(self, regime_event: RegimeEvent, price_event: PriceEvent) -> SignalEvent:
        returns = price_event.returns
        tickers = list(returns.columns)
        proba = regime_event.probabilities
        history = regime_event.labels_history

        mu_regime: dict[RegimeLabel, pd.Series] = {}
        unconditional = returns.mean() * 252

        aligned = history.reindex(returns.index).ffill().dropna() if not history.empty else pd.Series(dtype="int64")

        for r in RegimeLabel:
            if history.empty:
                mu_regime[r] = unconditional
                continue
            mask = aligned == r
            if mask.sum() >= _MIN_REGIME_OBS:
                mu_regime[r] = returns.loc[mask].mean() * 252
            else:
                mu_regime[r] = unconditional

        views = sum(
            float(proba[r.value]) * mu_regime[r] for r in RegimeLabel
        ).reindex(tickers).fillna(0.0)

        regime_names = [r.name for r in RegimeLabel]
        confidence = pd.Series(proba, index=regime_names)

        logger.info("SignalAgent: views per %d asset | regime=%s", len(views), regime_event.label.name)
        return SignalEvent(date=regime_event.date, views=views, confidence=confidence)
