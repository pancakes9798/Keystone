"""NoTradeBandOverlay — ribilancia solo se il turnover proposto supera la soglia.

Sprint M (2026-06-11). Statefulness: l'overlay ricorda gli ultimi pesi ESEGUITI
(stessa istanza attraverso i rebalance dell'engine). Se ||w_new − w_last||_1
< threshold, ritorna w_last (nessun trade). Il primo rebalance esegue sempre.
ATTENZIONE: istanza nuova per ogni run di backtest (lo stato non è condiviso).
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class NoTradeBandOverlay:
    def __init__(self, base_optimizer: Any, threshold: float) -> None:
        self._base = base_optimizer
        self.threshold = threshold
        self._last_weights: pd.Series | None = None

    def optimize(self, returns: pd.DataFrame) -> pd.Series:
        new_w = self._base.optimize(returns)
        if self._last_weights is None:
            self._last_weights = new_w
            return new_w
        last = self._last_weights.reindex(new_w.index, fill_value=0.0)
        proposed_turnover = float((new_w - last).abs().sum())
        if proposed_turnover < self.threshold:
            logger.debug("NoTradeBand: turnover proposto %.3f < %.3f — niente trade.",
                         proposed_turnover, self.threshold)
            return self._last_weights.copy()
        self._last_weights = new_w
        return new_w
