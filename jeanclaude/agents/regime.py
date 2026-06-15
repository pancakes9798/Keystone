from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from jeanclaude.data.transform.features import build_state_variables
from jeanclaude.signals.macro.detector import RegimeDetector
from jeanclaude.signals.macro.labels import RegimeLabel

from .base import BaseAgent
from .events import PriceEvent, RegimeEvent

logger = logging.getLogger(__name__)

_MIN_MACRO_ROWS = 60


class RegimeAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__()
        self._detector = RegimeDetector()
        self._last_regime: RegimeLabel | None = None

    async def run(self, event: PriceEvent) -> RegimeEvent:
        sv = build_state_variables(event.macro.loc[:event.date]).dropna()

        if len(sv) < _MIN_MACRO_ROWS:
            logger.warning(
                "RegimeAgent: solo %d righe macro (minimo %d) — default TRANSITION",
                len(sv), _MIN_MACRO_ROWS,
            )
            return self._default_event(event.date)

        try:
            result = self._detector.fit(sv)
        except Exception as exc:
            logger.warning("RegimeAgent: HMM fit fallito (%s) — default TRANSITION", exc)
            return self._default_event(event.date)

        proba = result.probabilities.iloc[-1].values
        label = RegimeLabel(int(np.argmax(proba)))
        changed = self._last_regime is not None and label != self._last_regime

        self._last_regime = label
        logger.info("RegimeAgent: label=%s changed=%s proba=%s", label.name, changed, proba.round(2))

        return RegimeEvent(
            date=event.date,
            label=label,
            probabilities=proba,
            labels_history=result.labels,
            changed=changed,
        )

    def _default_event(self, date: pd.Timestamp) -> RegimeEvent:
        return RegimeEvent(
            date=date,
            label=RegimeLabel.TRANSITION,
            probabilities=np.array([0.0, 0.0, 1.0]),
            labels_history=pd.Series(dtype="int64"),
            changed=False,
        )
