"""TickerSentiment — sentiment aggregato per singolo ticker (RIC)."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TickerSentiment:
    """Sentiment delle news per singolo ticker.

    Attributes
    ----------
    scores : pd.Series
        RIC → score ∈ [-1, 1]. Positivo = sentiment rialzista.
        Solo i ticker con almeno una headline nel periodo sono inclusi.
    confidence : pd.Series
        RIC → confidence ∈ [0, 1]: ``min(n_articles_i / 10, 1.0)``.
        Ticker senza news hanno confidence 0 (non inclusi).
    timestamp : pd.Timestamp
        Data dell'ultima headline nella finestra.
    n_articles : int
        Totale headlines nell'intera finestra.
    """

    scores: pd.Series       # index = RIC
    confidence: pd.Series   # index = RIC
    timestamp: pd.Timestamp
    n_articles: int

    @property
    def tickers(self) -> list[str]:
        return list(self.scores.index)

    def has_signal(self) -> bool:
        return len(self.scores) > 0 and self.confidence.max() > 0.0
