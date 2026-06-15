"""NewsSentiment — frozen dataclass output di NewsAggregator."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class NewsSentiment:
    """Sentiment aggregato delle news per asset class.

    Attributes
    ----------
    scores : pd.Series
        asset_class → score ∈ [-1, 1]. Positivo = sentiment rialzista.
    confidence : float
        ∈ [0, 1]: ``min(n_articles / 50, 1.0)``.
        Sotto 0.05 il CompositeSignalBuilder ignora questo segnale.
    timestamp : pd.Timestamp
        Data dell'ultima headline inclusa nella finestra.
    n_articles : int
        Numero totale di headlines nel periodo aggregato.
    """

    scores: pd.Series
    confidence: float
    timestamp: pd.Timestamp
    n_articles: int
