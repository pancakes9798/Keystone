"""NewsAggregator — rolling 5-day sentiment score per asset class."""
from __future__ import annotations

import logging

import pandas as pd

from .types import NewsSentiment

logger = logging.getLogger(__name__)

_CONFIDENCE_SCALE = 50  # n_articles that bring confidence to 1.0


class NewsAggregator:
    """Aggrega sentiment score degli ultimi ``window_days`` giorni per asset class.

    Parameters
    ----------
    window_days : int
        Numero di giorni di calendario da includere (default 5).
    """

    def __init__(self, window_days: int = 5) -> None:
        self.window_days = window_days

    def aggregate(
        self,
        scored_df: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> NewsSentiment:
        """Aggrega le headlines nell'ultimo ``window_days`` giorni.

        Parameters
        ----------
        scored_df : pd.DataFrame
            Output di ``FinBERTScorer.score()``. Deve avere un ``DatetimeIndex``
            e le colonne ``asset_class`` (str) e ``sentiment_score`` (float).
        as_of : pd.Timestamp
            Data di riferimento (estremo superiore della finestra, incluso).

        Returns
        -------
        NewsSentiment
            Se nessuna headline cade nella finestra: ``confidence=0.0``,
            ``scores`` vuoto, ``n_articles=0``.
        """
        if scored_df.empty:
            return NewsSentiment(
                scores=pd.Series(dtype=float),
                confidence=0.0,
                timestamp=as_of,
                n_articles=0,
            )

        cutoff = as_of - pd.Timedelta(days=self.window_days - 1)
        window = scored_df[
            (scored_df.index >= cutoff) & (scored_df.index <= as_of)
        ]

        if window.empty:
            return NewsSentiment(
                scores=pd.Series(dtype=float),
                confidence=0.0,
                timestamp=as_of,
                n_articles=0,
            )

        n_articles = len(window)
        confidence = min(n_articles / _CONFIDENCE_SCALE, 1.0)
        scores = window.groupby("asset_class")["sentiment_score"].mean()
        timestamp = window.index.max()

        logger.debug(
            "NewsAggregator: %d articles in %d-day window | confidence=%.2f",
            n_articles, self.window_days, confidence,
        )

        return NewsSentiment(
            scores=scores,
            confidence=confidence,
            timestamp=timestamp,
            n_articles=n_articles,
        )
