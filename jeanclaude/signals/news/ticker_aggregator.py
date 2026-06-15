"""TickerNewsAggregator — rolling sentiment per singolo ticker (RIC)."""
from __future__ import annotations

import logging

import pandas as pd

from .ticker_types import TickerSentiment

logger = logging.getLogger(__name__)

_CONFIDENCE_SCALE = 10  # n_articles per ticker che portano confidence a 1.0


class TickerNewsAggregator:
    """Aggrega sentiment FinBERT per singolo ticker nell'ultimo ``window_days`` giorni.

    Parameters
    ----------
    window_days : int
        Finestra mobile in giorni di calendario (default 5).

    Input DataFrame atteso (output di FinBERTScorer):
        - DatetimeIndex
        - colonne: ``ticker`` (str, RIC), ``sentiment_score`` (float ∈ [-1,1])
    """

    def __init__(self, window_days: int = 5) -> None:
        self.window_days = window_days

    def aggregate(
        self,
        scored_df: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> TickerSentiment:
        """Aggrega le headlines per ticker nell'ultimo ``window_days`` giorni.

        Parameters
        ----------
        scored_df : pd.DataFrame
            Deve avere DatetimeIndex e colonne ``ticker``, ``sentiment_score``.
        as_of : pd.Timestamp
            Estremo superiore della finestra (incluso).

        Returns
        -------
        TickerSentiment
            Scores e confidence per ogni ticker con almeno una headline.
            Se nessuna headline: scores/confidence vuoti, n_articles=0.
        """
        _empty = TickerSentiment(
            scores=pd.Series(dtype=float),
            confidence=pd.Series(dtype=float),
            timestamp=as_of,
            n_articles=0,
        )

        if scored_df.empty or "ticker" not in scored_df.columns:
            return _empty

        cutoff = as_of - pd.Timedelta(days=self.window_days - 1)
        window = scored_df[
            (scored_df.index >= cutoff) & (scored_df.index <= as_of)
        ]

        if window.empty:
            return _empty

        n_articles = len(window)
        grouped    = window.groupby("ticker")

        scores     = grouped["sentiment_score"].mean()
        counts     = grouped["sentiment_score"].count()
        confidence = (counts / _CONFIDENCE_SCALE).clip(upper=1.0)

        logger.debug(
            "TickerNewsAggregator: %d articles, %d tickers | window=%d days",
            n_articles, len(scores), self.window_days,
        )

        return TickerSentiment(
            scores=scores,
            confidence=confidence,
            timestamp=window.index.max(),
            n_articles=n_articles,
        )
