"""Signals module — macro regime detection, news sentiment, and composite signals."""
from __future__ import annotations

from .composite import CompositeSignal, CompositeSignalBuilder, DEFAULT_REGIME_TABLE, RegimeTable
from .macro import RegimeDetector, RegimeLabel, RegimeResult
from .news import NewsSentiment, FinBERTScorer, NewsAggregator, RefinitivNewsSource

__all__ = [
    "CompositeSignal",
    "CompositeSignalBuilder",
    "DEFAULT_REGIME_TABLE",
    "RegimeTable",
    "RegimeDetector",
    "RegimeLabel",
    "RegimeResult",
    "NewsSentiment",
    "FinBERTScorer",
    "NewsAggregator",
    "RefinitivNewsSource",
]
