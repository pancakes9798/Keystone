"""News sentiment signal — FinBERT-based sentiment from Refinitiv headlines."""
from .types import NewsSentiment
from .ticker_types import TickerSentiment
from .scorer import FinBERTScorer
from .aggregator import NewsAggregator
from .ticker_aggregator import TickerNewsAggregator
from .fetcher import RefinitivNewsSource

__all__ = [
    "NewsSentiment", "TickerSentiment",
    "FinBERTScorer",
    "NewsAggregator", "TickerNewsAggregator",
    "RefinitivNewsSource",
]
