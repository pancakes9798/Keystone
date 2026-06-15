"""
Data ingestion connectors.

Priority order (configured in DataLoader):
1. Refinitiv/LSEG  — primary, highest quality
2. FRED            — macro-only, public API
3. Yahoo Finance   — fallback for prices, prototype use

Usage::

    from jeanclaude.data.ingestion import RefinitivSource, FREDSource, YahooSource
"""

from .base import DataSource
from .fred import FREDSource
from .refinitiv import RefinitivSource
from .yahoo import YahooSource

__all__ = ["DataSource", "RefinitivSource", "FREDSource", "YahooSource"]
