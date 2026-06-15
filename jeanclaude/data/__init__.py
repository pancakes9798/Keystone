"""
Data module — ingestion, storage, and transformation.

Main entry point::

    from jeanclaude.data import DataLoader, DataConfig

    loader = DataLoader(DataConfig(
        cache_dir="~/jeanclaude_data",
        price_source="refinitiv",   # or "yahoo" for fallback
        macro_source="fred",
    ))

    prices = loader.get_prices(["SPY.P", "TLT.P"], start="2020-01-01", end="2024-12-31")
    macro  = loader.get_macro(["VIXCLS", "T10Y2Y"], start="2020-01-01", end="2024-12-31")
"""

from .ingestion import DataSource, FREDSource, RefinitivSource, YahooSource
from .loader import DataConfig, DataLoader
from .storage import ParquetStore
from .transform import (
    DEFAULT_MACRO_SERIES,
    add_momentum_features,
    align_to_prices,
    annualized_stats,
    build_state_variables,
    resample_returns,
    to_log_returns,
    to_simple_returns,
)

__all__ = [
    # Loader (main entry point)
    "DataLoader",
    "DataConfig",
    # Sources
    "DataSource",
    "RefinitivSource",
    "FREDSource",
    "YahooSource",
    # Storage
    "ParquetStore",
    # Transforms
    "to_log_returns",
    "to_simple_returns",
    "resample_returns",
    "annualized_stats",
    "build_state_variables",
    "add_momentum_features",
    "align_to_prices",
    "DEFAULT_MACRO_SERIES",
]
