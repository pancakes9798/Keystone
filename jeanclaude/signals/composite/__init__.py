"""Composite signal module — regime-based asset class scores per Black-Litterman."""

from .builder import CompositeSignalBuilder
from .types import CompositeSignal, DEFAULT_REGIME_TABLE, RegimeTable

__all__ = ["CompositeSignal", "CompositeSignalBuilder", "DEFAULT_REGIME_TABLE", "RegimeTable"]
