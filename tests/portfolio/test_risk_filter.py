"""Tests for RiskFilter — migrated to jeanclaude.portfolio.risk."""
from jeanclaude.portfolio.risk import RiskFilter


def test_import():
    """RiskFilter importabile dal nuovo path."""
    assert RiskFilter is not None
