"""Tests for PaperBroker nav property.

Note: apply_cost() was removed (non-idempotent, wall-clock-dated, zero production callers).
The two apply_cost-only tests that existed here have been deleted; equivalent invariants
are now covered by TestApplyRebalanceCost in test_broker_nav_accounting.py.
"""
import pytest
import pandas as pd
from jeanclaude.execution.paper.broker import PaperBroker
from jeanclaude.data.storage.parquet_store import ParquetStore


def _make_broker(tmp_path) -> PaperBroker:
    store = ParquetStore(str(tmp_path))
    return PaperBroker(initial_capital=100_000.0, store=store)


def test_nav_property_returns_initial_capital(tmp_path):
    broker = _make_broker(tmp_path)
    assert broker.nav == pytest.approx(100_000.0)


def test_nav_property_after_record_daily_nav(tmp_path):
    broker = _make_broker(tmp_path)
    weights = pd.Series({"SPY": 0.5, "TLT": 0.5})
    returns = pd.Series({"SPY": 0.01, "TLT": 0.005})

    # set weights first — use explicit date so _weights_before(d_nav) can find them
    prices = pd.Series({"SPY": 400.0, "TLT": 100.0})
    d_rebalance = pd.Timestamp("2026-01-01")
    d_nav = pd.Timestamp("2026-01-02")
    broker.execute_rebalance(weights, prices, transaction_cost_bps=0.0, date=d_rebalance)
    broker.record_daily_nav(returns, date=d_nav)

    expected_nav = 100_000.0 * (1 + 0.5 * 0.01 + 0.5 * 0.005)
    assert broker.nav == pytest.approx(expected_nav, rel=1e-4)
