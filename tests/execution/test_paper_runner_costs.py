# tests/execution/test_paper_runner_costs.py
import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock

from jeanclaude.costs import AlmgrenChrissModel, CostConfig
from jeanclaude.costs.data import VolumeSpreadLoader
from jeanclaude.execution.paper.runner import DailyRunner, DailyRunConfig
from jeanclaude.execution.paper.stubs import (
    NoopRegimeDetector, NoopRiskFilter, NoopReportBuilder,
)


def _make_prices(tickers, n=300):
    rng = np.random.default_rng(0)
    idx = pd.date_range("2025-01-02", periods=n, freq="B")
    data = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, (n, len(tickers))), axis=0))
    return pd.DataFrame(data, index=idx, columns=tickers)


def _make_runner_with_cost_model(broker, data_loader, cost_model, vsl):
    optimizer = MagicMock()
    optimizer.optimize.return_value = pd.Series(
        {"AAPL.O": 0.5, "TLT.O": 0.5}
    )
    config = DailyRunConfig(
        assets=["AAPL.O", "TLT.O"],
        macro_series=[],
        rebalance_freq="ME",
        send_report=False,
        transaction_cost_bps=10.0,
        min_history=60,
        history_start="2025-01-02",
    )
    return DailyRunner(
        broker=broker,
        optimizer=optimizer,
        risk_filter=NoopRiskFilter(),
        regime_detector=NoopRegimeDetector(),
        data_loader=data_loader,
        report_builder=NoopReportBuilder(),
        mailer=None,
        config=config,
        cost_model=cost_model,
        volume_spread_loader=vsl,
    )


def test_runner_accepts_cost_model(tmp_path):
    """DailyRunner should accept cost_model and volume_spread_loader without error."""
    from jeanclaude.data.storage.parquet_store import ParquetStore
    from jeanclaude.execution.paper.broker import PaperBroker

    broker = PaperBroker(100_000.0, ParquetStore(str(tmp_path)))
    tickers = ["AAPL.O", "TLT.O"]
    prices = _make_prices(tickers)

    data_loader = MagicMock()
    data_loader.get_prices.return_value = prices
    data_loader.get_returns.return_value = prices.pct_change().dropna()
    data_loader.get_macro.return_value = pd.DataFrame()

    vsl = MagicMock(spec=VolumeSpreadLoader)
    vsl.get_adv.return_value = pd.Series({"AAPL.O": 1_000_000.0, "TLT.O": 500_000.0})
    vsl.get_spread.return_value = pd.Series({"AAPL.O": 0.001, "TLT.O": 0.002})

    cost_model = AlmgrenChrissModel(CostConfig())
    runner = _make_runner_with_cost_model(broker, data_loader, cost_model, vsl)

    # Use a month-end date that triggers rebalance
    result = runner.run(pd.Timestamp("2025-07-31"))
    assert result is not None


def test_cost_model_called_on_rebalance_day(tmp_path):
    """VolumeSpreadLoader.get_adv and get_spread must be called on rebalance day."""
    from jeanclaude.data.storage.parquet_store import ParquetStore
    from jeanclaude.execution.paper.broker import PaperBroker

    broker = PaperBroker(100_000.0, ParquetStore(str(tmp_path)))
    tickers = ["AAPL.O", "TLT.O"]
    prices = _make_prices(tickers)

    data_loader = MagicMock()
    data_loader.get_prices.return_value = prices
    data_loader.get_returns.return_value = prices.pct_change().dropna()
    data_loader.get_macro.return_value = pd.DataFrame()

    vsl = MagicMock(spec=VolumeSpreadLoader)
    vsl.get_adv.return_value = pd.Series({"AAPL.O": 1_000_000.0, "TLT.O": 500_000.0})
    vsl.get_spread.return_value = pd.Series({"AAPL.O": 0.001, "TLT.O": 0.002})

    cost_model = AlmgrenChrissModel(CostConfig())
    runner = _make_runner_with_cost_model(broker, data_loader, cost_model, vsl)

    result = runner.run(pd.Timestamp("2025-07-31"))

    if result.rebalanced:
        vsl.get_adv.assert_called_once()
        vsl.get_spread.assert_called_once()
