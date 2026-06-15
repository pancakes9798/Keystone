import asyncio
from unittest.mock import patch

import numpy as np
import pandas as pd

from jeanclaude.agents.config import AgentConfig
from jeanclaude.agents.data import DataAgent
from jeanclaude.agents.events import PriceEvent


def _make_fake_prices(tickers: list[str], n: int = 10) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n)
    data = {t: np.linspace(100, 110, n) for t in tickers}
    return pd.DataFrame(data, index=dates)


def _make_fake_macro(n: int = 10) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n)
    return pd.DataFrame({"VIX": np.linspace(15, 20, n)}, index=dates)


def test_data_agent_returns_price_event():
    cfg = AgentConfig(universe=["XLK", "TLT"])
    agent = DataAgent(cfg)

    fake_prices = _make_fake_prices(["XLK", "TLT"])
    fake_macro_raw = _make_fake_macro()

    with patch.object(agent, "_fetch_prices", return_value=fake_prices), \
         patch.object(agent, "_fetch_macro", return_value=fake_macro_raw):
        event = asyncio.run(agent.run(pd.Timestamp("2024-01-10")))

    assert isinstance(event, PriceEvent)
    assert event.date == pd.Timestamp("2024-01-10")
    assert set(event.prices.columns) == {"XLK", "TLT"}
    assert len(event.returns) == len(fake_prices) - 1
    assert not event.returns.isnull().any().any()


def test_data_agent_macro_failure_continues():
    """Se yf.download fallisce per un ticker macro, DataAgent continua con gli altri."""
    import yfinance as yf

    cfg = AgentConfig(universe=["XLK"], macro_tickers={"VIX": "^VIX", "BadTicker": "XXXX=X"})
    agent = DataAgent(cfg)

    fake_prices = _make_fake_prices(["XLK"])
    fake_macro_series = pd.Series(
        [15.0] * 10,
        index=pd.date_range("2024-01-01", periods=10),
        name="Close",
    )

    def _side_effect(ticker, **kwargs):
        if ticker == "XXXX=X":
            raise ValueError("bad ticker")
        df = pd.DataFrame({"Close": fake_macro_series})
        df.index = pd.to_datetime(df.index)
        return df

    with patch.object(agent, "_fetch_prices", return_value=fake_prices), \
         patch("jeanclaude.agents.data.yf.download", side_effect=_side_effect):
        event = asyncio.run(agent.run(pd.Timestamp("2024-01-10")))

    assert isinstance(event, PriceEvent)
    assert "VIX" in event.macro.columns
    assert "BadTicker" not in event.macro.columns
