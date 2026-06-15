"""Smoke + unit test dell'entry-point ETF (mai testato prima del 2026-06)."""
import pandas as pd

import scripts.run_paper_etf as etf
from jeanclaude.data.storage.parquet_store import ParquetStore
from jeanclaude.execution.paper import PaperBroker


def test_module_constants_sane():
    assert etf._FLAT_COST_BPS > 0.0, "costi a zero in produzione (audit P0)"
    assert len(etf.ETF_TICKERS) == 15


def test_compute_asof_skips_weekend():
    idx = pd.bdate_range("2026-06-01", "2026-06-05")
    prices = pd.DataFrame({"XLK": range(5)}, index=idx)
    asof = etf._compute_asof(pd.Timestamp("2026-06-06"), prices.index)  # sabato
    assert asof == pd.Timestamp("2026-06-05")


def test_period_returns_gap_aware(tmp_path):
    """Il runner ETF ora delega a PaperBroker.period_returns: finestra ancorata
    alla riga NAV prima di asof (qui 06-02), gap-aware fino ad asof (06-05)."""
    idx = pd.bdate_range("2026-06-01", "2026-06-05")
    prices = pd.DataFrame({"XLK": [100.0, 101.0, 102.0, 103.0, 104.0]}, index=idx)
    broker = PaperBroker(initial_capital=100_000.0, store=ParquetStore(tmp_path))
    broker.record_daily_nav(pd.Series(dtype=float), date=pd.Timestamp("2026-06-02"))  # base NAV @ 06-02
    rets = broker.period_returns(prices, asof=pd.Timestamp("2026-06-05"), assets=["XLK"])
    assert rets["XLK"] == (104.0 / 101.0 - 1)
