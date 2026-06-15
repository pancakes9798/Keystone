"""P0 regression test del DailyRunner: idempotenza, gap, ordine record→rebalance, costi."""
import pandas as pd
import pytest
from unittest.mock import MagicMock

from jeanclaude.data.storage.parquet_store import ParquetStore
from jeanclaude.execution.paper import (
    DailyRunConfig, DailyRunner, NoopRegimeDetector, NoopReportBuilder, NoopRiskFilter, PaperBroker,
)
from jeanclaude.costs import AlmgrenChrissModel, CostConfig
from jeanclaude.costs.data import VolumeSpreadLoader

ASSETS = ["AAA", "BBB"]


class FakeLoader:
    """DataLoader in-memory con calendario controllato."""

    def __init__(self, prices: pd.DataFrame):
        self._prices = prices

    def get_prices(self, tickers, start, end):
        return self._prices.loc[:end, tickers]

    def get_returns(self, prices):
        return prices.pct_change().dropna(how="all")

    def get_macro(self, series, start, end):
        return pd.DataFrame()


class FixedOptimizer:
    def __init__(self, weights):
        self._w = weights

    def optimize(self, returns):
        return self._w


@pytest.fixture
def prices():
    idx = pd.bdate_range("2026-01-01", "2026-03-31")
    data = {
        "AAA": pd.Series(100.0 * (1.001 ** pd.RangeIndex(len(idx))).values, index=idx),
        "BBB": pd.Series(50.0 * (1.0005 ** pd.RangeIndex(len(idx))).values, index=idx),
    }
    return pd.DataFrame(data)


def _make_runner(tmp_path, prices, cost_bps=10.0):
    store = ParquetStore(tmp_path)
    broker = PaperBroker(initial_capital=100_000.0, store=store)
    config = DailyRunConfig(
        assets=ASSETS, macro_series=[], rebalance_freq="ME",
        send_report=False, transaction_cost_bps=cost_bps,
        min_history=20, history_start="2026-01-01",
    )
    runner = DailyRunner(
        broker=broker,
        optimizer=FixedOptimizer(pd.Series({"AAA": 0.5, "BBB": 0.5})),
        risk_filter=NoopRiskFilter(),
        regime_detector=NoopRegimeDetector(),
        data_loader=FakeLoader(prices),
        report_builder=NoopReportBuilder(),
        mailer=None,
        config=config,
    )
    return runner, broker


def test_double_run_same_day_is_idempotent(tmp_path, prices):
    runner, broker = _make_runner(tmp_path, prices)
    day = pd.Timestamp("2026-02-02")  # primo bday dopo il month-end di gennaio
    r1 = runner.run(day)
    r2 = runner.run(day)
    assert r1.rebalanced and r2.rebalanced  # il re-run riesegue, non salta
    assert r1.nav == pytest.approx(r2.nav)  # incluso il costo: stato identico
    nav_history = broker._store.load("paper_trading", "nav_history", "state", "state")
    assert nav_history.index.nunique() == len(nav_history)
    orders = broker._store.load("paper_trading", "orders", "state", "state")
    assert len(orders) == len(ASSETS)  # upsert: niente ordini duplicati


class FlakyOptimizer:
    """Fallisce alla prima chiamata, poi funziona — simula un crash mid-rebalance."""

    def __init__(self, weights):
        self._w = weights
        self.calls = 0

    def optimize(self, returns):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("optimizer crash simulato")
        return self._w


def test_rerun_recovers_after_partial_failure(tmp_path, prices):
    """NAV registrato ma rebalance crashato: il run successivo DEVE ribilanciare."""
    runner, broker = _make_runner(tmp_path, prices)
    runner._optimizer = FlakyOptimizer(pd.Series({"AAA": 0.5, "BBB": 0.5}))
    day = pd.Timestamp("2026-02-02")
    with pytest.raises(RuntimeError):
        runner.run(day)
    # stato parziale: NAV scritto, nessun rebalance
    assert broker.last_nav_date() == day
    assert broker.last_rebalance_date() is None
    # recovery: il re-run completa il ciclo
    result = runner.run(day)
    assert result.rebalanced
    assert broker.last_rebalance_date() == day


def test_asof_is_last_price_date_not_run_date(tmp_path, prices):
    """Run di sabato → la riga NAV è datata venerdì (ultima data di mercato)."""
    runner, broker = _make_runner(tmp_path, prices)
    runner.run(pd.Timestamp("2026-02-07"))  # sabato
    assert broker.last_nav_date() == pd.Timestamp("2026-02-06")  # venerdì


def test_gap_aware_returns_no_lost_pnl(tmp_path, prices):
    """Giorni saltati: il run successivo applica il ritorno multi-periodo, non solo l'ultimo giorno."""
    runner, broker = _make_runner(tmp_path, prices, cost_bps=0.0)
    runner.run(pd.Timestamp("2026-02-02"))   # alloca
    nav_before = broker.nav
    # salta 3 giorni di run, poi riprende
    runner.run(pd.Timestamp("2026-02-06"))
    p = prices
    multi_ret = (
        0.5 * (p.loc["2026-02-06", "AAA"] / p.loc["2026-02-02", "AAA"] - 1)
        + 0.5 * (p.loc["2026-02-06", "BBB"] / p.loc["2026-02-02", "BBB"] - 1)
    )
    assert broker.nav == pytest.approx(nav_before * (1 + multi_ret), rel=1e-9)


def test_rebalance_day_pnl_attributed_to_old_weights(tmp_path, prices):
    """Il return del giorno di rebalance matura sui pesi vecchi, non sui nuovi."""
    runner, broker = _make_runner(tmp_path, prices, cost_bps=0.0)
    runner.run(pd.Timestamp("2026-02-02"))
    # cambio optimizer: al prossimo rebalance va tutto su AAA
    runner._optimizer = FixedOptimizer(pd.Series({"AAA": 1.0, "BBB": 0.0}))
    runner.run(pd.Timestamp("2026-02-27"))
    nav_before_rebal = broker.nav
    result = runner.run(pd.Timestamp("2026-03-02"))  # rebalance day (feb completato)
    assert result.rebalanced
    p = prices
    old_w_ret = (
        0.5 * (p.loc["2026-03-02", "AAA"] / p.loc["2026-02-27", "AAA"] - 1)
        + 0.5 * (p.loc["2026-03-02", "BBB"] / p.loc["2026-02-27", "BBB"] - 1)
    )
    assert result.nav == pytest.approx(nav_before_rebal * (1 + old_w_ret), rel=1e-9)


def test_flat_cost_charged_when_no_cost_model(tmp_path, prices):
    """Senza AC model il costo flat bps viene davvero addebitato al NAV."""
    runner, broker = _make_runner(tmp_path, prices, cost_bps=10.0)
    result = runner.run(pd.Timestamp("2026-02-02"))
    assert result.rebalanced
    # allocazione iniziale: turnover = somma pesi = 1.0 → 10 bps su NAV pre-costo
    nav_history = broker._store.load("paper_trading", "nav_history", "state", "state")
    nav_day = float(nav_history["nav"].iloc[-1])
    assert nav_day < 100_000.0  # costo applicato


def test_ac_failure_fallback_charges_flat_cost_not_zero(tmp_path, prices):
    """Quando il cost model AC solleva un'eccezione, il runner deve addebitare il costo
    flat bps — NON zero. Questo era il bug auditato dal P0 2026-06-10."""
    store = ParquetStore(tmp_path)
    broker = PaperBroker(initial_capital=100_000.0, store=store)
    config = DailyRunConfig(
        assets=ASSETS, macro_series=[], rebalance_freq="ME",
        send_report=False, transaction_cost_bps=10.0,
        min_history=20, history_start="2026-01-01",
    )

    # Cost model che solleva sempre un'eccezione
    failing_cost_model = MagicMock(spec=AlmgrenChrissModel)
    failing_cost_model.config = CostConfig()
    failing_cost_model.estimate.side_effect = RuntimeError("AC model failed intentionally")

    vsl = MagicMock(spec=VolumeSpreadLoader)
    vsl.get_adv.return_value = pd.Series({"AAA": 1_000_000.0, "BBB": 500_000.0})
    vsl.get_spread.return_value = pd.Series({"AAA": 0.001, "BBB": 0.002})

    runner = DailyRunner(
        broker=broker,
        optimizer=FixedOptimizer(pd.Series({"AAA": 0.5, "BBB": 0.5})),
        risk_filter=NoopRiskFilter(),
        regime_detector=NoopRegimeDetector(),
        data_loader=FakeLoader(prices),
        report_builder=NoopReportBuilder(),
        mailer=None,
        config=config,
        cost_model=failing_cost_model,
        volume_spread_loader=vsl,
    )

    result = runner.run(pd.Timestamp("2026-02-02"))
    assert result.rebalanced

    # Il fallback flat bps deve essere addebitato: NAV < 100_000
    nav_history = broker._store.load("paper_trading", "nav_history", "state", "state")
    nav_day = float(nav_history["nav"].iloc[-1])
    assert nav_day < 100_000.0, (
        "AC failure fallback deve addebitare il costo flat bps, non zero — "
        "P0 2026-06-10: il vecchio codice addebitava 0 in caso di failure"
    )
    # Verifica che l'AC model sia stato davvero chiamato (non saltato)
    failing_cost_model.estimate.assert_called_once()


# ---------------------------------------------------------------------------
# P0 2026-06-10 — validazione news corretta (ticker senza aggregator asset-class)
# ---------------------------------------------------------------------------

def test_runner_accepts_ticker_news_without_asset_class_aggregator(tmp_path, prices):
    """Sprint H wiring: news ticker-level senza news_aggregator asset-class (bug live 2026-06-10)."""
    store = ParquetStore(tmp_path)
    broker = PaperBroker(initial_capital=100_000.0, store=store)
    config = DailyRunConfig(assets=ASSETS, macro_series=[], rebalance_freq="ME",
                            send_report=False, min_history=20, history_start="2026-01-01")
    # non deve sollevare
    DailyRunner(
        broker=broker, optimizer=FixedOptimizer(pd.Series({"AAA": 0.5, "BBB": 0.5})),
        risk_filter=NoopRiskFilter(), regime_detector=NoopRegimeDetector(),
        data_loader=FakeLoader(prices), report_builder=NoopReportBuilder(), mailer=None,
        config=config, store=store,
        news_fetcher=MagicMock(), news_scorer=MagicMock(),
        ticker_queries={"AAA": ["query"]}, ticker_aggregator=MagicMock(),
        ticker_bl_optimizer=None,
    )


def test_runner_rejects_asset_class_queries_without_aggregator(tmp_path, prices):
    """asset_class_queries senza news_aggregator deve sollevare ValueError."""
    store = ParquetStore(tmp_path)
    broker = PaperBroker(initial_capital=100_000.0, store=store)
    config = DailyRunConfig(assets=ASSETS, macro_series=[], rebalance_freq="ME",
                            send_report=False, min_history=20, history_start="2026-01-01")
    with pytest.raises(ValueError, match="asset_class_queries"):
        DailyRunner(
            broker=broker, optimizer=FixedOptimizer(pd.Series({"AAA": 0.5, "BBB": 0.5})),
            risk_filter=NoopRiskFilter(), regime_detector=NoopRegimeDetector(),
            data_loader=FakeLoader(prices), report_builder=NoopReportBuilder(), mailer=None,
            config=config, store=store,
            news_fetcher=MagicMock(), news_scorer=MagicMock(),
            asset_class_queries={"equity": ["q"]}, news_aggregator=None,
        )


def test_warns_when_nav_base_date_missing_from_prices(tmp_path, prices, caplog):
    """last_nav_date sparita dall'indice → warning forte, niente perdita silenziosa."""
    import logging as _logging
    runner, broker = _make_runner(tmp_path, prices, cost_bps=0.0)
    runner.run(pd.Timestamp("2026-02-02"))
    # simula cache ricostruita senza i primi giorni: il loader vede solo da metà feb
    runner._data_loader = FakeLoader(prices.loc["2026-02-04":])
    with caplog.at_level(_logging.WARNING):
        runner.run(pd.Timestamp("2026-02-06"))
    assert any("P&L window" in r.message for r in caplog.records)
