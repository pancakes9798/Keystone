"""Idempotenza e date logiche del PaperBroker — regression test per gli store corrotti del 2026-06."""
import pandas as pd
import pytest

from jeanclaude.data.storage.parquet_store import ParquetStore
from jeanclaude.execution.paper import PaperBroker


@pytest.fixture
def broker(tmp_path):
    return PaperBroker(initial_capital=100_000.0, store=ParquetStore(tmp_path))


D1 = pd.Timestamp("2026-06-01")
D2 = pd.Timestamp("2026-06-02")
D3 = pd.Timestamp("2026-06-03")
RETS = pd.Series({"AAA": 0.01, "BBB": -0.005})
PRICES = pd.Series({"AAA": 100.0, "BBB": 50.0})
W = pd.Series({"AAA": 0.6, "BBB": 0.4})


def test_record_daily_nav_same_date_twice_is_idempotent(broker):
    """Il bug che ha corrotto gli store live: doppio run = doppia composizione."""
    result = broker.execute_rebalance(W, PRICES, transaction_cost_bps=0.0, date=D1)
    nav_first = broker.record_daily_nav(RETS, date=D2)
    nav_second = broker.record_daily_nav(RETS, date=D2)
    assert nav_first == pytest.approx(nav_second)

    nav_history = broker._store.load("paper_trading", "nav_history", "state", "state")
    assert nav_history.index.nunique() == len(nav_history)  # nessuna data duplicata
    assert len(nav_history[nav_history.index == D2]) == 1


def test_record_daily_nav_uses_base_strictly_before_date(broker):
    broker.execute_rebalance(W, PRICES, transaction_cost_bps=0.0, date=D1)
    broker.record_daily_nav(pd.Series({"AAA": 0.0, "BBB": 0.0}), date=D1)
    nav_d2 = broker.record_daily_nav(RETS, date=D2)
    expected = 100_000.0 * (1.0 + (0.01 * 0.6 + -0.005 * 0.4))
    assert nav_d2 == pytest.approx(expected)


def test_rerun_on_rebalance_day_uses_old_weights(broker):
    """Re-run del giorno di rebalance: il P&L del giorno va sui pesi VECCHI."""
    broker.execute_rebalance(W, PRICES, transaction_cost_bps=0.0, date=D1)
    broker.record_daily_nav(pd.Series({"AAA": 0.0, "BBB": 0.0}), date=D1)
    # D2: rebalance verso 100% AAA, poi NAV — poi RE-RUN dello stesso giorno
    new_w = pd.Series({"AAA": 1.0, "BBB": 0.0})
    nav_run1 = broker.record_daily_nav(RETS, date=D2)
    broker.execute_rebalance(new_w, PRICES, transaction_cost_bps=0.0, date=D2)
    # re-run identico: positions contiene già i pesi nuovi a D2, ma il P&L di D2
    # deve ancora essere calcolato sui pesi di D1
    nav_run2 = broker.record_daily_nav(RETS, date=D2)
    broker.execute_rebalance(new_w, PRICES, transaction_cost_bps=0.0, date=D2)
    assert nav_run1 == pytest.approx(nav_run2)


def test_execute_rebalance_returns_cost_and_does_not_touch_nav_history(broker):
    result = broker.execute_rebalance(W, PRICES, transaction_cost_bps=10.0, date=D1)
    assert result.orders  # lista ordini non vuota
    # turnover dal portafoglio vuoto = 1.0 → costo = 100k * 1.0 * 10bps
    assert result.cost_eur == pytest.approx(100_000.0 * 1.0 * 10.0 / 10_000)
    nav_history = broker._store.load("paper_trading", "nav_history", "state", "state")
    assert nav_history is None or nav_history.empty


def test_apply_rebalance_cost_is_idempotent_via_record(broker):
    broker.execute_rebalance(W, PRICES, transaction_cost_bps=0.0, date=D1)
    broker.record_daily_nav(pd.Series(0.0, index=W.index), date=D1)
    nav_before = broker.nav
    broker.apply_rebalance_cost(date=D1, cost_eur=100.0)
    assert broker.nav == pytest.approx(nav_before - 100.0)
    # re-run del ciclo: record resetta la riga, il costo viene riapplicato una volta sola
    broker.record_daily_nav(pd.Series(0.0, index=W.index), date=D1)
    broker.apply_rebalance_cost(date=D1, cost_eur=100.0)
    assert broker.nav == pytest.approx(nav_before - 100.0)


def test_orders_upsert_per_date(broker):
    broker.execute_rebalance(W, PRICES, transaction_cost_bps=0.0, date=D1)
    broker.execute_rebalance(W, PRICES, transaction_cost_bps=0.0, date=D1)
    orders = broker._store.load("paper_trading", "orders", "state", "state")
    assert len(orders) == len(W)  # non 2×len(W)


def test_last_nav_date_and_last_rebalance_date(broker):
    assert broker.last_nav_date() is None
    assert broker.last_rebalance_date() is None
    broker.execute_rebalance(W, PRICES, transaction_cost_bps=0.0, date=D1)
    broker.record_daily_nav(RETS, date=D2)
    assert broker.last_rebalance_date() == D1
    assert broker.last_nav_date() == D2


def test_record_daily_nav_raises_on_nan_return_for_invested_asset(broker):
    """NaN su un asset investito = dato rotto a monte: mai inghiottire P&L."""
    broker.execute_rebalance(W, PRICES, transaction_cost_bps=0.0, date=D1)
    broker.record_daily_nav(pd.Series(0.0, index=W.index), date=D1)
    bad = pd.Series({"AAA": float("nan"), "BBB": 0.01})
    with pytest.raises(ValueError, match="AAA"):
        broker.record_daily_nav(bad, date=D2)


def test_record_daily_nav_tolerates_nan_on_zero_weight_asset(broker):
    broker.execute_rebalance(pd.Series({"AAA": 1.0, "BBB": 0.0}), PRICES,
                             transaction_cost_bps=0.0, date=D1)
    broker.record_daily_nav(pd.Series(0.0, index=W.index), date=D1)
    bad = pd.Series({"AAA": 0.01, "BBB": float("nan")})
    nav = broker.record_daily_nav(bad, date=D2)
    assert nav == pytest.approx(100_000.0 * 1.01)


def test_execute_rebalance_rejects_nan_weights(broker):
    bad_w = pd.Series({"AAA": 0.5, "BBB": float("nan")})
    with pytest.raises(ValueError, match="NaN"):
        broker.execute_rebalance(bad_w, PRICES, transaction_cost_bps=0.0, date=D1)


def test_execute_rebalance_rejects_leverage(broker):
    bad_w = pd.Series({"AAA": 0.8, "BBB": 0.4})
    with pytest.raises(ValueError, match="leva"):
        broker.execute_rebalance(bad_w, PRICES, transaction_cost_bps=0.0, date=D1)


def test_multi_day_invariant_nav_compounds_correctly(broker):
    """Invariante contabile: NAV(t) = NAV(t-1) * (1 + r·w) su sequenza di giorni."""
    broker.execute_rebalance(W, PRICES, transaction_cost_bps=0.0, date=D1)
    broker.record_daily_nav(pd.Series(0.0, index=W.index), date=D1)
    expected = 100_000.0
    for i, r in enumerate([0.01, -0.02, 0.003]):
        day = pd.Timestamp("2026-06-02") + pd.Timedelta(days=i)
        rets = pd.Series(r, index=W.index)
        nav = broker.record_daily_nav(rets, date=day)
        expected *= 1.0 + r  # pesi sommano a 1 → return di portafoglio = r
        assert nav == pytest.approx(expected)
