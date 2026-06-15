"""Tests for PaperBroker."""
from __future__ import annotations

import pandas as pd
import pytest

from jeanclaude.data.storage.parquet_store import ParquetStore
from jeanclaude.execution.paper.broker import Order, PortfolioState, PaperBroker


@pytest.fixture
def store(tmp_path):
    return ParquetStore(tmp_path)


@pytest.fixture
def broker(store):
    return PaperBroker(initial_capital=100_000.0, store=store)


def test_broker_initial_state_nav(broker):
    state = broker.get_state()
    assert state.nav == pytest.approx(100_000.0)


def test_broker_initial_state_empty_weights(broker):
    state = broker.get_state()
    assert len(state.weights) == 0


def test_broker_initial_state_cash_equals_capital(broker):
    state = broker.get_state()
    assert state.cash == pytest.approx(100_000.0)


def test_execute_rebalance_updates_weights(broker):
    prices = pd.Series({"SPY": 400.0, "TLT": 95.0, "GLD": 180.0})
    new_weights = pd.Series({"SPY": 0.5, "TLT": 0.3, "GLD": 0.2})
    broker.execute_rebalance(new_weights, prices, transaction_cost_bps=10.0)
    state = broker.get_state()
    assert state.weights["SPY"] == pytest.approx(0.5)
    assert state.weights["TLT"] == pytest.approx(0.3)
    assert state.weights["GLD"] == pytest.approx(0.2)


def test_execute_rebalance_cash_is_unallocated(broker):
    prices = pd.Series({"SPY": 400.0, "TLT": 95.0})
    # weights sum to 0.7 → cash = 30% of NAV
    new_weights = pd.Series({"SPY": 0.5, "TLT": 0.2})
    broker.execute_rebalance(new_weights, prices)
    state = broker.get_state()
    assert state.cash == pytest.approx(state.nav * 0.3, rel=1e-3)


def test_execute_rebalance_returns_orders(broker):
    prices = pd.Series({"SPY": 400.0, "TLT": 95.0})
    new_weights = pd.Series({"SPY": 0.6, "TLT": 0.4})
    # execute_rebalance now returns RebalanceResult; access .orders for the list
    result = broker.execute_rebalance(new_weights, prices)
    orders = result.orders
    assert len(orders) == 2
    assert all(isinstance(o, Order) for o in orders)
    spy_order = next(o for o in orders if o.asset == "SPY")
    assert spy_order.old_weight == pytest.approx(0.0)
    assert spy_order.new_weight == pytest.approx(0.6)


def test_execute_rebalance_logs_orders_to_parquet(store, broker):
    prices = pd.Series({"SPY": 400.0})
    broker.execute_rebalance(pd.Series({"SPY": 1.0}), prices)
    orders_df = store.load("paper_trading", "orders", "state", "state")
    assert orders_df is not None
    assert len(orders_df) == 1
    assert "asset" in orders_df.columns


def test_record_daily_nav_no_position(broker):
    # With no position, NAV should remain unchanged
    daily_returns = pd.Series({"SPY": 0.01, "TLT": -0.005})
    nav = broker.record_daily_nav(daily_returns)
    assert nav == pytest.approx(100_000.0)


def test_record_daily_nav_with_position(store, broker):
    # Rebalance first: 60% SPY, 40% TLT — use explicit date so weights are
    # visible to record_daily_nav on a strictly later date (upsert contract).
    prices = pd.Series({"SPY": 400.0, "TLT": 95.0})
    d_rebalance = pd.Timestamp("2026-01-01")
    d_nav = pd.Timestamp("2026-01-02")
    broker.execute_rebalance(pd.Series({"SPY": 0.6, "TLT": 0.4}), prices, date=d_rebalance)
    initial_nav = broker.nav  # 100_000 (no TC)

    # SPY +1%, TLT -0.5%
    daily_returns = pd.Series({"SPY": 0.01, "TLT": -0.005})
    nav = broker.record_daily_nav(daily_returns, date=d_nav)
    expected_return = 0.6 * 0.01 + 0.4 * (-0.005)
    assert nav == pytest.approx(initial_nav * (1 + expected_return), rel=1e-4)


def test_record_daily_nav_persists_to_parquet(store, broker):
    broker.record_daily_nav(pd.Series({"SPY": 0.0}))
    nav_df = store.load("paper_trading", "nav_history", "state", "state")
    assert nav_df is not None
    assert "nav" in nav_df.columns
    assert len(nav_df) == 1


from unittest.mock import MagicMock, patch
import numpy as np
from jeanclaude.execution.paper.runner import DailyRunConfig, DailyRunResult, DailyRunner
from jeanclaude.signals.macro.labels import RegimeLabel


def _make_runner(store, send_report=False):
    """Build a DailyRunner with mocked dependencies."""
    broker = PaperBroker(initial_capital=100_000.0, store=store)

    optimizer = MagicMock()
    optimizer.optimize.return_value = pd.Series({"SPY": 0.6, "TLT": 0.4})

    risk_filter = MagicMock()
    risk_filter.apply.return_value = pd.Series({"SPY": 0.6, "TLT": 0.4})

    regime_detector = MagicMock()
    regime_detector.current_regime.return_value = (RegimeLabel.EXPANSION, np.array([0.8, 0.1, 0.1]))

    # 65 business days so min_history=60 is satisfied
    dates = pd.bdate_range("2025-01-01", periods=65)
    rng = np.random.default_rng(42)
    prices = pd.DataFrame(
        rng.lognormal(size=(65, 2)) * 100,
        index=dates,
        columns=["SPY", "TLT"],
    )
    data_loader = MagicMock()
    data_loader.get_prices.return_value = prices
    data_loader.get_returns.return_value = prices.pct_change().dropna()
    data_loader.get_macro.return_value = pd.DataFrame(
        np.random.default_rng(0).normal(size=(65, 2)),
        index=dates,
        columns=["VIX", "TEDRATE"],
    )

    report_builder = MagicMock()
    report_builder.build.return_value = "<html>test</html>"

    mailer = MagicMock() if send_report else None

    config = DailyRunConfig(
        assets=["SPY", "TLT"],
        macro_series=["VIX", "TEDRATE"],
        rebalance_freq="W",
        send_report=send_report,
        report_recipients=["test@example.com"],
        min_history=60,
        history_start="2025-01-01",
    )
    return DailyRunner(
        broker=broker,
        optimizer=optimizer,
        risk_filter=risk_filter,
        regime_detector=regime_detector,
        data_loader=data_loader,
        report_builder=report_builder,
        mailer=mailer,
        config=config,
    )


def test_daily_runner_returns_result(store):
    runner = _make_runner(store)
    result = runner.run()
    assert isinstance(result, DailyRunResult)
    assert result.nav > 0


def test_daily_runner_non_rebalance_day_skips_optimizer(store):
    # P0 2026-06-10: rebalance avviene al PRIMO run dopo che un periodo è completato.
    # Con la nuova semantica "periodo completato", il run di lunedì 2025-01-06 è
    # l'allocazione iniziale (last_rebalance=None → due=True). Un secondo run lo stesso
    # giorno è bloccato dall'idempotency guard. Il giorno successivo (martedì 2025-01-07)
    # non ha periodi completati da ribilanciare → optimizer non chiamato.
    runner = _make_runner(store)
    dates = pd.bdate_range("2025-01-06", periods=65)  # starts Monday
    rng = np.random.default_rng(1)
    prices = pd.DataFrame(
        rng.lognormal(size=(65, 2)) * 100,
        index=dates,
        columns=["SPY", "TLT"],
    )
    runner._data_loader.get_prices.return_value = prices
    runner._data_loader.get_returns.return_value = prices.pct_change().dropna()
    # Prima allocazione il lunedì (consuma il "last_rebalance=None" trigger)
    runner.run(date=pd.Timestamp("2025-01-06"))
    runner._optimizer.optimize.reset_mock()
    # Martedì: la settimana Jan 6-10 non è ancora completata → no rebalance
    result = runner.run(date=pd.Timestamp("2025-01-07"))
    runner._optimizer.optimize.assert_not_called()
    assert result.rebalanced is False


def test_daily_runner_rebalance_day_calls_optimizer(store):
    # P0 2026-06-10: con la nuova schedulazione "periodo completato", il rebalance
    # si innesca al primo run dopo che il periodo precedente è chiuso. La settimana
    # Jan 6-10 si chiude venerdì 10; il primo run utile è lunedì 13 gennaio.
    runner = _make_runner(store)
    dates = pd.bdate_range("2025-01-06", periods=65)  # starts Monday Jan 6
    rng = np.random.default_rng(2)
    prices = pd.DataFrame(
        rng.lognormal(size=(65, 2)) * 100,
        index=dates,
        columns=["SPY", "TLT"],
    )
    runner._data_loader.get_prices.return_value = prices
    runner._data_loader.get_returns.return_value = prices.pct_change().dropna()
    # Prima allocazione su Jan 6 (consuma il trigger iniziale, min_history=60 non serve)
    runner.run(date=pd.Timestamp("2025-01-06"))
    runner._optimizer.optimize.reset_mock()
    # 2025-03-31 (lunedì): primo run dopo la fine della settimana Mar 24-28 con ≥60 ritorni.
    # P0 2026-06-10: rebalance avviene al primo run con periodo completato E storia sufficiente.
    rebalance_day = pd.Timestamp("2025-03-31")
    result = runner.run(date=rebalance_day)
    runner._optimizer.optimize.assert_called_once()
    assert result.rebalanced is True


def test_daily_runner_sends_report_when_configured(store):
    runner = _make_runner(store, send_report=True)
    runner.run()
    runner._report_builder.build.assert_called_once()
    runner._mailer.send.assert_called_once()


def test_daily_runner_no_report_when_mailer_is_none(store):
    runner = _make_runner(store, send_report=False)
    runner.run()
    runner._report_builder.build.assert_not_called()


# ---------------------------------------------------------------------------
# BL wiring tests
# ---------------------------------------------------------------------------

from jeanclaude.signals.composite.types import CompositeSignal, DEFAULT_REGIME_TABLE
from jeanclaude.signals.macro.labels import RegimeLabel, RegimeResult


def _make_regime_result(dates):
    """Minimal RegimeResult with all EXPANSION labels and high confidence."""
    labels = pd.Series(
        [RegimeLabel.EXPANSION.value] * len(dates),
        index=dates,
        dtype="int64",
    )
    probabilities = pd.DataFrame(
        {"0": 0.9, "1": 0.05, "2": 0.05},
        index=dates,
    ).rename(columns={"0": 0, "1": 1, "2": 2})
    return RegimeResult(labels=labels, probabilities=probabilities, model=MagicMock())


def _make_composite_signal(dates):
    return CompositeSignal(
        scores=pd.Series({"equity": 0.8, "bonds": -0.3, "gold": 0.1, "cash": -0.2}),
        confidence=0.9,
        regime=RegimeLabel.EXPANSION,
        regime_proba=np.array([0.9, 0.05, 0.05]),
        timestamp=dates[-1],
    )


def _make_bl_runner(store):
    """Build a DailyRunner wired with BL (signal_builder + prior_optimizer)."""
    broker = PaperBroker(initial_capital=100_000.0, store=store)

    dates = pd.bdate_range("2025-01-06", periods=65)
    rng = np.random.default_rng(7)
    prices = pd.DataFrame(
        rng.lognormal(size=(65, 2)) * 100,
        index=dates,
        columns=["SPY", "TLT"],
    )
    returns = prices.pct_change().dropna()

    regime_result = _make_regime_result(dates)
    signal = _make_composite_signal(dates)

    # Regime detector: fit() returns a RegimeResult
    regime_detector = MagicMock()
    regime_detector.fit.return_value = regime_result

    # Signal builder: build() returns a CompositeSignal
    signal_builder = MagicMock()
    signal_builder.build.return_value = signal

    # Prior optimizer: returns equal-weight
    prior_optimizer = MagicMock()
    prior_optimizer.optimize.return_value = pd.Series({"SPY": 0.5, "TLT": 0.5})

    # BL optimizer: fit() + optimize() — both mocked
    bl_optimizer = MagicMock()
    bl_optimizer.fit.return_value = bl_optimizer
    bl_optimizer.optimize.return_value = pd.Series({"SPY": 0.65, "TLT": 0.35})

    risk_filter = MagicMock()
    risk_filter.apply.return_value = pd.Series({"SPY": 0.65, "TLT": 0.35})

    data_loader = MagicMock()
    data_loader.get_prices.return_value = prices
    data_loader.get_returns.return_value = returns
    data_loader.get_macro.return_value = pd.DataFrame(
        rng.normal(size=(65, 2)),
        index=dates,
        columns=["VIX", "TEDRATE"],
    )

    report_builder = MagicMock()
    report_builder.build.return_value = "<html>test</html>"

    config = DailyRunConfig(
        assets=["SPY", "TLT"],
        macro_series=["VIX", "TEDRATE"],
        rebalance_freq="W",
        send_report=False,
        min_history=60,
        history_start="2025-01-01",
    )

    return DailyRunner(
        broker=broker,
        optimizer=bl_optimizer,
        risk_filter=risk_filter,
        regime_detector=regime_detector,
        data_loader=data_loader,
        report_builder=report_builder,
        mailer=None,
        config=config,
        signal_builder=signal_builder,
        prior_optimizer=prior_optimizer,
    )


def test_bl_runner_calls_regime_detector_fit(store):
    """On a rebalance day, regime_detector.fit() must be called (not current_regime).
    P0 2026-06-10: rebalance avviene al primo run con periodo completato E ≥60 ritorni;
    con dati da Jan 6 (65 giorni), il primo giorno valido è 2025-03-31 (lunedì)."""
    runner = _make_bl_runner(store)
    # Prima allocazione (consuma il trigger iniziale last_rebalance=None)
    runner.run(date=pd.Timestamp("2025-01-06"))
    runner._regime_detector.fit.reset_mock()
    runner._regime_detector.current_regime.reset_mock()
    # Primo run con ≥60 ritorni e periodo completato
    runner.run(date=pd.Timestamp("2025-03-31"))
    runner._regime_detector.fit.assert_called_once()
    runner._regime_detector.current_regime.assert_not_called()


def test_bl_runner_calls_signal_builder(store):
    """signal_builder.build() must be called with the RegimeResult from detector.fit().
    P0 2026-06-10: data spostata a 2025-03-31 (primo run valido post-storia sufficiente)."""
    runner = _make_bl_runner(store)
    runner.run(date=pd.Timestamp("2025-01-06"))
    runner._signal_builder.build.reset_mock()
    runner.run(date=pd.Timestamp("2025-03-31"))
    runner._signal_builder.build.assert_called_once()


def test_bl_runner_calls_prior_optimizer(store):
    """prior_optimizer.optimize() must be called to produce BL prior weights.
    P0 2026-06-10: data spostata a 2025-03-31."""
    runner = _make_bl_runner(store)
    runner.run(date=pd.Timestamp("2025-01-06"))
    runner._prior_optimizer.optimize.reset_mock()
    runner.run(date=pd.Timestamp("2025-03-31"))
    runner._prior_optimizer.optimize.assert_called_once()


def test_bl_runner_calls_bl_fit_and_optimize(store):
    """BLOptimizer.fit() and .optimize() must both be called on rebalance.
    P0 2026-06-10: data spostata a 2025-03-31."""
    runner = _make_bl_runner(store)
    runner.run(date=pd.Timestamp("2025-01-06"))
    runner._optimizer.fit.reset_mock()
    runner._optimizer.optimize.reset_mock()
    runner.run(date=pd.Timestamp("2025-03-31"))
    runner._optimizer.fit.assert_called_once()
    runner._optimizer.optimize.assert_called_once()


def test_bl_runner_uses_bl_weights(store):
    """The weights written to broker must come from BLOptimizer, not prior.
    P0 2026-06-10: data spostata a 2025-03-31."""
    runner = _make_bl_runner(store)
    runner.run(date=pd.Timestamp("2025-01-06"))
    result = runner.run(date=pd.Timestamp("2025-03-31"))
    assert result.rebalanced is True
    # risk_filter passes BL weights through unchanged in this mock
    state = runner._broker.get_state()
    assert state.weights["SPY"] == pytest.approx(0.65)
    assert state.weights["TLT"] == pytest.approx(0.35)


def test_bl_runner_non_rebalance_day_skips_bl(store):
    """On a non-rebalance day, neither fit() nor optimize() should be called.
    P0 2026-06-10: eseguiamo prima l'allocazione iniziale (Jan 6), poi il martedì
    (Jan 7) non ha periodi completati da ribilanciare."""
    runner = _make_bl_runner(store)
    # Prima allocazione (consuma il trigger last_rebalance=None)
    runner.run(date=pd.Timestamp("2025-01-06"))
    runner._optimizer.fit.reset_mock()
    runner._optimizer.optimize.reset_mock()
    # Martedì 7 gennaio: la settimana Jan 6-10 non è ancora completata → no rebalance
    runner.run(date=pd.Timestamp("2025-01-07"))
    runner._optimizer.fit.assert_not_called()
    runner._optimizer.optimize.assert_not_called()


def test_bl_runner_fallback_on_bl_failure(store):
    """When BLOptimizer.fit() raises, prior weights are used as fallback.
    P0 2026-06-10: data spostata a 2025-03-31 (primo run con ≥60 ritorni)."""
    runner = _make_bl_runner(store)
    runner._optimizer.fit.side_effect = ValueError("not enough obs per regime")
    # Prima allocazione (Jan 6) — il fallback usa prior_optimizer o equal-weight
    runner.run(date=pd.Timestamp("2025-01-06"))
    runner._prior_optimizer.optimize.reset_mock()
    result = runner.run(date=pd.Timestamp("2025-03-31"))
    # Should still rebalance (with prior weights from prior_optimizer)
    assert result.rebalanced is True
    runner._prior_optimizer.optimize.assert_called_once()


def test_align_labels_fills_gaps():
    """_align_labels should forward-fill macro labels onto returns index."""
    from jeanclaude.execution.paper.runner import DailyRunner

    # Macro: weekly labels
    macro_dates = pd.date_range("2025-01-06", periods=4, freq="W-FRI")
    labels = pd.Series(
        [RegimeLabel.EXPANSION.value] * 4,
        index=macro_dates,
        dtype="int64",
    )

    # Returns: daily
    returns_index = pd.bdate_range("2025-01-06", periods=20)

    aligned = DailyRunner._align_labels(labels, returns_index)

    # All days after first macro date should have a label
    assert not aligned.empty
    assert aligned.isna().sum() == 0
    assert all(v == RegimeLabel.EXPANSION.value for v in aligned.values)


# ---------------------------------------------------------------------------
# Task 7 — DailyRunner news integration
# ---------------------------------------------------------------------------

def _make_news_runner(store):
    """DailyRunner con news_fetcher + news_scorer + news_aggregator mockati."""
    dates = pd.bdate_range("2025-01-06", periods=65)
    rng = np.random.default_rng(99)
    prices = pd.DataFrame(
        rng.lognormal(size=(65, 2)) * 100,
        index=dates,
        columns=["SPY", "TLT"],
    )
    returns = prices.pct_change().dropna()

    from jeanclaude.signals.macro.labels import RegimeLabel, RegimeResult
    from jeanclaude.signals.composite.types import CompositeSignal

    regime_result = MagicMock()
    regime_result.labels = pd.Series(
        [RegimeLabel.EXPANSION.value] * len(dates), index=dates, dtype="int64"
    )
    regime_result.probabilities = pd.DataFrame(
        {0: 0.9, 1: 0.05, 2: 0.05}, index=dates
    )

    regime_detector = MagicMock()
    regime_detector.fit.return_value = regime_result

    signal_builder = MagicMock()
    signal_builder.build.return_value = CompositeSignal(
        scores=pd.Series({"equity": 0.8, "bonds": -0.3, "gold": 0.1, "cash": -0.2}),
        confidence=0.9,
        regime=RegimeLabel.EXPANSION,
        regime_proba=np.array([0.9, 0.05, 0.05]),
        timestamp=dates[-1],
    )

    bl_optimizer = MagicMock()
    bl_optimizer.fit.return_value = bl_optimizer
    bl_optimizer.optimize.return_value = pd.Series({"SPY": 0.65, "TLT": 0.35})

    risk_filter = MagicMock()
    risk_filter.apply.return_value = pd.Series({"SPY": 0.65, "TLT": 0.35})

    # News mocks
    news_fetcher = MagicMock()
    news_fetcher.get_headlines.return_value = pd.DataFrame(
        {"asset_class": ["equity"], "headline": ["Markets up"], "story_id": ["x1"]},
        index=pd.DatetimeIndex(["2025-01-10"], name="date"),
    )

    news_scorer = MagicMock()
    news_scorer.score.return_value = pd.DataFrame(
        {
            "asset_class": ["equity"],
            "headline": ["Markets up"],
            "story_id": ["x1"],
            "sentiment_score": [1.0],
        },
        index=pd.DatetimeIndex(["2025-01-10"], name="date"),
    )

    from jeanclaude.signals.news.types import NewsSentiment
    news_aggregator = MagicMock()
    news_aggregator.aggregate.return_value = NewsSentiment(
        scores=pd.Series({"equity": 0.8}),
        confidence=0.9,
        timestamp=pd.Timestamp("2025-01-10"),
        n_articles=45,
    )

    data_loader = MagicMock()
    data_loader.get_prices.return_value = prices
    data_loader.get_returns.return_value = returns
    data_loader.get_macro.return_value = pd.DataFrame(
        np.random.default_rng(0).normal(size=(65, 2)),
        index=dates,
        columns=["VIX", "TEDRATE"],
    )

    report_builder = MagicMock()
    report_builder.build.return_value = "<html>test</html>"

    broker = PaperBroker(initial_capital=100_000.0, store=store)

    config = DailyRunConfig(
        assets=["SPY", "TLT"],
        macro_series=["VIX", "TEDRATE"],
        rebalance_freq="W",
        send_report=False,
        min_history=60,
        history_start="2025-01-01",
    )

    return DailyRunner(
        broker=broker,
        optimizer=bl_optimizer,
        risk_filter=risk_filter,
        regime_detector=regime_detector,
        data_loader=data_loader,
        report_builder=report_builder,
        mailer=None,
        config=config,
        signal_builder=signal_builder,
        prior_optimizer=None,
        news_fetcher=news_fetcher,
        news_scorer=news_scorer,
        news_aggregator=news_aggregator,
        asset_class_queries={"equity": ["SPY"], "bonds": ["TLT"]},
        store=store,
    )


def test_news_runner_fetches_and_scores_headlines_daily(store):
    runner = _make_news_runner(store)
    friday = pd.Timestamp("2025-01-10")
    runner.run(date=friday)
    runner._news_fetcher.get_headlines.assert_called_once()
    runner._news_scorer.score.assert_called_once()


def test_news_runner_persists_headlines_to_parquet(store):
    runner = _make_news_runner(store)
    friday = pd.Timestamp("2025-01-10")
    runner.run(date=friday)
    saved = store.load("paper_trading", "headlines", "all", "all")
    assert saved is not None
    assert "sentiment_score" in saved.columns


def test_news_runner_aggregates_at_rebalance_day(store):
    # P0 2026-06-10: rebalance al primo run con periodo completato E ≥60 ritorni.
    # Prima allocazione (Jan 6), poi 2025-03-31 è il primo rebalance valido.
    runner = _make_news_runner(store)
    runner.run(date=pd.Timestamp("2025-01-06"))
    runner._news_aggregator.aggregate.reset_mock()
    runner.run(date=pd.Timestamp("2025-03-31"))
    runner._news_aggregator.aggregate.assert_called_once()


def test_news_runner_does_not_aggregate_on_non_rebalance_day(store):
    # P0 2026-06-10: dopo l'allocazione iniziale (Jan 6), il martedì 7 non ribilancia.
    runner = _make_news_runner(store)
    runner.run(date=pd.Timestamp("2025-01-06"))
    runner._news_aggregator.aggregate.reset_mock()
    runner.run(date=pd.Timestamp("2025-01-07"))
    runner._news_aggregator.aggregate.assert_not_called()


def test_news_runner_passes_news_sentiment_to_signal_builder(store):
    # P0 2026-06-10: data spostata a 2025-03-31 (primo rebalance con ≥60 ritorni).
    runner = _make_news_runner(store)
    runner.run(date=pd.Timestamp("2025-01-06"))
    runner._signal_builder.build.reset_mock()
    runner.run(date=pd.Timestamp("2025-03-31"))
    # signal_builder.build deve essere chiamato con news_sentiment keyword arg
    call_kwargs = runner._signal_builder.build.call_args
    assert "news_sentiment" in call_kwargs.kwargs
    assert call_kwargs.kwargs["news_sentiment"] is not None


def test_news_runner_graceful_degradation_on_fetch_failure(store):
    """Se il fetch delle news fallisce, il runner continua senza bloccarsi."""
    runner = _make_news_runner(store)
    runner._news_fetcher.get_headlines.side_effect = ConnectionError("API unavailable")
    friday = pd.Timestamp("2025-01-10")
    result = runner.run(date=friday)
    # Il runner deve completare normalmente
    assert result.nav > 0
