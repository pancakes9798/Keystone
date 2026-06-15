"""Execution lag: segnale a t, esecuzione a t+lag — niente ritorno del giorno del segnale ai pesi nuovi.

Fixture design
--------------
- Dati dal 2019-11-01 al 2020-06-30, min_history=45 → OOS a partire da ~2019-12-31.
- Primo rebalance (segnale ME): fine dicembre 2019 → SwitchOptimizer call 1 → A=1.0, B=0.0.
- Secondo rebalance: fine gennaio 2020 → call 2 → A=0.0, B=1.0.
- B ha ritorno costante +5% ogni giorno di febbraio.
- Con lag=1: il segnale "B=1.0" è calcolato il 31/01 ma eseguito il 03/02 (primo bday feb).
  - Il 31/01 il portafoglio è ancora 100% A → ritorno piccolo (zero noise).
  - Il 03/02 il portafoglio è 100% B → ritorno = 0.05.
- Con lag=0: il segnale "B=1.0" è eseguito immediatamente il 31/01 → il 31/01 è già in B,
  ma il 31/01 B rende 0.0 (non è febbraio) → il 03/02 B rende 0.05.
"""
import numpy as np
import pandas as pd
import pytest

from jeanclaude.backtest.engine import BacktestConfig, BacktestEngine


class SwitchOptimizer:
    """Prima chiamata: tutto su A. Seconda: tutto su B. Per tracciare l'attribuzione."""

    def __init__(self):
        self.calls = 0

    def optimize(self, returns):
        self.calls += 1
        if self.calls == 1:
            return pd.Series({"A": 1.0, "B": 0.0})
        return pd.Series({"A": 0.0, "B": 1.0})


@pytest.fixture
def returns():
    # Start Nov 2019; min_history=22 → OOS starts 2019-12-03.
    # First ME rebalance: 2019-12-31 (call 1 → A=1.0, B=0.0).
    # Second ME rebalance: 2020-01-31 (call 2 → A=0.0, B=1.0).
    # B ha ritorno +5% ogni giorno di febbraio.
    # Con lag=1: esecuzione B=1.0 il 2020-02-03 (primo bday dopo 2020-01-31).
    idx = pd.bdate_range("2019-11-01", "2020-06-30")
    rng = np.random.default_rng(7)
    a = pd.Series(rng.normal(0.0, 0.001, len(idx)), index=idx)
    # B ha un ritorno enorme riconoscibile in ogni giorno di febbraio
    b = pd.Series(0.0, index=idx)
    b.loc["2020-02"] = 0.05
    return pd.DataFrame({"A": a, "B": b})


def _run(returns, lag):
    config = BacktestConfig(
        rebalance_freq="ME", transaction_cost_bps=0.0,
        min_history=22, execution_lag=lag,
    )
    return BacktestEngine(SwitchOptimizer(), config=config).run(returns)


def test_lag1_rebalance_day_return_on_old_weights(returns):
    """Con lag=1, il giorno del 2° rebalance (fine gen) il portafoglio è ancora su A."""
    result = _run(returns, lag=1)
    daily = result.portfolio_returns  # serie dei ritorni giornalieri del portafoglio
    # il segnale B=1.0 arriva il 31/01 ma viene eseguito il 03/02 con lag=1
    rebal_jan = daily.loc["2020-01"].index[-1]
    first_feb = daily.loc["2020-02"].index[0]
    # il giorno del segnale (ultimo bday di gennaio): portafoglio ancora su A → ritorno piccolo
    assert abs(daily.loc[rebal_jan]) < 0.01
    # dal primo giorno di febbraio in poi: portafoglio su B → ritorno = 0.05
    assert daily.loc[first_feb] == pytest.approx(0.05, rel=1e-6)


def test_lag0_reproduces_legacy_same_day_execution(returns):
    """lag=0 mantiene il comportamento storico (pesi nuovi dal giorno del segnale).

    Con lag=0, il segnale B=1.0 viene applicato SUBITO il 31/01.
    Il 31/01 B rende 0.0 (il +5% è solo in febbraio), ma i pesi nuovi sono GIA' loggati
    alla data del segnale — differenza chiave vs lag=1 dove sarebbero loggati il 03/02.
    """
    result = _run(returns, lag=0)
    rebal_jan = result.portfolio_returns.loc["2020-01"].index[-1]
    # con lag=0 la data del segnale e' nell'indice dei pesi con il secondo segnale (B=1.0)
    assert rebal_jan in result.weights_history.index
    assert result.weights_history.loc[rebal_jan, "B"] == pytest.approx(1.0)


def test_lag1_weights_log_effective_date(returns):
    """Il log dei pesi riflette la data di EFFICACIA, non quella del segnale."""
    result = _run(returns, lag=1)
    rebal_jan = result.portfolio_returns.loc["2020-01"].index[-1]
    first_feb = result.portfolio_returns.loc["2020-02"].index[0]
    # il 31/01 (giorno del segnale) NON è nell'indice dei pesi — o ha ancora i pesi vecchi
    # il 03/02 (giorno dell'esecuzione) ha i nuovi pesi B=1.0
    assert result.weights_history.loc[first_feb, "B"] == pytest.approx(1.0)
    # il 31/01 non deve avere B=1.0 nei pesi effettivi
    if rebal_jan in result.weights_history.index:
        assert result.weights_history.loc[rebal_jan, "B"] == pytest.approx(0.0)


def test_default_lag_is_one():
    assert BacktestConfig().execution_lag == 1


def test_no_starvation_when_lag_exceeds_rebalance_interval(returns):
    """lag > intervallo di rebalance: si esegue comunque (il pending non viene mai resettato).

    Con freq="W" e lag=10 ogni segnale settimanale sostituisce quello pendente prima che
    il countdown scada — prima del fix il countdown veniva resettato ad ogni nuovo segnale
    e il portafoglio rimaneva piatto per tutta la simulazione.
    """
    import warnings as _w
    config = BacktestConfig(
        rebalance_freq="W",
        transaction_cost_bps=0.0,
        min_history=10,
        execution_lag=10,
    )
    with _w.catch_warnings():
        _w.simplefilter("ignore", UserWarning)
        result = BacktestEngine(SwitchOptimizer(), config=config).run(returns)
    # il portafoglio DEVE aver tradato: pesi non tutti zero a fine serie
    final_w = result.weights_history.iloc[-1]
    assert final_w.abs().sum() > 0.0
