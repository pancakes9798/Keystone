"""Schedulazione dei rebalance: logica 'periodo completato' con catch-up.

Sostituisce il vecchio _is_rebalance_day, che aveva due bug:
- con dati aggiornati a oggi, l'ultimo giorno del periodo PARZIALE corrente
  coincide sempre con oggi → rebalance ogni giorno;
- con dati in ritardo (oggi non nell'indice), non ribilanciava mai, e un
  month-end mancato slittava di un intero periodo.

Regola: si ribilancia al primo run in cui esiste un fine-periodo COMPLETATO
successivo all'ultimo rebalance. Il rebalance avviene quindi al primo giorno
di trading del periodo nuovo, usando dati fino a ieri sera — più realistico
del vecchio 'decidi ed esegui sulla stessa chiusura'.
"""
from __future__ import annotations

import pandas as pd

_FREQ_TO_PERIOD = {"ME": "M", "M": "M", "W": "W", "QE": "Q", "Q": "Q"}


def completed_period_ends(
    asof: pd.Timestamp, price_index: pd.DatetimeIndex, freq: str
) -> pd.DatetimeIndex:
    """Date di fine-periodo completate rispetto ad ``asof``.

    Una data è un fine-periodo completato se è l'ultimo giorno di trading di
    un periodo strettamente precedente al periodo di ``asof``.
    """
    period_freq = _FREQ_TO_PERIOD.get(freq, freq)
    past = price_index[price_index <= asof]
    if len(past) == 0:
        return pd.DatetimeIndex([])
    periods = past.to_period(period_freq)
    current_period = asof.to_period(period_freq)
    df = pd.DataFrame({"date": past, "period": periods})
    ends = df.groupby("period")["date"].max()
    ends = ends[ends.index < current_period]
    return pd.DatetimeIndex(ends.values)


def rebalance_due(
    asof: pd.Timestamp,
    price_index: pd.DatetimeIndex,
    freq: str,
    last_rebalance: pd.Timestamp | None,
) -> bool:
    """True se al run di ``asof`` è dovuto un rebalance.

    - ``last_rebalance is None`` → allocazione iniziale: sempre due.
    - Altrimenti: due se esiste un fine-periodo completato > last_rebalance.
      Recupera automaticamente i fine-periodo mancati (cron fermo).
    """
    if last_rebalance is None:
        return True
    ends = completed_period_ends(asof, price_index, freq)
    if len(ends) == 0:
        return False
    return bool(ends.max() > last_rebalance.normalize())
