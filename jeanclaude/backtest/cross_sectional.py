"""Cross-sectional monthly backtest on a point-in-time universe.

Pure functions — no network.  The caller supplies the daily total-return
matrix (date x RIC, fractional returns) and the point-in-time baskets
(formation month-end -> frozenset of member RICs, from
``jeanclaude.data.constituents.ConstituentHistory``).

Conventions (Sprint O frozen spec,
``docs/superpowers/plans/2026-06-12-sprint-o-momentum-pit.md``):

- Signal at formation T uses ONLY data with index <= T.
- Momentum 12-1: compound return over the last 252 trading days up to T,
  excluding the most recent 21 (short-term reversal skip).
- Execution lag t+1: the first trading day of the holding month is NOT
  accrued (position assumed entered at that day's close).
- Delisted mid-month: available daily returns are compounded, missing
  days count as cash — the partial-month P&L IS credited (Lehman's
  September 2008 must hit the portfolio).
"""
from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)

LOOKBACK_DAYS = 252   # finestra totale (giorni di borsa)
SKIP_DAYS = 21        # mese più recente escluso (reversal)
MIN_OBS = 200         # ritorni validi minimi nella finestra di 231 giorni


def momentum_12_1(
    daily_returns: pd.DataFrame,
    asof: pd.Timestamp,
    lookback: int = LOOKBACK_DAYS,
    skip: int = SKIP_DAYS,
    min_obs: int = MIN_OBS,
) -> pd.Series:
    """Momentum 12-1: compound return over the (T-252, T-21] trading-day window.

    Uses ONLY rows with index <= asof.  RICs with fewer than ``min_obs``
    valid returns in the window get NaN (not eligible).

    Parameters
    ----------
    daily_returns : pd.DataFrame
        Daily fractional total returns, index = trading dates, columns = RICs.
    asof : pd.Timestamp
        Formation date — no data after this date is used.
    lookback, skip, min_obs : int
        Window length, recent-days skip, minimum valid observations.

    Returns
    -------
    pd.Series
        Signal per RIC; NaN where not eligible.
    """
    window = daily_returns.loc[:asof].iloc[-lookback:-skip]
    valid = window.notna().sum()
    mom = (1.0 + window).prod(skipna=True) - 1.0
    return mom.where(valid >= min_obs)


def realized_volatility(
    daily_returns: pd.DataFrame,
    asof: pd.Timestamp,
    lookback: int = LOOKBACK_DAYS,
    min_obs: int = MIN_OBS,
) -> pd.Series:
    """Volatilità realizzata: dev.std dei ritorni nella finestra (T-lookback, T], T=asof.

    Usa SOLO righe con index <= asof.  NESSUNO skip (a differenza del momentum:
    per la volatilità non c'è effetto reversal da escludere).  RIC con meno di
    ``min_obs`` ritorni validi nella finestra ottengono NaN (non eleggibili).

    La strategia low-vol usa ``-realized_volatility`` come punteggio di bontà
    (più alto = meno volatile), così riusa :func:`top_decile_weights` (nlargest)
    senza una direzione di sort separata.

    Parameters
    ----------
    daily_returns : pd.DataFrame
        Ritorni frazionari giornalieri, index = date di borsa, colonne = RIC.
    asof : pd.Timestamp
        Data di formation — nessun dato dopo questa data viene usato.
    lookback, min_obs : int
        Lunghezza finestra e osservazioni valide minime.

    Returns
    -------
    pd.Series
        Volatilità per RIC; NaN dove non eleggibile.
    """
    window = daily_returns.loc[:asof].iloc[-lookback:]
    valid = window.notna().sum()
    vol = window.std()  # ddof=1, skipna per colonna
    return vol.where(valid >= min_obs)


def top_decile_weights(
    signal: pd.Series,
    decile: float = 0.10,
    min_names: int = 20,
) -> pd.Series:
    """Equal-weight long-only top decile of the eligible (non-NaN) signal.

    Returns an EMPTY series (cash month) if the decile has fewer than
    ``min_names`` names — flagged upstream, never silent.
    """
    eligible = signal.dropna()
    n_top = math.ceil(len(eligible) * decile)
    if n_top < min_names:
        return pd.Series(dtype=float)
    top = eligible.nlargest(n_top)
    return pd.Series(1.0 / n_top, index=top.index)


@dataclass(frozen=True)
class CrossSectionalResult:
    """Output of :func:`simulate_monthly`.

    Attributes
    ----------
    monthly_returns : pd.Series
        Net portfolio returns, indexed at the month-end of each HOLDING month.
    diagnostics : pd.DataFrame
        One row per formation date: n_members, n_eligible, n_held,
        coverage, turnover, cost.
    weights_history : dict[pd.Timestamp, pd.Series]
        Formation date -> portfolio weights used for the following month.
    """

    monthly_returns: pd.Series
    diagnostics: pd.DataFrame
    weights_history: dict[pd.Timestamp, pd.Series]


def simulate_monthly(
    daily_returns: pd.DataFrame,
    baskets: dict[pd.Timestamp, frozenset[str]],
    signal_fn: Callable[[pd.DataFrame, pd.Timestamp], pd.Series] | None = None,
    decile: float = 0.10,
    min_names: int = 20,
    tc_bps: float = 10.0,
    lookback: int = LOOKBACK_DAYS,
    skip: int = SKIP_DAYS,
    min_obs: int = MIN_OBS,
) -> CrossSectionalResult:
    """Backtest mensile cross-sectional sui panieri point-in-time.

    ``signal_fn(daily_returns, asof) -> pd.Series`` (punteggio: più alto = preferito)
    sostituisce il segnale; ``None`` (default) usa il momentum 12-1. Con ``signal_fn``
    i parametri ``lookback``/``skip``/``min_obs`` sono ignorati (valgono solo per il
    momentum di default — il chiamante chiude su eventuali parametri del proprio segnale).

    Per ogni formation T (chiavi di ``baskets``, month-end): segnale sui
    membri di basket_at(T) con soli dati <= T, top-decile equal weight,
    holding nel mese successivo con esecuzione t+1 (primo giorno di borsa
    del mese di holding NON accreditato), costi ``tc_bps`` one-way sul
    nozionale scambiato rispetto ai pesi precedenti driftati.

    Mese in cash (pochi nomi eleggibili): ritorno 0 meno l'eventuale costo
    di liquidazione del portafoglio precedente.

    Formations senza giorni di holding disponibili nei dati vengono
    saltate: il chiamante deve passare solo mesi di holding COMPLETI.
    """
    formations = sorted(baskets)
    rows: list[dict] = []
    rets_out: dict[pd.Timestamp, float] = {}
    weights_hist: dict[pd.Timestamp, pd.Series] = {}
    w_prev = pd.Series(dtype=float)
    month_ret_prev = pd.Series(dtype=float)

    for t in formations:
        members = sorted(baskets[t])
        month_end = t + pd.offsets.MonthEnd(1)
        # Giorni di borsa del mese di holding (dopo T), primo escluso (lag t+1)
        hold = daily_returns.loc[t + pd.Timedelta(days=1): month_end]
        if len(hold) < 2:
            logger.warning("Formation %s saltata: mese di holding senza dati", t.date())
            continue
        hold = hold.iloc[1:]

        present = [m for m in members if m in daily_returns.columns]
        if signal_fn is None:
            sig = momentum_12_1(
                daily_returns[present], asof=t,
                lookback=lookback, skip=skip, min_obs=min_obs,
            )
        else:
            sig = signal_fn(daily_returns[present], t)
        w_new = top_decile_weights(sig, decile=decile, min_names=min_names)
        if w_new.empty and len(sig.dropna()) > 0:
            logger.warning(
                "Formation %s: %d eleggibili -> decile sotto min_names=%d, mese in CASH",
                t.date(), int(sig.notna().sum()), min_names,
            )

        # Ritorno del mese di holding per RIC: compound dei giorni validi.
        # Colonne all-NaN (delisted prima del mese) -> prodotto vuoto -> 0 (cash).
        month_ret = (1.0 + hold).prod(skipna=True) - 1.0

        # Coverage del paniere: membri con almeno un ritorno valido nel mese
        n_priced = int(hold[present].notna().any().sum()) if present else 0
        coverage = n_priced / len(members) if members else 0.0

        port_ret = float((w_new * month_ret.reindex(w_new.index).fillna(0.0)).sum())

        # Turnover vs pesi precedenti driftati dai ritorni del mese appena chiuso
        if w_prev.empty:
            traded = float(w_new.abs().sum())
        else:
            grown = w_prev * (1.0 + month_ret_prev.reindex(w_prev.index).fillna(0.0))
            total = float(grown.sum())
            w_drift = grown / total if total > 0 else w_prev
            traded = float(w_new.subtract(w_drift, fill_value=0.0).abs().sum())
        cost = tc_bps / 1e4 * traded
        port_ret -= cost

        rets_out[month_end] = port_ret
        weights_hist[t] = w_new
        rows.append({
            "formation": t,
            "n_members": len(members),
            "n_eligible": int(sig.notna().sum()),
            "n_held": len(w_new),
            "coverage": coverage,
            "turnover": traded,
            "cost": cost,
        })
        w_prev = w_new
        month_ret_prev = month_ret

    diags = pd.DataFrame(rows).set_index("formation") if rows else pd.DataFrame()
    return CrossSectionalResult(
        monthly_returns=pd.Series(rets_out, dtype=float).sort_index(),
        diagnostics=diags,
        weights_history=weights_hist,
    )


def align_to_calendar(
    daily_returns: pd.DataFrame,
    calendar: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DatetimeIndex, pd.DatetimeIndex]:
    """Align the return matrix to a real trading calendar.

    Refinitiv TR.TotalReturn1D can emit PHANTOM rows on exchange holidays
    (values duplicating the previous trading day) and occasionally MISS a
    genuine trading day.  Phantom rows corrupt the t+1 execution lag and
    double-count returns; missing days silently shift positional windows.
    (Pre-run review Sprint O, 2026-06-12: 21 phantom rows and 4 missing
    NYSE days in the 2002-2007 segment.)

    Parameters
    ----------
    daily_returns : pd.DataFrame
        Raw return matrix (date x RIC).
    calendar : pd.DatetimeIndex
        Authoritative trading days (e.g. SPY's traded dates).

    Returns
    -------
    (aligned, extra, missing)
        ``aligned``: matrix reindexed on the calendar (phantom rows dropped,
        missing days inserted as all-NaN).  ``extra``: dates that were in the
        matrix but not in the calendar.  ``missing``: calendar days that had
        no row in the matrix.
    """
    extra = daily_returns.index.difference(calendar)
    missing = calendar.difference(daily_returns.index)
    aligned = daily_returns.reindex(calendar)
    if len(extra) or len(missing):
        logger.warning(
            "Calendar alignment: %d righe fantasma rimosse, %d giorni di borsa mancanti (NaN)",
            len(extra), len(missing),
        )
    return aligned, extra, missing
