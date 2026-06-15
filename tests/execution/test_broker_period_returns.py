"""Regression: il NAV di una data passata fissa deve essere idempotente tra run.

Bug di produzione (ETF store, 2026-06): un gap su 06-11 + due run sullo stesso
``asof`` = 06-12 (venerdì sera / sabato / domenica puntano tutti all'ultimo giorno
di mercato = venerdì) producevano due NAV diversi per la STESSA data:

- Run 1 (prima volta per asof): finestra gap-aware 06-10→06-12  → €101.576,57
- Run 2 (re-run stesso asof):   finestra collassata a 1 giorno   → €99.823,04

Causa: la finestra del ritorno era ancorata a ``last_nav_date()`` (ultima riga
NAV ASSOLUTA), che dopo il primo run è ``asof`` stesso → ``last_nav_date < asof``
falso → fallback 1-giorno, mentre ``record_daily_nav`` ancora la base alla riga
strettamente precedente ad ``asof``. Le due definizioni divergono solo sul re-run.

Fix: ``PaperBroker.period_returns`` ancora la finestra alla STESSA data-base di
``record_daily_nav`` (ultima riga NAV strettamente prima di ``asof``).
"""
from __future__ import annotations

import pandas as pd
import pytest

from jeanclaude.data.storage.parquet_store import ParquetStore
from jeanclaude.execution.paper import PaperBroker

D1 = pd.Timestamp("2026-06-10")  # ultima riga NAV prima del gap
D2 = pd.Timestamp("2026-06-11")  # giorno saltato (nessuna riga NAV)
D3 = pd.Timestamp("2026-06-12")  # asof ricalcolato due volte

# Prezzi congelati: D2 = +2% su D1, D3 = -0.98% su D2 (quindi +1% su D1).
PRICES = pd.DataFrame({"AAA": [100.0, 102.0, 101.0]}, index=pd.DatetimeIndex([D1, D2, D3], name="date"))


@pytest.fixture
def broker(tmp_path) -> PaperBroker:
    return PaperBroker(initial_capital=100_000.0, store=ParquetStore(tmp_path))


def _seed_invested_at_d1(broker: PaperBroker) -> float:
    """Portafoglio 100% AAA con base NAV stabilita al D1. Ritorna la base."""
    broker.execute_rebalance(pd.Series({"AAA": 1.0}), PRICES.loc[D1], transaction_cost_bps=0.0, date=D1)
    broker.record_daily_nav(pd.Series({"AAA": 0.0}), date=D1)
    return broker.nav  # 100_000


def test_period_returns_anchors_strictly_before_asof_even_after_asof_written(broker):
    """nav_date_before(asof) deve escludere la riga di asof stesso (re-run)."""
    _seed_invested_at_d1(broker)
    assert broker.nav_date_before(D3) == D1
    # Simula il primo run che scrive la riga D3:
    broker.record_daily_nav(broker.period_returns(PRICES, D3, ["AAA"]), date=D3)
    # Dopo che D3 esiste, l'ancora per asof=D3 deve restare D1, NON D3:
    assert broker.nav_date_before(D3) == D1


def test_nav_for_fixed_asof_is_idempotent_across_reruns_after_gap(broker):
    """Il bug: due run sullo stesso asof dopo un gap davano NAV diversi."""
    base = _seed_invested_at_d1(broker)

    # Run 1 (prima volta per asof=D3)
    r1 = broker.period_returns(PRICES, D3, ["AAA"])
    nav1 = broker.record_daily_nav(r1, date=D3)

    # Run 2 (re-run STESSO asof=D3; ora la riga D3 esiste già)
    r2 = broker.period_returns(PRICES, D3, ["AAA"])
    nav2 = broker.record_daily_nav(r2, date=D3)

    assert nav2 == pytest.approx(nav1), "re-run sullo stesso asof deve essere idempotente"
    # Valore corretto gap-aware: base × p(D3)/p(D1) = 100_000 × 101/100
    assert nav1 == pytest.approx(base * 101.0 / 100.0)
    # Il ritorno NON deve collassare al fallback 1-giorno (p(D3)/p(D2)) sul re-run:
    assert r2["AAA"] == pytest.approx(r1["AAA"])
