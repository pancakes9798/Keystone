"""Test della logica di schedulazione rebalance con catch-up."""
import pandas as pd

from jeanclaude.execution.paper.schedule import rebalance_due

# Calendario di trading fittizio: tutti i giorni feriali di apr-giu 2026
IDX = pd.bdate_range("2026-04-01", "2026-06-30")


def test_due_on_first_run_after_month_end():
    # Ultimo rebalance il 30/04; primo run di giugno → maggio è completato → due
    assert rebalance_due(
        asof=pd.Timestamp("2026-06-01"), price_index=IDX,
        freq="ME", last_rebalance=pd.Timestamp("2026-04-30"),
    )


def test_not_due_mid_month_after_recent_rebalance():
    assert not rebalance_due(
        asof=pd.Timestamp("2026-06-10"), price_index=IDX,
        freq="ME", last_rebalance=pd.Timestamp("2026-06-01"),
    )


def test_not_due_on_month_end_itself():
    """Il periodo corrente non è completo finché non inizia il successivo."""
    assert not rebalance_due(
        asof=pd.Timestamp("2026-05-29"), price_index=IDX,  # ultimo bday di maggio
        freq="ME", last_rebalance=pd.Timestamp("2026-04-30"),
    )


def test_catchup_after_missed_month_end():
    """Cron morto a cavallo del month-end: il primo run successivo recupera."""
    assert rebalance_due(
        asof=pd.Timestamp("2026-06-08"), price_index=IDX,
        freq="ME", last_rebalance=pd.Timestamp("2026-04-30"),
    )


def test_initial_allocation_when_never_rebalanced():
    assert rebalance_due(
        asof=pd.Timestamp("2026-06-10"), price_index=IDX,
        freq="ME", last_rebalance=None,
    )


def test_rerun_same_day_not_due_again():
    """Dopo il rebalance di oggi, un re-run oggi non è più due."""
    assert not rebalance_due(
        asof=pd.Timestamp("2026-06-01"), price_index=IDX,
        freq="ME", last_rebalance=pd.Timestamp("2026-06-01"),
    )


def test_weekly_frequency():
    idx = pd.bdate_range("2026-05-01", "2026-06-30")
    # Ultimo rebalance lunedì 01/06; run lunedì 08/06 → settimana precedente completata
    assert rebalance_due(
        asof=pd.Timestamp("2026-06-08"), price_index=idx,
        freq="W", last_rebalance=pd.Timestamp("2026-06-01"),
    )
    assert not rebalance_due(
        asof=pd.Timestamp("2026-06-04"), price_index=idx,
        freq="W", last_rebalance=pd.Timestamp("2026-06-01"),
    )
