"""Test della logica di staleness del watchdog."""
import pandas as pd

from scripts.check_paper_health import is_stale, last_expected_trading_day


def test_last_expected_trading_day_weekday():
    # mercoledì → atteso martedì (il run è serale)
    assert last_expected_trading_day(pd.Timestamp("2026-06-10")) == pd.Timestamp("2026-06-09")


def test_last_expected_trading_day_monday():
    # lunedì → atteso venerdì
    assert last_expected_trading_day(pd.Timestamp("2026-06-08")) == pd.Timestamp("2026-06-05")


def test_stale_when_nav_older_than_expected():
    assert is_stale(
        last_nav_date=pd.Timestamp("2026-05-31"),
        today=pd.Timestamp("2026-06-10"),
    )


def test_fresh_when_nav_at_expected_day():
    assert not is_stale(
        last_nav_date=pd.Timestamp("2026-06-09"),
        today=pd.Timestamp("2026-06-10"),
    )


def test_stale_when_no_nav_at_all():
    assert is_stale(last_nav_date=None, today=pd.Timestamp("2026-06-10"))
