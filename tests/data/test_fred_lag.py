"""FRED: le osservazioni diventano disponibili con lag di pubblicazione."""
import pandas as pd
import pytest
from unittest.mock import patch


def test_publication_lag_shifts_availability():
    from jeanclaude.data.ingestion.fred import FREDSource

    src = FREDSource.__new__(FREDSource)
    src._api_key = "fake"
    src._publication_lag_days = 1
    raw = pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.DatetimeIndex(["2026-06-01", "2026-06-02", "2026-06-03"]),
    )
    with patch.object(src, "_fetch_series", return_value=raw):
        out = src.get_macro(["VIXCLS"], start="2026-06-01", end="2026-06-10")
    s = out["VIXCLS"].dropna()
    # il valore osservato il 01/06 è DISPONIBILE dal 02/06 (shift di 1 bday)
    assert s.index[0] == pd.Timestamp("2026-06-02")
    assert s.iloc[0] == 1.0


def test_default_publication_lag_is_one():
    from jeanclaude.data.ingestion.fred import FREDSource

    src = FREDSource()
    assert src._publication_lag_days == 1


def test_zero_lag_no_shift():
    from jeanclaude.data.ingestion.fred import FREDSource

    src = FREDSource.__new__(FREDSource)
    src._api_key = "fake"
    src._publication_lag_days = 0
    raw = pd.Series(
        [1.0, 2.0],
        index=pd.DatetimeIndex(["2026-06-01", "2026-06-02"]),
    )
    with patch.object(src, "_fetch_series", return_value=raw):
        out = src.get_macro(["VIXCLS"], start="2026-06-01", end="2026-06-03")
    s = out["VIXCLS"].dropna()
    assert s.index[0] == pd.Timestamp("2026-06-01")
    assert s.iloc[0] == 1.0
