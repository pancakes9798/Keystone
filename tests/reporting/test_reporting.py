"""Tests for ReportBuilder and Mailer."""
from __future__ import annotations

import pandas as pd
import pytest
import numpy as np

from jeanclaude.data.storage.parquet_store import ParquetStore
from jeanclaude.reporting.builder import ReportBuilder


@pytest.fixture
def store_with_data(tmp_path):
    store = ParquetStore(tmp_path)
    dates = pd.date_range("2025-01-01", periods=30, freq="B")

    # nav_history
    nav_vals = 100_000.0 * np.cumprod(1 + np.random.default_rng(0).normal(0, 0.005, 30))
    nav_df = pd.DataFrame({"nav": nav_vals}, index=pd.DatetimeIndex(dates, name="date"))
    store.save("paper_trading", "nav_history", "state", "state", nav_df)

    # positions
    pos_df = pd.DataFrame(
        {"SPY": [0.6], "TLT": [0.4], "nav": [nav_vals[-1]], "cash": [0.0]},
        index=pd.DatetimeIndex([dates[-1]], name="date"),
    )
    store.save("paper_trading", "positions", "state", "state", pos_df)

    # orders
    ord_df = pd.DataFrame(
        {
            "asset": ["SPY", "TLT"],
            "old_weight": [0.0, 0.0],
            "new_weight": [0.6, 0.4],
            "price": [400.0, 95.0],
            "simulated_cost_bps": [10.0, 10.0],
        },
        index=pd.DatetimeIndex([dates[-1], dates[-1]], name="date"),
    )
    store.save("paper_trading", "orders", "state", "state", ord_df)
    return store


def test_report_builder_returns_html_string(store_with_data):
    builder = ReportBuilder(store_with_data)
    html = builder.build()
    assert isinstance(html, str)
    assert html.strip().startswith("<!DOCTYPE html")


def test_report_contains_nav(store_with_data):
    builder = ReportBuilder(store_with_data)
    html = builder.build()
    assert "NAV" in html


def test_report_contains_regime_section(store_with_data):
    builder = ReportBuilder(store_with_data)
    html = builder.build(regime="EXPANSION")
    assert "EXPANSION" in html


def test_report_contains_sharpe(store_with_data):
    builder = ReportBuilder(store_with_data)
    html = builder.build()
    assert "Sharpe" in html


def test_report_contains_equity_chart(store_with_data):
    builder = ReportBuilder(store_with_data)
    html = builder.build()
    assert "data:image/png;base64," in html


def test_report_empty_store_does_not_raise(tmp_path):
    store = ParquetStore(tmp_path)
    builder = ReportBuilder(store)
    html = builder.build()
    assert isinstance(html, str)


from unittest.mock import patch, MagicMock
from jeanclaude.reporting.mailer import Mailer


def test_mailer_send_calls_smtp(store_with_data):
    mailer = Mailer(smtp_user="sender@gmail.com", smtp_password="app-password")
    html = "<html><body>test</body></html>"

    with patch("jeanclaude.reporting.mailer.smtplib.SMTP") as mock_smtp_cls:
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)
        mailer.send(html, subject="Test", recipients=["r@example.com"])

    mock_smtp_cls.assert_called_once_with("smtp.gmail.com", 587)


def test_mailer_send_uses_starttls(store_with_data):
    mailer = Mailer(smtp_user="sender@gmail.com", smtp_password="app-password")

    with patch("jeanclaude.reporting.mailer.smtplib.SMTP") as mock_smtp_cls:
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)
        mailer.send("<html></html>", subject="X", recipients=["a@b.com"])

    mock_server.starttls.assert_called_once()
    mock_server.login.assert_called_once_with("sender@gmail.com", "app-password")


def test_mailer_send_multiple_recipients(store_with_data):
    mailer = Mailer(smtp_user="s@gmail.com", smtp_password="pw")

    with patch("jeanclaude.reporting.mailer.smtplib.SMTP") as mock_smtp_cls:
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)
        mailer.send("<html></html>", "subj", ["a@b.com", "c@d.com"])

    call_args = mock_server.sendmail.call_args
    assert call_args[0][1] == ["a@b.com", "c@d.com"]
