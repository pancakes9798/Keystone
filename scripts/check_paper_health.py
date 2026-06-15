#!/usr/bin/env python3
"""Watchdog del paper trading: alert email se gli store NAV non sono freschi.

Gira ogni sera via launchd DOPO i runner. Se uno store non ha la riga NAV
dell'ultimo giorno di trading atteso, manda un alert email. Exit code 1 se
almeno uno store è stale (visibile anche in `launchctl list`).

Usage:
    uv run python scripts/check_paper_health.py
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

from jeanclaude.data.storage.parquet_store import ParquetStore  # noqa: E402
from jeanclaude.reporting.mailer import Mailer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger("paper_health")

_REPO_ROOT = Path(__file__).parent.parent
_STORES = {
    "sprint_e": _REPO_ROOT / "data" / "paper_trading",
    "etf": _REPO_ROOT / "data" / "paper_etf",
    "sprint_l": _REPO_ROOT / "data" / "paper_sprint_l",
}


def last_expected_trading_day(today: pd.Timestamp) -> pd.Timestamp:
    """Ultimo giorno feriale STRETTAMENTE precedente a oggi (il run è serale)."""
    day = today.normalize() - pd.Timedelta(days=1)
    while day.weekday() >= 5:  # sab/dom
        day -= pd.Timedelta(days=1)
    return day


def is_stale(last_nav_date: pd.Timestamp | None, today: pd.Timestamp) -> bool:
    """True se il NAV non copre l'ultimo giorno di trading atteso.

    Tollera i festivi US: una singola giornata di buco non scatta (il run di
    un festivo registra comunque la riga con return 0 sotto l'ultima data di
    mercato). Due o più giorni feriali senza righe = stale.
    """
    if last_nav_date is None:
        return True
    expected = last_expected_trading_day(today)
    gap_days = len(pd.bdate_range(last_nav_date, expected)) - 1
    return gap_days > 1


def _send_alert(stale: dict[str, pd.Timestamp | None]) -> None:
    user = os.getenv("GMAIL_USER")
    password = os.getenv("GMAIL_APP_PASSWORD")
    recipients = [r.strip() for r in os.getenv("REPORT_RECIPIENTS", "").split(",") if r.strip()]
    if not (user and password and recipients):
        logger.warning("Alert non inviabile — credenziali email mancanti.")
        return
    rows = "".join(
        f"<tr><td>{name}</td><td>{d.date() if d is not None else 'MAI'}</td></tr>"
        for name, d in stale.items()
    )
    html = (
        "<h2 style='color:red'>&#9888;&#65039; JeanClaude — Paper trading FERMO</h2>"
        "<p>Gli store seguenti non hanno il NAV dell'ultimo giorno di trading atteso:</p>"
        f"<table border='1' cellpadding='4'><tr><th>Store</th><th>Ultimo NAV</th></tr>{rows}</table>"
        "<p>Vedi <code>docs/runbook.md</code> per il recovery.</p>"
    )
    try:
        Mailer(smtp_user=user, smtp_password=password).send(
            html, "JeanClaude — Paper trading FERMO", recipients
        )
        logger.info("Alert inviato a %s", recipients)
    except Exception as exc:
        logger.error("Invio alert fallito: %s", exc)


def main() -> None:
    today = pd.Timestamp.now()
    stale: dict[str, pd.Timestamp | None] = {}
    for name, root in _STORES.items():
        store = ParquetStore(root)
        nav = store.load("paper_trading", "nav_history", "state", "state")
        last = pd.Timestamp(nav.index[-1]) if nav is not None and not nav.empty else None
        if is_stale(last, today):
            stale[name] = last
            logger.error("STALE: %s — ultimo NAV: %s", name, last.date() if last else "mai")
        else:
            logger.info("OK: %s — ultimo NAV: %s", name, last.date())

    if stale:
        _send_alert(stale)
        sys.exit(1)
    logger.info("Tutti gli store sono freschi.")


if __name__ == "__main__":
    main()
