#!/usr/bin/env python3
"""
Sprint L paper trading runner — TrendMomentumOptimizer, frozen config, Yahoo TR.

Validated configuration (Sprint L — 2026-06-11):
  Universe:           20 ETFs (SPY, QQQ, IWM, IVE, EFA, EEM, EWJ, TLT, IEF, SHY,
                               TIP, LQD, HYG, EMB, GLD, SLV, DBC, VNQ, BIL, UUP)
  Optimizer:          TrendMomentumOptimizer
                        damp_factor=0.25, momentum_strength=0.5, equity_min=0.5
  Equity rics:        SPY, QQQ, IWM, IVE, EFA, EEM, EWJ (7 equity ETFs)
  Rebalance:          monthly end-of-month (ME)
  Transaction costs:  10 bps flat
  Risk filter:        NoopRiskFilter — see comment in _build_runner()
  Regime detector:    NoopRegimeDetector — no HMM (lezione Sprint K)

Data source — why Yahoo Finance, NOT Refinitiv (validate_total_return 2026-06-10):
  Refinitiv TRDPRC_1 is a PRICE-RETURN series (does not include dividends).
  The Sprint L backtest was run entirely on Yahoo Finance data with
  auto_adjust=True, which delivers TOTAL-RETURN adjusted prices (splits +
  dividends baked in).  Using Refinitiv here would introduce a data source
  mismatch: dividend-paying ETFs (e.g. SPY, TLT, IEF) would show lower
  cumulative returns in live paper trading than in the validated backtest,
  making the track record incomparable.  DataConfig(price_source="yahoo")
  routes to YahooSource(auto_adjust=True), matching the backtest exactly.

Run:
    uv run python scripts/run_paper_sprint_l.py
    uv run python scripts/run_paper_sprint_l.py --date 2026-06-10
"""
from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # noqa: E402

import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from jeanclaude.data import DataConfig, DataLoader  # noqa: E402
from jeanclaude.data.storage.parquet_store import ParquetStore  # noqa: E402
from jeanclaude.execution.paper import (  # noqa: E402
    DailyRunConfig,
    DailyRunner,
    PaperBroker,
)
from jeanclaude.execution.paper.stubs import (  # noqa: E402
    NoopRegimeDetector,
    NoopRiskFilter,
)
from jeanclaude.portfolio.optimizer import TrendMomentumOptimizer, TrendMomentumParams  # noqa: E402
from jeanclaude.reporting.builder import ReportBuilder  # noqa: E402
from jeanclaude.reporting.mailer import Mailer  # noqa: E402

# ── Config file ────────────────────────────────────────────────────────────
_REPO_ROOT   = Path(__file__).parent.parent
_CONFIG_PATH = _REPO_ROOT / "configs" / "experiments" / "2026-06-11-sprint-l-equity-structural.json"

# ── Logging ─────────────────────────────────────────────────────────────────
_LOG_FILE = _REPO_ROOT / "logs" / "paper_sprint_l.log"
_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

_fmt = logging.Formatter(
    "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_file_handler = logging.handlers.RotatingFileHandler(
    _LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
_file_handler.setFormatter(_fmt)
_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(_fmt)

logging.basicConfig(level=logging.INFO, handlers=[_file_handler, _stream_handler])
logger = logging.getLogger("paper_sprint_l")

# ── Load config JSON and derive universe + equity tickers ──────────────────
# Strip exchange suffixes from Refinitiv RICs → plain Yahoo Finance tickers.
# E.g. "SPY.P" → "SPY", "QQQ.O" → "QQQ".
# This avoids maintaining a second hardcoded list that could drift from the
# validated experiment config.

def _strip_ric_suffix(ric: str) -> str:
    """Remove Refinitiv exchange suffix from a RIC (e.g. 'SPY.P' → 'SPY')."""
    dot_idx = ric.rfind(".")
    if dot_idx > 0:
        return ric[:dot_idx]
    return ric


with open(_CONFIG_PATH) as _f:
    _cfg = json.load(_f)

UNIVERSE_TICKERS: list[str] = [_strip_ric_suffix(r) for r in _cfg["universe"]]
_EQUITY_TICKERS: frozenset[str] = frozenset(
    _strip_ric_suffix(r) for r in _cfg["equity_rics"]
)

# ── Frozen Sprint L parameters ─────────────────────────────────────────────
# Best validated IS-OOS config from the sprint-l-equity-structural experiment:
# damp=0.25, mom=0.5, equity_min=0.5 — see docs/reports/sprint_l_results.md.
_PARAMS = TrendMomentumParams(
    name="SprintL",
    damp_factor=0.25,
    momentum_strength=0.5,
    equity_min=0.5,
)

# ── Paths and operational constants ──────────────────────────────────────────
_DATA_DIR:       Path  = _REPO_ROOT / "data" / "paper_sprint_l"
_HISTORY_START:  str   = "2015-01-01"   # enough for HRP (756d) + momentum (252d) + margin
_INITIAL_NAV:    float = 100_000.0


# ── Email helpers (copied from run_paper_sprint_e.py) ─────────────────────

def _send_error_alert(exc: Exception) -> None:
    """Send an error alert email on runner failure."""
    user           = os.getenv("GMAIL_USER")
    password       = os.getenv("GMAIL_APP_PASSWORD")
    recipients_raw = os.getenv("REPORT_RECIPIENTS", "")
    recipients     = [r.strip() for r in recipients_raw.split(",") if r.strip()]

    if not (user and password and recipients):
        logger.warning("Alert email non inviata — credenziali o destinatari mancanti.")
        return

    tb   = traceback.format_exc()
    html = (
        "<h2 style='color:red'>&#9888;&#65039; JeanClaude Sprint L — Errore runner</h2>"
        f"<p><b>Errore:</b> {type(exc).__name__}: {exc}</p>"
        f"<pre style='background:#f5f5f5;padding:12px'>{tb}</pre>"
        f"<p>Log completo: <code>{_LOG_FILE}</code></p>"
    )
    try:
        Mailer(smtp_user=user, smtp_password=password).send(
            html,
            subject="&#9888;&#65039; JeanClaude Sprint L — Errore runner",
            recipients=recipients,
        )
        logger.info("Alert errore inviato a: %s", recipients)
    except Exception as mail_exc:
        logger.error("Impossibile inviare alert email: %s", mail_exc)


def _is_report_day(today: pd.Timestamp) -> bool:
    """True se oggi è venerdì (weekday 4) — giorno del report settimanale."""
    return today.weekday() == 4


def _build_mailer() -> Mailer | None:
    """Costruisce il Mailer se le credenziali Gmail sono nel .env; altrimenti None."""
    user     = os.getenv("GMAIL_USER")
    password = os.getenv("GMAIL_APP_PASSWORD")
    if user and password:
        return Mailer(smtp_user=user, smtp_password=password)
    logger.warning(
        "GMAIL_USER / GMAIL_APP_PASSWORD non trovati nel .env — email disabilitata."
    )
    return None


# ── Runner factory ─────────────────────────────────────────────────────────

def _build_runner(today: pd.Timestamp) -> tuple[DailyRunner, PaperBroker]:
    store  = ParquetStore(_DATA_DIR)
    broker = PaperBroker(initial_capital=_INITIAL_NAV, store=store)

    # Yahoo Finance — total-return adjusted prices (see module docstring).
    data_loader = DataLoader(DataConfig(price_source="yahoo"))

    optimizer = TrendMomentumOptimizer(params=_PARAMS, equity_rics=_EQUITY_TICKERS)

    # Risk filter — NoopRiskFilter: fedeltà alla config validata (hash nel json):
    # il backtest Sprint L non includeva risk filter; attivarlo qui divergerebbe
    # dal track record che vogliamo misurare.
    risk_filter = NoopRiskFilter()

    # Regime detector — NoopRegimeDetector: Sprint K dimostrava che HMM non porta
    # beneficio su questo universo ETF; Sprint L è stato validato senza macro.
    regime_detector = NoopRegimeDetector()

    send_today     = _is_report_day(today)
    recipients_raw = os.getenv("REPORT_RECIPIENTS", "")
    recipients     = [r.strip() for r in recipients_raw.split(",") if r.strip()]
    mailer         = _build_mailer() if (send_today and recipients) else None

    if send_today:
        if mailer and recipients:
            logger.info("Venerdì — report settimanale sarà spedito a: %s", recipients)
        else:
            logger.warning(
                "Venerdì ma email non configurata "
                "(GMAIL_USER / GMAIL_APP_PASSWORD / REPORT_RECIPIENTS)."
            )

    config = DailyRunConfig(
        assets=UNIVERSE_TICKERS,
        macro_series=[],                 # NoopRegimeDetector — nessuna macro
        rebalance_freq="ME",
        send_report=send_today and mailer is not None,
        report_recipients=recipients,
        transaction_cost_bps=10.0,
        min_history=252,
        history_start=_HISTORY_START,
    )

    runner = DailyRunner(
        broker=broker,
        optimizer=optimizer,
        risk_filter=risk_filter,
        regime_detector=regime_detector,
        data_loader=data_loader,
        report_builder=ReportBuilder(store=store),
        mailer=mailer,
        config=config,
    )
    return runner, broker


# ── Entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sprint L daily paper trading runner (TrendMomentum + Yahoo TR)"
    )
    parser.add_argument(
        "--date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Override today's date as YYYY-MM-DD (default: today); il run è idempotente",
    )
    args = parser.parse_args()

    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.date:
        try:
            today_ts = pd.Timestamp(args.date)
        except ValueError as exc:
            parser.error(f"Invalid --date value '{args.date}': {exc}")
    else:
        today_ts = pd.Timestamp.now().normalize()

    runner, broker = _build_runner(today=today_ts)
    result = runner.run(today_ts)
    state  = broker.get_state()

    print(f"\n{'═' * 60}")
    print(f"  Sprint L Paper Trading — {result.date.date()}")
    print(f"  Params:      damp={_PARAMS.damp_factor}, mom={_PARAMS.momentum_strength}, equity_min={_PARAMS.equity_min}")
    print(f"  NAV:         €{result.nav:,.2f}")
    print(f"  Rebalanced:  {result.rebalanced}")
    if result.rebalanced and result.orders:
        print(f"  Orders ({len(result.orders)}):")
        for o in sorted(result.orders, key=lambda x: abs(x.new_weight - x.old_weight), reverse=True):
            direction = "BUY " if o.new_weight > o.old_weight else "SELL"
            print(f"    {direction} {o.asset:<10}  {o.old_weight:.1%} → {o.new_weight:.1%}")
    if not state.weights.empty:
        print("  Top 5 weights:")
        for asset, w in state.weights.nlargest(5).items():
            print(f"    {asset:<12} {w:.1%}")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("Runner fallito con errore: %s", exc)
        _send_error_alert(exc)
        sys.exit(1)
