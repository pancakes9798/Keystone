#!/usr/bin/env python3
"""
ETF paper trading runner — HMM Regime + Black-Litterman.

Universe: 15 ETF settoriali/asset class (Yahoo Finance, zero costi dati)
Macro:    VIX / Oil / Copper / Gold / EURUSD (Yahoo Finance)
Strategy: HRP prior + HMM (5 macro features) → BL empirico per-ETF
Rebalance: fine mese (ME)
Capital:  €100,000

Sostituisce run_paper_sprint_e.py per il paper trading principale.
Nessuna dipendenza da Refinitiv — funziona con dati pubblici gratuiti.

Run:
    uv run python scripts/run_paper_etf.py
    uv run python scripts/run_paper_etf.py --date 2026-04-30   # dry run
"""
from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import sys
import traceback
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

from jeanclaude.data.storage.parquet_store import ParquetStore
from jeanclaude.data.transform.features import build_state_variables
from jeanclaude.execution.paper import PaperBroker
from jeanclaude.execution.paper.schedule import rebalance_due
from jeanclaude.portfolio.covariance import LedoitWolfCovariance
from jeanclaude.portfolio.optimizer.bl import _bl_posterior
from jeanclaude.portfolio.optimizer.hrp import HRPOptimizer
from jeanclaude.portfolio.risk.filter import RiskFilter
from jeanclaude.reporting.mailer import Mailer
from jeanclaude.signals.macro.detector import RegimeDetector
from jeanclaude.signals.macro.labels import RegimeLabel

# ── Logging ────────────────────────────────────────────────────────────────
_LOG_FILE = Path(__file__).parent.parent / "logs" / "paper_etf.log"
_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

_fmt = logging.Formatter(
    "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_fh = logging.handlers.RotatingFileHandler(
    _LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
_fh.setFormatter(_fmt)
_sh = logging.StreamHandler()
_sh.setFormatter(_fmt)
logging.basicConfig(level=logging.INFO, handlers=[_fh, _sh])
logger = logging.getLogger("paper_etf")

# ── Universe ───────────────────────────────────────────────────────────────
ETF_TICKERS = [
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP",  # US equity sectors
    "TLT", "IEF", "HYG",                                 # fixed income
    "GLD", "SLV", "USO",                                  # commodities
    "EEM", "EFA",                                          # international
]

MACRO_TICKERS: dict[str, str] = {
    "VIX":    "^VIX",
    "Oil":    "CL=F",
    "Copper": "HG=F",
    "Gold":   "GC=F",
    "EURUSD": "EURUSD=X",
}

# ── Config ─────────────────────────────────────────────────────────────────
_DATA_DIR      = Path(__file__).parent.parent / "data" / "paper_etf"
_HISTORY_START = "2010-01-01"   # 15+ anni per HMM stabile
_INITIAL_NAV   = 100_000.0
_DELTA         = 2.5
_TAU           = 0.025
_MAX_WEIGHT    = 0.25
_TP            = 0.02            # turnover penalty
_MIN_MACRO     = 60
_MIN_REGIME    = 20
_REBAL_FREQ    = "ME"
_FLAT_COST_BPS = 10.0
_CVAR_LIMIT    = 0.03
_DD_LIMIT      = 0.25


# ── Data fetching ──────────────────────────────────────────────────────────

def _fetch_prices(start: str, end: str) -> pd.DataFrame:
    raw = yf.download(ETF_TICKERS, start=start, end=end,
                      progress=False, auto_adjust=True)["Close"]
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    return raw.ffill().dropna(how="all")


def _fetch_macro(start: str, end: str) -> pd.DataFrame:
    frames = {}
    for name, ticker in MACRO_TICKERS.items():
        try:
            s = yf.download(ticker, start=start, end=end,
                            progress=False, auto_adjust=True)["Close"].squeeze()
            s.index = pd.to_datetime(s.index).tz_localize(None)
            frames[name] = s
        except Exception as e:
            logger.warning("Macro fetch failed for %s (%s): %s", name, ticker, e)
    df = pd.DataFrame(frames).ffill()
    return df


# ── ETF BL Optimizer ───────────────────────────────────────────────────────

def _bl_weights(
    returns: pd.DataFrame,
    macro_df: pd.DataFrame,
    detector: RegimeDetector,
) -> pd.Series:
    """HRP prior + HMM empirical views → BL weights. Falls back to HRP."""
    prior   = HRPOptimizer(LedoitWolfCovariance()).optimize(returns)
    tickers = list(returns.columns)
    N       = len(tickers)

    sv = build_state_variables(macro_df.loc[:returns.index[-1]]).dropna()
    if len(sv) < _MIN_MACRO:
        return prior

    try:
        result = detector.fit(sv)
    except Exception as exc:
        logger.warning("HMM fit fallito (%s) — fallback a prior HRP.", exc)
        return prior

    proba    = result.probabilities.iloc[-1].values
    combined = result.labels.index.union(returns.index)
    aligned  = (result.labels.reindex(combined).ffill()
                .reindex(returns.index).dropna())
    if aligned.empty:
        return prior

    aligned_rets = returns.loc[aligned.index]
    mu_regime: dict[RegimeLabel, pd.Series] = {}
    for r in RegimeLabel:
        mask = aligned == r
        mu_regime[r] = (aligned_rets[mask].mean() * 252
                        if mask.sum() >= _MIN_REGIME
                        else aligned_rets.mean() * 252)

    Q = (sum(float(proba[r.value]) * mu_regime[r] for r in RegimeLabel)
         .reindex(tickers).fillna(0.0).values)

    Sigma = returns.cov().values * 252
    w_mkt = prior.reindex(tickers).fillna(1.0 / N).values
    pi    = _DELTA * Sigma @ w_mkt
    tau_S = _TAU * Sigma

    try:
        mu_hat, Sigma_hat = _bl_posterior(pi, tau_S, np.eye(N), Q,
                                          np.diag(np.diag(tau_S)))
    except Exception as exc:
        logger.warning("BL posterior fallito (%s) — fallback a prior HRP.", exc)
        return prior

    ev, evec = np.linalg.eigh(Sigma + Sigma_hat)
    risk_psd = evec @ np.diag(np.maximum(ev, 1e-8)) @ evec.T
    risk_psd = (risk_psd + risk_psd.T) / 2

    w        = cp.Variable(N)
    w_anchor = np.full(N, 1.0 / N)
    prob     = cp.Problem(
        cp.Maximize(mu_hat @ w
                    - (_DELTA / 2) * cp.quad_form(w, risk_psd)
                    - _TP * cp.norm1(w - w_anchor)),
        [cp.sum(w) == 1, w >= 0, w <= _MAX_WEIGHT],
    )
    prob.solve()

    if w.value is None:
        logger.warning("cvxpy senza soluzione — fallback a prior HRP.")
        return prior
    return pd.Series(np.asarray(w.value), index=tickers)


# ── Rebalance / date helpers ───────────────────────────────────────────────

def _compute_asof(today: pd.Timestamp, price_index: pd.DatetimeIndex) -> pd.Timestamp:
    """Ultimo giorno di mercato ≤ today — data logica del ciclo."""
    available = price_index[price_index <= today]
    if len(available) == 0:
        raise ValueError(f"Nessun dato prezzi disponibile fino a {today.date()}.")
    return pd.Timestamp(available[-1])


def _is_report_day(today: pd.Timestamp) -> bool:
    return today.weekday() == 4  # venerdì


# ── Alert email ────────────────────────────────────────────────────────────

def _send_error_alert(exc: Exception) -> None:
    user     = os.getenv("GMAIL_USER")
    password = os.getenv("GMAIL_APP_PASSWORD")
    recipients = [r.strip() for r in os.getenv("REPORT_RECIPIENTS", "").split(",") if r.strip()]
    if not (user and password and recipients):
        return
    tb   = traceback.format_exc()
    html = (
        "<h2 style='color:red'>⚠️ JeanClaude ETF — Errore runner</h2>"
        f"<p><b>{type(exc).__name__}:</b> {exc}</p>"
        f"<pre style='background:#f5f5f5;padding:12px'>{tb}</pre>"
    )
    try:
        Mailer(smtp_user=user, smtp_password=password).send(
            html, "⚠️ JeanClaude ETF — Errore runner", recipients
        )
    except Exception as mail_exc:
        logger.error("Alert email failed: %s", mail_exc)


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ETF paper trading runner")
    parser.add_argument("--date", default=None, metavar="YYYY-MM-DD",
                        help="data logica del run (default: oggi); il run è idempotente")
    args = parser.parse_args()

    today = (pd.Timestamp(args.date) if args.date
             else pd.Timestamp.now().normalize())
    # yfinance end is exclusive → +1 day to include today
    end_str = (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    store  = ParquetStore(_DATA_DIR)
    broker = PaperBroker(initial_capital=_INITIAL_NAV, store=store)

    # ── Fetch data ─────────────────────────────────────────────────────
    logger.info("Fetching ETF prices (%s → %s)...", _HISTORY_START, end_str)
    prices = _fetch_prices(_HISTORY_START, end_str)
    # Align tickers — some might be missing early history
    available = [t for t in ETF_TICKERS if t in prices.columns]
    prices    = prices[available].ffill()
    returns   = np.log(prices / prices.shift(1)).dropna(how="all")  # log returns: SOLO per optimizer/risk filter; il NAV usa ritorni semplici da broker.period_returns
    logger.info("Prices: %d rows, %d ETFs", len(prices), len(available))

    logger.info("Fetching macro (Yahoo)...")
    macro = _fetch_macro(_HISTORY_START, end_str)
    logger.info("Macro: %d rows, features=%s", len(macro), list(macro.columns))

    asof = _compute_asof(today, prices.index)

    # ── 1. Daily NAV (P&L sui pesi vecchi, gap-aware, idempotente) ────
    # Finestra ancorata a nav_date_before(asof) dentro period_returns: un re-run
    # sullo stesso asof riproduce il primo run (vedi PaperBroker.period_returns).
    period_rets = broker.period_returns(prices, asof, available)
    nav = broker.record_daily_nav(period_rets, date=asof)

    rebalanced  = False
    new_weights = None
    regime      = None

    # ── 2. Rebalance a periodo completato (catch-up dei month-end persi) ──
    last_rebal = broker.last_rebalance_date()
    # Re-run dello stesso asof: riesegue il rebalance (upsert) e riapplica il costo
    # dopo il reset della riga NAV — stessa semantica del DailyRunner P0.
    due = rebalance_due(
        asof=asof, price_index=prices.index,
        freq=_REBAL_FREQ, last_rebalance=last_rebal,
    ) or (last_rebal is not None and last_rebal == asof)
    if due and len(returns.loc[:asof]) >= 252:
        detector = RegimeDetector()
        returns_asof = returns.loc[:asof]
        new_weights = _bl_weights(returns_asof, macro.loc[:asof], detector)
        new_weights = RiskFilter(cvar_limit=_CVAR_LIMIT, drawdown_limit=_DD_LIMIT).apply(
            new_weights, returns_asof
        )

        sv = build_state_variables(macro.loc[:asof]).dropna()
        if len(sv) >= _MIN_MACRO:
            try:
                result   = detector.fit(sv)
                proba    = result.probabilities.iloc[-1].values
                dominant = RegimeLabel(int(np.argmax(proba)))
                regime   = dominant.name
            except Exception as exc:
                logger.warning("Regime detection failed: %s", exc)

        rebal = broker.execute_rebalance(
            new_weights, prices.loc[asof],
            transaction_cost_bps=_FLAT_COST_BPS, date=asof,
        )
        if rebal.cost_eur > 0.0:
            nav = broker.apply_rebalance_cost(date=asof, cost_eur=rebal.cost_eur)
        rebalanced = True
        logger.info("Rebalanced %s | regime=%s | top=%s %.1f%% | cost €%.2f",
                    asof.date(), regime,
                    new_weights.idxmax(), new_weights.max() * 100, rebal.cost_eur)

    logger.info("NAV (%s): €%.2f", asof.date(), nav)

    # ── Weekly report ──────────────────────────────────────────────────
    if _is_report_day(asof):
        user     = os.getenv("GMAIL_USER")
        password = os.getenv("GMAIL_APP_PASSWORD")
        recipients = [r.strip() for r in
                      os.getenv("REPORT_RECIPIENTS", "").split(",") if r.strip()]
        if user and password and recipients:
            try:
                from jeanclaude.reporting.builder import ReportBuilder
                html = ReportBuilder(store=store).build(as_of=asof, regime=regime)
                Mailer(smtp_user=user, smtp_password=password).send(
                    html,
                    f"JeanClaude ETF Report — {asof.strftime('%Y-%m-%d')}",
                    recipients,
                )
                logger.info("Weekly report sent to %s", recipients)
            except Exception as e:
                logger.warning("Report failed: %s", e)

    # ── Print summary ──────────────────────────────────────────────────
    state = broker.get_state()
    print(f"\n{'═'*60}")
    print(f"  JeanClaude ETF — {asof.date()}")
    print(f"  NAV:        €{nav:,.2f}")
    print(f"  Rebalanced: {rebalanced}  |  Regime: {regime or 'n/a'}")
    if rebalanced and new_weights is not None:
        print(f"  Top 5 weights:")
        for etf, w in new_weights.nlargest(5).items():
            print(f"    {etf:<8}  {w:.1%}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("Runner crashed: %s", exc)
        _send_error_alert(exc)
        sys.exit(1)
