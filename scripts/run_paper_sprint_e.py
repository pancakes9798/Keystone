#!/usr/bin/env python3
"""
Sprint H paper trading runner — HRP + Trend + Momentum (prior) + HMM Regime + Black-Litterman.

IS-calibrated prior params (v16, OOS 2000-2026):
  Conservative  damp=0.40, mom=0.00   Sharpe 0.54, MaxDD -26.9%
  Balanced      damp=0.40, mom=1.00   Sharpe 0.57, MaxDD -26.6%
  Aggressive    damp=0.30, mom=2.00   Sharpe 0.58, MaxDD -27.7%

BL overlay: HMM (pomegranate, Student-t, 3 regimes) on 6 Refinitiv macro RICs →
  CompositeSignal → BLOptimizer (cvxpy max-Sharpe with turnover penalty).

Run daily (e.g. at 18:00 CET after US close):
  uv run python scripts/run_paper_sprint_e.py
  uv run python scripts/run_paper_sprint_e.py --strategy balanced
  uv run python scripts/run_paper_sprint_e.py --date 2026-04-30
"""
from __future__ import annotations

import argparse
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
from jeanclaude.data.transform.features import build_state_variables  # noqa: E402
from jeanclaude.execution.paper import (  # noqa: E402
    DailyRunConfig,
    DailyRunner,
    PaperBroker,
)
from jeanclaude.portfolio.risk.filter import RiskFilter  # noqa: E402
from jeanclaude.portfolio.optimizer import TrendMomentumOptimizer, TrendMomentumParams  # noqa: E402
from jeanclaude.signals.macro.detector import RegimeDetector  # noqa: E402
from jeanclaude.costs import AlmgrenChrissModel, CostConfig  # noqa: E402
from jeanclaude.costs.data import VolumeSpreadLoader          # noqa: E402
from jeanclaude.data.ingestion.refinitiv import RefinitivSource  # noqa: E402
from jeanclaude.reporting.builder import ReportBuilder  # noqa: E402
from jeanclaude.reporting.mailer import Mailer          # noqa: E402

_LOG_FILE = Path(__file__).parent.parent / "logs" / "paper_sprint_e.log"
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
logger = logging.getLogger("paper_sprint_e")

# ── Universe ───────────────────────────────────────────────────────────────
PRICE_RICS: list[str] = [
    "AAPL.O", "MSFT.O", "AMZN.O", "NVDA.O",
    "JPM.N",  "BRKb.N",
    "JNJ.N",  "PFE.N",
    "PG.N",   "KO.N",   "WMT.N",
    "XOM.N",  "CAT.N",
    "TM.N",   "NVS.N",  "SAP.N",
    "GLD.P",  "TLT.O",
]

_EQUITY_RICS: frozenset[str] = frozenset({
    "AAPL.O", "MSFT.O", "AMZN.O", "NVDA.O",
    "JPM.N",  "BRKb.N",
    "JNJ.N",  "PFE.N",
    "PG.N",   "KO.N",   "WMT.N",
    "XOM.N",  "CAT.N",
    "TM.N",   "NVS.N",  "SAP.N",
})

# ── IS-calibrated strategy params (Sprint E v16) ────────────────────────────
_STRATEGIES: dict[str, TrendMomentumParams] = {
    "conservative": TrendMomentumParams(
        name="Conservative", damp_factor=0.40, momentum_strength=0.00,
    ),
    "balanced": TrendMomentumParams(
        name="Balanced", damp_factor=0.40, momentum_strength=1.00,
    ),
    "aggressive": TrendMomentumParams(
        name="Aggressive", damp_factor=0.30, momentum_strength=2.00,
    ),
}

# ── Paths ──────────────────────────────────────────────────────────────────
_DATA_DIR: Path      = Path(__file__).parent.parent / "data" / "paper_trading"
_HISTORY_START: str  = "2010-01-01"   # 15+ years — enough for HMM + MA200 + 12m momentum
_INITIAL_NAV: float  = 100_000.0      # starting capital in €

# Risk filter — CLAUDE.md: "CVaR + drawdown limit prima di ogni ribilanciamento".
# Limiti larghi per una strategia equity aggressive; calibrazione fine in P1.
_CVAR_LIMIT: float = 0.05       # CVaR 95% giornaliero max 5%
_DRAWDOWN_LIMIT: float = 0.30   # drawdown 2y max 30% prima del de-risking proporzionale
_FLAT_COST_BPS: float = 10.0    # fallback se Almgren-Chriss non disponibile

# FinBERT ticker queries rimosse 2026-06-13 — path sentiment disabilitato
# (nessuno scope news sulla sessione platform): vedi
# docs/research/2026-06-13-sprint-p-fattibilita-news.md

# ── Macro series — Refinitiv RICs for HMM regime detection ─────────────────
# All market-based, daily, no rate-limit issues unlike FRED.
_MACRO_RICS: dict[str, str] = {
    "VIX":    "VXc1",   # equity volatility
    "MOVE":   ".MOVE",  # bond volatility
    "Oil":    "CLc1",   # cyclical / inflation
    "Copper": "HGc1",   # industrial activity
    "Gold":   "XAU=",   # safe haven
    "EURUSD": "EUR=",   # dollar strength
}
_MACRO_SERIES: list[str] = list(_MACRO_RICS.values())
_RIC_TO_FRIENDLY: dict[str, str] = {v: k for k, v in _MACRO_RICS.items()}


class _MacroRegimeDetector:
    """Wraps RegimeDetector: preprocesses raw Refinitiv macro df before HMM fit."""

    def __init__(self, detector: RegimeDetector) -> None:
        self._detector = detector

    def _preprocess(self, macro: pd.DataFrame) -> pd.DataFrame:
        renamed = macro.rename(columns=_RIC_TO_FRIENDLY)
        if renamed.shape[1] < 3:
            raise RuntimeError(
                f"Too few macro features available ({renamed.shape[1]}). "
                "Check Refinitiv session and RIC availability."
            )
        sv = build_state_variables(renamed)
        return sv.dropna()

    def fit(self, macro: pd.DataFrame):
        return self._detector.fit(self._preprocess(macro))

    def current_regime(self, macro: pd.DataFrame):
        return self._detector.current_regime(self._preprocess(macro))


def _send_error_alert(exc: Exception) -> None:
    """Manda una email di alert a tutti i destinatari in caso di errore del runner."""
    user     = os.getenv("GMAIL_USER")
    password = os.getenv("GMAIL_APP_PASSWORD")
    recipients_raw = os.getenv("REPORT_RECIPIENTS", "")
    recipients = [r.strip() for r in recipients_raw.split(",") if r.strip()]

    if not (user and password and recipients):
        logger.warning("Alert email non inviata — credenziali o destinatari mancanti.")
        return

    tb = traceback.format_exc()
    html = (
        "<h2 style='color:red'>⚠️ JeanClaude — Errore runner paper trading</h2>"
        f"<p><b>Errore:</b> {type(exc).__name__}: {exc}</p>"
        f"<pre style='background:#f5f5f5;padding:12px'>{tb}</pre>"
        f"<p>Log completo: <code>{_LOG_FILE}</code></p>"
    )
    try:
        mailer = Mailer(smtp_user=user, smtp_password=password)
        mailer.send(html, subject="⚠️ JeanClaude — Errore runner paper trading", recipients=recipients)
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
        "GMAIL_USER / GMAIL_APP_PASSWORD non trovati nel .env — email disabilitata. "
        "Vedi .env.example per la configurazione."
    )
    return None


def _build_runner(
    params: TrendMomentumParams,
    today: pd.Timestamp,
) -> tuple[DailyRunner, PaperBroker]:
    store  = ParquetStore(_DATA_DIR)
    broker = PaperBroker(initial_capital=_INITIAL_NAV, store=store)

    data_loader = DataLoader(DataConfig(
        price_source="refinitiv",
        macro_source="refinitiv",
        refinitiv_session="platform",
    ))

    # Almgren-Chriss cost model — reads NAV dynamically from PaperBroker
    refinitiv_source = RefinitivSource(session_type="platform")
    volume_spread_loader = VolumeSpreadLoader(source=refinitiv_source, store=store)
    cost_model = AlmgrenChrissModel(CostConfig(aum_source="dynamic"))

    # HMM regime detector (Student-t, 3 states, sticky transitions)
    regime_detector = _MacroRegimeDetector(RegimeDetector())

    # TrendMomentum as prior.
    # Path sentiment FinBERT DISABILITATO (2026-06-13): non ha mai funzionato in
    # produzione — veniva iniettato RefinitivSource (prezzi) al posto di
    # RefinitivNewsSource, e comunque la sessione platform NON ha lo scope news
    # (ScopeError su /data/news/v1/headlines — limite di licenza). Fallback sui
    # prior a ogni run dal lancio di Sprint H. Riattivare SOLO con un accesso
    # news verificato: docs/research/2026-06-13-sprint-p-fattibilita-news.md
    prior_optimizer    = TrendMomentumOptimizer(params=params, equity_rics=_EQUITY_RICS)

    # Report settimanale: invia solo il venerdì; recipients da env
    send_today     = _is_report_day(today)
    recipients_raw = os.getenv("REPORT_RECIPIENTS", "")
    recipients     = [r.strip() for r in recipients_raw.split(",") if r.strip()]
    mailer         = _build_mailer() if (send_today and recipients) else None

    if send_today:
        if mailer and recipients:
            logger.info("Venerdì — report settimanale sarà spedito a: %s", recipients)
        else:
            logger.warning("Venerdì ma email non configurata (GMAIL_USER/GMAIL_APP_PASSWORD/REPORT_RECIPIENTS).")

    config = DailyRunConfig(
        assets=PRICE_RICS,
        macro_series=_MACRO_SERIES,
        rebalance_freq="ME",
        send_report=send_today and mailer is not None,
        report_recipients=recipients,
        transaction_cost_bps=_FLAT_COST_BPS,
        min_history=252,
        history_start=_HISTORY_START,
    )

    runner = DailyRunner(
        broker=broker,
        optimizer=prior_optimizer,   # fallback se ticker BL non disponibile
        risk_filter=RiskFilter(cvar_limit=_CVAR_LIMIT, drawdown_limit=_DRAWDOWN_LIMIT),
        regime_detector=regime_detector,
        data_loader=data_loader,
        report_builder=ReportBuilder(store=store),
        mailer=mailer,
        config=config,
        prior_optimizer=prior_optimizer,
        cost_model=cost_model,
        volume_spread_loader=volume_spread_loader,
        store=store,
    )
    return runner, broker


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sprint E daily paper trading runner (HRP + Trend + Momentum)"
    )
    parser.add_argument(
        "--strategy",
        choices=list(_STRATEGIES),
        default="aggressive",
        help="Which strategy variant to run (default: aggressive)",
    )
    parser.add_argument(
        "--date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Override today's date as YYYY-MM-DD (default: today)",
    )
    args = parser.parse_args()

    params = _STRATEGIES[args.strategy]

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    if args.date:
        try:
            today_ts = pd.Timestamp(args.date)
        except ValueError as exc:
            parser.error(f"Invalid --date value '{args.date}': {exc}")
    else:
        today_ts = pd.Timestamp.now().normalize()

    runner, broker = _build_runner(params, today=today_ts)
    result = runner.run(today_ts)
    state = broker.get_state()

    print(f"\n{'═' * 60}")
    print(f"  Sprint H Paper Trading — {result.date.date()}")
    print(f"  Strategy:    {params.name} (prior)")
    print(f"  Regime:      {result.regime or 'n/a'}")
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
