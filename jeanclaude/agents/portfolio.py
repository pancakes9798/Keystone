from __future__ import annotations

import logging
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd

from jeanclaude.portfolio.covariance import LedoitWolfCovariance
from jeanclaude.portfolio.optimizer.bl import _bl_posterior
from jeanclaude.portfolio.optimizer.hrp import HRPOptimizer
from jeanclaude.portfolio.risk.filter import RiskFilter
from jeanclaude.portfolio.risk.metrics import cvar_historical

from .base import BaseAgent
from .config import AgentConfig
from .events import NoRebalanceEvent, PriceEvent, RegimeEvent, SignalEvent, WeightsEvent

logger = logging.getLogger(__name__)


class PortfolioAgent(BaseAgent):
    def __init__(self, config: AgentConfig, broker: object) -> None:
        super().__init__()
        self._config = config
        self._broker = broker
        self._risk_filter = RiskFilter(cvar_limit=config.rebalance.cvar_limit)
        self._hrp = HRPOptimizer(LedoitWolfCovariance())
        self._cooldown_file = Path(config.data_dir) / "last_rebalance_date.txt"
        self._last_rebalance_date: pd.Timestamp | None = self._load_last_rebalance_date(broker)

    async def run(
        self,
        signal_event: SignalEvent,
        price_event: PriceEvent,
        regime_event: RegimeEvent,
    ) -> WeightsEvent | NoRebalanceEvent:
        date = price_event.date
        returns = price_event.returns
        cfg = self._config.rebalance

        # Cooldown gate — blocca tutto, anche regime change
        if self._last_rebalance_date is not None:
            days_since = (date - self._last_rebalance_date).days
            if days_since < cfg.min_cooldown_days:
                logger.info("PortfolioAgent: cooldown (%d/%d giorni)", days_since, cfg.min_cooldown_days)
                return NoRebalanceEvent(date=date, reason="cooldown")

        # Calcola pesi candidati
        candidate = self._compute_bl_weights(signal_event, returns)
        if candidate is None:
            return NoRebalanceEvent(date=date, reason="optimization_failed")

        # Valuta trigger
        reasons: list[str] = []

        current_w = self._broker.get_state().weights
        if current_w.empty:
            reasons.append("empty_portfolio")

        if not reasons and cfg.regime_change_trigger and regime_event.changed:
            reasons.append("regime_change")

        if not reasons:
            if not current_w.empty:
                curr = current_w.reindex(candidate.index).fillna(0.0)
                turnover = (candidate - curr).abs().sum() / 2
                if turnover > cfg.signal_drift_threshold:
                    reasons.append("signal_drift")

        if cfg.risk_breach_trigger and not reasons:
            try:
                if not current_w.empty:
                    cols = [c for c in current_w.index if c in returns.columns]
                    if cols:
                        w = current_w.reindex(cols).fillna(0.0)
                        portfolio_cvar = cvar_historical(returns[cols] @ w, confidence=0.95)
                        if portfolio_cvar > cfg.cvar_limit:
                            reasons.append("risk_breach")
            except Exception as exc:
                logger.warning("PortfolioAgent: CVaR check fallito: %s", exc)

        if not reasons:
            return NoRebalanceEvent(date=date, reason="no_trigger")

        filtered = self._risk_filter.apply(candidate, returns)
        self._last_rebalance_date = date
        self._save_last_rebalance_date(date)
        reason = reasons[0]
        logger.info(
            "PortfolioAgent: ribilanciamento | reason=%s | top=%s %.1f%%",
            reason, filtered.idxmax(), filtered.max() * 100,
        )
        return WeightsEvent(date=date, weights=filtered, reason=reason)

    def _load_last_rebalance_date(self, broker: object) -> pd.Timestamp | None:  # noqa: ARG002
        """Carica l'ultima data di ribilanciamento dal file di cooldown.

        ``broker`` mantenuto per compatibilità firma con il costruttore.
        """
        if self._cooldown_file.exists():
            try:
                return pd.Timestamp(self._cooldown_file.read_text().strip())
            except (ValueError, OSError) as exc:
                logger.warning(
                    "PortfolioAgent: cooldown file illeggibile (%s) — cooldown resettato", exc
                )
        return None

    def _save_last_rebalance_date(self, date: pd.Timestamp) -> None:
        try:
            self._cooldown_file.parent.mkdir(parents=True, exist_ok=True)
            self._cooldown_file.write_text(str(date.date()))
        except Exception as exc:
            logger.warning("PortfolioAgent: impossibile salvare last_rebalance_date: %s", exc)

    def _compute_bl_weights(
        self, signal_event: SignalEvent, returns: pd.DataFrame
    ) -> pd.Series | None:
        tickers = list(returns.columns)
        N = len(tickers)
        cfg = self._config

        try:
            prior = self._hrp.optimize(returns)
        except Exception as exc:
            logger.warning("PortfolioAgent: HRP failed (%s) — equal weight", exc)
            prior = pd.Series(1.0 / N, index=tickers)

        Q = signal_event.views.reindex(tickers).fillna(0.0).values
        Sigma = returns.cov().values * 252
        w_mkt = prior.reindex(tickers).fillna(1.0 / N).values
        pi = cfg.delta * Sigma @ w_mkt
        tau_S = cfg.tau * Sigma

        try:
            mu_hat, Sigma_hat = _bl_posterior(pi, tau_S, np.eye(N), Q, np.diag(np.diag(tau_S)))
        except Exception as exc:
            logger.warning("PortfolioAgent: BL posterior failed (%s) — usando prior HRP", exc)
            return prior

        ev, evec = np.linalg.eigh(Sigma + Sigma_hat)
        risk_psd = evec @ np.diag(np.maximum(ev, 1e-8)) @ evec.T
        risk_psd = (risk_psd + risk_psd.T) / 2

        w = cp.Variable(N)
        w_anchor = np.full(N, 1.0 / N)
        prob = cp.Problem(
            cp.Maximize(
                mu_hat @ w
                - (cfg.delta / 2) * cp.quad_form(w, risk_psd)
                - cfg.turnover_penalty * cp.norm1(w - w_anchor)
            ),
            [cp.sum(w) == 1, w >= 0, w <= cfg.max_weight],
        )
        prob.solve()

        if w.value is None:
            logger.warning("PortfolioAgent: cvxpy no solution — usando prior HRP")
            return prior
        return pd.Series(np.asarray(w.value), index=tickers)
