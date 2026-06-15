"""RiskFilter — applica CVaR cap e drawdown limit tramite truncation."""
from __future__ import annotations

import logging

import pandas as pd

from .metrics import cvar_historical, component_var

logger = logging.getLogger(__name__)


class RiskFilter:
    """Pre-rebalancing risk gate con truncation degli asset più rischiosi.

    Quando il CVaR del portafoglio supera ``cvar_limit``, riduce
    iterativamente il peso dell'asset con maggiore component VaR.
    Se il drawdown storico supera ``drawdown_limit``, scala tutti i pesi
    proporzionalmente (posizione parzialmente cash).

    Parameters
    ----------
    cvar_limit : float
        CVaR giornaliero massimo accettabile (default 2%).
    drawdown_limit : float
        Drawdown storico massimo accettabile (default 15%).
    confidence : float
        Livello di confidenza per VaR/CVaR (default 0.95).
    truncation_step : float
        Frazione del peso corrente rimossa ad ogni iterazione (default 10%).
    min_weight : float
        Peso minimo per asset — floor della truncation (default 0.0).
    """

    def __init__(
        self,
        cvar_limit: float = 0.02,
        drawdown_limit: float = 0.15,
        confidence: float = 0.95,
        truncation_step: float = 0.1,
        min_weight: float = 0.0,
    ) -> None:
        self.cvar_limit = cvar_limit
        self.drawdown_limit = drawdown_limit
        self.confidence = confidence
        self.truncation_step = truncation_step
        self.min_weight = min_weight

    def apply(
        self,
        weights: pd.Series,
        returns: pd.DataFrame,
    ) -> pd.Series:
        """Applica CVaR cap e drawdown limit ai pesi proposti.

        Parameters
        ----------
        weights : pd.Series
            Pesi proposti dall'optimizer (sommanti a 1).
        returns : pd.DataFrame
            Storia dei rendimenti per asset (colonne = nomi asset).
            Richiede almeno 20 osservazioni per stime VaR significative.

        Returns
        -------
        pd.Series
            Pesi filtrati. Sommano a 1 se solo CVaR scatta,
            a < 1 se anche drawdown scatta (eccesso = cash).

        Notes
        -----
        Con meno di 20 osservazioni in ``returns`` i pesi vengono restituiti
        invariati: le stime VaR su serie brevi sono statisticamente prive di
        significato.
        """
        if len(returns) < 20:
            return weights.copy()
        weights = weights.copy().astype(float)
        weights = self._apply_cvar_truncation(weights, returns)
        weights = self._apply_drawdown_cap(weights, returns)
        return weights

    def _apply_cvar_truncation(
        self,
        weights: pd.Series,
        returns: pd.DataFrame,
    ) -> pd.Series:
        """Riduce iterativamente il peso dell'asset con max component VaR."""
        r_p = returns[list(weights.index)] @ weights
        current_cvar = cvar_historical(r_p, self.confidence)

        if current_cvar <= self.cvar_limit:
            return weights

        max_iter = len(weights) * 50
        converged = False
        for _ in range(max_iter):
            comp = component_var(weights, returns[list(weights.index)], self.confidence)
            worst = comp.idxmax()

            new_w = float(weights[worst]) * (1.0 - self.truncation_step)
            weights = weights.copy()
            weights[worst] = max(new_w, self.min_weight)
            weights = self._renormalize(weights)

            r_p = returns[list(weights.index)] @ weights
            current_cvar = cvar_historical(r_p, self.confidence)

            if current_cvar <= self.cvar_limit:
                converged = True
                break

            if (weights <= self.min_weight + 1e-10).all():
                break

        if not converged:
            logger.warning(
                "RiskFilter: CVaR truncation did not converge "
                "(current CVaR=%.4f, limit=%.4f). Returning best-effort weights.",
                current_cvar,
                self.cvar_limit,
            )

        return weights

    def _apply_drawdown_cap(
        self,
        weights: pd.Series,
        returns: pd.DataFrame,
        lookback: int = 504,
    ) -> pd.Series:
        """Scala tutti i pesi se il drawdown recente supera il limite.

        Il drawdown è calcolato sul portafoglio *proposto* applicato agli
        ultimi ``lookback`` giorni (default 504 = ~2 anni). Usare solo la
        finestra recente è corretto sia per il live trading che per il
        walk-forward backtest: evita che un crash di 10+ anni fa condizioni
        ogni rebalance futuro.
        """
        recent = returns.iloc[-lookback:]
        r_p = recent[list(weights.index)] @ weights
        dd = self._max_drawdown(r_p)

        if dd <= self.drawdown_limit:
            return weights

        scale = self.drawdown_limit / dd
        return weights * scale

    @staticmethod
    def _renormalize(weights: pd.Series) -> pd.Series:
        """Ri-normalizza i pesi a somma 1."""
        total = weights.sum()
        if total <= 0:
            return weights
        return weights / total

    @staticmethod
    def _max_drawdown(returns: pd.Series) -> float:
        """Drawdown massimo storico dalla serie di rendimenti."""
        cumulative = (1.0 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (running_max - cumulative) / running_max.clip(lower=1e-10)
        return float(drawdown.max())
