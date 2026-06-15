"""VolTargetOverlay — scala i pesi di un base optimizer a una vol target (Moreira-Muir 2017).

Stima causale: vol realizzata del portafoglio base sugli ultimi ``vol_lookback``
giorni della storia ricevuta (che nell'engine è già strictly < data del segnale).
scale = clip(target_vol / realized_vol, 0, max_leverage). Il de-lever (<1) lascia
il resto in cash; il lever (>1) presuppone funding — vedi BacktestConfig.

Duck-types l'interfaccia standard ``optimize(returns) -> pd.Series`` in modo che
BacktestEngine e DailyRunner possano usarlo senza modifiche.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class VolTargetOverlay:
    """Wrappa un optimizer base e scala i pesi alla vol target.

    Parameters
    ----------
    base_optimizer : object with .optimize(returns: pd.DataFrame) -> pd.Series
        Optimizer base da wrappare. Deve implementare la stessa interfaccia
        duck-type del BacktestEngine.
    target_vol : float
        Volatilità annualizzata target (es. 0.10 = 10%).
    max_leverage : float
        Cap superiore allo scaling (leva massima). Default 2.0.
    vol_lookback : int
        Finestra (in giorni di borsa) per la stima della vol realizzata.
        Default 63 (~3 mesi). Se la storia disponibile è inferiore a questa
        finestra, restituisce i pesi base invariati.
    """

    def __init__(
        self,
        base_optimizer: Any,
        target_vol: float,
        max_leverage: float = 2.0,
        vol_lookback: int = 63,
    ) -> None:
        self._base = base_optimizer
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.vol_lookback = vol_lookback

    def optimize(self, returns: pd.DataFrame) -> pd.Series:
        """Calcola i pesi scalati alla vol target.

        Parameters
        ----------
        returns : pd.DataFrame
            Storia dei rendimenti disponibile al momento del segnale
            (strictly < data del segnale nell'engine — nessun lookahead).

        Returns
        -------
        pd.Series
            Pesi scalati. Σw ≤ max_leverage se scale ≥ 1 (leva),
            Σw < base Σw se scale < 1 (de-lever, eccesso = cash).
            Se la storia è insufficiente restituisce i pesi base invariati.
        """
        base_w = self._base.optimize(returns)
        window = returns.reindex(columns=base_w.index).tail(self.vol_lookback)

        if len(window) < self.vol_lookback:
            logger.info(
                "VolTarget: storia insufficiente (%d < %d) — pesi base invariati.",
                len(window),
                self.vol_lookback,
            )
            return base_w

        port_ret = (window.fillna(0.0) * base_w).sum(axis=1)
        realized = float(port_ret.std() * np.sqrt(252))

        if realized <= 1e-8:
            logger.warning("VolTarget: volatilità realizzata quasi nulla — pesi base invariati.")
            return base_w

        scale = min(self.target_vol / realized, self.max_leverage)
        logger.debug(
            "VolTarget: realized=%.2f%% target=%.2f%% scale=%.4f",
            realized * 100,
            self.target_vol * 100,
            scale,
        )
        return base_w * scale
