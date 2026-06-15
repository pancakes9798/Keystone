"""Protocol per tutti gli optimizer di portafoglio."""
from __future__ import annotations

from typing import Protocol

import pandas as pd


class Optimizer(Protocol):
    """Interface condivisa per tutti gli optimizer di portafoglio.

    Qualsiasi oggetto con optimize(returns) -> pd.Series soddisfa questo Protocol.
    """

    def optimize(self, returns: pd.DataFrame) -> pd.Series:
        """Calcola i pesi ottimali a partire dalla storia dei rendimenti.

        Parameters
        ----------
        returns : pd.DataFrame
            DataFrame indicizzato per data (righe=date, colonne=nomi asset).

        Returns
        -------
        weights : pd.Series
            Pesi non negativi indicizzati per nome asset, sommanti a 1.
        """
        ...
