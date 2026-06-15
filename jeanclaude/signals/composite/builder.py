"""
CompositeSignalBuilder — converte RegimeResult in score per asset class.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from jeanclaude.signals.macro.labels import RegimeLabel, RegimeResult
from .types import CompositeSignal, RegimeTable

if TYPE_CHECKING:
    from jeanclaude.signals.news.types import NewsSentiment as _NewsSentiment

_ALL_REGIMES = frozenset(RegimeLabel)


class CompositeSignalBuilder:
    """Costruisce un CompositeSignal da un RegimeResult con una regime table configurabile.

    Parameters
    ----------
    regime_table : RegimeTable
        Mapping da ogni RegimeLabel a {asset_class: score ∈ [-1, 1]}.
        Deve contenere tutti e 3 i RegimeLabel con chiavi asset class identiche.
    """

    def __init__(self, regime_table: RegimeTable) -> None:
        self._validate_table(regime_table)
        self._table = regime_table

    @staticmethod
    def _validate_table(table: RegimeTable) -> None:
        # Tutti e 3 i regimi presenti
        missing = _ALL_REGIMES - set(table.keys())
        if missing:
            names = ", ".join(r.name for r in missing)
            raise ValueError(
                f"regime_table manca il/i regime(i): {names}. "
                "Tutti e 3 i RegimeLabel (EXPANSION, CONTRACTION, TRANSITION) sono obbligatori."
            )
        # Stesse asset class in tutti i regimi
        asset_sets = [frozenset(v.keys()) for v in table.values()]
        if len(set(asset_sets)) != 1:
            raise ValueError(
                "regime_table ha asset class inconsistenti tra i regimi. "
                "Ogni regime deve avere lo stesso set di asset class."
            )
        # Score in [-1, 1]
        for regime, assets in table.items():
            for asset, score in assets.items():
                if not (-1.0 <= score <= 1.0):
                    raise ValueError(
                        f"Score fuori range per asset '{asset}' nel regime "
                        f"{regime.name}: {score}. Deve essere in [-1, 1]."
                    )

    def build(
        self,
        result: RegimeResult,
        news_sentiment: "_NewsSentiment | None" = None,
        macro_weight: float = 0.7,
    ) -> CompositeSignal:
        """Produce un CompositeSignal dall'ultima osservazione del RegimeResult.

        Parameters
        ----------
        result : RegimeResult
            Output di RegimeDetector.fit() o RegimeDetector.predict().
        news_sentiment : NewsSentiment, optional
            Se fornito e confidence >= 0.05, viene blendato col segnale macro:
            ``final = macro_weight·macro + (1-macro_weight)·news·confidence``.
            Se None o confidence < 0.05, viene ignorato (solo macro).
        macro_weight : float
            Peso del segnale macro nel blend (default 0.7).

        Returns
        -------
        CompositeSignal
        """
        # Guard: empty-result check
        if result.probabilities.empty:
            raise ValueError(
                "result.probabilities is empty. "
                "Call RegimeDetector.fit() with at least one observation."
            )

        # Verify column ordering matches RegimeLabel enum values BEFORE extracting data.
        # Columns must be [0, 1, 2] = [EXPANSION, CONTRACTION, TRANSITION].
        # This ordering is guaranteed by _build_result in detector.py.
        expected_cols = [r.value for r in RegimeLabel]
        if list(result.probabilities.columns) != expected_cols:
            raise ValueError(
                f"result.probabilities columns must be {expected_cols} "
                f"(matching RegimeLabel enum values). Got: {list(result.probabilities.columns)}"
            )

        # Probabilità posteriori per l'ultimo timestep: shape (3,)
        proba = result.probabilities.iloc[-1].values.astype(np.float64)

        # Score pesato per asset class: score(asset) = Σ_r p(r) * table[r][asset]
        asset_classes = list(next(iter(self._table.values())).keys())
        score_values = np.zeros(len(asset_classes))
        for label in RegimeLabel:
            regime_scores = np.array([self._table[label][a] for a in asset_classes])
            score_values += proba[label.value] * regime_scores
        macro_scores = pd.Series(score_values, index=asset_classes)

        # Confidence = 1 - entropia normalizzata (log naturale)
        n = len(RegimeLabel)  # 3
        p = np.clip(proba, 1e-12, 1.0)  # evita log(0)
        entropy_norm = -np.sum(p * np.log(p)) / np.log(n)
        confidence = float(np.clip(1.0 - entropy_norm, 0.0, 1.0))

        # Regime dominante
        regime = RegimeLabel(int(np.argmax(proba)))

        # Timestamp = ultima data nel RegimeResult
        timestamp = result.labels.index[-1]

        # --- News blend ---
        if news_sentiment is not None and news_sentiment.confidence >= 0.05:
            news_scores = news_sentiment.scores.reindex(macro_scores.index, fill_value=0.0)
            alpha = float(macro_weight)
            final_scores = alpha * macro_scores + (1.0 - alpha) * news_scores * news_sentiment.confidence
        else:
            final_scores = macro_scores

        return CompositeSignal(
            scores=final_scores,
            confidence=confidence,
            regime=regime,
            regime_proba=proba,
            timestamp=timestamp,
        )
