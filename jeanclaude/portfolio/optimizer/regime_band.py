"""RegimeBandOverlay — banda equity [floor, cap] condizionata al regime HMM.

Sprint K (2026-06-10). Il base optimizer produce pesi Σ=1; l'overlay vincola la
quota equity nella banda del regime corrente scalando proporzionalmente i due
sleeve (equity e non-equity). La detection è PRE-CALCOLATA walk-forward con
``walkforward_regimes`` (fit espandente su macro ≤ t — causale; le bande non
influenzano la detection, quindi i fit si riusano su tutte le combo di griglia).

NOTE sulla cache di ``walkforward_regimes``
-------------------------------------------
``pd.DataFrame.attrs`` non sopravvive al round-trip parquet (viene perso alla
lettura). La cache viene quindi validata confrontando SOLO le date dell'indice
con quelle richieste. Il chiamante è responsabile di includere il ``random_state``
nel nome del file (es. ``wf_regimes_{config_hash}_{random_state}.parquet``) per
evitare collisioni tra esperimenti con seed diversi.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

logger = logging.getLogger(__name__)


class RegimeBandOverlay:
    """Wrappa un base optimizer e vincola la quota equity nella banda del regime.

    Parameters
    ----------
    base_optimizer : Any
        Qualsiasi oggetto con metodo ``optimize(returns: pd.DataFrame) -> pd.Series``.
    equity_rics : frozenset[str] | set[str]
        Insieme dei ticker considerati «equity sleeve».
    bands : dict[str, tuple[float, float]]
        Mappa regime_name → (floor, cap). Chiavi devono corrispondere ai nomi
        restituiti da ``regime_lookup`` (es. "EXPANSION", "CONTRACTION", "TRANSITION").
        Regimi non presenti nella mappa usano i pesi base senza modifiche.
    regime_lookup : Callable[[pd.Timestamp], str]
        Funzione che data una data ritorna il nome del regime corrente.
        Tipicamente una closure su un ``pd.Series`` pre-calcolato da
        ``walkforward_regimes``.
    """

    def __init__(
        self,
        base_optimizer: Any,
        equity_rics: frozenset[str] | set[str],
        bands: dict[str, tuple[float, float]],
        regime_lookup: Callable[[pd.Timestamp], str],
    ) -> None:
        self._base = base_optimizer
        self._equity = frozenset(equity_rics)
        self._bands = bands
        self._regime_lookup = regime_lookup

    def optimize(self, returns: pd.DataFrame) -> pd.Series:
        """Applica l'overlay al risultato del base optimizer.

        Parameters
        ----------
        returns : pd.DataFrame
            Matrice dei rendimenti (stessa interfaccia del base optimizer).
            L'ultima data dell'indice è usata come data di riferimento per
            la lookup del regime.

        Returns
        -------
        pd.Series
            Pesi finali con Σ=1 (eccetto il caso limite equity-only + cap=0).
        """
        base_w = self._base.optimize(returns)
        as_of = pd.Timestamp(returns.index[-1])
        regime = self._regime_lookup(as_of)
        band = self._bands.get(regime)
        if band is None:
            logger.warning("RegimeBand: regime %r senza banda — pesi base.", regime)
            return base_w
        floor, cap = band
        eq_mask = base_w.index.isin(self._equity)
        eq_sum = float(base_w[eq_mask].sum())
        target = min(max(eq_sum, floor), cap)
        if abs(target - eq_sum) < 1e-12:
            return base_w
        w = base_w.copy()
        non_eq_sum = float(base_w[~eq_mask].sum())
        if eq_sum > 0:
            w[eq_mask] = base_w[eq_mask] * (target / eq_sum)
        elif target > 0:
            # base senza equity ma floor > 0: equal-weight sullo sleeve equity disponibile
            eq_assets = [c for c in base_w.index if c in self._equity]
            if eq_assets:
                w[eq_assets] = target / len(eq_assets)
        remainder = 1.0 - target
        if non_eq_sum > 0:
            w[~eq_mask] = base_w[~eq_mask] * (remainder / non_eq_sum)
        else:
            # nessuno sleeve difensivo di base: il resto resta cash (Σ < 1)
            w[~eq_mask] = 0.0
        logger.debug(
            "RegimeBand: %s equity %.2f→%.2f (banda %.2f-%.2f)",
            regime, eq_sum, target, floor, cap,
        )
        return w


def walkforward_regimes(
    macro: pd.DataFrame,
    rebalance_dates: Iterable[pd.Timestamp],
    random_state: int = 42,
    cache_path: str | Path | None = None,
    min_obs: int = 120,
) -> pd.Series:
    """Regime HMM a ogni data di rebalance, fit espandente su macro ≤ t (causale).

    Ritorna una ``pd.Series`` con indice di date (Timestamp) e valori stringa
    (nomi di ``RegimeLabel``, es. "EXPANSION"). Le date vengono processate in
    ordine e ogni fit usa solo macro ≤ t — nessun look-ahead.

    Cache
    -----
    Con ``cache_path`` i risultati sono persistiti su parquet e riusati se le
    date dell'indice coincidono con quelle richieste. Poiché ``pd.DataFrame.attrs``
    non sopravvive al round-trip parquet, il ``random_state`` NON viene verificato
    dalla cache: il chiamante deve includere il seed nel nome del file (es.
    ``wf_regimes_{config_hash}_{random_state}.parquet``).

    Parameters
    ----------
    macro : pd.DataFrame
        Serie macro grezze (date-indexed). Le colonne vengono passate a
        ``build_state_variables`` che applica normalizzazione expanding causale.
    rebalance_dates : Iterable[pd.Timestamp]
        Date di rebalance per cui calcolare il regime.
    random_state : int
        Seed per ``RegimeDetector`` — garantisce determinismo tra run.
    cache_path : str | Path | None
        Percorso file parquet di cache. Se None non viene usata la cache.
        Includere il ``random_state`` nel nome per evitare collisioni.
    min_obs : int
        Minimo numero di osservazioni (post-dropna) richieste per fittare
        il modello. Date con storia insufficiente ricevono regime "UNKNOWN".

    Returns
    -------
    pd.Series
        Serie ``date → regime_name`` (str). Indice: DatetimeIndex, nome="date".
        Valori: nomi di ``RegimeLabel`` (es. "EXPANSION") o "UNKNOWN".
    """
    from jeanclaude.data.transform.features import build_state_variables
    from jeanclaude.signals.macro.detector import RegimeDetector
    from jeanclaude.signals.macro.labels import RegimeLabel

    dates = [pd.Timestamp(d) for d in rebalance_dates]
    cache_path = Path(cache_path) if cache_path is not None else None

    # Cache hit: valida solo sulle date (attrs non sopravvive al parquet).
    # Il random_state deve essere codificato nel nome del file dal chiamante.
    if cache_path is not None and cache_path.exists():
        cached = pd.read_parquet(cache_path)["regime"]
        cached.index = pd.to_datetime(cached.index)
        if list(cached.index) == dates:
            logger.info("walkforward_regimes: cache hit (%d date).", len(dates))
            return cached

    out: dict[pd.Timestamp, str] = {}
    for i, d in enumerate(dates):
        sv = build_state_variables(macro.loc[:d]).dropna()
        if len(sv) < min_obs:
            out[d] = "UNKNOWN"
            continue
        try:
            detector = RegimeDetector(random_state=random_state)
            result = detector.fit(sv)
            out[d] = RegimeLabel(int(result.labels.iloc[-1])).name
        except Exception as exc:
            logger.warning(
                "walkforward_regimes: fit fallito a %s (%s) — UNKNOWN.", d.date(), exc
            )
            out[d] = "UNKNOWN"
        if (i + 1) % 25 == 0:
            logger.info("walkforward_regimes: %d/%d", i + 1, len(dates))

    series = pd.Series(out, name="regime")
    series.index.name = "date"
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        series.to_frame().to_parquet(cache_path)
    return series
