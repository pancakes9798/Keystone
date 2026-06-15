"""Generatore bounded dello spazio di ricerca formulaico returns-only (Sprint T).

Ogni candidato è uno StrategyConfig con un signal_fn(daily_returns, asof)->pd.Series
(indice RIC, più alto = preferito), pronto per simulate_monthly. Enumerazione
dichiarata e contata; IC pre-screen e corr-pruning sono applicati a valle
dall'harness sul solo TRAIN.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from jeanclaude.signals.formulaic import fields, operators as ops

SignalFn = Callable[[pd.DataFrame, pd.Timestamp], pd.Series]


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    tier: str
    signal_fn: SignalFn


# ---- caratteristiche base (ritornano una Series RIC->valore grezzo a asof) ----

def _char_retvol(d):
    def f(dr, asof):
        w = dr.loc[:asof].iloc[-d:]
        return -w.std(ddof=1)  # -vol: più alto = meno volatile
    return f

def _char_idiovol(d):
    def f(dr, asof):
        resid = fields.resid_return(dr.loc[:asof], window=60)
        return -resid.iloc[-d:].std(ddof=1)
    return f

def _char_maxret(d):
    def f(dr, asof):
        return -dr.loc[:asof].iloc[-d:].max()  # lottery aversion
    return f

def _char_skew(d):
    def f(dr, asof):
        return -dr.loc[:asof].iloc[-d:].skew()
    return f

def _char_autocorr(d):
    def f(dr, asof):
        w = dr.loc[:asof].iloc[-(d + 1):]
        return w.apply(lambda c: c.autocorr(lag=1))
    return f

def _char_momentum_signed(mb, ms):
    sign = -1.0 if (mb, ms) in {(1, 0), (24, 13), (36, 13)} else 1.0  # reversal families
    def f(dr, asof):
        return sign * fields.cumret(dr, asof, mb, ms)
    return f


def _build_chars() -> dict:
    """~24 famiglie caratteristica con griglia di lookback (dichiarata)."""
    chars = {}
    for mb, ms in [(12, 1), (9, 1), (6, 1), (3, 1), (1, 0), (24, 13), (36, 13)]:
        chars[f"mom{mb}_{ms}"] = _char_momentum_signed(mb, ms)
    for d in [21, 63, 126, 189, 252]:
        chars[f"retvol{d}"] = _char_retvol(d)
    for d in [63, 126, 252]:
        chars[f"idiovol{d}"] = _char_idiovol(d)
    for d in [21, 63, 126]:
        chars[f"maxret{d}"] = _char_maxret(d)
    for d in [63, 126, 252]:
        chars[f"skew{d}"] = _char_skew(d)
    for d in [21, 63, 126]:
        chars[f"autocorr{d}"] = _char_autocorr(d)
    return chars  # 7+5+3+3+3+3 = 24 famiglie


_NORMALIZERS = {"rank": ops.cs_rank, "z": ops.cs_zscore}


def _wrap(char_fn, normalizer):
    def f(dr, asof):
        return normalizer(char_fn(dr, asof).dropna())
    return f


def _tier_b() -> list[StrategyConfig]:
    """24 famiglie × 2 normalizzatori = 48 config depth-1."""
    out = []
    for cname, cfn in _build_chars().items():
        for nname, nfn in _NORMALIZERS.items():
            out.append(StrategyConfig(f"B:{cname}:{nname}", "B", _wrap(cfn, nfn)))
    return out


def _tier_a() -> list[StrategyConfig]:
    # 3 seed reversal returns-only ispirati a Kakushadze (#9/#19/#34 ridotti).
    def k_rev_close1(dr, asof):
        w = dr.loc[:asof]
        return ops.cs_rank(-(w.iloc[-1]))
    def k_reversion_trend(dr, asof):
        # NB: è REVERSAL — vende i winner con trend su, compra i loser con trend giù (#19-like).
        lp = fields.log_price(dr.loc[:asof])
        trend = np.sign(lp.iloc[-1] - lp.iloc[-8])
        mom = (1 + dr.loc[:asof].iloc[-250:]).prod() - 1
        return ops.cs_rank(-(trend) * (1 + ops.cs_rank(mom)))
    def k_volratio(dr, asof):
        w = dr.loc[:asof]
        vr = w.iloc[-2:].std(ddof=1) / w.iloc[-5:].std(ddof=1)
        return (1 - ops.cs_rank(vr)) + (1 - ops.cs_rank(w.iloc[-1]))
    return [StrategyConfig("A:k_rev_close1", "A", k_rev_close1),
            StrategyConfig("A:k_reversion_trend", "A", k_reversion_trend),
            StrategyConfig("A:k_volratio", "A", k_volratio)]


def _aligned_rank(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    """cs_rank di due caratteristiche grezze sull'INTERSEZIONE dei RIC validi.

    Evita i NaN da footprint-NaN divergenti: senza intersezione, un RIC valido in
    `a` ma NaN in `b` produrrebbe NaN nel prodotto → riduzione silenziosa dell'universo.
    """
    a, b = a.dropna(), b.dropna()
    common = a.index.intersection(b.index)
    return ops.cs_rank(a[common]), ops.cs_rank(b[common])


def _tier_c() -> list[StrategyConfig]:
    def mom_x_lowvol(dr, asof):
        # `-_char_retvol` annulla la negazione interna → vol grezza; rank alto = vol ALTA;
        # (1-v) = rank vol BASSA. m*(1-v) = momentum × low-vol (interazione moltiplicativa).
        m, v = _aligned_rank(fields.cumret(dr, asof, 12, 1), -_char_retvol(126)(dr, asof))
        return m * (1 - v)
    def mom_x_lowidiovol(dr, asof):
        m, v = _aligned_rank(fields.cumret(dr, asof, 12, 1), -_char_idiovol(126)(dr, asof))
        return m * (1 - v)
    def revol_x_skew(dr, asof):
        # entrambe già "alto = preferito" (-vol, -skew); prodotto premia chi è buono su ENTRAMBE.
        rv, sk = _aligned_rank(_char_retvol(126)(dr, asof), _char_skew(252)(dr, asof))
        return rv * sk
    return [StrategyConfig("C:mom_x_lowvol", "C", mom_x_lowvol),
            StrategyConfig("C:mom_x_lowidiovol", "C", mom_x_lowidiovol),
            StrategyConfig("C:revol_x_skew", "C", revol_x_skew)]


def build_search_space() -> list[StrategyConfig]:
    """Spazio completo dichiarato (Tier A + B + C). N = len(...)."""
    return _tier_a() + _tier_b() + _tier_c()
