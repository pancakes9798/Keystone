#!/usr/bin/env python3
"""Confronta Refinitiv TRDPRC_1 (raw e adjusted) con Yahoo auto_adjust.

Domanda: gli adjustments CORAX di Refinitiv catturano i dividendi ordinari
(→ total return) o solo split/eventi straordinari (→ price return)?

Metodo: per ticker ad alto dividend yield (KO, JNJ, XOM) e uno a basso yield
(AMZN), confronto del CAGR 2015-2025 tra:
  - Refinitiv senza adjustments (price return puro)
  - Refinitiv con CCH,CRE,RTS,RPO
  - Yahoo auto_adjust=True (split+dividend adjusted = total return proxy)
Se CAGR(Refinitiv adj) ≈ CAGR(Yahoo) → adjusted è total return.
Se CAGR(Refinitiv adj) ≈ CAGR(Refinitiv raw) + ~0 → i dividendi NON ci sono:
il backtest deve usare Yahoo per il total return o un campo TR dedicato.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

from jeanclaude.data.ingestion.refinitiv import RefinitivSource  # noqa: E402
from jeanclaude.data.ingestion.yahoo import YahooSource  # noqa: E402

PAIRS = {"KO.N": "KO", "JNJ.N": "JNJ", "XOM.N": "XOM", "AMZN.O": "AMZN"}
START, END = "2015-01-02", "2025-12-31"


def _cagr(prices: pd.Series) -> float:
    prices = prices.dropna()
    years = (prices.index[-1] - prices.index[0]).days / 365.25
    return (prices.iloc[-1] / prices.iloc[0]) ** (1 / years) - 1


def main() -> None:
    ref = RefinitivSource(session_type="platform")
    yah = YahooSource()
    rics, yts = list(PAIRS), list(PAIRS.values())

    raw = ref.get_prices(rics, start=START, end=END)
    adj = ref.get_prices(rics, start=START, end=END,
                         adjustments=("CCH", "CRE", "RTS", "RPO"))
    ytr = yah.get_prices(yts, start=START, end=END)

    print(f"\n{'Ticker':<8} {'Ref raw':>9} {'Ref adj':>9} {'Yahoo TR':>9}  verdetto")
    print("-" * 55)
    verdicts = []
    for ric, yt in PAIRS.items():
        c_raw, c_adj, c_y = _cagr(raw[ric]), _cagr(adj[ric]), _cagr(ytr[yt])
        # se l'adjusted recupera >70% del gap dividendi vs Yahoo → TR ok
        gap_total = c_y - c_raw
        gap_closed = (c_adj - c_raw) / gap_total if abs(gap_total) > 1e-4 else 1.0
        verdict = "TR OK" if gap_closed > 0.7 else "SOLO PRICE RETURN"
        verdicts.append(gap_closed > 0.7)
        print(f"{ric:<8} {c_raw:>8.2%} {c_adj:>8.2%} {c_y:>8.2%}  {verdict} (gap chiuso {gap_closed:.0%})")

    print("\nVERDETTO COMPLESSIVO:",
          "Refinitiv adjusted ≈ total return — usare adjustments nel backtest."
          if all(verdicts) else
          "Refinitiv adjusted NON è total return — per i backtest usare Yahoo "
          "auto_adjust (o un campo TR dedicato) e documentarlo nella config.")


if __name__ == "__main__":
    main()
