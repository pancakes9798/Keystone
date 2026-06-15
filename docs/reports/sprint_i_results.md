# Sprint I — Risultati Validazione Onesta 20 ETF

**Config hash:** `882c5ac1cedc8087`  
**Universo:** 20 ETF ex-ante (design doc 2026-04-28)  
**IS:** 2005-01-01 → 2011-12-31  |  **OOS:** 2012-01-01 → 2026-06-01  
**Fonte prezzi:** Yahoo auto_adjust=True (total return)  
**Note:** Universo ex-ante dal design doc 2026-04-28. Griglie originali, NON quelle estese v16. OOS toccato una sola volta. Fonte total return: Yahoo auto_adjust=True (RIC suffix stripped: SPY.P->SPY etc.). Refinitiv CORAX e' solo price return — verificato empiricamente 2026-06-10 con scripts/validate_total_return.py (gap dividendi non chiuso). Refinitiv resta la fonte live/paper.  

## Parametri IS congelati

- **Conservative**: damp=0.50, mom=0.00
- **Balanced**: damp=0.25, mom=0.50
- **Aggressive**: damp=0.25, mom=0.70

## Metriche OOS

| Strategy | CAGR | Sharpe | Sortino | MaxDD | Calmar | $10k→ |
|---|---|---|---|---|---|---|
| SPY B&H | 15.0% | 0.92 | 1.12 | -33.7% | 0.45 | $65,198 |
| 60/40 SPY/IEF | 9.8% | 1.00 | 1.25 | -21.0% | 0.46 | $38,094 |
| Conservative | 2.4% | 1.49 | 2.00 | -3.1% | 0.77 | $13,650 |
| Balanced | 5.1% | 0.95 | 1.13 | -12.9% | 0.40 | $19,488 |
| Aggressive | 5.2% | 0.94 | 1.13 | -13.3% | 0.39 | $19,725 |

## Deflated Sharpe Ratio (sensibilità)

**n_trials reale (registry):** 21  

| Strategy | n_trials | DSR | Pass (≥0.75)? |
|---|---|---|---|
| Conservative (real) | 21 | 1.000 | ✓ |
| Conservative (×2) | 42 | 0.999 | ✓ |
| Conservative (×4) | 84 | 0.998 | ✓ |
| Balanced (real) | 21 | 0.931 | ✓ |
| Balanced (×2) | 42 | 0.886 | ✓ |
| Balanced (×4) | 84 | 0.830 | ✓ |
| Aggressive (real) | 21 | 0.930 | ✓ |
| Aggressive (×2) | 42 | 0.884 | ✓ |
| Aggressive (×4) | 84 | 0.828 | ✓ |

## CPCV con refit (proxy: griglia Balanced) — CPCVConfig n_splits=6, n_test_splits=2, 15 fold

> Approximazione documentata: train=concatenazione segmenti contigui purged+embargoed; calibrate_fn = grid-search Balanced su segmenti ≥252 obs, Sharpe medio; evaluate_fn = engine run sul test fold, Sharpe aggregato. Overlap di covarianza tra fold accettato (LdP: purging riguarda le label, non le feature).

| Fold | n_train | n_test | best_damp | best_mom | test_sharpe |
|---|---|---|---|---|---|
| 0 | 3536 | 1796 | 0.25 | 0.3 | 0.893 |
| 1 | 3462 | 1796 | 0.25 | 0.3 | 0.747 |
| 2 | 3463 | 1795 | 0.25 | 0.7 | 1.022 |
| 3 | 3463 | 1795 | 0.0 | 0.3 | 0.693 |
| 4 | 3516 | 1795 | 0.25 | 0.3 | 1.518 |
| 5 | 3515 | 1796 | 0.0 | 0.3 | 0.990 |
| 6 | 3442 | 1795 | 0.25 | 0.7 | 1.175 |
| 7 | 3442 | 1795 | 0.0 | 0.3 | 0.794 |
| 8 | 3495 | 1795 | 0.0 | 0.3 | 1.711 |
| 9 | 3516 | 1795 | 0.25 | 0.5 | 0.757 |
| 10 | 3442 | 1795 | 0.0 | 0.3 | 0.371 |
| 11 | 3495 | 1795 | 0.25 | 0.5 | 1.328 |
| 12 | 3517 | 1794 | 0.25 | 0.3 | 0.535 |
| 13 | 3496 | 1794 | 0.25 | 0.7 | 1.424 |
| 14 | 3570 | 1794 | 0.25 | 0.7 | 1.285 |

**Frazione fold con Sharpe > 0:** 100%

## VERDETTO

**EDGE CONFERMATO — almeno una strategia batte SPY in Sharpe OOS con DSR(n_trials reale) ≥ 0.75**

> Criteri: EDGE CONFERMATO se Sharpe OOS di almeno una strategia > Sharpe SPY E DSR(n_trials reale) ≥ 0.75 per quella strategia.

_Report generato automaticamente da scripts/backtest_sprint_i.py_
## Interpretazione onesta (sintesi del controller, 2026-06-10)

Il criterio formale dello script (Sharpe OOS > SPY con DSR ≥ 0.75) è soddisfatto da tutte e tre le strategie, ma la lettura economica corretta è più sfumata:

1. **Il CAGR del vecchio Sprint E era l'universo, non la strategia.** Su 16 mega-cap scelte ex-post il tearsheet dichiarava CAGR 12.7%; sull'universo ex-ante di 20 ETF l'Aggressive rende il 5.2%. La differenza (~7.5 punti/anno) era selection bias.
2. **Lo Sharpe invece sopravvive.** 0.94–0.95 onesto contro 0.97 del backtest viziato: la meccanica HRP+trend+momentum produce davvero un profilo di rischio migliore di SPY B&H (DD -13% vs -34%), e il DSR regge anche a n_trials ×4.
3. **Ma il 60/40 resta imbattuto in Sharpe** (1.00 vs 0.95, a parità di costi): la macchina complessa non batte il benchmark ingenuo risk-adjusted. Batte SPY, non il 60/40.
4. **Conservative (Sharpe 1.49, MaxDD -3.1%, CAGR 2.4%)** è di fatto un portafoglio difensivo quasi-obbligazionario: Sharpe eccellente ma rendimento da T-bill+. Interessante come sleeve difensiva, non come strategia singola.
5. **CPCV con refit: 15/15 fold positivi** (mean Sharpe 1.02, min 0.37) — la robustezza della meccanica è genuina, e i parametri selezionati per fold sono stabili (damp 0/0.25, mom 0.3–0.7).

**Conclusione:** la strategia è un *risk-managed diversified portfolio* legittimo con Sharpe da 0.95, non una macchina da alpha. Il claim onesto è "migliore di SPY risk-adjusted, comparabile ma inferiore a un 60/40, con drawdown molto più contenuti di entrambi". I prossimi sforzi dovrebbero puntare a battere il 60/40, non SPY.
