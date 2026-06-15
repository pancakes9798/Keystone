# Sprint L — Base Strutturale Equity + De-risk Trend: dominare il 60/40?

**Config hash:** `227fef2fdc4f7055`  
**Universo:** 20 ETF (identico a Sprint I — ex-ante dal design doc)  
**IS:** 2005-01-01 → 2011-12-31  |  **OOS:** 2012-01-01 → 2026-06-01  
**Fonte prezzi:** Yahoo auto_adjust=True (total return)  
**Note:** Sprint L: floor strutturale equity_min ∈ {0.5, 0.6, 0.7} × damp ∈ {0.25, 0.5}, mom=0.5 congelato. Criterio successo ex-ante: Sharpe ≥ 60/40 AND MaxDD ≤ 60/40 AND CAGR ≥ 8%. Fonte TR Yahoo. Lezioni I/J/K: no leva, no HMM.  

## Parametri selezionati (IS)

- `equity_min` = 0.5  (IS grid: [0.5, 0.6, 0.7])
- `damp_factor` = 0.25  (IS grid: [0.25, 0.5])
- `momentum_strength` = 0.5  (CONGELATO da Sprint I Balanced)

> ⚠️  **AVVISO BORDO GRIGLIA:** il combo selezionato (equity_min=0.5, damp=0.25) è sul bordo della griglia IS. Il parametro ottimale potrebbe essere fuori griglia — interpretare con cautela.

## Statistiche OOS

- **Equity share media effettiva OOS:** 45.8%
- **Turnover annuo OOS:** 160.3%

## Metriche OOS

| Strategy | CAGR | Sharpe | Sortino | MaxDD | Calmar | $10k→ |
|---|---|---|---|---|---|---|
| SPY B&H | 15.0% | 0.92 | 1.12 | -33.7% | 0.45 | $65,198 |
| 60/40 SPY/IEF | 9.8% | 1.00 | 1.25 | -21.0% | 0.46 | $38,094 |
| Sprint I Balanced (unlevered) | 5.1% | 0.95 | 1.13 | -12.9% | 0.40 | $19,487 |
| Sprint L (em=0.5, d=0.25) | 7.3% | 0.91 | 1.14 | -16.4% | 0.45 | $25,753 |

## Deflated Sharpe Ratio

**n_trials questo esperimento (config_hash):** 6  
**n_trials cumulativi (+ 33 da I+J+K):** 39  

| Strategy | n_trials | DSR | Pass (≥0.75)? |
|---|---|---|---|
| Sprint L (em=0.5, d=0.25) (n=L) | 6 | 0.976 | ✓ |
| Sprint L (em=0.5, d=0.25) (n=L+prior) | 39 | 0.868 | ✓ |

## VERDETTI (tre criteri ex-ante — dichiarati PRIMA del run OOS)

> Nessuna selezione a posteriori del criterio — tutti e tre riportati.

### Criterio (a): Sharpe OOS ≥ Sharpe 60/40

**FAIL ✗** — Sprint L Sharpe=0.91 vs 60/40 Sharpe=1.00

### Criterio (b): MaxDD OOS non più profondo del MaxDD 60/40

**PASS ✓** — Sprint L MaxDD=-16.4% vs 60/40 MaxDD=-21.0% (non più profondo)

### Criterio (c): CAGR OOS ≥ 8%

**FAIL ✗** — Sprint L CAGR=7.3% vs soglia=8.0%

### Verdetto complessivo: **FAIL (almeno uno non soddisfatto)**


---
_Report generato automaticamente da scripts/backtest_sprint_l.py_
## Interpretazione onesta (controller, 2026-06-11)

Verdetto ex-ante: **FAIL complessivo** (1 criterio su 3). Ma è il fallimento più interessante della serie:

- **(b) PASSA**: MaxDD -16.4% contro -21.0% del 60/40 — il de-risk trend continua a fare il suo lavoro.
- **(a) e (c) falliscono di poco**: Sharpe 0.91 vs 1.00, CAGR 7.3% vs 8% — Sprint L è il punto più vicino al 60/40 mai raggiunto dalla macchina (Sprint I: CAGR 5.1%; J: 6.4% con Sharpe distrutto; K: 6.4% con whipsaw).
- Due osservazioni meccaniche contano più dei numeri:
  1. **Il floor non tiene**: equity share media effettiva OOS 45.9% contro il floor 0.5 selezionato — damp, momentum tilt e renorm finale erodono il floor (by design, ma più del previsto). L'equity strutturale REALE consegnata è da 45/55, non da 50/50+.
  2. **Turnover 160%/anno**: l'enforcement mensile del floor contro la deriva HRP genera scambi continui — contro la filosofia low-turnover, e a 10bps costa ~30bps/anno.
  3. **Bordo griglia**: il combo selezionato (em=0.5, damp=0.25) è sul bordo in entrambe le dimensioni — con damp_grid a 2 valori l'avviso è strutturale, ma em=0.5 al minimo della griglia suggerisce che floor più ALTI peggiorano l'IS (il 2008 punisce l'equity strutturale in-sample).

**Lettura cumulativa I+J+K+L (39 trial):** la frontiera onesta della macchina su questo universo è ~Sharpe 0.9-0.95 con CAGR 5-7% e DD -13/-16%. Il 60/40 resta marginalmente sopra in Sharpe perché non paga né turnover né tracking del floor. La prossima mossa più promettente NON è un'altra variante di allocazione: è ridurre l'attrito (enforcement del floor solo a soglie di deviazione, turnover penalty) o riconoscere che il valore della macchina è il profilo DD e venderlo come tale.
