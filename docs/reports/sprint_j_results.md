# Sprint J — Vol Targeting + Leva: battere SPY B&H?

**Config hash:** `e92470f2d2f8dd70`  
**Universo:** 20 ETF (identico a Sprint I — ex-ante dal design doc)  
**IS:** 2005-01-01 → 2011-12-31  |  **OOS:** 2012-01-01 → 2026-06-01  
**Fonte prezzi:** Yahoo auto_adjust=True (total return)  
**Funding source:** FRED DGS3MO/100 + 0.005 (publication_lag=1)  
**Note:** Sprint J: vol targeting + leva sui parametri base CONGELATI di Sprint I (Balanced damp=0.25 mom=0.5; Aggressive damp=0.25 mom=0.7 — NON ricalibrati). Griglia IS solo su target_vol {0.08,0.10,0.12} x arm. Arm A: VolTarget(TrendMomentum). Arm B: VolTarget(TrendMomentum con equity_min=0.60). max_leverage=2.0, funding=DGS3MO+50bps. Fonte TR: Yahoo auto_adjust. OOS toccato una volta.  

## Parametri congelati

**Base da Sprint I (NON ricalibrati):**  
- Arm A (Balanced): damp=0.25, mom=0.5
- Arm B (Aggressive + equity_floor): damp=0.25, mom=0.7, equity_min=0.60

**IS grid → target_vol congelato:**  
- Arm A: target_vol = 0.08
- Arm B: target_vol = 0.08

## Metriche OOS

| Strategy | CAGR | Sharpe | Sortino | MaxDD | Calmar | $10k→ |
|---|---|---|---|---|---|---|
| SPY B&H | 15.0% | 0.92 | 1.12 | -33.7% | 0.45 | $65,198 |
| 60/40 SPY/IEF | 9.8% | 1.00 | 1.25 | -21.0% | 0.46 | $38,094 |
| Sprint I Balanced (unlevered) | 5.1% | 0.95 | 1.13 | -12.9% | 0.40 | $19,487 |
| Arm A (VolTarget Balanced) | 5.9% | 0.67 | 0.76 | -24.7% | 0.24 | $21,394 |
| Arm B (VolTarget Aggressive+floor) | 7.0% | 0.74 | 0.87 | -25.4% | 0.28 | $24,860 |

## Deflated Sharpe Ratio

**n_trials questo esperimento (config_hash):** 6  
**n_trials cumulativi (+ 21 Sprint I):** 27  

| Strategy | n_trials | DSR | Pass (≥0.75)? |
|---|---|---|---|
| Arm A (VolTarget Balanced) (n=J) | 6 | 0.864 | ✓ |
| Arm A (VolTarget Balanced) (n=J+SpI) | 27 | 0.654 | ✗ |
| Arm B (VolTarget Aggressive+floor) (n=J) | 6 | 0.915 | ✓ |
| Arm B (VolTarget Aggressive+floor) (n=J+SpI) | 27 | 0.747 | ✗ |

## VERDETTI (due criteri espliciti ex-ante)

> Nessuna selezione a posteriori del criterio — entrambi riportati.

### Criterio A: CAGR OOS ≥ SPY CAGR?

| Arm | CAGR | SPY CAGR | Batte SPY? |
|---|---|---|---|
| Arm A (VolTarget Balanced) | 5.9% | 15.0% | NO |
| Arm B (VolTarget Aggressive+floor) | 7.0% | 15.0% | NO |

### Criterio B: Sharpe netto ≥ SPY Sharpe E MaxDD < SPY MaxDD?

| Arm | Sharpe | SPY Sharpe | MaxDD | SPY MaxDD | Criterio B? |
|---|---|---|---|---|---|
| Arm A (VolTarget Balanced) | 0.67 | 0.92 | -24.7% | -33.7% | NO |
| Arm B (VolTarget Aggressive+floor) | 0.74 | 0.92 | -25.4% | -33.7% | NO |

---
_Report generato automaticamente da scripts/backtest_sprint_j.py_
## Interpretazione onesta (controller, 2026-06-10)

L'ipotesi dichiarata ex-ante ("vol targeting + leva ≤2x porta il CAGR verso SPY mantenendo lo Sharpe") è **confutata** su entrambi i criteri:

1. **CAGR**: 5.9–7.0% contro 15.0% di SPY — la leva ha chiuso solo ~1.5 punti del gap di ~10.
2. **Sharpe**: la leva l'ha DISTRUTTO (0.95 unlevered → 0.67–0.74) e il MaxDD è raddoppiato (-13% → -25%).

Perché ha fallito (diagnosi, utile per i prossimi esperimenti):
- **Pro-ciclicità della stima**: la vol realizzata 63d è bassa nei mercati calmi → leva massima esattamente PRIMA degli spike (2020, 2022). Con ribilanciamento mensile il de-lever arriva sistematicamente in ritardo.
- **Funding**: T-bill+50bps nell'era 2022-2026 (~5%+) rende il prestito su 1x di NAV un drag di ~5 punti l'anno proprio quando la leva è alta.
- **tv=0.08 selezionato sul bordo griglia** con IS Sharpe basse (~0.43): il vol targeting non aggiungeva valore nemmeno in-sample — segnale che la trasmissione Sharpe→CAGR via leva non è gratis su questa macchina.

**Conclusione cumulativa (Sprint I + J):** la macchina HRP+trend+momentum ha uno Sharpe genuino ~0.95 a bassa vol, ma (a) non batte il 60/40 risk-adjusted, (b) la leva semplice non converte lo Sharpe in CAGR. Il gap verso SPY B&H 2012-2026 è in gran parte beta di un bull market eccezionale. Le direzioni di ricerca rimaste oneste: equity floor dinamico regime-condizionato (0–100% invece di damp moltiplicativo), orizzonti di ribilanciamento multipli, carry/duration come fonte di rendimento aggiuntiva, o accettare il profilo difensivo e misurarsi contro il benchmark giusto (60/40), non contro SPY.
