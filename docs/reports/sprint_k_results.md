# Sprint K — Banda Equity Regime-Condizionata: CAGR vs SPY con DD contenuto?

**Config hash:** `85f003cfeb8f1990`  
**Universo:** 20 ETF (identico a Sprint I — ex-ante dal design doc)  
**IS:** 2005-01-01 → 2011-12-31  |  **OOS:** 2012-01-01 → 2026-06-01  
**Fonte prezzi:** Yahoo auto_adjust=True (total return)  
**Macro:** VIX, Oil, Copper, Gold, EURUSD (Yahoo Finance)  
**Note:** Sprint K: banda equity regime-condizionata (HMM walk-forward causale) su parametri base CONGELATI di Sprint I (Balanced damp=0.25 mom=0.5; Aggressive damp=0.25 mom=0.7 — NON ricalibrati). Griglia IS SOLO sulle 3 mappe banda x 2 basi = 6 combo. Macro Yahoo: VIX(^VIX), Oil(CL=F), Copper(HG=F), Gold(GC=F), EURUSD(EURUSD=X). UNKNOWN -> nessuna banda (pesi base). Tre mappe banda: K1 aggressiva: EXPANSION(0.8,1.0) TRANSITION(0.4,0.7) CONTRACTION(0.0,0.2). K2 moderata: EXPANSION(0.7,1.0) TRANSITION(0.5,0.8) CONTRACTION(0.0,0.3). K3 tenue: EXPANSION(0.9,1.0) TRANSITION(0.5,0.9) CONTRACTION(0.1,0.4). Fonte TR: Yahoo auto_adjust. OOS toccato una volta.  

## Parametri base congelati (da Sprint I — NON ricalibrati)

- Arm A (Balanced): damp=0.25, mom=0.5
- Arm B (Aggressive): damp=0.25, mom=0.7

**IS grid → mappa banda congelata:**  
- Arm A (Balanced): mappa = K1
  - EXPANSION: [0.8, 1.0]
  - TRANSITION: [0.4, 0.7]
  - CONTRACTION: [0.0, 0.2]
- Arm B (Aggressive): mappa = K1
  - EXPANSION: [0.8, 1.0]
  - TRANSITION: [0.4, 0.7]
  - CONTRACTION: [0.0, 0.2]

## Distribuzione Regimi

### IS (2005-2011)

| Regime | # date rebalance |
|---|---|
| CONTRACTION | 52 |
| EXPANSION | 6 |
| TRANSITION | 26 |

### OOS (2012-2026)

| Regime | # date rebalance |
|---|---|
| CONTRACTION | 62 |
| EXPANSION | 22 |
| TRANSITION | 89 |

## Turnover Medio Annuo

| Strategy | Turnover annuo (one-way) |
|---|---|
| Arm A (Balanced+Band) | 164.1% |
| Arm B (Aggressive+Band) | 169.2% |
| Sprint I Balanced (unlevered) | 50.8% |

> **Nota whipsaw:** alto turnover con basso miglioramento sullo Sharpe indica che le band producono segnali contraddittori (whipsaw) — la modalità di fallimento attesa.

## Metriche OOS

| Strategy | CAGR | Sharpe | Sortino | MaxDD | Calmar | $10k→ |
|---|---|---|---|---|---|---|
| SPY B&H | 15.0% | 0.92 | 1.12 | -33.7% | 0.45 | $65,197 |
| 60/40 SPY/IEF | 9.8% | 1.00 | 1.25 | -21.0% | 0.46 | $38,094 |
| Sprint I Balanced (unlevered) | 5.1% | 0.95 | 1.13 | -12.9% | 0.40 | $19,487 |
| Arm A (Balanced+Band) | 6.3% | 0.81 | 0.97 | -18.0% | 0.35 | $22,655 |
| Arm B (Aggressive+Band) | 6.4% | 0.81 | 0.97 | -18.0% | 0.35 | $22,826 |

## Deflated Sharpe Ratio

**n_trials questo esperimento (config_hash):** 6  
**n_trials cumulativi (+ 27 da I+J):** 33  

| Strategy | n_trials | DSR | Pass (≥0.75)? |
|---|---|---|---|
| Arm A (Balanced+Band) (n=K) | 6 | 0.947 | ✓ |
| Arm A (Balanced+Band) (n=K+SpI+SpJ) | 33 | 0.796 | ✓ |
| Arm B (Aggressive+Band) (n=K) | 6 | 0.949 | ✓ |
| Arm B (Aggressive+Band) (n=K+SpI+SpJ) | 33 | 0.801 | ✓ |

## VERDETTI (due criteri espliciti ex-ante)

> Nessuna selezione a posteriori del criterio — entrambi riportati.

### Criterio A: CAGR OOS ≥ SPY CAGR?

| Arm | CAGR | SPY CAGR | Batte SPY? |
|---|---|---|---|
| Arm A (Balanced+Band) | 6.3% | 15.0% | NO |
| Arm B (Aggressive+Band) | 6.4% | 15.0% | NO |

### Criterio B: Sharpe ≥ SPY Sharpe E MaxDD < SPY MaxDD?

| Arm | Sharpe | SPY Sharpe | MaxDD | SPY MaxDD | Criterio B? |
|---|---|---|---|---|---|
| Arm A (Balanced+Band) | 0.81 | 0.92 | -18.0% | -33.7% | NO |
| Arm B (Aggressive+Band) | 0.81 | 0.92 | -18.0% | -33.7% | NO |

### Confronto con Sprint I (contributo della banda)

| Arm | CAGR | SpI CAGR | Sharpe | SpI Sharpe | MaxDD | SpI MaxDD | Migliora SpI? |
|---|---|---|---|---|---|---|---|
| Arm A (Balanced+Band) | 6.3% | 5.1% | 0.81 | 0.95 | -18.0% | -12.9% | NO |
| Arm B (Aggressive+Band) | 6.4% | 5.1% | 0.81 | 0.95 | -18.0% | -12.9% | NO |

---
_Report generato automaticamente da scripts/backtest_sprint_k.py_
## Interpretazione onesta (controller, 2026-06-10)

Entrambi i criteri di verdetto falliscono anche qui: CAGR 6.3-6.4% (< SPY 15%), Sharpe 0.81 (< SPY 0.92 e < Sprint I 0.95). La banda regime-condizionata ha alzato il CAGR di +1.2 punti rispetto al damp moltiplicativo, ma pagando 14 punti di Sharpe, 5 punti di MaxDD e un turnover esploso a ~165%/anno (contro la priorità low-turnover del progetto).

Diagnosi del meccanismo (più informativa del risultato):
- **Il detector HMM-VIX è troppo pessimista come segnale di allocazione**: EXPANSION rilevata solo 28/257 mesi (11%), CONTRACTION 62/173 mesi OOS — nel mezzo di uno dei più grandi bull market della storia. Con la mappa K1 selezionata, l'equity è rimasta cappata sotto 0.7 per ~87% dei mesi OOS.
- **Whipsaw**: il rimbalzo TRANSITION↔CONTRACTION mensile genera i 165%/anno di turnover — il classico fallimento previsto ex-ante nel piano.
- Il CAGR sale comunque perché i floor forzano più equity della HRP naturale (bond-heavy), non perché il timing aggiunga valore.

**Quadro cumulativo I+J+K (33 trial, tutti nel registry):** lo Sharpe ~0.95 del damp moltiplicativo unlevered resta il miglior risultato risk-adjusted della famiglia. Né la leva (J) né il regime timing HMM (K) lo migliorano. L'evidenza converge: su questo universo, in questo periodo, l'alpha contro SPY B&H non è in questi meccanismi.
