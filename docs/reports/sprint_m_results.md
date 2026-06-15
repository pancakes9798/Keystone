# Sprint M — No-Trade Band su Base Sprint L: abbattere il friction cost?

**Config hash:** `8ebbff9d04160153`  
**Universo:** 20 ETF (identico a Sprint L — ex-ante dal design doc)  
**IS:** 2005-01-01 → 2011-12-31  |  **OOS:** 2012-01-01 → 2026-06-01  
**Fonte prezzi:** Yahoo auto_adjust=True (total return)  
**Note:** Sprint M: no-trade band su base Sprint L congelata (em=0.5, damp=0.25, mom=0.5). Ipotesi ex-ante: threshold ∈ {0.05, 0.10, 0.20} riduce turnover ≥50% e porta Sharpe ≥ 60/40 AND MaxDD ≤ 60/40 AND CAGR ≥ 8%. Parametri base CONGELATI da Sprint L: em=0.5, damp=0.25, mom=0.5. Fonte TR Yahoo. DSR cumulativo: +39 da I+J+K+L → 42 prima di IS.  

## Ipotesi ex-ante

> Una no-trade band sull'intero vettore pesi (si ribilancia solo se il turnover proposto supera una soglia) riduce il turnover di almeno il 50% e porta la base Sprint L (em=0.5, damp=0.25, mom=0.5 CONGELATI) a Sharpe ≥ 1.00 mantenendo MaxDD ≤ 60/40 e avvicinando CAGR 8%.

## Parametri selezionati (IS)

- `threshold` = 0.05  (IS grid: [0.05, 0.1, 0.2])
- `equity_min` = 0.5  (CONGELATO da Sprint L)
- `damp_factor` = 0.25  (CONGELATO da Sprint L)
- `momentum_strength` = 0.5  (CONGELATO da Sprint I/L)

## Statistiche Turnover OOS

| | Turnover annuo |
|---|---|
| Sprint M (con band, threshold=0.05) | 153.7% |
| Sprint L base (senza band) | 160.7% |
| Riduzione | 4.4% |
| Ipotesi ≥50% riduzione | **FAIL ✗** |

- **Rebalance skippati OOS:** 29 su 159 totali (18.2%)

## Metriche OOS

| Strategy | CAGR | Sharpe | Sortino | MaxDD | Calmar | $10k→ |
|---|---|---|---|---|---|---|
| SPY B&H | 15.0% | 0.92 | 1.12 | -33.7% | 0.45 | $65,198 |
| 60/40 SPY/IEF | 9.8% | 1.00 | 1.25 | -21.0% | 0.46 | $38,094 |
| Sprint L (no band) | 7.3% | 0.91 | 1.14 | -16.4% | 0.45 | $25,758 |
| Sprint M (band=0.05) | 7.3% | 0.91 | 1.14 | -16.6% | 0.44 | $25,720 |

## Deflated Sharpe Ratio

**n_trials questo esperimento (config_hash):** 3  
**n_trials cumulativi (+ 39 da I+J+K+L):** 42  

| Strategy | n_trials | DSR | Pass (≥0.75)? |
|---|---|---|---|
| Sprint M (band=0.05) (n=M) | 3 | 0.992 | ✓ |
| Sprint M (band=0.05) (n=M+prior) | 42 | 0.861 | ✓ |

## VERDETTI (tre criteri ex-ante — dichiarati PRIMA del run OOS)

> Nessuna selezione a posteriori del criterio — tutti e tre riportati.

### Criterio (a): Sharpe OOS ≥ Sharpe 60/40

**FAIL ✗** — Sprint M Sharpe=0.91 vs 60/40 Sharpe=1.00

### Criterio (b): MaxDD OOS non più profondo del MaxDD 60/40

**PASS ✓** — Sprint M MaxDD=-16.6% vs 60/40 MaxDD=-21.0% (non più profondo)

### Criterio (c): CAGR OOS ≥ 8%

**FAIL ✗** — Sprint M CAGR=7.3% vs soglia=8.0%

### Verdetto complessivo: **FAIL (almeno uno non soddisfatto)**


---
_Report generato automaticamente da scripts/backtest_sprint_m.py_
## Interpretazione onesta (controller, 2026-06-11)

Ipotesi confutata su tutta la linea: riduzione turnover 4.4% (dichiarato ≥50%), metriche invariate (Sharpe 0.91, CAGR 7.3%). L'IS ha scelto la banda più stretta della griglia (0.05, bordo) — cioè "quasi nessuna banda": i rebalance mensili sono mosse GRANDI e genuine (ricomposizione HRP + enforcement floor + momentum), non rumore filtrabile. Diagnosi rivista: il gap verso il 60/40 NON è attrito (160%/anno × 10bps ≈ 16bps ≈ 0.03 Sharpe) ma qualità di allocazione.

**Chiusura della serie I→M (42 trial cumulativi, DSR cumulativo 0.861):** cinque ipotesi pre-registrate, cinque verdetti onesti. La frontiera della famiglia HRP+trend+momentum su 20 ETF è Sharpe 0.91-0.95, CAGR 5-7.3%, MaxDD -13/-16%. Né leva, né regime HMM, né floor strutturale, né riduzione attrito la spostano. Continuare con varianti di allocazione sarebbe data mining. Le opzioni strategiche reali: (1) mandare in paper trading la macchina validata (Sprint L: miglior CAGR della famiglia con DD -16%) sull'universo onesto e accumulare track record; (2) se si cerca ancora alpha, cambiare FONTE di rendimento (asset, frequenza, segnali di selezione titoli), non meccanica di allocazione.
