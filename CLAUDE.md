# JeanClaude — Mini Hedge Fund

Questo è un progetto di asset management quantitativo multi-asset sviluppato da Emanuele e collega.

## Cos'è questo progetto

Un framework Python modulare per gestire un portafoglio multi-asset con allocazione dinamica guidata da segnali macro e news. L'approccio è **Global Macro** con frequenza di ribilanciamento settimanale/mensile e focus su low turnover.

**Fasi del progetto:**
1. Framework di ricerca e backtesting (in corso)
2. Paper trading
3. Live trading

## Architettura

Il repo è organizzato in moduli indipendenti:

```
jeanclaude/
├── data/           # Ingestion (Refinitiv + FRED), storage Parquet, transform
├── signals/        # Macro regime detection (pomegranate HMM), news sentiment (FinBERT), composite
├── portfolio/      # HRP optimizer + BL views + TrendMomentum + risk filter
├── backtest/       # Engine simulazione storica, CPCV, metriche (Sharpe, Sortino, Calmar)
├── execution/      # Paper trading engine
└── reporting/      # Dashboard P&L, allocazione, risk metrics
```

## Strategia di portafoglio

- **Ottimizzatore:** Hierarchical Risk Parity (HRP) come baseline
- **Regime detection:** Hidden Markov Model su indicatori macro
- **View injection:** Black-Litterman con segnali macro/news come "views"
- **Risk filter:** CVaR + drawdown limit prima di ogni ribilanciamento

## Fonti dati

- **Refinitiv / LSEG** — fonte primaria (prezzi, news, macro, earnings)
- **FRED** — macro US (fredapi)
- **ECB Data Portal** — macro EU
- **Yahoo Finance** — fallback via yfinance

## Stack tecnologico

`pandas` / `cvxpy` / `pomegranate` (HMM) / `transformers` (FinBERT) / `lseg-data` / `yfinance` / `fredapi` / `pytest` / `uv`

## Collaborazione

- Branch strategy: `main` (stabile) + feature branch per modulo
- Dal 2026-06 il progetto è gestito dal solo Emanuele (Filippo non è più coinvolto)
- Design doc completo: `docs/superpowers/specs/2026-04-05-hedge-fund-design.md`

## Note per Claude

- Quando scrivi codice, segui la struttura modulare sopra — ogni modulo ha un'interfaccia pulita
- Le interfacce tra moduli sono un contratto: non cambiarle senza motivarlo
- Priorità al low turnover e alla robustezza rispetto alla performance massima
- I dati Refinitiv sono disponibili tramite API del collega — non hardcodare credenziali
- Prima di implementare qualcosa di nuovo, controlla se esiste già nei moduli del repo prima di scrivere codice nuovo
- Runbook operativo del paper trading: `docs/runbook.md` — i run giornalieri sono idempotenti e keyed per data di mercato (asof); i giorni saltati vengono recuperati automaticamente al run successivo
- Audit di progetto 2026-06-10: `docs/reports/2026-06-10-audit.md` — i risultati di backtest pre-audit non sono affidabili (selection bias + tuning iterativo); da rifare in P1
