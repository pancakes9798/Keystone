# Runbook operativo — JeanClaude paper trading

## Architettura operativa
- **Runner Sprint E** (`scripts/run_paper_sprint_e.py`): launchd 22:45, store `data/paper_trading/`, log `logs/paper_sprint_e.log`
- **Runner ETF** (`scripts/run_paper_etf.py`): launchd 22:50, store `data/paper_etf/`, log `logs/paper_etf.log`
- **Runner Sprint L** (`scripts/run_paper_sprint_l.py`): launchd 22:55, store `data/paper_sprint_l/`, log `logs/paper_sprint_l.log`
  - Fonte dati: **Yahoo Finance (total return)** — `DataConfig(price_source="yahoo")` con `auto_adjust=True`. Il backtest Sprint L è stato validato su dati Yahoo TR; usare Refinitiv (TRDPRC_1 = price return, no dividendi) creerebbe un mismatch sul track record.
  - Config congelata: `damp_factor=0.25`, `momentum_strength=0.5`, `equity_min=0.5`, no risk filter, no HMM.
- **Watchdog** (`scripts/check_paper_health.py`): launchd 23:30, alert email se gli store non sono freschi

## Controlli rapidi
| Domanda | Comando |
|---|---|
| I job sono caricati? | `launchctl list \| grep jeanclaude` |
| Il run di ieri è andato? | `tail -20 logs/paper_sprint_e.log` |
| NAV aggiornato? | `uv run python scripts/check_paper_health.py` |

## Se arriva l'alert "Paper trading FERMO"
1. `tail -50 logs/paper_sprint_e.log logs/paper_etf.log` — cercare l'eccezione.
2. Refinitiv giù / credenziali scadute → sistemare `.env`, poi rilanciare a mano (punto 4).
3. Mac spento all'orario del job → launchd recupera al risveglio; se il giorno è passato, punto 4.
4. **Recovery manuale** (i run sono idempotenti, si possono rilanciare senza paura):
   `uv run python scripts/run_paper_sprint_e.py`, `uv run python scripts/run_paper_etf.py` e `uv run python scripts/run_paper_sprint_l.py`
   I giorni saltati NON vanno rilanciati uno a uno: il primo run recupera il P&L
   dell'intero gap (ritorno multi-periodo) e gli eventuali rebalance mancati (catch-up).
5. Se lo stato sembra corrotto: `uv run python scripts/reset_paper_stores.py` (dry-run) e leggere l'output PRIMA di `--apply`.

## Cambi di configurazione
- Orari: editare i plist in `ops/launchd/`, ricopiarli in `~/Library/LaunchAgents/` e `launchctl unload && launchctl load`.
- Destinatari email: variabile `REPORT_RECIPIENTS` in `.env` (comma-separated).
- I plist sotto `ops/launchd/` sono la fonte di verità versionata: non editare direttamente quelli in `~/Library/LaunchAgents/`.

---

## ⚠️ Automazione launchd BLOCCATA da macOS TCC (diagnosi 2026-06-13)

**Sintomo:** i job `com.jeanclaude.paper-*` escono con `exit code 78 (EX_CONFIG)` e
**non scrivono nulla nel log**. `launchctl print` mostra `runs > 0` ma gli store NAV non
si aggiornano. L'health-check rileva `STALE` e manda l'alert.

**Causa radice (verificata):** il progetto vive in `~/Desktop/JeanClaude`. Su macOS recente
la cartella **Desktop è protetta da TCC** (Transparency, Consent & Control). Un LaunchAgent
non ha il permesso di accedervi: viene bloccato *prima* di eseguire python, quindi nessun log.
Test definitivo: un job launchd che scrive in `/tmp` gira (exit 0) ma `ls ~/Desktop/JeanClaude/data`
dentro lo stesso job fallisce (`DESKTOP_FAIL`). Gli script funzionano perfettamente lanciati a
mano o con `env -i` (ambiente minimale) — **non è un bug del codice, è un permesso di sistema**.

**Conseguenza:** l'automazione notturna NON ha mai funzionato via launchd; gli unici NAV
registrati venivano da run manuali. I plist sono stati aggiornati a `.venv/bin/python` diretto
(no `uv run`, evita il lock-race del venv) ma questo NON basta finché TCC blocca il Desktop.

**Fix (richiede azione GUI dell'utente — una delle due):**
1. **Concedi Full Disk Access** a `/bin/sh` o al python del venv:
   System Settings → Privacy & Security → Full Disk Access → `+` → aggiungi il binario.
   (Fragile: il python del venv è gestito da uv e può cambiare path.)
2. **Sposta il progetto fuori da Desktop** (consigliato), es. `~/jeanclaude`:
   `git` e dati seguono; aggiornare i path assoluti nei plist e ricaricare i job.
   Le cartelle Home non-protette (non Desktop/Documents/Downloads) non sono soggette a TCC.

**Recovery manuale immediato (idempotente, keyed per data di mercato):**
```bash
cd ~/Desktop/JeanClaude
.venv/bin/python scripts/run_paper_sprint_e.py
.venv/bin/python scripts/run_paper_etf.py
.venv/bin/python scripts/run_paper_sprint_l.py
```
