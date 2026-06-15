# NAV non-deterministico per data fissa — root cause & fix

**Data:** 2026-06-14
**Branch:** `fix/nav-gap-window-idempotency`
**Severità:** alta (minava la fiducia nel track record paper trading — prerequisito ai soldi veri)
**Stato:** RISOLTO (fix + test + suite verde 563; auto-repair dei dati vivi via run schedulato)

## Sintomo

Il NAV registrato per una **data di mercato passata fissa** (`asof`) cambiava tra run.
Esempio store ETF, `asof = 2026-06-12`: oscillava **€101.576,57 ↔ €99.823,04** (≈ 1 giorno di mercato).
Idem su tutti e 3 gli store attivi (`paper_etf`, `paper_sprint_l`, `paper_trading`).

## Root cause (provata)

La finestra del ritorno di periodo era ancorata a `PaperBroker.last_nav_date()` (ultima riga
NAV **assoluta**), mentre `record_daily_nav` ancora la **base** alla riga NAV *strettamente
precedente* ad `asof` (`nav_df[index < asof]`). Le due definizioni divergono **solo su una
re-run** dello stesso `asof` dopo un gap:

| Run | `last_nav_date` prima del run | ramo `_period_returns` | NAV[06-12] |
|-----|------|------|------|
| 1 (sab 13/06, prima volta per asof=06-12) | 06-10 (gap su 06-11) | `06-10 < 06-12` ✓ → ritorno **2-giorni** `p12/p10` | €101.576,57 ✓ |
| 2 (dom 14/06, re-run) | **06-12** (scritto da Run 1) | `06-12 < 06-12` ✗ → fallback **1-giorno** `p12/p11` | €99.823,04 ✗ |

Innesco: nel weekend `asof = ultimo giorno di mercato ≤ today` resta **venerdì 06-12** per i run
di ven/sab/dom, quindi più job launchd riscrivono la stessa riga. Il secondo run vede
`last_nav_date == asof`, **collassa silenziosamente** la finestra a 1 giorno (nessun warning,
perché 06-12 *è* nell'indice prezzi) e perde il P&L del giorno nel gap (06-11). La base resta
06-10 → il P&L 06-10→06-11 non è né nella base né nel ritorno.

Confermato in due modi indipendenti:
- **Riproduzione no-network** con prezzi congelati: 101.000 (Run 1) vs 99.019 (Run 2).
- **Log di produzione** `paper_etf.log`: "Daily NAV recorded for 2026-06-12: 101576.57" (06-13)
  poi "99823.04" (06-14), base 99900 invariata in entrambi.

## Fix

Centralizzato l'accounting nel broker, eliminando i due `_period_returns` duplicati
(`scripts/run_paper_etf.py` + `jeanclaude/execution/paper/runner.py`):

- `PaperBroker.nav_date_before(date)` — ultima riga NAV **strettamente prima** di `date`
  (stessa semantica della base di `record_daily_nav`).
- `PaperBroker.period_returns(prices, asof, assets)` — finestra ancorata a
  `nav_date_before(asof)`, **non** all'ultima riga assoluta. Un re-run sullo stesso `asof`
  riproduce esattamente il primo run → NAV idempotente per data fissa.

Regression test: `tests/execution/test_broker_period_returns.py` (idempotenza + ancora stabile
anche dopo che la riga di `asof` esiste). Suite intera: **563 passed**.
Code review: APPROVE, 0 CRITICAL/HIGH.

## Residuo investigato: yfinance `auto_adjust=True`

Domanda: anche col fix, prezzi ri-scaricati live possono rivedere retroattivamente le chiusure
aggiustate (dividendi/split) → il NAV di una data passata può ancora derivare?

**Misura empirica (06-14):** ricalcolo gap-aware del 06-12 con prezzi yfinance freschi =
**€101.576,57**, identico al primo run del 06-13 → **deriva = +0,00 bps**.

**Perché è ~nullo:** una data passata viene riscritta solo finché è `asof`, cioè durante la
finestra **senza contrattazioni** (es. weekend). In quell'intervallo non maturano nuovi
dividendi/split, quindi le chiusure aggiustate di `base_date` e `asof` sono congelate e il
rapporto `p[asof]/p[base_date]` è invariante. Superato il weekend, `asof` avanza e la riga
passata non viene più riscritta.

**Decisione:** nessuna guardia aggiuntiva nel motore (sarebbe over-engineering contro un caso a
0 bps). L'esposizione teorica residua (correzioni vendor retroattive su corporate action già
settled) si chiude solo con una **fonte prezzi point-in-time (Refinitiv)** — già la direzione
dichiarata del progetto, fuori scope da questo fix.

## Pulizia collaterale

- `apply_rebalance_cost`: rimossa l'unica mutazione in-place del DataFrame (ricostruzione
  immutabile come `record_daily_nav`). Behavior-preserving, coperto da `TestApplyRebalanceCost`.

## Riparazione dati vivi

I 3 store hanno il valore **sbagliato** (re-run) salvato per 06-12. Col fix, il run schedulato
con `asof = 06-12` ricalcola il valore corretto in modo idempotente (auto-repair):

| store | salvato (errato) | corretto (auto-repair) |
|-------|------------------|------------------------|
| `paper_etf` | €99.823,04 | €101.576,57 |
| `paper_sprint_l` | €100.185,51 | €101.843,96 |
| `paper_trading` | €100.141,93 | €100.315,07 |
