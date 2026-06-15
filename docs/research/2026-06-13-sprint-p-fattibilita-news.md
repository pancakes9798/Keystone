# Sprint P (parte 1) — Fattibilità News Sentiment: VERDETTO ROSSO con l'accesso attuale

**Data probe:** 2026-06-13
**Scopo:** verificare se l'accesso LSEG attuale supporta il backtest FinBERT cross-sectional
previsto dal brief Fase 2 (`docs/research/2026-06-11-direzione-alpha-fase2.md`, punto 2).

---

## Riepilogo esecutivo

| Capacità | Esito | Evidenza |
|---|---|---|
| News API su sessione **platform** (credenziali Refinitiv platform) | **ROSSO** | `ScopeError: Insufficient scope for key=/data/news/v1/headlines` — la licenza non include lo scope `trapi.data.news.*` |
| Campi sentiment pre-calcolati (TR.NewsSentiment, MIESentiment, BuzzScore / MarketPsych) | **ROSSO** | `Unable to resolve all requested fields` — non licenziati |
| News API su sessione **desktop** | **NON TESTABILE ORA** | Richiede l'app LSEG Workspace in esecuzione (`localhost:9000` connection refused) — serve un login Workspace |
| Sentiment ticker nel paper trading live (Sprint H) | **MAI FUNZIONATO** | Bug di iniezione (sotto) + comunque ScopeError sulla sessione platform |

**Conclusione: il backtest FinBERT cross-sectional NON è fattibile con le credenziali attuali.**
Anche nello scenario migliore (Workspace desktop funzionante), le news headline API Refinitiv
coprono tipicamente ~15 mesi di storia: un backtest pluriennale resterebbe impossibile — il
massimo ottenibile sarebbe la validazione forward (live) del segnale.

---

## Evidenza dettagliata

### 1. Platform session: scope mancante

```
ld.news.get_headlines("R:AAPL.OQ", start=..., end=..., count=100)
→ ScopeError: Insufficient scope for key=/data/news/v1/headlines, method=GET.
  Required scopes: {'trapi.data.news...'}
```

Identico per query RIC (`R:AAPL.OQ`) e free-text, su tutte le finestre temporali
(da 7 giorni a 18 anni fa). È un limite di **licenza**, non di codice: le credenziali
Refinitiv platform non includono il data layer news.

### 2. Sentiment pre-calcolato: non licenziato

`TR.NewsSentiment`, `TR.MIESentiment`, `TR.BuzzScore` → "Unable to resolve all
requested fields". I prodotti Refinitiv di sentiment (MarketPsych/MI) sono
add-on a pagamento, non inclusi.

### 3. Bug di produzione scoperto (collaterale): il sentiment live è codice morto

`scripts/run_paper_sprint_e.py` iniettava come `news_fetcher` la classe
**`RefinitivSource`** (prezzi) invece di **`RefinitivNewsSource`** (news):

```
WARNING jeanclaude.execution.paper.runner — Ticker news fetch/score failed:
'RefinitivSource' object has no attribute 'get_ticker_headlines' — skipping.
INFO — No ticker sentiment available — using prior weights
```

**Ogni run notturno dal lancio di Sprint H è andato in fallback sui prior** — le views
BL da FinBERT non sono mai entrate in produzione. Nota: anche correggendo la classe,
la sessione platform notturna prenderebbe comunque ScopeError (punto 1). Decisione:
path sentiment DISABILITATO esplicitamente nel runner di produzione (config a None,
riferimento a questo report) finché non esiste un accesso news funzionante — meglio
un'assenza dichiarata di un fallimento notturno mascherato da warning.

---

## Opzioni per la direzione Fase 2 (decisione per Emanuele)

1. **Test Workspace desktop** (gratis, 10 minuti): se Emanuele ha un login LSEG Workspace,
   avviare l'app e rilanciare il probe (`/tmp/probe_news_history.py` con sessione desktop).
   Se le news funzionano: profondità attesa ~15 mesi → niente backtest storico, ma si può
   costruire la RACCOLTA LIVE (fetch giornaliero via desktop → archivio proprio → dopo N
   mesi un backtest sul proprio archivio). Lento ma onesto, e l'infrastruttura FinBERT esiste.
2. **Upgrade licenza news/MarketPsych** — costo da verificare con LSEG; sblocca il backtest
   storico vero. Probabilmente sproporzionato per un progetto a 100k paper.
3. **Fonte news alternativa gratuita/economica** (es. archivio GDELT, Alpha Vantage news,
   EDGAR 8-K filings come proxy event) — qualità/copertura da validare, nuova fattibilità.
4. **Cambiare segnale differenziante** restando sul data layer PIT già validato: fondamentali
   point-in-time via Refinitiv (TR.* con SDate — ATTENZIONE allo stesso left-anchoring bug,
   da verificare), low-vol/quality cross-sectional, oppure doppio sort momentum×low-vol.
   Qualunque scelta = nuovo sprint con probe di fattibilità e criteri ex-ante.

*Probe: `/tmp/probe_news_history.py` (sessione platform), campi TR testati inline.*
