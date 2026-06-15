# Sprint N — Verdetto Fattibilità: Constituent Point-in-Time e Total Return Refinitiv

**Data:** 2026-06-11  
**Autore:** JeanClaude / probe automatico  
**Scopo:** Determinare se l'accesso LSEG/Refinitiv attuale supporta (A) costituenti storici point-in-time e (B) prezzi total-return per titoli delistati. Questo decide la direzione della Fase 2.

---

## Riepilogo Esecutivo

| Capacità | Esito | Note |
|---|---|---|
| A1. Dated chain syntax (point-in-time) | ROSSO | Ritorna sempre il paniere corrente |
| A2. JL change-log (leavers history) | GIALLO | 657 eventi 2000-2026 + 167 pre-2000; richiede workaround per ricostruzione |
| B1. Prezzi storici RIC delistati | VERDE | LEH.N^I08, GM.N^F09 — dati completi |
| B2. Total Return (TR.TotalReturn1D) | VERDE | CAGR KO 8.06% vs Yahoo 8.09% — gap dividendi chiuso |
| B3. Rate limit 10 RIC simultaneous | VERDE | 1.90s per 5 anni × 10 titoli, nessun throttling |

**VERDETTO FINALE: GIALLO** — La capacità B (total return + delistati) è completamente funzionante. La capacità A (constituenti point-in-time) richiede un workaround tramite JL change-log; non è disponibile come query diretta a data arbitraria.

---

## Setup e Credenziali

- Sessione: **platform** (credenziali `.env`: `LSEG_APP_KEY`, `LSEG_USERNAME`, `LSEG_PASSWORD`)  
- Libreria: `lseg-data` (Python, `.venv`)  
- Stato sessione: **OK** — connessione aperta con successo in tutti i test

---

## Probe A — Costituenti Storici Point-in-Time

### Metodo 1 — Dated chain syntax (FALLITO come PIT)

**Chiamata testata:**
```python
ld.get_data(universe="0#.SPX(20200101)", fields=["TR.RIC", "TR.CommonName"])
```
**Risposta:** 503 righe  
**Esito:** Tecnicamente la chiamata ha successo (HTTP 200, dati restituiti), ma **non è point-in-time**. La cross-verifica ha dimostrato che le liste per `20080630`, `20151231`, `20210101` e oggi sono **identiche** (503 titoli, zero differenze). TSLA.OQ risulta nel paniere 2008 — impossibile storicamente, visto che TSLA è entrata nell'S&P 500 solo a dicembre 2020. La sintassi della data viene ignorata dall'API con l'accesso attuale; viene restituita la membership corrente.

### Metodo 2 — `TR.IndexConstituentRIC` con `SDate` su `0#.SPX` (IDEM)

```python
ld.get_data(universe="0#.SPX", fields=["TR.IndexConstituentRIC"],
            parameters={"SDate": "2020-01-01"})
```
**Risposta:** 503 righe, colonna `Constituent RIC` vuota (tutti `<NA>`).  
**Esito:** Stesso paniere attuale, senza arricchimento di informazione.

### Metodo 3 — `.SPX` con `TR.IndexConstituentRIC` + `SDate` (IDEM)

```python
ld.get_data(universe=".SPX", fields=["TR.IndexConstituentRIC", "TR.IndexConstituentName"],
            parameters={"SDate": "2020-01-01"})
```
**Risposta:** 506 righe (include il ticker-indice `.SPX`), ma lista attuale.  
**Esito:** Nessun point-in-time.

### Metodo 4 — Joined-Leavers `TR.IndexJLConstituentChangeDate` (PARZIALMENTE FUNZIONANTE)

**Chiamata testata:**
```python
ld.get_data(
    universe=".SPX",
    fields=["TR.IndexJLConstituentChangeDate", "TR.IndexJLConstituentRIC",
            "TR.IndexJLConstituentChangeType", "TR.IndexJLConstituentName"],
    parameters={"SDate": "2000-01-01", "EDate": "2026-12-31"}
)
```
**Risposta:** 657 righe (eventi di cambio 2000-2026) + 167 eventi pre-2000 (1995-1999).  
**Colonne restituite:** `Instrument`, `Date`, `Constituent RIC`, `Constituent Name`  
**Nota:** il campo `ChangeType` (join/leave) **non viene restituito** — la colonna è assente dalla risposta, rendendo impossibile distinguere ingressi da uscite senza inferenza.

**Spot check:**
- `GM.N^F09` presente con data `2009-06-03` — corretto (GM uscita dopo bankruptcy)
- `LEH.N^I08` presente con data `2008-09-17` — corretto (Lehman uscita dopo il crack)
- `TSLA.OQ` **assente** dal JL change-log — corretto, TSLA è ancora nel paniere (non è mai uscita), quindi non appare come leaver/joiner recente nella finestra testata

**Logica di ricostruzione PIT via JL:**  
La lista JL contiene solo i **leavers** (titoli che sono usciti). Per ricostruire il paniere a data T:
1. Partire dal paniere corrente (503 titoli — API oggi)
2. Prendere tutti i titoli JL con `Date > T` → questi erano nel paniere a T ma non oggi
3. Prendere tutti i titoli JL con `Date <= T` → questi sono usciti prima di T e non vanno inclusi

**Limitazione critica:** il JL restituisce 657 eventi dal 2000, ma il numero di sostituzioni storiche nell'S&P 500 dal 2000 è dell'ordine di ~800-1000 (circa 20-30 cambi/anno × 25 anni). La copertura del 70-80% suggerisce che alcuni eventi potrebbero mancare (potenzialmente quelli con SDate < 1995). Per date ante-2000 la ricostruzione è imprecisa.

### Costituenti per data — Conteggi

| Data | Count totale | Note |
|---|---|---|
| 2008-06-30 | 503 | Lista attuale (non PIT) |
| 2015-12-31 | 503 | Lista attuale (non PIT) |
| 2020-01-01 | 503 | Lista attuale (non PIT) |
| 2021-01-01 | 503 | Lista attuale (non PIT) |

---

## Probe B — Total Return per Titoli Delistati

### B1 — Prezzi storici RIC delistati

**Lehman Brothers (`LEH.N^I08`):**
```python
ld.get_history(universe=["LEH.N^I08"], fields=["TRDPRC_1"],
               start="2007-01-01", end="2008-12-31")
```
- **Risposta:** 428 barre giornaliere, 2007-01-03 → 2008-09-12
- **Ultimo prezzo:** $3.65 il 12 settembre 2008 (2 giorni prima del filing Chapter 11)
- **Esito:** VERDE — dati completi e coerenti

**GM (`GM.N^F09`):**
```python
ld.get_history(universe=["GM.N^F09"], fields=["TRDPRC_1"],
               start="2005-01-01", end="2009-12-31")
```
- **Risposta:** 1110 barre giornaliere, 2005-01-03 → 2009-06-01
- **Esito:** VERDE — dati completi fino all'uscita dall'indice

### B2 — Total Return con dividendi (`TR.TotalReturn1D`)

**Chiamata che funziona:**
```python
ld.get_data(
    universe=["KO.N"],
    fields=["TR.TotalReturn1D", "TR.TotalReturn1D.calcdate"],
    parameters={"SDate": "2015-01-01", "EDate": "2025-12-31", "Frq": "D"}
)
```
- **Risposta:** 2766 osservazioni giornaliere con data (`Calc Date`) e rendimento % totale giornaliero
- **Colonne:** `Instrument`, `Daily Total Return`, `Calc Date`

**Validazione CAGR KO 2015-2025:**
- Composizione dei 2766 ritorni giornalieri: crescita 2.3427x
- CAGR calcolato: **8.06%**
- Riferimento Yahoo Finance TR: **8.09%**
- Scarto: **0.03 pp** — trascurabile, validazione superata
- CAGR prezzi puri (senza dividendi): **4.71%** (sia raw che CORAX-adjusted)
- Gap chiuso dai dividendi: **+3.35 pp/anno** — il campo cattura correttamente la componente dividend

**Alternative testate:**

| Campo | Metodo | Esito | Note |
|---|---|---|---|
| `TR.TotalReturn1D` | `get_data` con `Frq=D` | VERDE | Campo principale; ritorna % giornaliero con data |
| `TR.TotalReturn` scalare | `get_data` senza `Frq` | VERDE | Ritorna 134.27% cumulativo 2015-2025 → CAGR 8.05% |
| `TR.TotalReturn` time-series | `get_history` | GIALLO | Ritorna solo 1 riga (valore finale), non serie temporale |
| `TR.TotalReturn` con `RateBasis` | `get_data` | ROSSO | Parametro non riconosciuto dall'API |
| CORAX (`CCH+CRE+RTS+RPO`) | `get_history` | ROSSO | CAGR 4.71% = identico al prezzo raw; i dividendi ordinari NON sono inclusi nelle rettifiche CORAX |

**Conclusione B2:** Il campo corretto per total return giornaliero è `TR.TotalReturn1D` via `get_data` con `Frq=D`.

### B3 — Rate Limit: 10 RIC simultanei

```python
ld.get_history(universe=rics_10, fields=["TRDPRC_1"], start="2020-01-01", end="2025-12-31")
```
- RIC testati: KO.N, AAPL.OQ, MSFT.OQ, JPM.N, JNJ.N, PG.N, UNH.N, XOM.N, V.N, MA.N
- **Risposta:** 1508 barre × 10 colonne
- **Tempo:** 1.90 secondi
- **Throttling:** nessuno rilevato
- **Esito:** VERDE — nessuna limitazione osservata per batch di 10 titoli su 5 anni

---

## Analisi e Raccomandazioni per Fase 2

### Capacità A — Point-in-Time Constituents

**Situazione:** La sintassi dated chain non funziona come PIT con il nostro tier di accesso. Il workaround via JL change-log è fattibile ma con limitazioni:

**Workaround consigliato:**
```
paniere_at_T = (paniere_oggi) 
               + (JL_leavers con data > T)    # titoli usciti dopo T → erano presenti
               - (JL_leavers con data <= T)   # titoli usciti prima/su T → non erano presenti
```

**Limiti del workaround:**
1. I JL events dal 2000 sono 657; la copertura pre-2000 è parziale (167 eventi dal 1995)
2. Il campo `ChangeType` (join/leave) non viene restituito → tutti i 657 eventi sono trattati come leavers, ma alcuni potrebbero essere re-insertions
3. Backdating affidabile stimato: **2003-oggi** (20+ anni di storia utilizzabile)
4. Per backtesting ante-2003 si consiglia integrazione con fonte esterna (es. CRSP via WRDS)

**Azione raccomandata:** Usare il JL workaround per ricerca Fase 2 con universo S&P 500 dal 2003 in poi. Documentare l'assenza del ChangeType come rischio di survivorship parziale.

### Capacità B — Total Return + Delistati

**Situazione:** Completamente operativa.

- `TR.TotalReturn1D` via `get_data` con `Frq=D`: serie giornaliera con date, validata con errore < 0.03 pp su benchmark Yahoo Finance
- Prezzi per delisted RIC (suffisso `^`): disponibili (Lehman, GM testati)
- I RIC delistati usciti dall'S&P 500 sono identificabili dal JL change-log (suffisso `^MMMYY`)
- Nessun throttling rilevato per batch di 10 titoli

---

## VERDETTO FINALE: GIALLO

**La Fase 2 è fattibile con workaround.**

| | Stato |
|---|---|
| **B — Total Return (delistati inclusi)** | VERDE — procedi subito |
| **A — Point-in-Time Constituents** | GIALLO — workaround JL funzionante, limitazioni note |

**Azioni immediate:**
1. Implementare il modulo `constituent_history.py` basato su JL change-log per ricostruire panieri storici
2. Usare `TR.TotalReturn1D` come campo standard per tutti i ritorni (non CORAX, non prezzi raw)
3. Definire l'orizzonte di backtest a partire dal **2003** per massimizzare l'affidabilità del PIT

**Blocchi residui:**
- Il ChangeType mancante nel JL espone a un rischio di bias nel periodo 2000-2003; tollerabile se il backtest parte dal 2003
- Per ricerca accademica rigorosa (pre-2000 o survivorship-free garantito) serve integrazione CRSP/WRDS

---

*Probe eseguito il 2026-06-11. Script di test: `/tmp/probe_a_constituents.py`, `/tmp/probe_a_jl_deep.py`, `/tmp/probe_b_total_return.py`, `/tmp/probe_b_ko_validate.py`*

---

## ADDENDUM 2026-06-12 — Verdetto aggiornato: VERDE

**Il blocco residuo della capacità A è risolto.** Il probe del parametro `IC` (Sprint N parte 3)
ha trovato la causa del log incompleto: **il default dell'API è leavers-only**. Con:

```python
ld.get_data(
    universe=".SPX",
    fields=["TR.IndexJLConstituentChangeDate",
            "TR.IndexJLConstituentRIC.change",   # → colonna Change: Joiner/Leaver
            "TR.IndexJLConstituentRIC",
            "TR.IndexJLConstituentName"],
    parameters={"SDate": "1995-01-01", "EDate": "2099-12-31", "IC": "B"},  # B = Both
)
```

il log passa da 824 a **1668 eventi (844 Joiner + 824 Leaver)** e la colonna `Change`
distingue esplicitamente la direzione (il campo `TR.IndexJLConstituentChangeType` resta
non restituito — la via giusta è il suffisso `.change`).

**Sentinel:** TSLA.OQ Joiner 2020-12-21 ✓ · LEH.N^I08 Joiner 1998-01-12 + Leaver 2008-09-17 ✓ ·
GM.N^F09 Leaver 2009-06-03 (join pre-1995, correttamente fuori finestra) ✓

**Validazione post-fix** (`docs/research/2026-06-12-validazione-constituents.md`):
year-end 2003-2025 tutti tra **501 e 505 membri** (prima: 990 nel 2003), spot check **6/6 PASS**
(incluso TSLA assente nel 2015, il FAIL della validazione 2026-06-11).

| Capacità | Esito aggiornato |
|---|---|
| A — Point-in-Time Constituents | **VERDE** — JL log completo con IC=B, semantica esplicita join/leave |
| B — Total Return (delistati inclusi) | VERDE (invariato) |

**VERDETTO FINALE AGGIORNATO: VERDE** — si procede con Sprint O (momentum baseline come
validation gate, criteri ex-ante).

### Review adversarial post-fix (workflow 6 agenti, 2026-06-12)

Tre finding confermati e **tutti risolti in giornata**:

1. **Membro fantasma EVHC.N^L16 in ogni paniere 2003-01→2016-11 (HIGH).** Il log contiene 10 coppie
   join+leave stesso giorno/stesso RIC (flash entry tipo Envision alla merger AmSurg); il backward walk
   le risolveva per ordine riga API (proprietà non contrattuale — un refetch avrebbe potuto eliminare
   silenziosamente membri storici come PEG.N, nell'indice dal 1957). **Fix:** le coppie same-day sono
   cancellate come no-op netti prima del walk (`_cancel_same_day_pairs`). Post-fix i conteggi year-end
   combaciano ESATTAMENTE con la verità storica: **500 piatto 2003-2013, 502 (2014), 504 (2015)**.
2. **Null nella colonna `change` trattato silenziosamente come leave (MEDIUM).** La validazione usava
   `.dropna()`. **Fix:** null → `ValueError` loud-fail.
3. **`TR.TotalReturn1D` mai validato sui RIC morti (HIGH — gate Sprint O).** Il 42% del paniere 2005
   è composto da RIC delistati (`^`). **Probe live 2026-06-12 sul percorso di produzione
   `get_total_returns`: 5/5 RIC morti con copertura piena** (LEH -78.6% a fine finestra, GM -96.9%,
   DISCK, VIAB, EVHC — ritorni plausibili, dividendi inclusi). Resta per Sprint O il report di
   copertura per-snapshot con soglia esplicita di fallimento (la degradazione silenziosa di
   `_fetch_total_return_one` su RIC non prezzabili è documentata, non ancora gated).

Migliorie collaterali dalla review: validazione con **ancore esatte** (500/502/504) al posto della
banda ±10 che aveva lasciato passare il +1 sistematico per 13 anni; nota su FJ.N^K00 (unico gap
genuino del log: due leave consecutivi senza join); RIC retroattivi nei panieri (il log usa il RIC
ODIERNO anche per il passato, es. META.OQ porta il join 2013 di FB — membership corretta, continuità
prezzi sul RIC corrente da tenere d'occhio in Sprint O); la claim "copertura 70-80% pre-2003" della
prima probe era un artefatto del fetch leavers-only — con IC=B il log è essenzialmente completo dal 1995.

Diagnostica riproducibile: `scripts/check_jl_integrity.py`.
