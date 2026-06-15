# Sprint Q (parte 1) — Fattibilità Fondamentali Point-in-Time: GIALLO

**Data probe:** 2026-06-13
**Scopo:** verificare se i fondamentali Refinitiv (value/quality) supportano un backtest
cross-sectional point-in-time sull'universo S&P 500 già validato (Sprint N/O).

---

## Riepilogo esecutivo

| Capacità | Esito | Evidenza |
|---|---|---|
| Campi value/quality disponibili | **VERDE** | P/E, P/B, EV/EBITDA, Gross Margin, Debt/EBITDA — tutti risolti per AAPL |
| Serie storica con date di periodo | **VERDE** | P/B trimestrale FQ0..FQ-12: 13 trimestri con period-end date vere (2026-03-31 → 2023-06-30), nessun left-anchoring visibile |
| Fondamentali sui **delisted** | **ROSSO** | LEH.N^I08 tutto `<NA>`; GM.N^F09 ha P/B *contaminati* (valori 2019-2021 sul RIC della vecchia GM fallita nel 2009) |
| Cross-section a data storica assoluta | **ROSSO** | `SDate=2010-06-30` su 5 RIC → **timeout 20s** (performance inadeguata per ~500 nomi/mese) |
| As-reported vs restated (look-ahead) | **NON RISOLTO** | il campo `.originalannouncedate` non si risolve sul nome provato; serve trovare il campo di filing date per gestire il reporting lag |

**VERDETTO: GIALLO** — i fondamentali sono tecnicamente recuperabili e con date di periodo
corrette, ma **due problemi bloccanti** e **una domanda aperta critica** impediscono di
procedere direttamente al backtest.

---

## I tre ostacoli (in ordine di gravità)

### 1. Delisted contaminati → survivorship bias rientra dalla finestra (ROSSO)

Il 42% del paniere 2005 è composto da RIC delistati. Il probe mostra che i loro fondamentali
sono **inaffidabili**: Lehman (`LEH.N^I08`) ritorna tutto `<NA>`; GM (`GM.N^F09`, la vecchia GM
fallita nel 2009) ritorna P/B datati 2019-2021 — sono i fondamentali della **nuova** GM
(re-IPO 2010) erroneamente associati al RIC del titolo morto. Un backtest che usa questi valori
selezionerebbe su dati sbagliati o escluderebbe i morti → **lo stesso survivorship bias che il
PIT elimina rientrerebbe dal layer fondamentali**. È il fallimento esatto già visto con
`get_data Frq=D` sui prezzi (left-anchoring), in forma diversa.

### 2. Cross-section storica troppo lenta (ROSSO operativo)

La query naturale per un backtest cross-sectional — "P/B di tutti i membri a una data passata" —
è andata in **timeout (20s) su soli 5 RIC**. Su ~500 nomi × ~270 mesi sarebbe inutilizzabile
senza una strategia di fetch diversa (per-RIC serie storica + allineamento locale, come fatto
per i total return, non query cross-section per data).

### 3. As-reported vs restated: il look-ahead nascosto (DA RISOLVERE PRIMA DEL BACKTEST)

I ratio P/B "Daily Time Series Ratio" usano il prezzo (noto ogni giorno) sul book value per share
dell'ultimo report. **Domanda critica non risolta:** il book value restituito è quello *noto a
quella data* (as-first-reported) o quello *corrente ristrutturato* (restated)? E con quale lag
diventa noto (un report Q1 è depositato ~6 settimane dopo la fine trimestre)? Se il backtest usa
il valore del 2026-03-31 come noto il 2026-03-31, è **look-ahead** pari al reporting lag, più
l'eventuale look-ahead da restatement. Il campo `.originalannouncedate`/filing date va trovato e
validato PRIMA di qualsiasi backtest — è la differenza tra un backtest onesto e uno inquinato.

---

## Raccomandazione

Sprint Q è **fattibile ma non ancora pronto**. Prima di un backtest servono tre verifiche
(parte 2 del probe, nessun backtest):

1. **Filing date / as-reported:** trovare il campo Refinitiv che dà la data di deposito e
   verificare se i valori sono as-first-reported; in subordine, applicare un lag prudenziale
   fisso (es. 90 giorni dopo period-end) e documentarlo.
2. **Copertura delisted:** quantificare quanti dei ~480 RIC delistati hanno fondamentali validi
   e non contaminati; se la copertura è bassa, il segnale fondamentale non è PIT-onesto.
3. **Fetch per-RIC:** replicare il pattern di `fetch_pit_total_returns.py` (serie storica per RIC,
   resumabile, allineamento locale) invece della cross-section per data che va in timeout.

Solo se i tre passano: spec congelata + criteri ex-ante + pre-run review, come Sprint O.

**Alternativa a costo zero di rischio-dati:** un segnale **low-volatility** cross-sectional usa
solo i prezzi total-return già scaricati e validati (nessun nuovo data layer, nessun problema di
reporting lag o restatement). Il low-vol anomaly ha letteratura robusta e viva nelle large cap.
Candidato naturale per il prossimo sprint se non si vuole affrontare i tre ostacoli fondamentali.

*Probe: `/tmp/probe_fundamentals_pit.py` (sessione platform).*
