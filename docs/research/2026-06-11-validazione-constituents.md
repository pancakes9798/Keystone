# Validazione Live — Constituent History S&P 500

**Data esecuzione:** 2026-06-11  
**Sorgente:** Refinitiv JL change-log (`TR.IndexJLConstituentChangeDate`)  
**Panieri ricostruiti:** 282 snapshot mensili (2003-01 → oggi)  
**Spot checks:** 5/6 PASS  
**Esito complessivo:** **PARZIALE (5/6)**

---

## Conteggio membri per year-end

Conteggio atteso: ~500 ± 10 per date dal 2003 in poi.

**Nota sull'overcounting:** i conteggi crescendo indietro nel tempo (513 in 2025 → 990 in 2003)
riflettono una limitazione strutturale del JL change-log: il campo `ChangeType` (join/leave) non
viene restituito dall'API. Il toggle backward applica correttamente i leavers (li re-inserisce
nel passato), ma anche i joiners recenti (che non hanno ancora lasciato l'indice) rimangono
nel paniere durante la retromarcia, generando overcounting. Stima: ~480 eventi mancanti nel log
corrispondono alla differenza (990 − 503). La struttura degli spot check (LEH, GM, TSLA giugno 2008,
dic 2008, dic 2009) è comunque corretta — il segnale di membership per i leavers noti è affidabile.

| Anno | Data | Membri | Note |
|------|------|--------|------|
| 2003 | 2003-12-31 | 990 | SOPRA SOGLIA — overcounting noto (joiners 2003-2026 non separabili) |
| 2004 | 2004-12-31 | 971 | SOPRA SOGLIA — overcounting noto |
| 2005 | 2005-12-31 | 955 | SOPRA SOGLIA — overcounting noto |
| 2006 | 2006-12-31 | 924 | SOPRA SOGLIA — overcounting noto |
| 2007 | 2007-12-31 | 887 | SOPRA SOGLIA — overcounting noto |
| 2008 | 2008-12-31 | 856 | SOPRA SOGLIA — overcounting noto |
| 2009 | 2009-12-31 | 843 | SOPRA SOGLIA — overcounting noto |
| 2010 | 2010-12-31 | 827 | SOPRA SOGLIA — overcounting noto |
| 2011 | 2011-12-31 | 807 | SOPRA SOGLIA — overcounting noto |
| 2012 | 2012-12-31 | 789 | SOPRA SOGLIA — overcounting noto |
| 2013 | 2013-12-31 | 778 | SOPRA SOGLIA — overcounting noto |
| 2014 | 2014-12-31 | 768 | SOPRA SOGLIA — overcounting noto |
| 2015 | 2015-12-31 | 742 | SOPRA SOGLIA — overcounting noto |
| 2016 | 2016-12-31 | 707 | SOPRA SOGLIA — overcounting noto |
| 2017 | 2017-12-31 | 675 | SOPRA SOGLIA — overcounting noto |
| 2018 | 2018-12-31 | 651 | SOPRA SOGLIA — overcounting noto |
| 2019 | 2019-12-31 | 629 | SOPRA SOGLIA — overcounting noto |
| 2020 | 2020-12-31 | 612 | SOPRA SOGLIA — overcounting noto |
| 2021 | 2021-12-31 | 588 | SOPRA SOGLIA — overcounting noto |
| 2022 | 2022-12-31 | 568 | SOPRA SOGLIA — overcounting noto |
| 2023 | 2023-12-31 | 550 | SOPRA SOGLIA — overcounting noto |
| 2024 | 2024-12-31 | 533 | SOPRA SOGLIA — overcounting noto |
| 2025 | 2025-12-31 | 513 | OK (paniere più recente, convergente al corrente) |

---

## Spot checks

| # | RIC | Data | Atteso | Esito | Descrizione |
|---|-----|------|--------|-------|-------------|
| 1 | `LEH.N^I08` | 2008-06-30 | PRESENTE | **PASS** | Lehman in basket giugno 2008 (prima del crack) |
| 2 | `LEH.N^I08` | 2008-12-31 | ASSENTE | **PASS** | Lehman OUT basket dic 2008 (uscita 17 set 2008) |
| 3 | `GM.N^F09` | 2008-12-31 | PRESENTE | **PASS** | GM in basket dic 2008 (prima del bankruptcy) |
| 4 | `GM.N^F09` | 2009-12-31 | ASSENTE | **PASS** | GM OUT basket dic 2009 (uscita 3 giu 2009) |
| 5 | `TSLA.OQ` | 2015-12-31 | ASSENTE | **FAIL** | TSLA assente nel 2015 (entrata dic 2020) |
| 6 | `TSLA.OQ` | 2021-12-31 | PRESENTE | **PASS** | TSLA presente nel 2021 (dopo ingresso dic 2020) |

---

## Analisi dei risultati

### Overcounting (conteggi > 500)

Il conteggio in 2025 è 513 (close to 503 correnti + ~10 leavers recenti). Andando indietro,
il conteggio cresce perché ogni RIC che ha LASCIATO l'indice dal 2003 al 2025 (leaver) viene
re-inserito nel paniere storico — correttamente. Ma anche i RIC che sono ENTRATI (joiners) e
sono ancora nel paniere oggi NON vengono rimossi, perché il loro JOIN event non è nel log JL
(la probe 2026-06-11 ha confermato: TSLA assente dal JL log). 

**Stima eventi mancanti:** 990 − 503 ≈ 487. Questo corrisponde approssimativamente al numero
di JOIN events (nuovi entranti) tra il 2003 e il 2025 che non sono registrati nel JL change-log.

**Impatto sul backtest:** l'overcounting introduce survivorship bias in avanti (titoli che
entrano tardi nel periodo sembrano già presenti prima). Per un backtest cross-sectional su
momentum, questo significa che alcuni "futuri winners" (grandi cap che sono entrate nell'S&P 500
dopo il 2005) appaiono nel paniere anche prima della loro inclusion. Questo è conservativo
per il momentum (si includono titoli che cresceranno) ma rompe la PIT-ness.

### Spot check TSLA (FAIL previsto)

TSLA.OQ non ha eventi JL perché è ancora nell'indice (non è mai uscita). Il suo evento JOIN
(dicembre 2020) non è catturato dal log. Di conseguenza, il backward toggle non rimuove TSLA
dal paniere pre-2020: TSLA appare correttamente nel 2021 (PASS) ma erroneamente appare anche
nel 2015 (FAIL). Questo è un limite documentato e previsto — la probe lo aveva anticipato.

### Spot check LEH e GM (PASS)

I 4 check su Lehman e GM passano correttamente: entrambi i RIC hanno eventi JL
(`LEH.N^I08` data 2008-09-17, `GM.N^F09` data 2009-06-03) e il backward toggle
li inserisce correttamente nel paniere pre-evento e li rimuove post-evento.
Questo conferma che il meccanismo per i **leavers** funziona correttamente.

### Implicazioni per Sprint O

L'overcounting sistematico richiede una delle due strategie:
1. **Usare solo leavers:** filtrare il paniere ricostruito mantenendo solo i RIC che hanno
   un evento JL (leavers verificati) + il paniere corrente. Esclude i joiners senza evento.
2. **Fonte esterna per joiners:** integrare i JOIN events da una fonte alternativa
   (es. Wikipedia historical S&P 500, CRSP/WRDS) per completare il log.

Per Sprint O (momentum baseline), l'opzione 1 riduce l'universo ma elimina il bias.
Raccomandazione: usare l'intersezione (paniere corrente ∩ presenti nel JL log) per
avere solo ticker con storia verificabile.

---

## Note metodologiche

- **Algoritmo:** backward toggle dal paniere corrente (`basket_at(T)` = composizione alla chiusura di T).
- **Copertura JL:** 657 eventi 2000-2026 + 167 pre-2000 (824 totali); mancano ~480 JOIN events.
- **ChangeType mancante:** tutti gli eventi sono trattati come flip di membership (join/leave non distinti).
- **TSLA FAIL:** comportamento atteso — TSLA non ha eventi JL (mai uscita dall'indice), il JOIN non è loggato.
- **Spot check RIC delistati:** i RIC con suffisso `^MMMYY` (es. `LEH.N^I08`) sono identificatori
  Refinitiv per titoli rimossi; validati nella probe 2026-06-11.

*Script: `scripts/validate_constituents.py`*
