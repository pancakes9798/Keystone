# Validazione Live — Constituent History S&P 500

**Data esecuzione:** 2026-06-12  
**Sorgente:** Refinitiv JL change-log (`TR.IndexJL*`, `IC=B`)  
**Panieri ricostruiti:** 282 snapshot mensili (2003-01 → oggi)  
**Spot checks:** 6/6 PASS  
**Conteggi year-end fuori attesa:** 0  
**Esito complessivo:** **PASS**  

---

## Conteggio membri per year-end

Dove la verità storica è nota con certezza il check è ESATTO: 500 ticker
fino al 2013 (sostituzioni one-for-one), 502 a fine 2014 (GOOG/GOOGL +
DISCK), 504 a fine 2015. Per gli anni successivi (share class multiple)
si usa la banda [490, 515].

| Anno | Data | Membri | Atteso | Esito |
|------|------|--------|--------|-------|
| 2003 | 2003-12-31 | 500 | 500 | OK |
| 2004 | 2004-12-31 | 500 | 500 | OK |
| 2005 | 2005-12-31 | 500 | 500 | OK |
| 2006 | 2006-12-31 | 500 | 500 | OK |
| 2007 | 2007-12-31 | 500 | 500 | OK |
| 2008 | 2008-12-31 | 500 | 500 | OK |
| 2009 | 2009-12-31 | 500 | 500 | OK |
| 2010 | 2010-12-31 | 500 | 500 | OK |
| 2011 | 2011-12-31 | 500 | 500 | OK |
| 2012 | 2012-12-31 | 500 | 500 | OK |
| 2013 | 2013-12-31 | 500 | 500 | OK |
| 2014 | 2014-12-31 | 502 | 502 | OK |
| 2015 | 2015-12-31 | 504 | 504 | OK |
| 2016 | 2016-12-31 | 505 | [490, 515] | OK |
| 2017 | 2017-12-31 | 505 | [490, 515] | OK |
| 2018 | 2018-12-31 | 505 | [490, 515] | OK |
| 2019 | 2019-12-31 | 505 | [490, 515] | OK |
| 2020 | 2020-12-31 | 505 | [490, 515] | OK |
| 2021 | 2021-12-31 | 505 | [490, 515] | OK |
| 2022 | 2022-12-31 | 503 | [490, 515] | OK |
| 2023 | 2023-12-31 | 503 | [490, 515] | OK |
| 2024 | 2024-12-31 | 503 | [490, 515] | OK |
| 2025 | 2025-12-31 | 503 | [490, 515] | OK |

---

## Spot checks

| # | RIC | Data | Atteso | Esito | Descrizione |
|---|-----|------|--------|-------|-------------|
| 1 | `LEH.N^I08` | 2008-06-30 | PRESENTE | **PASS** | Lehman in basket giugno 2008 (prima del crack) |
| 2 | `LEH.N^I08` | 2008-12-31 | ASSENTE | **PASS** | Lehman OUT basket dic 2008 (uscita 17 set 2008) |
| 3 | `GM.N^F09` | 2008-12-31 | PRESENTE | **PASS** | GM in basket dic 2008 (prima del bankruptcy) |
| 4 | `GM.N^F09` | 2009-12-31 | ASSENTE | **PASS** | GM OUT basket dic 2009 (uscita 3 giu 2009) |
| 5 | `TSLA.OQ` | 2015-12-31 | ASSENTE | **PASS** | TSLA assente nel 2015 (entrata dic 2020) |
| 6 | `TSLA.OQ` | 2021-12-31 | PRESENTE | **PASS** | TSLA presente nel 2021 (dopo ingresso dic 2020) |

---

## Note metodologiche

- **Algoritmo:** backward walk dal paniere corrente (basket_at(T) = composizione alla chiusura di T)
  con semantica esplicita join/leave: il fetch usa `IC=B` (joiners+leavers) e il campo
  `TR.IndexJLConstituentRIC.change` (colonna `Change`: Joiner/Leaver).
- **Copertura:** con `IC=B` il log è essenzialmente completo dal 1995 (densità eventi 1995-2002
  coerente col turnover storico pubblico). La finestra di backtest dal 2003 resta una convenzione
  conservativa, non un limite del log.
- **Join pre-1995:** i membri entrati prima dell'inizio del log (es. GM) non hanno evento join:
  restano nel paniere fino all'inizio della finestra — comportamento corretto.
- **Coppie same-day:** le coppie join+leave stesso giorno/stesso RIC (10 nel log, es. EVHC.N^L16)
  sono cancellate come no-op netti prima del walk — l'esito non dipende dall'ordine riga API.
- **Anomalia nota:** FJ.N^K00 (Fort James) ha due leave consecutivi (1997, 2000) senza join in mezzo —
  unico gap genuino del log; con la semantica esplicita non corrompe il walk (add idempotente).
- **Spot check RIC delistati:** i RIC con suffisso `^MMMYY` (es. `LEH.N^I08`) sono identificatori
  Refinitiv per titoli rimossi; la loro presenza nel JL log è stata verificata nella probe 2026-06-11.

*Script: `scripts/validate_constituents.py`*