"""Verifica integrità del JL change-log S&P 500 scaricato da Refinitiv.

Controlli:
1. Conteggi totali (eventi, join, leave)
2. Righe duplicate esatte (date, ric, change)
3. Coppie same-day stesso RIC con change diversi (ordine un-apply ambiguo)
4. Eventi consecutivi nella stessa direzione per RIC
5. Consistenza ultimo evento vs paniere corrente
6. Sentinel noti (TSLA, Lehman, GM)
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data" / "constituents"


def _latest(pattern: str) -> Path:
    """Ritorna il file di cache più recente per il pattern (ordinato per nome)."""
    candidates = sorted(DATA_DIR.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"Nessun file '{pattern}' in {DATA_DIR} — esegui prima "
            "scripts/validate_constituents.py per popolare la cache."
        )
    return candidates[-1]


JL_PATH = _latest("jl_events__*.parquet")
BASKET_PATH = _latest("current_basket__*.parquet")

SENTINELS = {
    "TSLA.OQ": {"join": ["2020-12-21"], "leave": []},
    "LEH.N^I08": {"join": ["1998-01-12"], "leave": ["2008-09-17"]},
    "GM.N^F09": {"join": [], "leave": ["2009-06-03"]},
}


def check_counts(jl: pd.DataFrame) -> None:
    print("=== 1. Conteggi totali ===")
    counts = jl["change"].value_counts()
    print(f"Eventi totali: {len(jl)}")
    print(f"Join:  {counts.get('join', 0)}")
    print(f"Leave: {counts.get('leave', 0)}")
    unexpected = set(jl["change"].unique()) - {"join", "leave"}
    print(f"Valori change inattesi: {sorted(unexpected) if unexpected else 'nessuno'}")
    print(f"Range date: {jl['date'].min().date()} -> {jl['date'].max().date()}")
    print(f"NaN: date={jl['date'].isna().sum()}, ric={jl['ric'].isna().sum()}, "
          f"change={jl['change'].isna().sum()}")


def check_exact_duplicates(jl: pd.DataFrame) -> None:
    print("\n=== 2. Righe duplicate esatte (date, ric, change) ===")
    dup_mask = jl.duplicated(subset=["date", "ric", "change"], keep=False)
    dups = jl[dup_mask].sort_values(["ric", "date"])
    n_extra = jl.duplicated(subset=["date", "ric", "change"], keep="first").sum()
    print(f"Righe coinvolte in duplicati: {len(dups)} (righe in eccesso: {n_extra})")
    if len(dups):
        print(dups.to_string(index=False))


def check_same_day_pairs(jl: pd.DataFrame) -> None:
    print("\n=== 3. Coppie same-day stesso RIC con change diversi ===")
    grouped = jl.groupby(["date", "ric"])["change"].nunique()
    pairs = grouped[grouped > 1]
    print(f"Coppie (date, ric) con join+leave stesso giorno: {len(pairs)}")
    for (date, ric) in pairs.index:
        print(f"  {date.date()}  {ric}")


def check_consecutive_same_direction(jl: pd.DataFrame) -> None:
    print("\n=== 4. Eventi consecutivi nella stessa direzione per RIC ===")
    jl_sorted = jl.sort_values(["ric", "date"], kind="mergesort")
    offenders = []
    for ric, grp in jl_sorted.groupby("ric"):
        changes = grp["change"].tolist()
        dates = grp["date"].tolist()
        runs = [
            (dates[i].date(), dates[i + 1].date(), changes[i])
            for i in range(len(changes) - 1)
            if changes[i] == changes[i + 1]
        ]
        if runs:
            offenders.append((ric, runs))
    print(f"RIC con eventi consecutivi nella stessa direzione: {len(offenders)}")
    for ric, runs in offenders[:10]:
        runs_str = "; ".join(f"{c} {d1} -> {c} {d2}" for d1, d2, c in runs)
        print(f"  {ric}: {runs_str}")


def check_basket_consistency(jl: pd.DataFrame, basket: set[str]) -> None:
    print("\n=== 5. Consistenza ultimo evento vs paniere corrente ===")
    jl_sorted = jl.sort_values(["ric", "date"], kind="mergesort")
    violations = []
    ambiguous = []
    for ric, grp in jl_sorted.groupby("ric"):
        last_date = grp["date"].iloc[-1]
        last_day_changes = set(grp.loc[grp["date"] == last_date, "change"])
        in_basket = ric in basket
        if len(last_day_changes) > 1:
            ambiguous.append((ric, last_date.date(), in_basket))
            continue
        last_change = last_day_changes.pop()
        expected = "join" if in_basket else "leave"
        if last_change != expected:
            violations.append((ric, last_date.date(), last_change, in_basket))
    rics_with_events = set(jl["ric"].unique())
    basket_no_events = sorted(basket - rics_with_events)
    print(f"RIC con eventi: {len(rics_with_events)}; "
          f"RIC nel paniere senza eventi (mai cambiati dal 1995): {len(basket_no_events)}")
    print(f"Violazioni (ultimo evento incoerente col paniere): {len(violations)}")
    for ric, date, change, in_basket in violations[:10]:
        where = "IN paniere" if in_basket else "NON in paniere"
        print(f"  {ric} ({where}): ultimo evento {change} il {date}")
    print(f"Casi ambigui (join+leave nello stesso ultimo giorno): {len(ambiguous)}")
    for ric, date, in_basket in ambiguous[:10]:
        where = "IN paniere" if in_basket else "NON in paniere"
        print(f"  {ric} ({where}): join+leave entrambi il {date}")


def check_sentinels(jl: pd.DataFrame) -> None:
    print("\n=== 6. Sentinel ===")
    for ric, expected in SENTINELS.items():
        events = jl[jl["ric"] == ric].sort_values("date")
        actual = {
            kind: [d.strftime("%Y-%m-%d")
                   for d in events.loc[events["change"] == kind, "date"]]
            for kind in ("join", "leave")
        }
        ok = all(actual[kind] == expected[kind] for kind in ("join", "leave"))
        status = "OK" if ok else "MISMATCH"
        print(f"  {ric}: {status} — atteso {expected}, trovato {actual}")


def main() -> None:
    jl = pd.read_parquet(JL_PATH)
    basket = set(pd.read_parquet(BASKET_PATH)["ric"])
    print(f"JL log: {JL_PATH.name} ({len(jl)} righe)")
    print(f"Paniere corrente: {BASKET_PATH.name} ({len(basket)} RIC)\n")
    check_counts(jl)
    check_exact_duplicates(jl)
    check_same_day_pairs(jl)
    check_consecutive_same_direction(jl)
    check_basket_consistency(jl, basket)
    check_sentinels(jl)


if __name__ == "__main__":
    main()
