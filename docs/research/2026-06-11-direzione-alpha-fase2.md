# Caccia all'alpha — Fase 2: cambiare fonte di rendimento

**Stato:** brief approvato per kickoff. **Premessa:** la serie I→M (42 trial, `data/research/trials.jsonl`) ha chiuso la famiglia "meccaniche di allocazione su 20 ETF": frontiera Sharpe 0.91–0.95, CAGR 5–7.3%, DD −13/−16%. Ogni ulteriore variante di allocazione = data mining. La Fase 2 cambia FONTE di rendimento.

## Direzione raccomandata: selezione titoli cross-sectional su universo point-in-time

**L'edge è il dato, non il segnale.** Il selection bias (la lezione n.1 dell'audit) si risolve solo con i constituent storici as-of: **Refinitiv ha la composizione storica degli indici** (chain `0#.SPX` con date di ingresso/uscita) — è l'asset più prezioso dell'accesso LSEG e finora non l'abbiamo usato. Con l'universo S&P 500 point-in-time si possono testare onestamente segnali cross-sectional:

1. **Momentum 12-1 cross-sectional** (il classico, baseline da battere internamente)
2. **FinBERT news sentiment cross-sectional** — lo scorer e l'aggregatore per-ticker ESISTONO GIÀ (`jeanclaude/signals/news/`, Sprint H) ma non sono mai stati validati come segnale di selezione; le news storiche Refinitiv ne permettono il backtest
3. Combinazioni momentum × sentiment (doppio sort)

**Disciplina invariata:** ExperimentConfig congelata, registry (n_trials cumulativo riparte dal conteggio attuale 45 record/42 distinti), DSR, criteri ex-ante, pre-run review, OOS una volta.

## Lavoro preparatorio (Sprint N — data layer, nessun backtest)

1. `RefinitivSource.get_index_constituents(index_ric, as_of)` — verificare il supporto API (`ld.get_data` su chain storiche o `TR.IndexConstituentRIC` con SDate) e costruire lo snapshot mensile 2005→oggi, cache parquet versionata.
2. Validazione: confronto spot dei constituent in 3 date note (es. 2008, 2015, 2020) contro fonti pubbliche; conteggio ~500 ±tolleranza per data.
3. Prezzi total return per i constituent: il problema dividendi di Refinitiv (TRDPRC_1 = price return) qui NON è aggirabile con Yahoo (i delisted non ci sono su Yahoo — è proprio il punto del point-in-time). Indagare il campo TR Refinitiv (`TR.TotalReturn` daily o CLOSEPRICE adjusted) PRIMA di qualunque backtest; in subordine, accettare price return + stima dividendi di settore documentata.
4. Solo dopo: Sprint O = primo backtest selezione titoli (momentum baseline) con criteri ex-ante.

## Criteri ex-ante suggeriti per Sprint O (da congelare al kickoff)

- Long-only top-decile momentum, ribilancio mensile, costi 10bps+slippage, universo point-in-time: CAGR ≥ SPY−2pt con DD ≤ SPY (il momentum storico promette questo; se non lo consegna sul nostro setup, la pipeline ha un problema) — è un *validation gate* della pipeline più che una strategia finale.

## Rischi noti

- Constituent history potrebbe richiedere permessi LSEG specifici → verificare subito (Sprint N task 1).
- Costi realistici su 50-100 titoli > ETF; Almgren-Chriss già nel repo, da ricalibrare.
- Capacity/realismo: con paper trading a 100k il punto è il segnale, non la capacità.
