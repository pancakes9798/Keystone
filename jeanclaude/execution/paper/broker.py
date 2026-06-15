"""PaperBroker — simulated order execution and portfolio state persistence.

Contratto di accounting (P0, 2026-06-10):
- ``nav_history`` è la fonte di verità del NAV: una riga per data logica (upsert).
- ``record_daily_nav(date=D)`` calcola dalla base strettamente precedente a D
  usando i pesi in vigore prima di D → i re-run sono idempotenti.
- ``execute_rebalance`` non tocca nav_history: ritorna il costo, che il chiamante
  applica con ``apply_rebalance_cost`` DOPO record_daily_nav.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from jeanclaude.data.storage.parquet_store import ParquetStore

logger = logging.getLogger(__name__)

_CAT = "paper_trading"
_STATE_KEY = "state"


@dataclass
class Order:
    """A single simulated order generated at rebalance time."""
    date: pd.Timestamp
    asset: str
    old_weight: float
    new_weight: float
    price: float
    simulated_cost_bps: float


@dataclass
class RebalanceResult:
    """Esito di execute_rebalance: ordini + costo flat stimato (NON ancora applicato al NAV)."""
    orders: list[Order] = field(default_factory=list)
    cost_eur: float = 0.0


@dataclass
class PortfolioState:
    """Current portfolio snapshot."""
    date: pd.Timestamp
    weights: pd.Series   # asset → weight (sums to ≤ 1; remainder = cash)
    nav: float           # portfolio NAV in €
    cash: float          # unallocated cash in €


def _as_date(date: pd.Timestamp | None) -> pd.Timestamp:
    return (date if date is not None else pd.Timestamp.now()).normalize()


class PaperBroker:
    """Simulates order execution and manages portfolio state on Parquet.

    State is stored in three files under ``store`` root:
    - ``paper_trading/positions__state__state.parquet``
    - ``paper_trading/nav_history__state__state.parquet``
    - ``paper_trading/orders__state__state.parquet``

    Parameters
    ----------
    initial_capital : float
        Starting capital in €. Used only when no state file exists yet.
    store : ParquetStore
        Parquet storage backend.
    """

    def __init__(self, initial_capital: float, store: ParquetStore) -> None:
        self._initial_capital = initial_capital
        self._store = store

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    def get_state(self) -> PortfolioState:
        """Return the current portfolio state.

        If no state file exists, returns the initial state (empty weights,
        full capital as NAV and cash).
        """
        positions = self._store.load(_CAT, "positions", _STATE_KEY, _STATE_KEY)
        if positions is None or positions.empty:
            return PortfolioState(
                date=pd.Timestamp.now().normalize(),
                weights=pd.Series(dtype=float),
                nav=self._initial_capital,
                cash=self._initial_capital,
            )
        latest = positions.iloc[-1]
        weight_cols = [c for c in positions.columns if c not in ("nav", "cash")]
        weights = latest[weight_cols].astype(float)
        return PortfolioState(
            date=latest.name if isinstance(latest.name, pd.Timestamp) else pd.Timestamp(latest.name),
            weights=weights,
            nav=float(latest["nav"]),
            cash=float(latest["cash"]),
        )

    def weights_before(self, date: pd.Timestamp) -> pd.Series:
        """Pesi in vigore PRIMA di ``date`` (ultima riga positions con index < date).

        Garantisce che il P&L del giorno D sia attribuito ai pesi vecchi anche
        quando il rebalance di D è già stato persistito (re-run dello stesso giorno).
        """
        positions = self._store.load(_CAT, "positions", _STATE_KEY, _STATE_KEY)
        if positions is None or positions.empty:
            return pd.Series(dtype=float)
        prior = positions[positions.index < date]
        if prior.empty:
            return pd.Series(dtype=float)
        latest = prior.iloc[-1]
        weight_cols = [c for c in positions.columns if c not in ("nav", "cash")]
        return latest[weight_cols].astype(float)

    def last_nav_date(self) -> pd.Timestamp | None:
        """Data dell'ultima riga di nav_history, o None se vuota."""
        nav_df = self._store.load(_CAT, "nav_history", _STATE_KEY, _STATE_KEY)
        if nav_df is None or nav_df.empty:
            return None
        return pd.Timestamp(nav_df.index[-1])

    def nav_date_before(self, date: pd.Timestamp) -> pd.Timestamp | None:
        """Data dell'ultima riga nav_history STRETTAMENTE precedente a ``date``.

        È la stessa base che :meth:`record_daily_nav` usa per il NAV (riga con
        index < date). Ancorare qui la finestra del ritorno (vedi
        :meth:`period_returns`) garantisce che un re-run sullo stesso ``date`` —
        quando la riga di ``date`` esiste già — usi la STESSA finestra del primo
        run, invece di collassare a 1 giorno. È la differenza con
        :meth:`last_nav_date`, che ritorna l'ultima riga ASSOLUTA (= ``date``
        stesso dopo il primo run).
        """
        nav_df = self._store.load(_CAT, "nav_history", _STATE_KEY, _STATE_KEY)
        if nav_df is None or nav_df.empty:
            return None
        prior = nav_df[nav_df.index < date]
        if prior.empty:
            return None
        return pd.Timestamp(prior.index[-1])

    def period_returns(
        self,
        prices: pd.DataFrame,
        asof: pd.Timestamp,
        assets: list[str],
    ) -> pd.Series:
        """Ritorni semplici gap-aware dalla data-base NAV ad ``asof``.

        La finestra è ``[nav_date_before(asof), asof]`` — ancorata alla riga NAV
        STRETTAMENTE precedente ad ``asof``, la stessa base di
        :meth:`record_daily_nav`. NON all'ultima riga NAV assoluta: così un
        re-run sullo stesso ``asof`` (riga già scritta) riproduce esattamente il
        primo run → NAV idempotente per una data fissa. Regression:
        ``tests/execution/test_broker_period_returns.py``.

        Se la data-base è sparita dall'indice prezzi (cambio universo, cache
        ricostruita), il fallback a 1 giorno PERDE il P&L del gap: logga un
        warning forte invece di mis-contabilizzare in silenzio.
        """
        base_date = self.nav_date_before(asof)
        if base_date is not None and base_date in prices.index and base_date < asof:
            return (prices.loc[asof] / prices.loc[base_date] - 1).reindex(assets, fill_value=0.0)
        if base_date is not None and base_date < asof and base_date not in prices.index:
            logger.warning(
                "P&L window: data-base NAV %s non è nell'indice prezzi — "
                "fallback a ritorno 1-giorno, possibile P&L perso nel gap %s → %s. "
                "Verificare la cache prezzi/universo.",
                base_date.date(), base_date.date(), asof.date(),
            )
        window = prices.loc[:asof]
        if len(window) >= 2:
            return (window.iloc[-1] / window.iloc[-2] - 1).reindex(assets, fill_value=0.0)
        return pd.Series(0.0, index=assets)

    def last_rebalance_date(self) -> pd.Timestamp | None:
        """Data dell'ultimo rebalance persistito (ultima riga positions), o None."""
        positions = self._store.load(_CAT, "positions", _STATE_KEY, _STATE_KEY)
        if positions is None or positions.empty:
            return None
        return pd.Timestamp(positions.index[-1])

    @property
    def nav(self) -> float:
        """Current portfolio NAV in currency units.

        Reads the latest value from nav_history (updated daily by record_daily_nav).
        Falls back to positions.nav when nav_history is empty, then to
        initial_capital when neither file exists.

        # Gli accessor ricaricano da Parquet a ogni chiamata: a 1 ciclo/giorno è il
        # design giusto (stato sempre fresco). Valutare caching solo per replay loop.
        """
        nav_history = self._store.load(_CAT, "nav_history", _STATE_KEY, _STATE_KEY)
        if nav_history is not None and not nav_history.empty:
            return float(nav_history["nav"].iloc[-1])
        return self.get_state().nav

    # ------------------------------------------------------------------
    # Writers (date-aware, upsert per data)
    # ------------------------------------------------------------------

    def execute_rebalance(
        self,
        new_weights: pd.Series,
        prices: pd.Series,
        transaction_cost_bps: float = 10.0,
        date: pd.Timestamp | None = None,
    ) -> RebalanceResult:
        """Simulate a portfolio rebalance for the logical ``date``.

        Persiste positions e orders con semantica upsert (una sola versione per
        data). NON modifica nav_history e NON applica il costo: il costo flat
        stimato è in ``RebalanceResult.cost_eur`` e va applicato dal chiamante
        con :meth:`apply_rebalance_cost` (oppure sostituito dalla stima
        Almgren-Chriss quando disponibile).

        Parameters
        ----------
        new_weights : pd.Series
            Target weights (asset → float, sum ≤ 1).
        prices : pd.Series
            Current prices for each asset (used only for order logging).
        transaction_cost_bps : float
            One-way transaction cost in basis points (per il costo flat).
        date : pd.Timestamp | None
            Data logica del rebalance (default: oggi).

        Returns
        -------
        RebalanceResult
        """
        when = _as_date(date)
        if new_weights.isna().any():
            raise ValueError(
                f"execute_rebalance: pesi NaN per {list(new_weights[new_weights.isna()].index)}"
            )
        total_w = float(new_weights.sum())
        if total_w > 1.0 + 1e-6:
            raise ValueError(
                f"execute_rebalance: somma pesi {total_w:.4f} > 1 — leva non supportata dal PaperBroker."
            )
        old_weights_now = self.weights_before(when)
        old_weights = old_weights_now.reindex(new_weights.index, fill_value=0.0)
        current_nav = self.nav

        orders: list[Order] = []
        for asset, new_w in new_weights.items():
            orders.append(Order(
                date=when,
                asset=str(asset),
                old_weight=float(old_weights.get(asset, 0.0)),
                new_weight=float(new_w),
                price=float(prices.get(asset, float("nan"))),
                simulated_cost_bps=float(transaction_cost_bps),
            ))

        missing_prices = [o.asset for o in orders if pd.isna(o.price)]
        if missing_prices:
            logger.warning(
                "execute_rebalance: prezzo mancante per %s alla data %s — "
                "registrato NaN nell'order log.", missing_prices, when.date()
            )

        # Costo flat dal turnover (union di vecchi e nuovi asset: i drop contano come sell)
        all_assets = old_weights_now.index.union(new_weights.index)
        old_full = old_weights_now.reindex(all_assets, fill_value=0.0)
        new_full = new_weights.reindex(all_assets, fill_value=0.0)
        turnover = float((new_full - old_full).abs().sum())
        cost_eur = current_nav * (transaction_cost_bps / 10_000) * turnover

        # Upsert positions alla data
        positions = self._store.load(_CAT, "positions", _STATE_KEY, _STATE_KEY)
        row = new_weights.to_dict()
        row["nav"] = current_nav
        row["cash"] = current_nav * max(0.0, 1.0 - float(new_weights.sum()))
        new_row = pd.DataFrame([row], index=pd.DatetimeIndex([when], name="date"))
        if positions is not None and not positions.empty:
            positions = positions[positions.index != when]
            updated = pd.concat([positions, new_row]).sort_index()
        else:
            updated = new_row
        self._store.save(_CAT, "positions", _STATE_KEY, _STATE_KEY, updated)

        # Upsert orders alla data
        orders_df = pd.DataFrame(
            [
                {
                    "date": o.date,
                    "asset": o.asset,
                    "old_weight": o.old_weight,
                    "new_weight": o.new_weight,
                    "price": o.price,
                    "simulated_cost_bps": o.simulated_cost_bps,
                }
                for o in orders
            ]
        ).set_index("date")
        existing_orders = self._store.load(_CAT, "orders", _STATE_KEY, _STATE_KEY)
        if existing_orders is not None and not existing_orders.empty:
            existing_orders = existing_orders[existing_orders.index != when]
            updated_orders = pd.concat([existing_orders, orders_df]).sort_index()
        else:
            updated_orders = orders_df
        self._store.save(_CAT, "orders", _STATE_KEY, _STATE_KEY, updated_orders)

        logger.info(
            "Rebalanced %d assets on %s | NAV %.2f | turnover %.2f%% | flat cost %.2f EUR",
            len(orders), when.date(), current_nav, turnover * 100, cost_eur,
        )
        return RebalanceResult(orders=orders, cost_eur=cost_eur)

    def record_daily_nav(
        self,
        daily_returns: pd.Series,
        pre_applied_cost: float = 0.0,
        date: pd.Timestamp | None = None,
    ) -> float:
        """Upsert del NAV per la data logica ``date``.

        Base = ultima riga di nav_history strettamente precedente a ``date``
        (fallback: positions/initial capital). Pesi = quelli in vigore prima di
        ``date``. Una riga esistente alla stessa data viene sovrascritta →
        chiamate ripetute per la stessa data sono idempotenti.

        Attenzione (backfill): scrivere una data D1 PRECEDENTE a righe già
        esistenti non ricalcola le righe successive — le righe dopo D1 restano
        calcolate sulla vecchia base. Il runner giornaliero non fa mai backfill;
        per ricostruire la storia, ripartire da uno store vuoto in ordine
        cronologico.

        Nota (giorno zero): se non esiste alcuna riga positions precedente a
        ``date`` (prima allocazione il giorno stesso), i pesi sono vuoti e il
        return del giorno non viene applicato: il NAV resta la base — corretto,
        perché il capitale non era ancora investito prima di ``date``.

        Parameters
        ----------
        daily_returns : pd.Series
            Simple returns per asset per il periodo che termina in ``date``
            (può coprire più giorni se i run precedenti sono stati saltati).
        pre_applied_cost : float
            Costo già stimato per oggi (EUR), sottratto dal NAV.
        date : pd.Timestamp | None
            Data logica (default: oggi).

        Returns
        -------
        float
            Updated NAV in €.
        """
        when = _as_date(date)
        weights = self.weights_before(when)

        nav_df = self._store.load(_CAT, "nav_history", _STATE_KEY, _STATE_KEY)
        if nav_df is not None and not nav_df.empty:
            prior = nav_df[nav_df.index < when]
            current_nav = float(prior["nav"].iloc[-1]) if not prior.empty else self.get_state().nav
        else:
            current_nav = self.get_state().nav

        if weights.empty:
            new_nav = current_nav
        else:
            aligned = daily_returns.reindex(weights.index, fill_value=0.0)
            # Guardia NaN: pandas .sum() skipna inghiottirebbe silenziosamente il P&L di uno
            # sleeve investito. Fallire rumorosamente permette al cron di inviare l'alert e
            # rilanciare il run in modo idempotente con i prezzi corretti.
            invested = weights[weights != 0.0]
            nan_assets = aligned[invested.index][aligned[invested.index].isna()]
            if not nan_assets.empty:
                raise ValueError(
                    "record_daily_nav: ritorni NaN per asset investiti "
                    f"{list(nan_assets.index)} alla data {when.date()} — "
                    "dato mancante a monte; correggere i prezzi e rilanciare il run."
                )
            aligned = aligned.fillna(0.0)
            portfolio_return = float((aligned * weights).sum())
            new_nav = current_nav * (1.0 + portfolio_return)

        new_nav = max(0.0, new_nav - pre_applied_cost)

        new_row = pd.DataFrame({"nav": [new_nav]}, index=pd.DatetimeIndex([when], name="date"))
        if nav_df is not None and not nav_df.empty:
            nav_df = nav_df[nav_df.index != when]
            updated = pd.concat([nav_df, new_row]).sort_index()
        else:
            updated = new_row
        self._store.save(_CAT, "nav_history", _STATE_KEY, _STATE_KEY, updated)

        logger.info(
            "Daily NAV recorded for %s: %.2f (pre_applied_cost=%.2f)",
            when.date(), new_nav, pre_applied_cost,
        )
        return new_nav

    def apply_rebalance_cost(self, date: pd.Timestamp | None, cost_eur: float) -> float:
        """Sottrae ``cost_eur`` dalla riga nav_history di ``date`` (che deve esistere).

        Da chiamare DOPO record_daily_nav nello stesso ciclo. Idempotente a
        livello di ciclo: un re-run riparte da record_daily_nav che resetta la
        riga, quindi il costo viene applicato esattamente una volta.
        """
        when = _as_date(date)
        nav_df = self._store.load(_CAT, "nav_history", _STATE_KEY, _STATE_KEY)
        if nav_df is None or nav_df.empty or when not in nav_df.index:
            raise ValueError(
                f"apply_rebalance_cost: nessuna riga nav_history per {when.date()} — "
                "chiamare prima record_daily_nav."
            )
        new_nav = max(0.0, float(nav_df.loc[when, "nav"]) - cost_eur)
        # Immutabile: ricostruisci invece di mutare in place (coerente con record_daily_nav).
        new_row = pd.DataFrame({"nav": [new_nav]}, index=pd.DatetimeIndex([when], name="date"))
        updated = pd.concat([nav_df[nav_df.index != when], new_row]).sort_index()
        self._store.save(_CAT, "nav_history", _STATE_KEY, _STATE_KEY, updated)
        logger.info("Rebalance cost applied on %s: %.2f EUR | NAV → %.2f", when.date(), cost_eur, new_nav)
        return new_nav

