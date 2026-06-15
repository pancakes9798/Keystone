"""Walk-forward backtesting engine.

Simulates portfolio performance with periodic rebalancing. At each rebalance
date, only data strictly before that date is passed to the optimizer — no
lookahead. Transaction costs are applied as a fraction of turnover.

Usage::

    engine = BacktestEngine(optimizer=HRPOptimizer(EWMACovariance()))
    result = engine.run(returns, min_history=60)
    print(summary(result))
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    """Configuration for the backtesting engine."""
    rebalance_freq: str = "W"            # pandas offset alias: "W", "ME", "QE"
    initial_capital: float = 1_000_000.0
    transaction_cost_bps: float = 10.0   # one-way cost in basis points
    min_history: int = 60                # minimum observations before first rebalance
    execution_lag: int = 1               # Giorni di mercato tra il segnale (chiusura t) e l'esecuzione
                                         # (chiusura t+lag). 0 = comportamento legacy (stesso giorno, ottimistico).
    max_leverage: float = 1.0            # Leva massima ammessa (Σw). Default 1.0 = nessuna leva.
                                         # Pesi con Σw > max_leverage sollevano ValueError.
    funding_rate_annual: Union[float, "pd.Series"] = 0.0
                                         # Tasso annuo di funding sul capitale preso a prestito
                                         # max(0, Σw−1). Applicato /252 ogni giorno in cui i pesi
                                         # sono levered. Una pd.Series viene riallineata per data
                                         # con forward-fill.


@dataclass
class BacktestResult:
    """Output of a completed backtest run."""
    portfolio_returns: pd.Series    # daily return series
    weights_history: pd.DataFrame   # weights at each rebalance date
    turnover: pd.Series             # one-way turnover at each rebalance
    cost_history: pd.Series = None  # total_cost_pct per rebalance date; empty if no cost_model

    def __post_init__(self) -> None:
        if self.cost_history is None:
            self.cost_history = pd.Series(dtype=float)


class BacktestEngine:
    """Walk-forward portfolio backtesting engine.

    Parameters
    ----------
    optimizer : object with .optimize(returns: pd.DataFrame) -> pd.Series
        Portfolio optimizer. Called at each rebalance date with historical data.
    config : BacktestConfig
        Engine parameters.
    risk_filter : RiskFilter, optional
        Applied after optimization to scale down weights if risk limits breached.
    """

    def __init__(
        self,
        optimizer,
        config: Optional[BacktestConfig] = None,
        risk_filter=None,
        cost_model=None,
    ) -> None:
        self.optimizer = optimizer
        self.config = config or BacktestConfig()
        self.risk_filter = risk_filter
        self.cost_model = cost_model

    # ── Private helpers ───────────────────────────────────────────────────────

    def _funding_rate_at(self, date: pd.Timestamp) -> float:
        """Restituisce il tasso di funding annuo per la data *date*.

        Se ``funding_rate_annual`` è una pd.Series, usa il valore più recente
        disponibile fino a *date* (forward-fill implicito). Se è un float
        scalare, lo ritorna direttamente.
        """
        fr = self.config.funding_rate_annual
        if isinstance(fr, pd.Series):
            s = fr.loc[:date]
            return float(s.iloc[-1]) if len(s) else 0.0
        return float(fr)

    def _validate_leverage(self, weights: pd.Series) -> None:
        """Valida che la somma lorda dei pesi non superi max_leverage.

        Raises
        ------
        ValueError
            Se la somma lorda (pesi positivi) supera ``config.max_leverage``.
        """
        gross = float(weights.clip(lower=0).sum())
        if gross > self.config.max_leverage + 1e-9:
            raise ValueError(
                f"leva {gross:.4f} oltre max_leverage={self.config.max_leverage}. "
                "Aumentare BacktestConfig.max_leverage per consentire la leva."
            )

    def _charge_costs(
        self,
        old_weights: Optional[pd.Series],
        new_weights: pd.Series,
        date,
        history: pd.DataFrame,
        adv: Optional[pd.DataFrame],
        spread: Optional[pd.DataFrame],
        nav: float,
        cost_log: dict,
    ) -> float:
        """Compute transaction-cost drag for one rebalance and update *cost_log* in place.

        Parameters
        ----------
        old_weights : pd.Series or None
            Weights before the rebalance (None on first-ever rebalance from cash).
        new_weights : pd.Series
            Weights after the rebalance.
        date : Timestamp
            Execution date (used as key in *cost_log* and for adv/spread slicing).
        history : pd.DataFrame
            Returns history strictly before *date* (or up to *date* for legacy
            path — caller is responsible for the correct slice).
        adv, spread : pd.DataFrame or None
            Market microstructure inputs for the Almgren-Chriss cost model.
        nav : float
            Current NAV, used as AUM in AC impact estimation.
        cost_log : dict
            Mutable dict; an entry is added when the AC model is used.

        Returns
        -------
        float
            tc_drag to subtract from the day's portfolio return.
        """
        tc_bps = self.config.transaction_cost_bps / 10_000

        if old_weights is None:
            return 0.0

        to = float(
            (new_weights - old_weights.reindex(new_weights.index, fill_value=0.0))
            .abs()
            .sum()
        )

        if (
            self.cost_model is not None
            and adv is not None
            and spread is not None
        ):
            vol_window = self.cost_model.config.vol_window
            vol = history.tail(vol_window).std()
            adv_slice = adv.reindex(columns=history.columns).ffill()
            adv_slice = adv_slice.loc[adv_slice.index <= date]
            spread_slice = spread.reindex(columns=history.columns).ffill()
            spread_slice = spread_slice.loc[spread_slice.index <= date]

            if adv_slice.empty or spread_slice.empty:
                return tc_bps * to

            adv_today = adv_slice.iloc[-1]
            spread_today = spread_slice.iloc[-1]
            impact = self.cost_model.estimate(
                old_weights, new_weights,
                adv_today, spread_today, vol,
                aum=nav,
            )
            cost_log[date] = impact.total_cost_pct
            return impact.total_cost_pct

        return tc_bps * to

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        returns: pd.DataFrame,
        min_history: Optional[int] = None,
        adv: Optional[pd.DataFrame] = None,
        spread: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        """Run walk-forward backtest.

        Parameters
        ----------
        returns : pd.DataFrame
            Full return history (date-indexed, asset columns).
        min_history : int, optional
            Minimum observations required before the first rebalance.
            If None, uses config.min_history (default 60).
        adv : pd.DataFrame, optional
            Average daily volume per asset (for Almgren-Chriss cost model).
        spread : pd.DataFrame, optional
            Bid-ask spread per asset (for Almgren-Chriss cost model).

        Returns
        -------
        BacktestResult
        """
        if min_history is None:
            min_history = self.config.min_history

        lag = self.config.execution_lag

        # Build rebalance dates from actual OOS index dates to avoid
        # calendar/business-day mismatches (e.g. "W" produces Sundays).
        # We group the OOS index by the requested frequency and take the last
        # actual trading date in each period. Convert to Timestamp for
        # consistent comparison with the DatetimeIndex.
        oos_dummy = pd.Series(0, index=returns.index[min_history:])
        rebal_dates = set(
            pd.DatetimeIndex(
                oos_dummy.resample(self.config.rebalance_freq)
                .apply(lambda x: x.index[-1] if len(x) > 0 else pd.NaT)
                .dropna()
                .values
            )
        )

        weights_log: dict = {}
        turnover_log: dict = {}
        cost_log: dict = {}

        # We build a return for each OOS date exactly
        oos_index = returns.index[min_history:]
        daily_rets: list[float] = []

        current_weights: Optional[pd.Series] = None
        nav = self.config.initial_capital

        # Execution lag staging: holds (new_weights, countdown) while waiting
        # for the execution date to arrive.
        pending_weights: Optional[pd.Series] = None
        pending_countdown: int = 0

        for date in oos_index:
            # ── Step 1: Execute a staged rebalance if its countdown reached zero ──
            tc_drag = 0.0
            if pending_weights is not None:
                pending_countdown -= 1
                if pending_countdown <= 0:
                    # Turnover and costs are charged at the EXECUTION date.
                    # AC model inputs (adv/spread/vol) are sampled here using
                    # data available on the execution date (adv/spread up to and
                    # including this date; returns vol uses history up to date).
                    exec_history = returns.loc[returns.index < date]
                    if current_weights is not None:
                        to = float(
                            (pending_weights - current_weights.reindex(
                                pending_weights.index, fill_value=0.0
                            )).abs().sum()
                        )
                        turnover_log[date] = to

                    tc_drag = self._charge_costs(
                        current_weights, pending_weights, date,
                        exec_history, adv, spread, nav, cost_log,
                    )

                    current_weights = pending_weights
                    pending_weights = None
                    # Log effective weights at the execution date
                    weights_log[date] = current_weights.copy()

            # ── Step 2: Signal — compute new weights at rebalance dates ──────────
            if date in rebal_dates:
                # Strictly before the rebalance date — no lookahead
                history = returns.loc[returns.index < date]
                new_weights = self.optimizer.optimize(history)

                if self.risk_filter is not None:
                    # Risk filter applied at signal time on data strictly before t
                    new_weights = self.risk_filter.apply(new_weights, history)

                # Validate leverage at signal time so errors surface immediately.
                self._validate_leverage(new_weights)

                if lag == 0:
                    # Legacy same-day execution: apply immediately
                    if current_weights is not None:
                        to = float((new_weights - current_weights).abs().sum())
                        turnover_log[date] = to

                    tc_drag = self._charge_costs(
                        current_weights, new_weights, date,
                        history, adv, spread, nav, cost_log,
                    )

                    current_weights = new_weights
                    weights_log[date] = current_weights.copy()
                else:
                    # Stage the new weights: they become effective lag trading days later.
                    # If a previous signal is still pending, replace ONLY the weights —
                    # the countdown (= execution date) stays fixed to avoid starvation
                    # when the rebalance interval is shorter than execution_lag.
                    if pending_weights is not None:
                        warnings.warn(
                            f"Nuovo segnale a {date.date()} sostituisce un rebalance pendente: "
                            "i pesi più recenti verranno eseguiti alla data di esecuzione già fissata.",
                            UserWarning,
                            stacklevel=2,
                        )
                        pending_weights = new_weights          # countdown NON resettato
                    else:
                        pending_weights = new_weights
                        pending_countdown = lag

            # ── Step 3: Compute daily return using currently effective weights ───
            if current_weights is not None:
                daily_ret = float((returns.loc[date] * current_weights).sum())
                # Funding drag on borrowed fraction — applied BEFORE tc_drag and
                # EVERY day when weights are levered (not only at rebalance).
                # Skipped when max_leverage == 1.0 (legacy path) to preserve
                # bit-identical results for unlevered portfolios.
                borrowed = max(0.0, float(current_weights.sum()) - 1.0)
                if borrowed > 0.0:
                    daily_ret -= borrowed * self._funding_rate_at(date) / 252.0
                daily_ret -= tc_drag
            else:
                daily_ret = 0.0

            daily_rets.append(daily_ret)
            nav *= (1 + daily_ret)

        portfolio_returns = pd.Series(daily_rets, index=oos_index)

        weights_history = pd.DataFrame(weights_log).T
        if not weights_history.empty:
            weights_history = weights_history.reindex(columns=returns.columns, fill_value=0.0)

        turnover = pd.Series(turnover_log)

        return BacktestResult(
            portfolio_returns=portfolio_returns,
            weights_history=weights_history,
            turnover=turnover,
            cost_history=pd.Series(cost_log),
        )
