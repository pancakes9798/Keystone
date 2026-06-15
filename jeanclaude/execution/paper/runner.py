"""DailyRunner — orchestrates the daily paper trading cycle."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

from .broker import Order, PaperBroker
from .schedule import rebalance_due

if TYPE_CHECKING:
    from jeanclaude.reporting.builder import ReportBuilder
    from jeanclaude.reporting.mailer import Mailer
    from jeanclaude.signals.composite.builder import CompositeSignalBuilder
    from jeanclaude.portfolio.optimizer.bl import BLOptimizer, TickerBLOptimizer
    from jeanclaude.signals.news.fetcher import RefinitivNewsSource
    from jeanclaude.signals.news.scorer import FinBERTScorer
    from jeanclaude.signals.news.aggregator import NewsAggregator
    from jeanclaude.signals.news.ticker_aggregator import TickerNewsAggregator
    from jeanclaude.data.storage.parquet_store import ParquetStore
    from jeanclaude.signals.news.types import NewsSentiment as _NewsSentiment
    from jeanclaude.signals.news.ticker_types import TickerSentiment as _TickerSentiment
    from jeanclaude.costs.model import AlmgrenChrissModel
    from jeanclaude.costs.data import VolumeSpreadLoader

logger = logging.getLogger(__name__)


@dataclass
class DailyRunConfig:
    """Configuration for DailyRunner."""
    assets: list[str]
    macro_series: list[str]
    rebalance_freq: str = "W"
    send_report: bool = True
    report_recipients: list[str] = field(default_factory=list)
    transaction_cost_bps: float = 10.0
    min_history: int = 60
    history_start: str = "2020-01-01"


@dataclass
class DailyRunResult:
    """Output of a single daily run."""
    date: pd.Timestamp
    nav: float
    rebalanced: bool
    orders: list[Order]
    regime: str | None


class DailyRunner:
    """Orchestrates the daily paper trading cycle.

    On each call to ``run()``:
    1. Fetch prices and compute returns via DataLoader.  Logical date ``asof``
       = last market date ≤ requested date (Saturday run → Friday record).
    2. Record P&L on OLD weights from last_nav_date to asof (gap-aware
       multi-day return) via PaperBroker.record_daily_nav.
    3. If a completed period exists since last rebalance → optimize → risk
       filter → execute_rebalance(date=asof); apply the real cost (AC model
       when available, flat bps fallback otherwise) via apply_rebalance_cost.
    4. Optionally build and email the HTML report.

    Parameters
    ----------
    broker : PaperBroker
    optimizer : BLOptimizer | any optimizer with .optimize(returns)
        When ``signal_builder`` is set this must be a ``BLOptimizer``;
        otherwise any object with ``optimize(returns) -> pd.Series`` works.
    risk_filter :
        Object with ``apply(weights, returns) -> pd.Series``.
    regime_detector :
        Object with ``fit(macro) -> RegimeResult`` and
        ``current_regime(macro) -> (RegimeLabel, np.ndarray)``.
    data_loader :
    report_builder : ReportBuilder
    mailer : Mailer | None
    config : DailyRunConfig
    signal_builder : CompositeSignalBuilder | None
        When provided, activates the full BL wiring path.
    prior_optimizer : any optimizer with .optimize(returns) | None
        Used as the prior for BL (e.g. HRPOptimizer).
        When None and BL path is active, equal-weight is used as prior.
    """

    def __init__(
        self,
        broker: PaperBroker,
        optimizer: Any,
        risk_filter: Any,
        regime_detector: Any,
        data_loader: Any,
        report_builder: "ReportBuilder",
        mailer: "Mailer | None",
        config: DailyRunConfig,
        signal_builder: "CompositeSignalBuilder | None" = None,
        prior_optimizer: Any = None,
        news_fetcher: "RefinitivNewsSource | None" = None,
        news_scorer: "FinBERTScorer | None" = None,
        news_aggregator: "NewsAggregator | None" = None,
        asset_class_queries: "dict[str, list[str]] | None" = None,
        store: "ParquetStore | None" = None,
        cost_model: "AlmgrenChrissModel | None" = None,
        volume_spread_loader: "VolumeSpreadLoader | None" = None,
        ticker_queries: "dict[str, list[str]] | None" = None,
        ticker_aggregator: "TickerNewsAggregator | None" = None,
        ticker_bl_optimizer: "TickerBLOptimizer | None" = None,
    ) -> None:
        self._broker = broker
        self._optimizer = optimizer
        self._risk_filter = risk_filter
        self._regime_detector = regime_detector
        self._data_loader = data_loader
        self._report_builder = report_builder
        self._mailer = mailer
        self._config = config
        self._signal_builder = signal_builder
        self._prior_optimizer = prior_optimizer
        self._news_fetcher = news_fetcher
        self._news_scorer = news_scorer
        self._news_aggregator = news_aggregator
        self._asset_class_queries = asset_class_queries
        self._store = store
        self._cost_model = cost_model
        self._volume_spread_loader = volume_spread_loader
        self._ticker_queries = ticker_queries
        self._ticker_aggregator = ticker_aggregator
        self._ticker_bl_optimizer = ticker_bl_optimizer

        if news_fetcher is not None and store is None:
            raise ValueError(
                "store must be provided when news_fetcher is set — "
                "headlines need a ParquetStore to persist daily."
            )
        if asset_class_queries is not None and (
            news_fetcher is None or news_scorer is None or news_aggregator is None
        ):
            raise ValueError(
                "asset_class_queries richiede news_fetcher, news_scorer e news_aggregator."
            )
        if ticker_queries is not None and (news_fetcher is None or news_scorer is None):
            raise ValueError(
                "ticker_queries richiede news_fetcher e news_scorer."
            )
        if ticker_queries is not None and store is None:
            raise ValueError(
                "store must be provided when ticker_queries is set."
            )

    def run(self, date: pd.Timestamp | None = None) -> DailyRunResult:
        """Execute the daily cycle.

        La data logica del ciclo è ``asof`` = ultimo giorno di mercato ≤ date:
        un run di sabato sera o in ritardo viene registrato sotto la data di
        mercato corretta. Ordine del ciclo: (1) P&L sui pesi vecchi fino ad
        asof, (2) eventuale rebalance, (3) costo applicato alla riga di asof.
        Il ciclo è idempotente: due run sullo stesso asof producono lo stesso stato.
        """
        today = (date or pd.Timestamp.now()).normalize()
        end_str = today.strftime("%Y-%m-%d")
        yesterday_str = (today - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        prices = self._data_loader.get_prices(
            tickers=self._config.assets,
            start=self._config.history_start,
            end=end_str,
        )
        returns = self._data_loader.get_returns(prices)

        available = prices.index[prices.index <= today]
        if len(available) == 0:
            raise ValueError(f"Nessun dato prezzi disponibile fino a {end_str}.")
        asof = pd.Timestamp(available[-1])

        # --- Daily news fetch + score (asset-class and/or ticker) ---
        self._fetch_and_store_news(asof, yesterday_str, end_str)
        self._fetch_and_store_ticker_news(asof, yesterday_str, end_str)

        # --- 1. P&L del periodo dalla base NAV (gap-aware), pesi vecchi ---
        # Ancora la finestra a nav_date_before(asof), NON a last_nav_date(): un
        # re-run sullo stesso asof resta idempotente (vedi PaperBroker.period_returns).
        period_returns = self._broker.period_returns(prices, asof, self._config.assets)

        nav = self._broker.record_daily_nav(period_returns, date=asof)

        # --- 2. Rebalance se un periodo è stato completato dall'ultimo rebalance ---
        orders: list[Order] = []
        regime: str | None = None
        rebalanced = False

        last_rebal = self._broker.last_rebalance_date()
        # Un re-run dello stesso asof DEVE rieseguire il rebalance già registrato:
        # record_daily_nav ha appena resettato la riga NAV (upsert), quindi il costo
        # va riapplicato; execute_rebalance è upsert → stato identico, mai duplicato.
        due = rebalance_due(
            asof=asof,
            price_index=prices.index,
            freq=self._config.rebalance_freq,
            last_rebalance=last_rebal,
        ) or (last_rebal is not None and last_rebal == asof)

        if due and len(returns.loc[:asof]) >= self._config.min_history:
            returns_asof = returns.loc[:asof]
            macro = self._data_loader.get_macro(
                series=self._config.macro_series,
                start=self._config.history_start,
                end=asof.strftime("%Y-%m-%d"),
            )

            if self._ticker_bl_optimizer is not None:
                new_weights, regime = self._run_ticker_bl_rebalance(returns_asof, macro, asof)
            elif self._signal_builder is not None:
                new_weights, regime = self._run_bl_rebalance(returns_asof, macro, asof)
            else:
                new_weights, regime = self._run_simple_rebalance(returns_asof, macro)

            new_weights = self._risk_filter.apply(new_weights, returns_asof)
            prices_asof = prices.loc[asof]

            result = self._broker.execute_rebalance(
                new_weights, prices_asof, self._config.transaction_cost_bps, date=asof
            )
            orders = result.orders
            cost_eur = result.cost_eur

            # Stima Almgren-Chriss quando disponibile: SOSTITUISCE il costo flat.
            if self._cost_model is not None and self._volume_spread_loader is not None:
                try:
                    vol = returns_asof.tail(self._cost_model.config.vol_window).std()
                    adv = self._volume_spread_loader.get_adv(
                        self._config.assets,
                        self._config.history_start,
                        end_str,
                        window=self._cost_model.config.adv_window,
                    )
                    spread = self._volume_spread_loader.get_spread(
                        self._config.assets,
                        self._config.history_start,
                        end_str,
                    )
                    old_weights = self._broker.weights_before(asof)
                    impact = self._cost_model.estimate(
                        old_weights,
                        new_weights,
                        adv,
                        spread,
                        vol,
                        aum=nav,
                    )
                    cost_eur = impact.total_cost_eur
                    logger.info(
                        "AC cost: %.2f EUR (%.2f bps) | turnover %.1f%%",
                        impact.total_cost_eur,
                        impact.total_cost_pct * 10_000,
                        impact.turnover_pct * 100,
                    )
                except Exception as cost_exc:
                    logger.warning(
                        "AC cost model failed (%s) — uso costo flat %.2f EUR (%.1f bps).",
                        cost_exc, cost_eur, self._config.transaction_cost_bps,
                    )

            # --- 3. Applica il costo alla riga NAV di asof (una sola volta) ---
            if cost_eur > 0.0:
                nav = self._broker.apply_rebalance_cost(date=asof, cost_eur=cost_eur)

            rebalanced = True
            logger.info("Rebalanced on %s | regime=%s", asof.date(), regime)

        # Optional report
        if self._config.send_report and self._mailer is not None:
            html = self._report_builder.build(as_of=asof, regime=regime)
            subject = f"JeanClaude Daily Report — {asof.strftime('%Y-%m-%d')}"
            self._mailer.send(html, subject, self._config.report_recipients)

        return DailyRunResult(
            date=asof,
            nav=nav,
            rebalanced=rebalanced,
            orders=orders,
            regime=regime,
        )

    # ------------------------------------------------------------------
    # Rebalance helpers
    # ------------------------------------------------------------------

    def _run_simple_rebalance(
        self, returns: pd.DataFrame, macro: pd.DataFrame
    ) -> tuple[pd.Series, str | None]:
        """Backward-compatible path: generic optimizer, no BL."""
        regime_label, _ = self._regime_detector.current_regime(macro)
        regime = regime_label.name
        new_weights = self._optimizer.optimize(returns)
        return new_weights, regime

    def _fetch_and_store_news(
        self, today: pd.Timestamp, yesterday_str: str, end_str: str
    ) -> None:
        """Fetch today's asset-class headlines, score them, append to Parquet."""
        if self._news_fetcher is None or self._asset_class_queries is None:
            return
        try:
            headlines = self._news_fetcher.get_headlines(
                self._asset_class_queries or {},
                start=yesterday_str,
                end=end_str,
            )
            scored = self._news_scorer.score(headlines)  # type: ignore[union-attr]
            existing = self._store.load("paper_trading", "headlines", "all", "all")  # type: ignore[union-attr]
            merged = pd.concat([existing, scored]) if existing is not None else scored
            self._store.save("paper_trading", "headlines", "all", "all", merged)  # type: ignore[union-attr]
        except Exception as exc:
            logger.warning("News fetch/score failed: %s — continuing without news.", exc)

    def _fetch_and_store_ticker_news(
        self, today: pd.Timestamp, yesterday_str: str, end_str: str
    ) -> None:
        """Fetch today's per-ticker headlines, score with FinBERT, append to Parquet."""
        if self._ticker_queries is None or self._news_scorer is None or self._store is None:
            return
        try:
            headlines = self._news_fetcher.get_ticker_headlines(  # type: ignore[union-attr]
                self._ticker_queries,
                start=yesterday_str,
                end=end_str,
            )
            scored = self._news_scorer.score(headlines)  # type: ignore[union-attr]
            existing = self._store.load("paper_trading", "ticker_headlines", "all", "all")  # type: ignore[union-attr]
            merged = pd.concat([existing, scored]) if existing is not None else scored
            self._store.save("paper_trading", "ticker_headlines", "all", "all", merged)  # type: ignore[union-attr]
            logger.info("Ticker news stored: %d headlines for %d tickers",
                        len(scored), scored["ticker"].nunique() if not scored.empty else 0)
        except Exception as exc:
            logger.warning("Ticker news fetch/score failed: %s — skipping.", exc)

    def _get_ticker_sentiment(self, today: pd.Timestamp) -> "_TickerSentiment | None":
        """Load persisted ticker headlines and aggregate to TickerSentiment."""
        if self._ticker_aggregator is None or self._store is None:
            return None
        try:
            all_headlines = self._store.load("paper_trading", "ticker_headlines", "all", "all")
            if all_headlines is not None and not all_headlines.empty:
                return self._ticker_aggregator.aggregate(all_headlines, as_of=today)
        except Exception as exc:
            logger.warning("Ticker sentiment aggregation failed: %s", exc)
        return None

    def _run_ticker_bl_rebalance(
        self, returns: pd.DataFrame, macro: pd.DataFrame, today: pd.Timestamp
    ) -> tuple[pd.Series, str | None]:
        """Ticker-level BL path: FinBERT sentiment → TickerBLOptimizer.

        Uses TrendMomentum as prior. Regime is detected via HMM (for logging).
        Falls back to prior if no sentiment or optimization fails.
        """
        from jeanclaude.signals.macro.labels import RegimeLabel as _RL

        # Prior weights
        if self._prior_optimizer is not None:
            prior_weights = self._prior_optimizer.optimize(returns)
        else:
            n = len(returns.columns)
            prior_weights = pd.Series(1.0 / n, index=returns.columns)

        # Regime detection (for logging only)
        regime: str | None = None
        try:
            regime_result = self._regime_detector.fit(macro)
            regime = _RL(int(regime_result.labels.iloc[-1])).name
        except Exception as exc:
            logger.warning("Regime detection failed: %s — continuing without regime label.", exc)

        # Ticker sentiment
        ticker_sentiment = self._get_ticker_sentiment(today)
        if ticker_sentiment is None or not ticker_sentiment.has_signal():
            logger.info("No ticker sentiment available — using prior weights (regime=%s)", regime)
            return prior_weights, regime

        # TickerBL optimize
        try:
            new_weights = self._ticker_bl_optimizer.optimize(  # type: ignore[union-attr]
                ticker_sentiment, prior_weights, returns
            )
            n_views = int((ticker_sentiment.confidence >= self._ticker_bl_optimizer.min_confidence).sum())  # type: ignore[union-attr]
            logger.info(
                "TickerBL rebalanced | regime=%s | %d ticker views | "
                "top=%s %.1f%%",
                regime, n_views,
                new_weights.idxmax(), new_weights.max() * 100,
            )
        except Exception as exc:
            logger.warning("TickerBL optimize failed: %s — using prior.", exc)
            new_weights = prior_weights

        return new_weights, regime

    def _get_news_sentiment(self, today: pd.Timestamp) -> "_NewsSentiment | None":
        """Load persisted headlines and aggregate to NewsSentiment. Returns None on failure."""
        if self._news_aggregator is None or self._store is None:
            return None
        try:
            all_headlines = self._store.load("paper_trading", "headlines", "all", "all")
            if all_headlines is not None and not all_headlines.empty:
                return self._news_aggregator.aggregate(all_headlines, as_of=today)
        except Exception as exc:
            logger.warning("News aggregation failed: %s — continuing without news.", exc)
        return None

    def _run_bl_rebalance(
        self, returns: pd.DataFrame, macro: pd.DataFrame, today: pd.Timestamp
    ) -> tuple[pd.Series, str | None]:
        """Full BL path: regime fit → composite signal (+ news) → BL optimize.

        Fits the HMM on ``macro``, builds a CompositeSignal, aligns regime
        labels to the returns index, fits BLOptimizer, and returns BL weights.

        Falls back to prior_optimizer (or equal-weight) if BL fitting fails
        (e.g. too few observations per regime after alignment).
        """
        # 1. Fit HMM → full RegimeResult
        # labels are stored as int64; convert to RegimeLabel for .name access
        from jeanclaude.signals.macro.labels import RegimeLabel as _RL
        regime_result = self._regime_detector.fit(macro)
        regime = _RL(int(regime_result.labels.iloc[-1])).name

        # 2. News sentiment (None if news not configured)
        news_sentiment = self._get_news_sentiment(today)

        # 3. Build CompositeSignal from the last posterior (+ optional news)
        signal = self._signal_builder.build(  # type: ignore[union-attr]
            regime_result, news_sentiment=news_sentiment
        )

        # 4. Prior weights (HRP/NCO or equal-weight)
        if self._prior_optimizer is not None:
            prior_weights = self._prior_optimizer.optimize(returns)
        else:
            n = len(returns.columns)
            prior_weights = pd.Series(1.0 / n, index=returns.columns)

        # 5. Align regime labels to returns index (forward-fill macro gaps)
        regime_labels = self._align_labels(regime_result.labels, returns.index)

        if regime_labels.empty:
            logger.warning(
                "BL path: no aligned regime labels — falling back to prior weights."
            )
            return prior_weights, regime

        # 6. Fit BLOptimizer on aligned returns × labels, then optimize
        try:
            self._optimizer.fit(returns.loc[regime_labels.index], regime_labels)
            new_weights = self._optimizer.optimize(signal, prior_weights, returns)
        except (ValueError, RuntimeError) as exc:
            logger.warning(
                "BL optimize failed (%s) — falling back to prior weights.", exc
            )
            new_weights = prior_weights

        return new_weights, regime

    @staticmethod
    def _align_labels(
        labels: pd.Series, returns_index: pd.DatetimeIndex
    ) -> pd.Series:
        """Forward-fill regime labels onto the returns index.

        Macro data often has a coarser or offset calendar vs daily prices.
        We union the two indexes, ffill, then reindex to returns_index,
        dropping any leading NaNs (before the first macro observation).
        """
        combined_index = labels.index.union(returns_index)
        aligned = labels.reindex(combined_index).ffill()
        return aligned.reindex(returns_index).dropna()

