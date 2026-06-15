"""ReportAgent — legge il NAV e invia email il venerdì (scrittura NAV delegata a ExecutionAgent)."""
from __future__ import annotations

import logging
import os
from typing import Any

import pandas as pd

from .base import BaseAgent
from .config import AgentConfig
from .events import NoRebalanceEvent, PriceEvent, WeightsEvent

logger = logging.getLogger(__name__)


class ReportAgent(BaseAgent):
    """Legge il NAV e invia email il venerdì (o a ogni ribilanciamento).

    La scrittura del NAV è responsabilità di ExecutionAgent (che chiama
    record_daily_nav prima nel ciclo). Questo agente legge solo la proprietà
    ``nav`` del broker — non chiama più ``record_daily_nav``.

    Parameters
    ----------
    config : AgentConfig
        Configurazione agente — usa ``report_recipients`` per abilitare email.
    broker : object
        Broker paper trading — deve esporre la proprietà ``nav -> float``.
    """

    def __init__(self, config: AgentConfig, broker: object) -> None:
        super().__init__()
        self._config = config
        self._broker = broker
        self._mailer: Any | None = None

        if config.report_recipients:
            smtp_user = os.environ.get("SMTP_USER", "")
            smtp_password = os.environ.get("SMTP_PASSWORD", "")
            if smtp_user and smtp_password:
                try:
                    from jeanclaude.reporting.mailer import Mailer
                    self._mailer = Mailer(smtp_user=smtp_user, smtp_password=smtp_password)
                    logger.info("ReportAgent: Mailer inizializzato (user=%s)", smtp_user)
                except ImportError:
                    logger.warning("ReportAgent: Mailer non disponibile — email disabilitata")
            else:
                logger.warning(
                    "ReportAgent: SMTP_USER/SMTP_PASSWORD non configurati — email disabilitata"
                )

    async def run(
        self,
        weights_result: WeightsEvent | NoRebalanceEvent,
        price_event: PriceEvent,
    ) -> None:
        """Aggiorna NAV e, se venerdì o ribilanciamento, invia email.

        Parameters
        ----------
        weights_result : WeightsEvent | NoRebalanceEvent
            Risultato del PortfolioAgent per oggi.
        price_event : PriceEvent
            Prezzi e rendimenti di oggi.
        """
        date = price_event.date
        is_friday = date.dayofweek == 4
        rebalanced = isinstance(weights_result, WeightsEvent)

        # Aggiorna NAV giornaliero
        nav = self._update_nav(price_event)

        logger.info(
            "ReportAgent: NAV=%.2f | date=%s | friday=%s | rebalanced=%s",
            nav, date.date(), is_friday, rebalanced,
        )

        # Invia email se venerdì o ribilanciamento e destinatari configurati
        if (is_friday or rebalanced) and self._config.report_recipients and self._mailer:
            self._send_report(weights_result, price_event, nav)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_nav(self, price_event: PriceEvent) -> float:
        """Legge il NAV corrente dal broker (la scrittura è dell'ExecutionAgent)."""
        try:
            return float(self._broker.nav)  # type: ignore[union-attr]
        except Exception as exc:
            logger.warning("ReportAgent: lettura NAV fallita (%s)", exc)
            return 0.0

    def _send_report(
        self,
        weights_result: WeightsEvent | NoRebalanceEvent,
        price_event: PriceEvent,
        nav: float,
    ) -> None:
        """Compone e invia l'email del report."""
        subject = f"JeanClaude Report — {price_event.date.date()}"
        html = self._build_html(weights_result, price_event, nav)
        try:
            self._mailer.send(  # type: ignore[union-attr]
                html=html,
                subject=subject,
                recipients=self._config.report_recipients,
            )
            logger.info("ReportAgent: email inviata a %s", self._config.report_recipients)
        except Exception as exc:
            logger.error("ReportAgent: invio email fallito (%s)", exc)

    def _build_html(
        self,
        weights_result: WeightsEvent | NoRebalanceEvent,
        price_event: PriceEvent,
        nav: float,
    ) -> str:
        """Restituisce un report HTML minimale."""
        date_str = price_event.date.date()

        if isinstance(weights_result, WeightsEvent):
            rebalance_line = (
                f"<p><b>Rebalanced:</b> YES (reason={weights_result.reason})</p>"
            )
            weights_rows = "".join(
                f"<tr><td>{ticker}</td><td>{w:.1%}</td></tr>"
                for ticker, w in weights_result.weights.items()
            )
            weights_table = (
                "<table border='1' cellpadding='4'>"
                "<tr><th>Asset</th><th>Weight</th></tr>"
                f"{weights_rows}"
                "</table>"
            )
        else:
            rebalance_line = (
                f"<p><b>Rebalanced:</b> NO (reason={weights_result.reason})</p>"
            )
            weights_table = ""

        return (
            f"<html><body>"
            f"<h2>JeanClaude Daily Report — {date_str}</h2>"
            f"<p><b>NAV:</b> {nav:,.2f} €</p>"
            f"{rebalance_line}"
            f"{weights_table}"
            f"</body></html>"
        )
