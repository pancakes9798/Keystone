"""ReportBuilder — generates a self-contained HTML portfolio report."""
from __future__ import annotations

import base64
import io
import logging
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before pyplot import
import matplotlib.pyplot as plt
import pandas as pd
from jinja2 import Environment, BaseLoader

from jeanclaude.data.storage.parquet_store import ParquetStore
from jeanclaude.backtest.metrics import (
    sharpe_ratio, sortino_ratio, calmar_ratio, max_drawdown
)

logger = logging.getLogger(__name__)

_CAT = "paper_trading"
_STATE_KEY = "state"

_TEMPLATE = """<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="utf-8">
<title>JeanClaude Report {{ date }}</title>
<style>
  body{font-family:Arial,sans-serif;max-width:1000px;margin:auto;padding:20px;background:#f5f5f5}
  .card{background:#fff;border-radius:8px;padding:20px;margin:16px 0;box-shadow:0 2px 6px rgba(0,0,0,.1)}
  h1{color:#1a1a2e;margin:0 0 8px}h2{color:#16213e;margin:0 0 12px}
  .metric{display:inline-block;margin:8px 24px 8px 0}
  .val{font-size:26px;font-weight:700;color:#0f3460}
  .lbl{font-size:11px;color:#888;text-transform:uppercase}
  table{border-collapse:collapse;width:100%}
  th,td{border:1px solid #e0e0e0;padding:8px 12px;text-align:left}
  th{background:#f7f7f7;font-weight:600}
  .regime{display:inline-block;padding:3px 12px;border-radius:12px;font-weight:700;font-size:13px}
  .expansion{background:#d4edda;color:#155724}
  .contraction{background:#f8d7da;color:#721c24}
  .transition{background:#fff3cd;color:#856404}
  .unknown{background:#e2e3e5;color:#383d41}
  img{width:100%;border-radius:4px}
  .pos{color:#28a745}.neg{color:#dc3545}
</style>
</head>
<body>
<div class="card">
  <h1>JeanClaude Portfolio Report</h1>
  <p style="color:#666;margin:4px 0 16px">{{ date }}
    {% if regime %}&nbsp;|&nbsp;Regime: <span class="regime {{ regime_class }}">{{ regime }}</span>{% endif %}
  </p>
  <div class="metric"><div class="val">{{ nav }}</div><div class="lbl">NAV (€)</div></div>
  <div class="metric"><div class="val {{ pnl_day_cls }}">{{ pnl_day }}</div><div class="lbl">P&L Giornaliero</div></div>
  <div class="metric"><div class="val {{ pnl_week_cls }}">{{ pnl_week }}</div><div class="lbl">P&L Settimanale</div></div>
  <div class="metric"><div class="val {{ pnl_inception_cls }}">{{ pnl_inception }}</div><div class="lbl">P&L da Inception</div></div>
</div>
{% if equity_chart %}
<div class="card"><h2>Equity Curve</h2><img src="data:image/png;base64,{{ equity_chart }}" alt="Equity curve"></div>
{% endif %}
{% if allocation_chart %}
<div class="card"><h2>Allocazione Corrente</h2><img src="data:image/png;base64,{{ allocation_chart }}" alt="Allocation"></div>
{% endif %}
<div class="card">
  <h2>Metriche di Rischio</h2>
  <table>
    <tr><th>Metrica</th><th>Valore</th></tr>
    <tr><td>Sharpe Ratio</td><td>{{ sharpe }}</td></tr>
    <tr><td>Sortino Ratio</td><td>{{ sortino }}</td></tr>
    <tr><td>Calmar Ratio</td><td>{{ calmar }}</td></tr>
    <tr><td>Max Drawdown</td><td>{{ max_dd }}</td></tr>
    <tr><td>VaR 95% (giornaliero)</td><td>{{ var_95 }}</td></tr>
  </table>
</div>
<div class="card">
  <h2>Ultimi Ordini</h2>
  {% if orders %}
  <table>
    <tr><th>Asset</th><th>Peso Prec.</th><th>Peso Nuovo</th><th>Costo (bps)</th></tr>
    {% for o in orders %}<tr>
      <td>{{ o.asset }}</td>
      <td>{{ "%.1f%%" | format(o.old_weight * 100) }}</td>
      <td>{{ "%.1f%%" | format(o.new_weight * 100) }}</td>
      <td>{{ o.simulated_cost_bps }}</td>
    </tr>{% endfor %}
  </table>
  {% else %}<p style="color:#888">Nessun ordine registrato.</p>{% endif %}
</div>
</body>
</html>"""


class ReportBuilder:
    """Builds a self-contained HTML portfolio report from Parquet state.

    Parameters
    ----------
    store : ParquetStore
        Must contain paper_trading data written by PaperBroker.
    """

    def __init__(self, store: ParquetStore) -> None:
        self._store = store
        self._env = Environment(loader=BaseLoader())

    def build(
        self,
        as_of: Optional[pd.Timestamp] = None,
        regime: Optional[str] = None,
    ) -> str:
        """Generate a self-contained HTML report.

        Parameters
        ----------
        as_of : pd.Timestamp, optional
            Report date (default: today).
        regime : str, optional
            Current regime label (e.g. "EXPANSION"). If None, omitted.

        Returns
        -------
        str
            Full HTML string with base64-encoded chart images.
        """
        today = (as_of or pd.Timestamp.now()).normalize()

        nav_history = self._store.load(_CAT, "nav_history", _STATE_KEY, _STATE_KEY)
        positions = self._store.load(_CAT, "positions", _STATE_KEY, _STATE_KEY)
        orders_df = self._store.load(_CAT, "orders", _STATE_KEY, _STATE_KEY)

        # --- NAV and P&L ---
        if nav_history is not None and not nav_history.empty:
            nav_series = nav_history["nav"]
            current_nav = float(nav_series.iloc[-1])
            inception_nav = float(nav_series.iloc[0])
            pnl_inception = current_nav - inception_nav

            daily_returns = nav_series.pct_change().dropna()
            pnl_day = float(nav_series.diff().iloc[-1]) if len(nav_series) >= 2 else 0.0
            week_nav = nav_series[nav_series.index >= today - pd.Timedelta(days=7)]
            pnl_week = float(week_nav.iloc[-1] - week_nav.iloc[0]) if len(week_nav) >= 2 else 0.0

            sharpe = round(sharpe_ratio(daily_returns), 2)
            sortino = round(sortino_ratio(daily_returns), 2)
            calmar = round(calmar_ratio(daily_returns), 2)
            mdd = round(max_drawdown(daily_returns) * 100, 2)
            var_95 = round(float(daily_returns.quantile(0.05)) * 100, 2)
        else:
            current_nav = 0.0
            pnl_day = pnl_week = pnl_inception = 0.0
            sharpe = sortino = calmar = 0.0
            mdd = var_95 = 0.0

        # --- Charts ---
        equity_chart = (
            self._build_equity_chart(nav_history["nav"])
            if nav_history is not None and not nav_history.empty
            else None
        )
        allocation_chart = (
            self._build_allocation_chart(positions)
            if positions is not None and not positions.empty
            else None
        )

        # --- Orders ---
        orders = []
        if orders_df is not None and not orders_df.empty:
            latest_date = orders_df.index.max()
            latest_orders = orders_df[orders_df.index == latest_date]
            orders = latest_orders.reset_index().to_dict("records")

        # --- Regime CSS class ---
        regime_class = (regime or "unknown").lower()

        def _fmt_pnl(v: float) -> str:
            return f"{'+'if v>=0 else ''}{v:,.0f} €"

        ctx = {
            "date": today.strftime("%d %B %Y"),
            "nav": f"{current_nav:,.0f} €",
            "pnl_day": _fmt_pnl(pnl_day),
            "pnl_day_cls": "pos" if pnl_day >= 0 else "neg",
            "pnl_week": _fmt_pnl(pnl_week),
            "pnl_week_cls": "pos" if pnl_week >= 0 else "neg",
            "pnl_inception": _fmt_pnl(pnl_inception),
            "pnl_inception_cls": "pos" if pnl_inception >= 0 else "neg",
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "max_dd": f"{mdd}%",
            "var_95": f"{var_95}%",
            "equity_chart": equity_chart,
            "allocation_chart": allocation_chart,
            "orders": orders,
            "regime": regime,
            "regime_class": regime_class,
        }

        tmpl = self._env.from_string(_TEMPLATE)
        return tmpl.render(**ctx)

    @staticmethod
    def _build_equity_chart(nav_series: pd.Series) -> str:
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot(nav_series.index, nav_series.values, color="#0f3460", linewidth=1.5)
        ax.fill_between(nav_series.index, nav_series.values, nav_series.values[0],
                        alpha=0.08, color="#0f3460")
        ax.set_title("Equity Curve", fontsize=12, pad=8)
        ax.set_ylabel("NAV (€)")
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()

    @staticmethod
    def _build_allocation_chart(positions: pd.DataFrame) -> str:
        weight_cols = [c for c in positions.columns if c not in ("nav", "cash")]
        latest = positions[weight_cols].iloc[-1].astype(float)
        latest = latest[latest > 0].sort_values(ascending=True)
        if latest.empty:
            return ""
        fig, ax = plt.subplots(figsize=(7, max(2, len(latest) * 0.5)))
        colors = plt.cm.Blues_r([i / max(len(latest), 1) for i in range(len(latest))])
        bars = ax.barh(latest.index, latest.values * 100, color=colors)
        ax.set_xlabel("Peso (%)")
        ax.set_title("Allocazione Corrente", fontsize=12, pad=8)
        for bar, val in zip(bars, latest.values):
            ax.text(val * 100 + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{val*100:.1f}%", va="center", fontsize=9)
        ax.set_xlim(0, 100)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()
