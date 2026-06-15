#!/usr/bin/env python3
"""Entry point multi-agent JeanClaude — sostituisce run_paper_etf.py."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jeanclaude.agents.config import AgentConfig, RebalanceConfig
from jeanclaude.agents.orchestrator import Orchestrator


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _load_config(config_path: Path | None) -> AgentConfig:
    if config_path is None:
        return AgentConfig()
    with config_path.open() as fh:
        raw = yaml.safe_load(fh) or {}
    rebalance_raw = raw.pop("rebalance", {})
    rebalance = RebalanceConfig(**rebalance_raw)
    return AgentConfig(**raw, rebalance=rebalance)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="JeanClaude multi-agent paper trading runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Data target YYYY-MM-DD (default: oggi)",
    )
    parser.add_argument(
        "--config-file",
        default=None,
        type=Path,
        help="YAML con override AgentConfig",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _setup_logging(args.log_level)

    logger = logging.getLogger(__name__)

    target_date: pd.Timestamp | None = None
    if args.date:
        try:
            target_date = pd.Timestamp(args.date)
        except ValueError:
            logger.error("Data non valida: %s — usa formato YYYY-MM-DD", args.date)
            sys.exit(1)

    config = _load_config(args.config_file)
    logger.info(
        "JeanClaude Agent avviato | date=%s | universe=%d ticker",
        target_date or "oggi",
        len(config.universe),
    )

    try:
        orch = Orchestrator(config)
        orch.run(target_date)
    except KeyboardInterrupt:
        logger.info("Interrotto dall'utente")
        sys.exit(0)
    except Exception as exc:
        logger.exception("Errore fatale: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
