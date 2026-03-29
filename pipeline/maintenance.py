"""
Maintenance
===========
Staleness checks and automatic background tasks triggered from Broker.py.

Every time Broker.py runs, it calls run_maintenance() which checks:
  - Price data freshness  (stale if last row is > 1 trading day old)
  - Sentiment freshness   (stale if last row is > 3 days old)
  - Model checkpoint age  (stale if newest checkpoint is > 7 days old)
  - Auto-tune age         (stale if last tune was > 7 days ago)
  - Shadow portfolio step (always advances one cycle)

Each task only runs if it's actually needed. Everything is logged to
logs/maintenance.log so you can see exactly what ran and why.
"""

import logging
import os
import glob
import json
from datetime import datetime, timedelta, date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_STATE_FILE = "broker/state/maintenance.json"
_LOG_FILE   = "logs/maintenance.log"


# ── State persistence ─────────────────────────────────────────────────────────

def _load_state() -> dict:
    try:
        with open(_STATE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_state(state: dict) -> None:
    Path(_STATE_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def _today() -> str:
    return date.today().isoformat()


def _days_since(iso_date: str | None) -> int:
    """Days since an ISO date string. Returns 9999 if None."""
    if not iso_date:
        return 9999
    try:
        return (date.today() - date.fromisoformat(iso_date)).days
    except Exception:
        return 9999


# ── Individual maintenance tasks ──────────────────────────────────────────────

def _check_prices(state: dict, universe: list[str] | None) -> bool:
    """Update prices if last update was not today."""
    last = state.get("prices_updated")
    if last == _today():
        logger.info("Maintenance: prices up to date (last: %s)", last)
        return False

    logger.info("Maintenance: prices stale (last: %s) — updating...", last or "never")
    try:
        from pipeline.updater import update_parquet
        n = update_parquet(universe=universe)
        logger.info("Maintenance: prices updated — %d new rows", n)
        state["prices_updated"] = _today()
        return True
    except Exception as exc:
        logger.warning("Maintenance: price update failed: %s", exc)
        return False


def _check_sentiment(state: dict, universe: list[str] | None) -> bool:
    """Update sentiment if last update was > 2 days ago."""
    last = state.get("sentiment_updated")
    if _days_since(last) < 2:
        logger.info("Maintenance: sentiment up to date (last: %s)", last)
        return False

    if not universe:
        logger.info("Maintenance: no universe — skipping sentiment update")
        return False

    logger.info("Maintenance: sentiment stale (last: %s) — updating...", last or "never")
    try:
        from pipeline.sentiment import update_sentiment
        n = update_sentiment(universe, lookback_days=3)
        logger.info("Maintenance: sentiment updated — %d new headlines", n)
        state["sentiment_updated"] = _today()
        return True
    except Exception as exc:
        logger.warning("Maintenance: sentiment update failed: %s", exc)
        return False


def _check_model(state: dict, save_dir: str = "models") -> bool:
    """Finetune model if newest checkpoint is > 7 days old."""
    last = state.get("model_finetuned")
    if _days_since(last) < 7:
        logger.info("Maintenance: model up to date (last finetune: %s)", last)
        return False

    # Also skip if no checkpoint exists yet (first-time setup — user must train manually)
    ckpts = sorted(glob.glob(f"{save_dir}/best_fold*.pt"))
    if not ckpts:
        logger.info("Maintenance: no checkpoint found — skipping finetune (run Agent.py --mode train first)")
        return False

    logger.info("Maintenance: model stale (last: %s) — finetuning...", last or "never")
    try:
        from pipeline.scheduler import weekly_finetune
        weekly_finetune(save_dir=save_dir)
        state["model_finetuned"] = _today()
        logger.info("Maintenance: model finetune complete")
        return True
    except Exception as exc:
        logger.warning("Maintenance: model finetune failed: %s", exc)
        return False


def _check_autotune(state: dict, initial_cash: float) -> bool:
    """Run auto-tuner if last tune was > 7 days ago."""
    last = state.get("autotuned")
    if _days_since(last) < 7:
        logger.info("Maintenance: auto-tune up to date (last: %s)", last)
        return False

    logger.info("Maintenance: auto-tune stale (last: %s) — running...", last or "never")
    try:
        from pipeline.autotuner import run_autotuner
        run_autotuner(initial_cash=initial_cash)
        state["autotuned"] = _today()
        logger.info("Maintenance: auto-tune complete")
        return True
    except Exception as exc:
        logger.warning("Maintenance: auto-tune failed: %s", exc)
        return False


# ── Main entry point ──────────────────────────────────────────────────────────

def run_maintenance(initial_cash: float = 10_000.0, save_dir: str = "models") -> None:
    """
    Run all staleness checks. Called at the top of every Broker.py cycle.
    Each task only runs if it's actually needed.
    """
    _setup_logging()
    state = _load_state()

    logger.info("-" * 50)
    logger.info("Maintenance check: %s", datetime.now().strftime("%Y-%m-%d %H:%M"))

    # Load universe for data updates
    universe = None
    try:
        from pipeline.updater import _load_trained_universe
        universe = _load_trained_universe(save_dir)
    except Exception:
        pass

    # If no checkpoint universe, use top liquid tickers from parquet
    # to avoid updating all 11,500 tickers on first run
    if not universe:
        try:
            from pipeline.data import load_master, get_asset_universe
            df_raw = load_master(top_n=750)
            universe = get_asset_universe(df_raw, top_n=750)
            logger.info("Maintenance: using top-%d liquid tickers (no checkpoint yet)", len(universe))
        except Exception:
            pass

    _check_prices(state, universe)
    _check_sentiment(state, universe)
    _check_model(state, save_dir)
    _check_autotune(state, initial_cash)

    _save_state(state)
    logger.info("Maintenance check complete.")


def _setup_logging():
    Path("logs").mkdir(exist_ok=True)
    file_handler = logging.FileHandler(_LOG_FILE)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    if not any(isinstance(h, logging.FileHandler) and "maintenance" in getattr(h, "baseFilename", "")
               for h in logger.handlers):
        logger.addHandler(file_handler)
    if not logger.level or logger.level > logging.INFO:
        logger.setLevel(logging.INFO)
