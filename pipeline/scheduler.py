"""
Scheduler — runs continuously and:
  1. Every trading day after market close: fetches new data
  2. Every Sunday night:                  fine-tunes the best checkpoint on recent data
  3. Logs everything to logs/scheduler.log

Run with:
    python -m pipeline.scheduler
or:
    python Agent.py --mode schedule
"""

import os
import glob
import time
import logging
import schedule
from datetime import datetime, timedelta
from pathlib import Path

import torch
import numpy as np
import pandas as pd

from pipeline.updater import update_parquet

logger = logging.getLogger(__name__)


def _setup_logging():
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("logs/scheduler.log"),
            logging.StreamHandler(),
        ],
    )


def _is_weekday() -> bool:
    return datetime.today().weekday() < 5   # Mon–Fri


def daily_update(universe: list[str] | None = None):
    """Fetch today's OHLCV + news sentiment and append to parquets/CSV."""
    if not _is_weekday():
        logger.info("Weekend — skipping daily update.")
        return

    logger.info("=== Daily data update ===")
    try:
        n = update_parquet(universe=universe)
        logger.info(f"Price update complete. {n} new rows added.")
    except Exception as e:
        logger.error(f"Price update failed: {e}", exc_info=True)

    # Sentiment update
    try:
        from pipeline.sentiment import update_sentiment
        tickers = universe or []
        if tickers:
            n_sent = update_sentiment(tickers, lookback_days=3)
            logger.info(f"Sentiment update complete. {n_sent} new headlines scored.")
    except Exception as e:
        logger.error(f"Sentiment update failed: {e}", exc_info=True)


def weekly_finetune(
    top_n: int = 150,
    finetune_steps: int = 20_000,
    lookback_months: int = 24,
    save_dir: str = "models",
    seed: int = 42,
):
    """
    Fine-tune the best existing checkpoint on the most recent `lookback_months`
    of data. Saves a new versioned checkpoint if val Sharpe improves.
    """
    logger.info("=== Weekly fine-tune ===")

    try:
        from pipeline.data     import load_master, get_asset_universe, walk_forward_split
        from pipeline.train    import train_fold, PPO_CFG
        from pipeline.backtest import load_model

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        df = load_master()
        asset_list = get_asset_universe(df, top_n=top_n)
        df = df[df.index.get_level_values("ticker").isin(asset_list)]

        # Use only recent data for fine-tuning
        dates     = sorted(df.index.get_level_values("date").unique())
        cutoff    = pd.Timestamp(dates[-1]) - pd.DateOffset(months=lookback_months)
        df_recent = df[df.index.get_level_values("date") >= cutoff]

        folds = walk_forward_split(df_recent, train_years=1, val_years=3, test_years=3)
        if not folds:
            logger.warning("Not enough recent data for a fine-tune fold.")
            return

        fold = folds[-1]

        # Load best existing checkpoint as starting point
        ckpts = sorted(glob.glob(f"{save_dir}/best_fold*.pt"))
        pretrained_state = None
        if ckpts:
            ckpt = torch.load(ckpts[-1], map_location=device, weights_only=False)
            pretrained_state = ckpt["model_state"]
            model_cfg        = ckpt["model_cfg"]
            logger.info(f"Fine-tuning from: {ckpts[-1]}")
        else:
            model_cfg = None
            logger.info("No existing checkpoint — training from scratch.")

        cfg = {**PPO_CFG, "total_steps": finetune_steps}

        ckpt_path, val_sharpe = train_fold(
            df_train        = fold["train"],
            df_val          = fold["val"],
            asset_list      = asset_list,
            fold_idx        = _finetune_version(),
            cfg             = cfg,
            model_cfg       = model_cfg,
            save_dir        = save_dir,
            device          = device,
            seed            = seed,
            pretrained_state= pretrained_state,
        )

        logger.info(f"Fine-tune complete. Val Sharpe={val_sharpe:.3f} | Saved: {ckpt_path}")

        # After every finetune, run the auto-tuner to update parameters and RL mode
        logger.info("=== Auto-tune pass (post-finetune) ===")
        try:
            from pipeline.autotuner import run_autotuner
            run_autotuner(save_dir=save_dir)
        except Exception as e:
            logger.error(f"Auto-tune pass failed: {e}", exc_info=True)

    except Exception as e:
        logger.error(f"Weekly fine-tune failed: {e}", exc_info=True)


def _finetune_version() -> int:
    """Increment fold index based on existing checkpoints."""
    ckpts = glob.glob("models/best_fold*.pt")
    return len(ckpts)


def run_scheduler(universe: list[str] | None = None):
    """
    Start the scheduler loop.
    - Daily update at 17:00 (after US market close)
    - Weekly fine-tune every Sunday at 20:00
    """
    _setup_logging()
    logger.info("Scheduler started.")
    logger.info("  Daily data update:  17:00 Mon–Fri")
    logger.info("  Weekly fine-tune:   Sunday 20:00")
    logger.info("  Auto-tune pass:     after each finetune (params + RL gate)")
    logger.info("Press Ctrl+C to stop.\n")

    schedule.every().day.at("17:00").do(daily_update, universe=universe)
    schedule.every().sunday.at("20:00").do(weekly_finetune)

    # Run an immediate update on startup to catch up
    logger.info("Running initial data update...")
    daily_update(universe=universe)

    while True:
        schedule.run_pending()
        time.sleep(60)


# Allow running as a module
if __name__ == "__main__":
    import pandas as pd
    run_scheduler()
