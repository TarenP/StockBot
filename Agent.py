"""
Stock Predictor Agent — Entry Point
=====================================

MODES
-----
  python Agent.py --mode train      Train from scratch (walk-forward folds)
  python Agent.py --mode finetune   Fine-tune best checkpoint on recent data
  python Agent.py --mode update     Fetch latest market data (run daily)
  python Agent.py --mode backtest   Evaluate best checkpoint vs benchmarks
  python Agent.py --mode predict    Print today's recommended portfolio
  python Agent.py --mode schedule   Start the always-on scheduler daemon

QUICK START
-----------
  1. pip install -r requirements.txt
  2. python Agent.py --mode train          # first-time training (~hours)
  3. python Agent.py --mode predict        # see today's picks
  4. python Agent.py --mode schedule       # keep data + model fresh forever
"""

import argparse
import glob
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if callable(_reconfigure):
        try:
            _reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Stock Predictor Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--mode", choices=["train", "finetune", "update", "backtest",
                                       "predict", "schedule", "train_screener",
                                       "screen", "replay", "ablation", "warmup"],
                   default="predict", help="What to do (default: predict)")
    p.add_argument("--top_n",        type=int,   default=None,      help="Universe size for portfolio agent (defaults to broker.config)")
    p.add_argument("--top_k",        type=int,   default=10,       help="Stocks to hold / show")
    p.add_argument("--folds",        type=int,   default=3,        help="Walk-forward folds")
    p.add_argument("--total_steps",  type=int,   default=100_000,  help="PPO steps per fold")
    p.add_argument("--finetune_steps", type=int, default=20_000,   help="PPO steps for fine-tune")
    p.add_argument("--checkpoint",   type=str,   default=None,     help="Path to .pt checkpoint")
    p.add_argument("--save_dir",     type=str,   default="models", help="Checkpoint directory")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--force_retrain", action="store_true",         help="Retrain folds even if completion markers already exist")
    p.add_argument("--force_refresh", action="store_true",
                   help="Re-download last 30 days (--mode update)")
    p.add_argument("--expand_universe", action="store_true",
                   help="Ignore checkpoint universe and re-bootstrap from ~1500 tickers (--mode update)")
    # Screener filters
    p.add_argument("--penny",        action="store_true",
                   help="Screen penny stocks only (price < $5)")
    p.add_argument("--min_price",    type=float, default=0.01,     help="Min price filter for screener")
    p.add_argument("--max_price",    type=float, default=None,     help="Max price filter for screener")
    p.add_argument("--min_volume",   type=float, default=10_000,   help="Min avg daily volume for screener")
    p.add_argument("--screener_top_n", type=int, default=50,       help="How many picks to show from screener")
    p.add_argument("--screener_epochs", type=int, default=10,      help="Epochs for screener training")
    p.add_argument("--skip_screener_train", action="store_true",   help="Skip screener retraining in --mode train")
    p.add_argument("--shadow_generations", type=int, default=5,    help="Generations for shadow warm-up")
    p.add_argument("--shadow_replay_years", type=int, default=3,   help="Replay years for shadow warm-up")
    p.add_argument("--shadow_validation_top_n", type=int, default=20, help="Top genomes to fully validate in shadow warm-up")
    p.add_argument("--debug_fast", action="store_true",            help="Use fast debug defaults to reach replay validation quickly")
    p.add_argument("--replay_years",  type=int, default=3,         help="Years of history to replay (--mode replay)")
    p.add_argument("--sensitivity",   action="store_true",         help="Run sensitivity sweep during replay")
    p.add_argument("--rl_checkpoint", type=str, default=None,      help="Path to RL checkpoint for ablation/RL modes")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _best_checkpoint(save_dir: str) -> str | None:
    from pipeline.checkpoints import resolve_checkpoint_path

    return resolve_checkpoint_path(save_dir=save_dir)


def _load_broker_config(path: str = "broker.config") -> dict:
    cfg = {}
    config_path = os.path.abspath(path)
    if not os.path.exists(config_path):
        return cfg

    with open(config_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            cfg[key.strip()] = value.split("#")[0].strip()
    return cfg


def _resolve_top_n(args) -> int:
    if args.top_n is not None:
        return int(args.top_n)

    cfg = _load_broker_config()
    typed_cfg = _load_typed_config()
    try:
        from pipeline.universe_resolver import (
            get_universe_mode,
            is_benchmark_constrained_mode,
            resolve_configured_universe,
        )

        universe_mode = get_universe_mode(typed_cfg)
        configured_universe = resolve_configured_universe(config=typed_cfg)
        if configured_universe and is_benchmark_constrained_mode(universe_mode):
            top_n = len(configured_universe)
            logger.info(
                "Using top_n=%d from configured universe mode=%s",
                top_n,
                universe_mode,
            )
            return top_n
    except Exception as exc:
        logger.warning("Could not resolve configured universe size for top_n: %s", exc)

    try:
        top_n = int(cfg.get("top_n", 500))
    except (TypeError, ValueError):
        top_n = 500
    logger.info("Using top_n=%d from broker.config", top_n)
    return top_n


def _load_typed_config(path: str = "broker.config") -> dict:
    cfg = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            value = value.split("#")[0].strip()
            cfg[key.strip()] = (
                True if value.lower() == "true" else
                False if value.lower() == "false" else
                (int(value) if value.lstrip("-").isdigit() else
                 (float(value) if value.replace(".", "", 1).lstrip("-").isdigit() else value))
            )
    return cfg


def _latest_price_panel_date(price_path: str = "MasterDS/stooq_panel.parquet") -> pd.Timestamp | None:
    if not os.path.exists(price_path):
        return None

    raw = pd.read_parquet(price_path)
    if isinstance(raw.index, pd.MultiIndex) and "date" in raw.index.names:
        dates = pd.DatetimeIndex(pd.to_datetime(raw.index.get_level_values("date"), utc=True, errors="coerce"))
    elif "date" in raw.columns:
        dates = pd.DatetimeIndex(pd.to_datetime(raw["date"], utc=True, errors="coerce"))
    else:
        return None

    if getattr(dates, "tz", None) is not None:
        dates = dates.tz_convert(None)
    dates = dates.dropna()
    if len(dates) == 0:
        return None
    return pd.Timestamp(dates.max()).normalize()


def _load_data_and_universe(
    top_n: int,
    include_raw_cols: bool = False,
    universe_as_of_date: pd.Timestamp | None = None,
):
    from pipeline.data import load_master, get_asset_universe
    from pipeline.universe_resolver import (
        get_investable_universe_filters,
        get_universe_mode,
        is_benchmark_constrained_mode,
        normalize_tickers,
        resolve_configured_universe,
    )

    cfg = _load_typed_config()
    investable_filters = get_investable_universe_filters(cfg)
    df = load_master(
        top_n=top_n,
        min_history_days=int(investable_filters["min_history_days"]),
        min_price=float(investable_filters["min_price"]),
        min_avg_volume=float(investable_filters["min_avg_volume"]),
        include_raw_cols=include_raw_cols,
        universe_as_of_date=universe_as_of_date,
        config=cfg,
    )
    if is_benchmark_constrained_mode(get_universe_mode(cfg)):
        configured_universe = normalize_tickers(
            resolve_configured_universe(
                as_of_date=universe_as_of_date,
                save_dir="models",
                config=cfg,
            )
        )
        available = set(df.index.get_level_values("ticker").unique())
        asset_list = [ticker for ticker in configured_universe if ticker in available]
    else:
        asset_list = []
    if not asset_list:
        asset_list = get_asset_universe(
            df,
            top_n=top_n,
            lookback_years=5,
            as_of_date=universe_as_of_date,
        )
    df = df[df.index.get_level_values("ticker").isin(asset_list)]
    return df, asset_list


def _bootstrap_universe_size(top_n: int) -> int:
    return max(int(top_n) * 3, 1_000)


def _ensure_price_data(top_n: int, save_dir: str = "models") -> None:
    from pipeline.updater import PARQUET_PATH, update_parquet

    if PARQUET_PATH.exists():
        return

    bootstrap_size = _bootstrap_universe_size(top_n)
    logger.info(
        "No local price parquet found. Bootstrapping fresh market data "
        "for ~%d candidate tickers...",
        bootstrap_size,
    )
    update_parquet(
        save_dir=save_dir,
        force_full_refresh=True,
        bootstrap_universe_size=bootstrap_size,
    )
    if not PARQUET_PATH.exists():
        raise RuntimeError(
            "Price-data bootstrap did not create MasterDS/stooq_panel.parquet. "
            "Check network access and retry."
        )


def _effective_debug_settings(args) -> dict:
    settings = {
        "folds": args.folds,
        "total_steps": args.total_steps,
        "screener_epochs": args.screener_epochs,
        "skip_screener_train": args.skip_screener_train,
        "shadow_generations": args.shadow_generations,
        "shadow_replay_years": args.shadow_replay_years,
        "shadow_validation_top_n": args.shadow_validation_top_n,
    }
    if args.debug_fast:
        settings.update({
            "folds": 1,
            "total_steps": min(args.total_steps, 2_048),
            "screener_epochs": 1,
            "skip_screener_train": True,
            "shadow_generations": 1,
            "shadow_replay_years": 2,
            "shadow_validation_top_n": 5,
        })
        logger.info(
            "Debug-fast enabled: folds=%d total_steps=%d screener_epochs=%d "
            "skip_screener=%s shadow_generations=%d shadow_replay_years=%d "
            "shadow_validation_top_n=%d",
            settings["folds"],
            settings["total_steps"],
            settings["screener_epochs"],
            settings["skip_screener_train"],
            settings["shadow_generations"],
            settings["shadow_replay_years"],
            settings["shadow_validation_top_n"],
        )
    return settings


# ── Mode: update ──────────────────────────────────────────────────────────────

def run_update(args):
    from pipeline.updater   import update_parquet
    from pipeline.sentiment import update_sentiment

    logger.info("Fetching latest market data...")
    cfg = _load_typed_config()

    # --expand_universe: use parquet + bootstrap to get the broadest possible universe
    explicit_universe = None
    if getattr(args, "expand_universe", False):
        from pipeline.updater import get_live_universe
        bootstrap_size = _bootstrap_universe_size(_resolve_top_n(args))
        logger.info("Expanding universe: reading parquet + bootstrapping up to %d tickers...", bootstrap_size)
        explicit_universe = get_live_universe(save_dir=args.save_dir, target_size=bootstrap_size)
        logger.info("Expanded universe: %d tickers", len(explicit_universe))

    n_price = update_parquet(
        universe=explicit_universe,
        save_dir=args.save_dir,
        force_full_refresh=args.force_refresh or getattr(args, "expand_universe", False),
        bootstrap_universe_size=_bootstrap_universe_size(_resolve_top_n(args)),
        config=cfg,
    )
    if n_price:
        logger.info(f"Price data: {n_price} new rows added.")
    else:
        logger.info("Price data already up to date.")

    # Sentiment update — use expanded universe if requested, else checkpoint
    try:
        from pipeline.updater import get_live_universe

        universe = explicit_universe
        if universe is None:
            universe = get_live_universe(
                save_dir=args.save_dir,
                target_size=_bootstrap_universe_size(_resolve_top_n(args)),
                config=cfg,
            )
        if universe is None:
            df, universe = _load_data_and_universe(_resolve_top_n(args))
        logger.info(f"Fetching news sentiment for {len(universe)} tickers...")
        n_sent = update_sentiment(
            universe,
            lookback_days=7 if args.force_refresh else 3,
            save_dir=args.save_dir,
        )
        logger.info(f"Sentiment: {n_sent} new headlines scored.")
    except Exception as e:
        logger.warning(f"Sentiment update skipped: {e}")


# ── Mode: train ───────────────────────────────────────────────────────────────

def run_train(args):
    from pipeline.data import walk_forward_split
    from pipeline.train import PPO_CFG, fold_is_complete, train_fold

    debug_settings = _effective_debug_settings(args)
    top_n = _resolve_top_n(args)
    _ensure_price_data(top_n, save_dir=args.save_dir)
    df, asset_list = _load_data_and_universe(top_n, include_raw_cols=True)
    logger.info(f"Universe: {len(asset_list)} tickers")

    folds = walk_forward_split(df, train_years=8, val_years=1, test_years=1)
    logger.info(f"Walk-forward folds available: {len(folds)}")

    cfg = {**PPO_CFG, "total_steps": debug_settings["total_steps"]}
    best_ckpts = []
    selected_folds = folds[:debug_settings["folds"]]

    # ── Train screener first (needed to build shortlist universe) ─────────────
    if not debug_settings["skip_screener_train"]:
        logger.info("\nTraining screener on full universe...")
        logger.info("The screener narrows 11,500+ tickers to a shortlist for the RL agent.")
        try:
            from pipeline.data import load_master as _load_all
            from pipeline.screener import train_screener

            df_all = _load_all(top_n=99_999, include_raw_cols=True)
            train_screener(
                df_all,
                device=DEVICE,
                epochs=debug_settings["screener_epochs"],
                force_rebuild_cache=args.force_retrain,
            )
            logger.info("Screener training complete.")
        except Exception as exc:
            logger.warning(f"Screener training failed (continuing): {exc}")

    # ── Build shortlist universe from screener (if screener is trained) ───────
    shortlist_universe = None
    if not debug_settings["skip_screener_train"]:
        try:
            from pipeline.screener import SCREENER_CKPT, load_screener
            from pipeline.train import build_shortlist_universe
            import os as _os
            if _os.path.exists(SCREENER_CKPT):
                logger.info("Building shortlist universe from trained screener...")
                screener_model = load_screener(DEVICE)
                shortlist_universe = build_shortlist_universe(
                    df_features=df,
                    screener_model=screener_model,
                    top_n=100,
                    device=DEVICE,
                )
                logger.info(f"Shortlist universe: {len(shortlist_universe)} tickers")
        except Exception as exc:
            logger.warning(f"Could not build shortlist universe ({exc}) — using full asset_list")

    with tqdm(
        total=len(selected_folds),
        desc="Training folds",
        unit="fold",
        colour="magenta",
        dynamic_ncols=True,
    ) as folds_pbar:
        for i, fold in enumerate(selected_folds):
            if fold_is_complete(args.save_dir, i) and not args.force_retrain:
                logger.info(f"Fold {i} already complete - skipping.")
                ckpt = f"{args.save_dir}/best_fold{i}.pt"
                if os.path.exists(ckpt):
                    import torch

                    meta = torch.load(ckpt, map_location="cpu", weights_only=False)
                    best_ckpts.append((ckpt, meta.get("val_sharpe", 0.0)))
                folds_pbar.update(1)
                folds_pbar.set_postfix(done=f"{i + 1}/{len(selected_folds)}")
                continue

            try:
                folds_pbar.set_postfix(current=f"{i + 1}/{len(selected_folds)}")
                ckpt_path, val_sharpe = train_fold(
                    df_train=fold["train"],
                    df_val=fold["val"],
                    asset_list=asset_list,
                    fold_idx=i,
                    cfg=cfg,
                    save_dir=args.save_dir,
                    device=DEVICE,
                    seed=args.seed,
                    top_n=top_n,
                    force_restart=args.force_retrain,
                    shortlist_universe=shortlist_universe,
                )
                best_ckpts.append((ckpt_path, val_sharpe))
                folds_pbar.update(1)
                folds_pbar.set_postfix(
                    done=f"{i + 1}/{len(selected_folds)}",
                    val_sharpe=f"{val_sharpe:.3f}",
                )
            except KeyboardInterrupt:
                logger.info("Training interrupted. Run the same command to resume.")
                break

    if best_ckpts:
        best = max(best_ckpts, key=lambda x: x[1])
        logger.info(f"\nBest checkpoint: {best[0]}  (val Sharpe={best[1]:.3f})")

        if debug_settings["skip_screener_train"]:
            logger.info("\nSkipping screener retraining for this run.")

        logger.info("\nRunning shadow warm-up on historical data...")
        logger.info("This finds the best broker parameters from history so you")
        logger.info("start with a pre-tuned config on day one. Takes ~10-20 min.")
        try:
            from broker.replay import _build_price_lookup
            from broker.shadows import run_historical_warmup

            live_config = _load_typed_config()
            price_lookup = _build_price_lookup()

            promoted = run_historical_warmup(
                df_features=df,
                price_lookup=price_lookup,
                checkpoint_path=best[0],
                live_config=live_config,
                generations=debug_settings["shadow_generations"],
                replay_years=debug_settings["shadow_replay_years"],
                validation_top_n=debug_settings["shadow_validation_top_n"],
            )
            if promoted:
                logger.info("Shadow warm-up complete. broker.config updated with best historical parameters.")
            else:
                logger.info("Shadow warm-up complete. broker.config left unchanged.")
        except Exception as exc:
            logger.warning(f"Shadow warm-up failed (broker.config unchanged): {exc}")


def run_finetune(args):
    from pipeline.data     import walk_forward_split
    from pipeline.train    import train_fold, PPO_CFG
    from pipeline.backtest import load_model

    top_n = _resolve_top_n(args)
    df, asset_list = _load_data_and_universe(top_n, include_raw_cols=True)

    # Use only the most recent 2 years for fine-tuning
    dates   = sorted(df.index.get_level_values("date").unique())
    cutoff  = pd.Timestamp(dates[-1]) - pd.DateOffset(months=24)
    df_recent = df[df.index.get_level_values("date") >= cutoff]

    folds = walk_forward_split(df_recent, train_years=1, val_years=3, test_years=3)
    if not folds:
        logger.error("Not enough recent data for fine-tuning (need ~18 months).")
        return

    fold = folds[-1]

    # Load pretrained weights
    ckpt_path = args.checkpoint or _best_checkpoint(args.save_dir)
    pretrained_state = None
    model_cfg        = None
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        pretrained_state = ckpt["model_state"]
        model_cfg        = ckpt["model_cfg"]
        logger.info(f"Fine-tuning from: {ckpt_path}")
    else:
        logger.info("No checkpoint found — training from scratch.")

    cfg = {**PPO_CFG, "total_steps": args.finetune_steps}

    # Version the fine-tuned checkpoint
    fold_idx = len(glob.glob(f"{args.save_dir}/best_fold*.pt"))

    new_ckpt, val_sharpe = train_fold(
        df_train         = fold["train"],
        df_val           = fold["val"],
        asset_list       = asset_list,
        fold_idx         = fold_idx,
        cfg              = cfg,
        model_cfg        = model_cfg,
        save_dir         = args.save_dir,
        device           = DEVICE,
        seed             = args.seed,
        pretrained_state = pretrained_state,
    )
    logger.info(f"Fine-tune done. Val Sharpe={val_sharpe:.3f} | Saved: {new_ckpt}")


# ── Mode: backtest ────────────────────────────────────────────────────────────

def run_backtest_mode(args):
    from pipeline.data import walk_forward_split
    from pipeline.backtest import load_model, run_backtest

    ckpt_path = args.checkpoint or _best_checkpoint(args.save_dir)
    if not ckpt_path:
        logger.error("No checkpoint found. Run --mode train first.")
        return

    import torch
    meta  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    top_n = meta.get("top_n", _resolve_top_n(args))

    df, asset_list = _load_data_and_universe(top_n, include_raw_cols=True)

    # Use checkpoint's asset_list if available (may be a focused shortlist)
    ckpt_asset_list = meta.get("asset_list")
    if ckpt_asset_list:
        # Filter to tickers that exist in current df
        available = set(df.index.get_level_values("ticker").unique())
        ckpt_asset_list = [t for t in ckpt_asset_list if t in available]
        if len(ckpt_asset_list) >= 10:
            asset_list = ckpt_asset_list
            logger.info("Backtest: using %d tickers from checkpoint asset_list.", len(asset_list))

    folds = walk_forward_split(df, train_years=8, val_years=1, test_years=1)
    if not folds:
        logger.error("Not enough data for a full fold.")
        return

    # Use the test fold that matches the checkpoint's fold index
    ckpt_fold = meta.get("fold", len(folds) - 1)
    if ckpt_fold < len(folds):
        df_test = folds[ckpt_fold]["test"]
        test_dates = sorted(df_test.index.get_level_values("date").unique())
        logger.info(
            "Backtest: fold %d test set (%s → %s)",
            ckpt_fold, test_dates[0].date(), test_dates[-1].date(),
        )
    else:
        df_test = folds[-1]["test"]
        logger.warning("Checkpoint fold %d out of range — using last fold.", ckpt_fold)

    model, ckpt_n_features = load_model(ckpt_path, DEVICE)
    logger.info("Backtest: evaluating raw RL policy on held-out test fold.")
    run_backtest(
        model=model,
        df_test=df_test,
        asset_list=asset_list,
        device=DEVICE,
        save_plot="plots/backtest.png",
        ckpt_n_features=ckpt_n_features,
    )


# ── Mode: predict ─────────────────────────────────────────────────────────────

def run_predict(args):
    from pipeline.backtest import load_model
    from pipeline.features import FEATURE_COLS

    ckpt_path = args.checkpoint or _best_checkpoint(args.save_dir)
    if not ckpt_path:
        logger.error("No checkpoint found. Run --mode train first.")
        return

    import torch
    meta   = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    configured_top_n = _resolve_top_n(args)
    top_n  = meta.get("top_n", configured_top_n)
    if args.top_n is None and top_n != configured_top_n:
        logger.info(f"Using universe size from checkpoint: {top_n} tickers")

    df, asset_list = _load_data_and_universe(top_n)

    # Use checkpoint's asset_list if available (may be a focused shortlist)
    ckpt_asset_list = meta.get("asset_list")
    if ckpt_asset_list:
        available = set(df.index.get_level_values("ticker").unique())
        ckpt_asset_list = [t for t in ckpt_asset_list if t in available]
        if len(ckpt_asset_list) >= 10:
            asset_list = ckpt_asset_list
            logger.info("Predict: using %d tickers from checkpoint asset_list.", len(asset_list))

    model, ckpt_n_features = load_model(ckpt_path, DEVICE)

    lookback = 20
    dates    = sorted(df.index.get_level_values("date").unique())
    if len(dates) < lookback:
        logger.error("Not enough data for prediction.")
        return

    recent_dates = dates[-lookback:]
    df_recent    = df[df.index.get_level_values("date").isin(recent_dates)]

    # ── Sentiment freshness warning ──────────────────────────────────────────
    last_date = recent_dates[-1]
    try:
        sent_raw  = pd.read_csv("Sentiment/analyst_ratings_with_sentiment.csv",
                                usecols=["date"])
        sent_last = pd.to_datetime(sent_raw["date"], utc=True,
                                   errors="coerce").dt.tz_convert(None).max()
        gap_days  = (pd.Timestamp(last_date) - sent_last).days
        if gap_days > 7:
            logger.warning(
                f"Sentiment data is {gap_days} days stale (last: {sent_last.date()}). "
                f"Run --mode update to fetch fresh news — sentiment signals will show n/a until then."
            )
    except Exception:
        pass

    # Slice feature columns to match what the checkpoint was trained on
    feature_cols_pred = FEATURE_COLS[:ckpt_n_features]
    # Only keep columns that exist in df
    feature_cols_pred = [c for c in feature_cols_pred if c in df.columns]
    n_features = len(feature_cols_pred)
    n_assets   = len(asset_list)
    asset_map  = {a: i for i, a in enumerate(asset_list)}
    obs        = np.zeros((lookback, n_assets, n_features), dtype=np.float32)

    for t_idx, date in enumerate(recent_dates):
        try:
            slice_df = df_recent[feature_cols_pred].loc[date]
            for ticker, row in slice_df.iterrows():
                if ticker in asset_map:
                    obs[t_idx, asset_map[ticker], :] = row.values.astype(np.float32)
        except KeyError:
            pass

    obs_t   = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    weights = model.get_weights(obs_t).squeeze(0).cpu().numpy()

    asset_weights = weights[:-1]
    cash_weight   = weights[-1]

    # ── Build ranked output table ────────────────────────────────────────────
    # Grab the most recent feature snapshot per ticker for signal context
    last_date    = recent_dates[-1]
    feature_cols = feature_cols_pred

    rows = []
    for idx, ticker in enumerate(asset_list):
        w = asset_weights[idx]
        if w < 1e-4:
            continue
        row = {"ticker": ticker, "weight": w}
        try:
            feats = df_recent.loc[(last_date, ticker)]
            # Map a few key signals to human-readable columns
            feat_dict = dict(zip(feature_cols, feats.values))
            row["momentum_20d"] = feat_dict.get("ret_20d",   np.nan)
            row["rsi"]          = feat_dict.get("rsi",        np.nan)
            row["sentiment"]    = feat_dict.get("sent_net",   np.nan)
            row["vol_ratio"]    = feat_dict.get("vol_ratio",  np.nan)
        except KeyError:
            row["momentum_20d"] = np.nan
            row["rsi"]          = np.nan
            row["sentiment"]    = np.nan
            row["vol_ratio"]    = np.nan
        rows.append(row)

    picks = (
        pd.DataFrame(rows)
          .sort_values("weight", ascending=False)
          .head(args.top_k)
          .reset_index(drop=True)
    )
    picks.index += 1   # 1-based rank

    # ── Signal legend (z-scored, so interpret relative to universe) ──────────
    print(f"\n{'='*72}")
    print(f"  Weekly Portfolio Picks  —  as of {last_date.date()}")
    print(f"  Universe: {n_assets} stocks  |  Showing top {args.top_k}")
    print(f"{'='*72}")
    print(f"  {'#':<4} {'Ticker':<8} {'Weight':>7}  {'Mom20d':>8}  {'RSI':>6}  {'Sentiment':>10}  {'Vol Ratio':>10}")
    print(f"  {'-'*66}")
    for rank, row in picks.iterrows():
        mom  = f"{row['momentum_20d']:+.2f}" if not np.isnan(row['momentum_20d']) else "  n/a"
        rsi  = f"{row['rsi']:+.2f}"          if not np.isnan(row['rsi'])          else "  n/a"
        sent = f"{row['sentiment']:+.2f}"    if not np.isnan(row['sentiment'])    else "  n/a"
        vol  = f"{row['vol_ratio']:+.2f}"    if not np.isnan(row['vol_ratio'])    else "  n/a"
        print(f"  {rank:<4} {row['ticker']:<8} {row['weight']:>6.2%}  {mom:>8}  {rsi:>6}  {sent:>10}  {vol:>10}")
    print(f"  {'-'*66}")
    print(f"  {'CASH':<12} {cash_weight:>6.2%}")
    print(f"{'='*72}")
    print(f"  Signals are cross-sectionally z-scored (0 = universe avg).")
    print(f"  Rebalance weekly. Not financial advice.\n")


# ── Mode: train_screener ──────────────────────────────────────────────────────

def run_train_screener(args):
    from pipeline.screener import train_screener
    from pipeline.data import load_master

    # Load ALL tickers — no universe cap for the screener
    logger.info("Loading full dataset for screener training (all tickers)...")
    df = load_master(
        min_price      = args.min_price,
        min_avg_volume = args.min_volume,
        top_n          = 99_999,   # effectively no cap
        include_raw_cols=True,
    )
    train_screener(
        df,
        device=DEVICE,
        epochs=args.screener_epochs,
        force_rebuild_cache=args.force_retrain,
    )


def run_screen(args):
    from pipeline.screener import run_screener, print_screener_results
    from pipeline.data import load_master

    min_price = 0.01
    max_price = None
    label     = "All stocks"

    if args.penny:
        max_price = 5.0
        label     = "Penny stocks (< $5)"
        logger.info("Penny stock mode — scanning stocks under $5")
    elif args.max_price:
        max_price = args.max_price
        label     = f"Stocks under ${max_price}"

    if args.min_price != 0.01:
        min_price = args.min_price

    logger.info("Loading full dataset for screening...")
    df = load_master(
        min_price      = min_price,
        min_avg_volume = args.min_volume,
        top_n          = 99_999,
        include_raw_cols=True,
    )

    results = run_screener(
        df         = df,
        device     = DEVICE,
        top_n      = args.screener_top_n,
        min_price  = min_price,
        max_price  = max_price,
        min_volume = args.min_volume,
    )

    print_screener_results(results, label=label)


def run_replay_mode(args):
    from broker.replay import run_full_replay
    from broker.broker import _resolve_checkpoint
    from pathlib import Path as _Path

    replay_universe_as_of = None
    latest_price_date = _latest_price_panel_date()
    if latest_price_date is not None:
        replay_universe_as_of = latest_price_date - pd.DateOffset(years=args.replay_years)
        logger.info(
            "Replay: selecting universe as of %s to avoid forward-looking membership drift.",
            replay_universe_as_of.date(),
        )

    df, asset_list = _load_data_and_universe(
        _resolve_top_n(args),
        universe_as_of_date=replay_universe_as_of,
    )
    live_config = _load_typed_config()

    # P1: Auto-enable universe snapshot for replay determinism unless explicitly disabled.
    # This ensures re-running the same replay always uses the same ticker set.
    replay_config = dict(live_config)
    if not replay_config.get("freeze_universe_snapshot", False):
        snapshot_path = str(replay_config.get(
            "universe_snapshot_path",
            "plots/live_universe_snapshot.json",
        ))
        replay_config["freeze_universe_snapshot"] = True
        replay_config["universe_snapshot_path"] = snapshot_path
        logger.info(
            "Replay: auto-enabled universe snapshot for determinism -> %s. "
            "Set freeze_universe_snapshot = true in broker.config to make this permanent.",
            snapshot_path,
        )

    run_full_replay(
        df_features          = df,
        initial_cash         = float(replay_config.get("cash", 10_000.0)),
        replay_years         = args.replay_years,
        run_sensitivity_sweep= args.sensitivity,
        save_plot            = "plots/replay.png",
        live_config          = replay_config,
        checkpoint_path      = _resolve_checkpoint(replay_config.get("rl_checkpoint_path")),
    )


def run_ablation_mode(args):
    from broker.replay import run_ablation, _build_price_lookup
    df_features, _ = _load_data_and_universe(_resolve_top_n(args))
    price_lookup = _build_price_lookup()
    report = run_ablation(
        df_features    = df_features,
        price_lookup   = price_lookup,
        checkpoint_path= args.rl_checkpoint,
    )
    print(report.to_string(index=False))


def run_warmup_mode(args):
    from broker.replay import _build_price_lookup
    from broker.shadows import run_historical_warmup
    import torch

    debug_settings = _effective_debug_settings(args)
    ckpt_path = args.checkpoint or _best_checkpoint(args.save_dir)
    if not ckpt_path:
        logger.error("No checkpoint found. Run --mode train first or pass --checkpoint.")
        return

    top_n = _resolve_top_n(args)
    try:
        meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        top_n = int(meta.get("top_n", top_n))
    except Exception:
        pass
    df, asset_list = _load_data_and_universe(top_n, include_raw_cols=True)
    logger.info("Warm-up mode using universe: %d tickers", len(asset_list))

    live_config = _load_typed_config()
    price_lookup = _build_price_lookup()
    promoted = run_historical_warmup(
        df_features=df,
        price_lookup=price_lookup,
        checkpoint_path=ckpt_path,
        live_config=live_config,
        generations=debug_settings["shadow_generations"],
        replay_years=debug_settings["shadow_replay_years"],
        validation_top_n=debug_settings["shadow_validation_top_n"],
    )
    if promoted:
        logger.info("Warm-up mode complete. broker.config updated.")
    else:
        logger.info("Warm-up mode complete. broker.config left unchanged.")


# ── Mode: schedule ────────────────────────────────────────────────────────────

def run_schedule(args):
    from pipeline.scheduler import run_scheduler
    # Pass the universe so the scheduler knows what to update
    try:
        from pipeline.updater import get_live_universe

        universe = get_live_universe(
            target_size=_bootstrap_universe_size(_resolve_top_n(args)),
            config=_load_typed_config(),
        )
    except Exception:
        universe = None   # scheduler will infer from parquet
    run_scheduler(universe=universe)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    logger.info(f"Mode: {args.mode} | Device: {DEVICE}")

    dispatch = {
        "update":         run_update,
        "train":          run_train,
        "finetune":       run_finetune,
        "backtest":       run_backtest_mode,
        "predict":        run_predict,
        "schedule":       run_schedule,
        "train_screener": run_train_screener,
        "screen":         run_screen,
        "replay":         run_replay_mode,
        "ablation":       run_ablation_mode,
        "warmup":         run_warmup_mode,
    }
    dispatch[args.mode](args)
