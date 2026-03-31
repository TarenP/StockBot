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

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

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
    ckpts = sorted(glob.glob(f"{save_dir}/best_fold*.pt"))
    return ckpts[-1] if ckpts else None


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


def _load_data_and_universe(top_n: int):
    from pipeline.data import load_master, get_asset_universe
    df = load_master(top_n=top_n)
    asset_list = get_asset_universe(df, top_n=top_n, lookback_years=5)
    df = df[df.index.get_level_values("ticker").isin(asset_list)]
    return df, asset_list


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
    n_price = update_parquet(save_dir=args.save_dir, force_full_refresh=args.force_refresh)
    if n_price:
        logger.info(f"Price data: {n_price} new rows added.")
    else:
        logger.info("Price data already up to date.")

    # Sentiment update — use universe from checkpoint
    try:
        from pipeline.updater import _load_trained_universe
        universe = _load_trained_universe(args.save_dir)
        if universe is None:
            df, universe = _load_data_and_universe(_resolve_top_n(args))
        logger.info(f"Fetching news sentiment for {len(universe)} tickers...")
        n_sent = update_sentiment(universe, lookback_days=7 if args.force_refresh else 3)
        logger.info(f"Sentiment: {n_sent} new headlines scored.")
    except Exception as e:
        logger.warning(f"Sentiment update skipped: {e}")


# ── Mode: train ───────────────────────────────────────────────────────────────

def run_train(args):
    from pipeline.data import walk_forward_split
    from pipeline.train import PPO_CFG, fold_is_complete, train_fold

    debug_settings = _effective_debug_settings(args)
    top_n = _resolve_top_n(args)
    df, asset_list = _load_data_and_universe(top_n)
    logger.info(f"Universe: {len(asset_list)} tickers")

    folds = walk_forward_split(df, train_years=8, val_years=1, test_years=1)
    logger.info(f"Walk-forward folds available: {len(folds)}")

    cfg = {**PPO_CFG, "total_steps": debug_settings["total_steps"]}
    best_ckpts = []
    selected_folds = folds[:debug_settings["folds"]]

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
        else:
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
    df, asset_list = _load_data_and_universe(top_n)

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
    from pipeline.data     import walk_forward_split
    from pipeline.backtest import run_backtest, load_model

    ckpt_path = args.checkpoint or _best_checkpoint(args.save_dir)
    if not ckpt_path:
        logger.error("No checkpoint found. Run --mode train first.")
        return

    import torch
    meta  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    top_n = meta.get("top_n", _resolve_top_n(args))

    df, asset_list = _load_data_and_universe(top_n)
    folds = walk_forward_split(df, train_years=8, val_years=1, test_years=1)
    if not folds:
        logger.error("Not enough data for a full fold.")
        return

    df_test   = folds[-1]["test"]
    ckpt_path = args.checkpoint or _best_checkpoint(args.save_dir)
    if not ckpt_path:
        logger.error("No checkpoint found. Run --mode train first.")
        return

    model = load_model(ckpt_path, DEVICE)
    run_backtest(
        model      = model,
        df_test    = df_test,
        asset_list = asset_list,
        device     = DEVICE,
        save_plot  = "plots/backtest.png",
    )


# ── Mode: predict ─────────────────────────────────────────────────────────────

def run_predict(args):
    from pipeline.backtest import load_model
    from pipeline.features import FEATURE_COLS

    ckpt_path = args.checkpoint or _best_checkpoint(args.save_dir)
    if not ckpt_path:
        logger.error("No checkpoint found. Run --mode train first.")
        return

    # Read top_n from checkpoint so universe always matches training
    import torch
    meta   = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    configured_top_n = _resolve_top_n(args)
    top_n  = meta.get("top_n", configured_top_n)
    if args.top_n is None and top_n != configured_top_n:
        logger.info(f"Using universe size from checkpoint: {top_n} tickers")

    df, asset_list = _load_data_and_universe(top_n)

    ckpt_path = args.checkpoint or _best_checkpoint(args.save_dir)
    if not ckpt_path:
        logger.error("No checkpoint found. Run --mode train first.")
        return

    model = load_model(ckpt_path, DEVICE)

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

    n_features = df.shape[1]
    n_assets   = len(asset_list)
    asset_map  = {a: i for i, a in enumerate(asset_list)}
    obs        = np.zeros((lookback, n_assets, n_features), dtype=np.float32)

    for t_idx, date in enumerate(recent_dates):
        try:
            slice_df = df_recent.loc[date]
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
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]

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
    df, asset_list = _load_data_and_universe(_resolve_top_n(args))
    run_full_replay(
        df_features          = df,
        initial_cash         = 10_000.0,
        replay_years         = args.replay_years,
        run_sensitivity_sweep= args.sensitivity,
        save_plot            = "plots/replay.png",
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
    df, asset_list = _load_data_and_universe(top_n)
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
        from pipeline.data import load_master, get_asset_universe
        df = load_master()
        universe = get_asset_universe(df, top_n=args.top_n)
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
