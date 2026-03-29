"""
AutoTuner
=========
Automatically optimises broker parameters and decides whether to enable RL
mode, based entirely on replay data. Called by the scheduler after each
weekly finetune. Writes results back to broker.config.

What it does:
  1. Runs the sensitivity sweep over the last 2 years of data and picks the
     parameter set with the best Sharpe ratio.
  2. Runs the ablation gate (screener_rl vs heuristics_only). If RL passes,
     sets rl_enabled = true in broker.config. If it fails, sets it to false.
  3. Logs every decision with the data that drove it to logs/autotuner.log.

The user never needs to touch broker.config for these settings — the system
manages them automatically.

Parameters that are NEVER auto-tuned (require human judgment):
  - cash           (depends on how much money you're deploying)
  - max_positions  (portfolio concentration preference)
  - no_options     (risk preference)
"""

import logging
import os
import re
import glob
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Parameter search grid ─────────────────────────────────────────────────────

_PARAM_GRID = [
    # min_score × stop_loss × take_profit × max_sector
    {"min_score": ms, "stop_loss_floor": sl, "take_profit": tp, "max_sector_pct": mx}
    for ms in [0.50, 0.55, 0.60, 0.65]
    for sl in [0.06, 0.08, 0.10]
    for tp in [0.30, 0.40, 0.50]
    for mx in [0.20, 0.30, 0.40]
]

# ── Config file I/O ───────────────────────────────────────────────────────────

def _read_config(path: str = "broker.config") -> dict:
    """Parse broker.config into a dict. Ignores comments and blank lines."""
    cfg = {}
    with open(path) as f:
        for line in f:
            line = line.split("#")[0].strip()
            if "=" in line:
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip()
    return cfg


def _config_int(cfg: dict, key: str, default: int) -> int:
    try:
        return int(cfg.get(key, default))
    except (TypeError, ValueError):
        return default


def _write_config_key(key: str, value: str, path: str = "broker.config") -> None:
    """Update a single key in broker.config in-place, preserving all comments."""
    with open(path) as f:
        content = f.read()

    # Match the key at the start of a line (with optional spaces around =)
    pattern = rf"^({re.escape(key)}\s*=\s*)([^\n#]*)(.*)$"
    replacement = rf"\g<1>{value}\g<3>"
    new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    if new_content == content:
        # Key not found — append it
        new_content = content.rstrip() + f"\n{key:<22} = {value}\n"

    with open(path, "w") as f:
        f.write(new_content)

    logger.info("broker.config updated: %s = %s", key, value)


# ── Best checkpoint helper ────────────────────────────────────────────────────

def _best_checkpoint(save_dir: str = "models") -> str | None:
    ckpts = sorted(glob.glob(f"{save_dir}/best_fold*.pt"))
    return ckpts[-1] if ckpts else None


# ── Step 1: Parameter optimisation ───────────────────────────────────────────

def tune_parameters(
    df_features: pd.DataFrame,
    price_lookup: pd.DataFrame,
    initial_cash: float = 10_000.0,
    replay_years: int = 2,
    config_path: str = "broker.config",
) -> dict:
    """
    Run a parameter grid search over the last `replay_years` of data.
    Writes the best min_score, stop_loss, take_profit, and max_sector
    back to broker.config.

    Returns the best parameter dict.
    """
    from broker.replay import run_replay
    from pipeline.benchmark import compute_metrics

    logger.info("AutoTuner: starting parameter optimisation (%d combinations)...",
                len(_PARAM_GRID))

    # Restrict to tuning window
    dates  = sorted(df_features.index.get_level_values("date").unique())
    cutoff = pd.Timestamp(dates[-1]) - pd.DateOffset(years=replay_years)
    df_tune = df_features[df_features.index.get_level_values("date") >= cutoff]

    best_sharpe = -np.inf
    best_params = _PARAM_GRID[0]
    results = []

    for i, params in enumerate(_PARAM_GRID):
        try:
            rets, _ = run_replay(
                df_tune,
                price_lookup,
                strategy="heuristics_only",
                initial_cash=initial_cash,
                min_score=params["min_score"],
                stop_loss_floor=params["stop_loss_floor"],
                take_profit=params["take_profit"],
                max_sector_pct=params["max_sector_pct"],
                label=f"tune_{i}",
            )
            m = compute_metrics(rets, label=str(params))
            sharpe = m["sharpe"]
            results.append({**params, "sharpe": sharpe, "max_drawdown": m["max_drawdown"]})

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = params

        except Exception as exc:
            logger.warning("AutoTuner: param combo %d failed: %s", i, exc)

    logger.info(
        "AutoTuner: best params — min_score=%.2f  stop=%.2f  tp=%.2f  "
        "max_sector=%.2f  Sharpe=%.3f",
        best_params["min_score"],
        best_params["stop_loss_floor"],
        best_params["take_profit"],
        best_params["max_sector_pct"],
        best_sharpe,
    )

    # Write winners back to config
    _write_config_key("min_score",   f"{best_params['min_score']:.2f}",    config_path)
    _write_config_key("stop_loss",   f"{best_params['stop_loss_floor']:.2f}", config_path)
    _write_config_key("take_profit", f"{best_params['take_profit']:.2f}",  config_path)
    _write_config_key("max_sector",  f"{best_params['max_sector_pct']:.2f}", config_path)

    # Save full results for audit
    Path("plots").mkdir(exist_ok=True)
    pd.DataFrame(results).sort_values("sharpe", ascending=False).to_csv(
        "plots/param_tuning.csv", index=False
    )
    logger.info("AutoTuner: full parameter results saved → plots/param_tuning.csv")

    return best_params


# ── Step 2: RL gate decision ──────────────────────────────────────────────────

def tune_rl_mode(
    df_features: pd.DataFrame,
    price_lookup: pd.DataFrame,
    initial_cash: float = 10_000.0,
    replay_years: int = 2,
    save_dir: str = "models",
    config_path: str = "broker.config",
) -> bool:
    """
    Run the ablation gate. If screener_rl beats heuristics_only by the
    required margin, sets rl_enabled = true in broker.config.
    Returns True if RL was enabled, False if disabled.
    """
    from broker.replay import run_ablation, _check_ablation_gate
    from pipeline.benchmark import compute_metrics

    checkpoint = _best_checkpoint(save_dir)
    if checkpoint is None:
        logger.warning("AutoTuner: no checkpoint found — keeping rl_enabled = false")
        _write_config_key("rl_enabled", "false", config_path)
        return False

    logger.info("AutoTuner: running RL ablation gate with checkpoint %s...", checkpoint)

    # Restrict to tuning window
    dates  = sorted(df_features.index.get_level_values("date").unique())
    cutoff = pd.Timestamp(dates[-1]) - pd.DateOffset(years=replay_years)
    df_tune = df_features[df_features.index.get_level_values("date") >= cutoff]

    try:
        report_df = run_ablation(
            df_tune,
            price_lookup,
            checkpoint_path=checkpoint,
            initial_cash=initial_cash,
            replay_years=replay_years,
        )
        gate = _check_ablation_gate(report_df)
    except Exception as exc:
        logger.error("AutoTuner: ablation failed — keeping rl_enabled = false: %s", exc)
        _write_config_key("rl_enabled", "false", config_path)
        return False

    if gate == "PASSED":
        _write_config_key("rl_enabled",         "true",      config_path)
        _write_config_key("rl_checkpoint_path",  checkpoint,  config_path)
        logger.info("AutoTuner: RL ENABLED — ablation gate passed.")
        return True
    else:
        _write_config_key("rl_enabled", "false", config_path)
        logger.info("AutoTuner: RL DISABLED — ablation gate failed.")
        return False


# ── Main entry point ──────────────────────────────────────────────────────────

def run_autotuner(
    initial_cash: float = 10_000.0,
    replay_years: int = 2,
    save_dir: str = "models",
    config_path: str = "broker.config",
) -> None:
    """
    Full auto-tune pass: parameter optimisation then RL gate decision.
    Called by the scheduler after each weekly finetune.
    """
    _setup_autotuner_logging()
    logger.info("=" * 60)
    logger.info("AutoTuner run started: %s", datetime.now().strftime("%Y-%m-%d %H:%M"))
    logger.info("=" * 60)

    try:
        from pipeline.data import load_master
        from broker.replay import _build_price_lookup

        cfg = _read_config(config_path)
        top_n = _config_int(cfg, "top_n", 500)
        logger.info("AutoTuner: loading market data...")
        df_features = load_master(top_n=top_n)
        price_lookup = _build_price_lookup()

        # Step 1: tune heuristic parameters
        tune_parameters(
            df_features, price_lookup,
            initial_cash=initial_cash,
            replay_years=replay_years,
            config_path=config_path,
        )

        # Step 2: decide RL mode
        tune_rl_mode(
            df_features, price_lookup,
            initial_cash=initial_cash,
            replay_years=replay_years,
            save_dir=save_dir,
            config_path=config_path,
        )

        logger.info("AutoTuner run complete.")

    except Exception as exc:
        logger.error("AutoTuner run failed: %s", exc, exc_info=True)


def _setup_autotuner_logging():
    Path("logs").mkdir(exist_ok=True)
    handler = logging.FileHandler("logs/autotuner.log")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    if not any(isinstance(h, logging.FileHandler) and "autotuner" in h.baseFilename
               for h in logger.handlers):
        logger.addHandler(handler)
