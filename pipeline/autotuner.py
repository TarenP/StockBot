"""
AutoTuner
=========
Automatically optimizes broker parameters and decides whether to enable RL
mode, based entirely on replay data. Called by the scheduler after each
weekly fine-tune. Writes results back to broker.config.
"""

import glob
import logging
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PARAM_OPTIONS = {
    "min_score": [0.52, 0.58, 0.64],
    "stop_loss_floor": [0.07, 0.10],
    "take_profit": [0.45, 0.60],
    "max_sector_pct": [0.30, 0.40],
    "max_position_pct": [0.12, 0.18, 0.22],
    "cash_floor": [0.01, 0.03],
    "max_gross_exposure": [0.95, 0.99],
    "target_volatility": [0.15, 0.22],
}

_PARAM_GRID = [
    {
        "min_score": ms,
        "stop_loss_floor": sl,
        "take_profit": tp,
        "max_sector_pct": mx,
        "max_position_pct": mpp,
        "cash_floor": cf,
        "max_gross_exposure": mge,
        "target_volatility": tv,
    }
    for ms in _PARAM_OPTIONS["min_score"]
    for sl in _PARAM_OPTIONS["stop_loss_floor"]
    for tp in _PARAM_OPTIONS["take_profit"]
    for mx in _PARAM_OPTIONS["max_sector_pct"]
    for mpp in _PARAM_OPTIONS["max_position_pct"]
    for cf in _PARAM_OPTIONS["cash_floor"]
    for mge in _PARAM_OPTIONS["max_gross_exposure"]
    for tv in _PARAM_OPTIONS["target_volatility"]
]

_HOLDOUT_FRACTION = 0.25
_MIN_HOLDOUT_DAYS = 63
_MAX_HOLDOUT_DAYS = 126
_HOLDOUT_FINALISTS = 12
_FAST_SEARCH_MAX_COMBINATIONS = 96
_FAST_HOLDOUT_FINALISTS = 6


def _read_config(path: str = "broker.config") -> dict:
    """Parse broker.config into a dict. Ignores comments and blank lines."""
    cfg = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.split("#")[0].strip()
            if "=" in line:
                key, value = line.split("=", 1)
                cfg[key.strip()] = value.strip()
    return cfg


def _config_int(cfg: dict, key: str, default: int) -> int:
    try:
        return int(cfg.get(key, default))
    except (TypeError, ValueError):
        return default


def _config_float(cfg: dict, key: str, default: float) -> float:
    try:
        return float(cfg.get(key, default))
    except (TypeError, ValueError):
        return default


def _config_str(cfg: dict, key: str, default: str) -> str:
    value = cfg.get(key, default)
    return str(value).strip().lower() if value is not None else default


def _current_param_values(cfg: dict) -> dict:
    return {
        "min_score": _config_float(cfg, "min_score", 0.58),
        "stop_loss_floor": _config_float(cfg, "stop_loss", 0.10),
        "take_profit": _config_float(cfg, "take_profit", 0.60),
        "max_sector_pct": _config_float(cfg, "max_sector", 0.40),
        "max_position_pct": _config_float(cfg, "max_position_pct", 0.18),
        "cash_floor": _config_float(cfg, "cash_floor", 0.03),
        "max_gross_exposure": _config_float(cfg, "max_gross_exposure", 0.99),
        "target_volatility": _config_float(cfg, "target_volatility", 0.22),
    }


def _param_signature(params: dict) -> tuple:
    return tuple((key, round(float(params[key]), 6)) for key in _PARAM_OPTIONS)


def _param_distance(params: dict, center: dict) -> tuple:
    distance = 0.0
    for key, options in _PARAM_OPTIONS.items():
        lo = min(options)
        hi = max(options)
        width = max(hi - lo, 1e-9)
        distance += abs(float(params[key]) - float(center[key])) / width
    return (distance, _param_signature(params))


def _build_param_grid(
    cfg: dict,
    search_mode: str | None = None,
    max_combinations: int | None = None,
) -> list[dict]:
    """
    Build the weekly parameter-search grid.

    The old behavior tried every point in the full Cartesian grid. The default
    fast mode keeps the current live settings plus the nearest deterministic
    candidates, which preserves a real correction pass without turning weekly
    maintenance into hundreds of full replays.
    """
    mode = (search_mode or _config_str(cfg, "autotune_search_mode", "fast")).lower()
    if mode == "full":
        return [dict(params) for params in _PARAM_GRID]

    try:
        budget = int(max_combinations or _config_int(
            cfg, "autotune_max_combinations", _FAST_SEARCH_MAX_COMBINATIONS,
        ))
    except (TypeError, ValueError):
        budget = _FAST_SEARCH_MAX_COMBINATIONS
    budget = max(1, min(budget, len(_PARAM_GRID) + 1))

    center = _current_param_values(cfg)
    selected: list[dict] = []
    seen: set[tuple] = set()

    def add(params: dict) -> None:
        sig = _param_signature(params)
        if sig not in seen:
            selected.append(dict(params))
            seen.add(sig)

    add(center)
    for params in sorted(_PARAM_GRID, key=lambda row: _param_distance(row, center)):
        add(params)
        if len(selected) >= budget:
            break

    return selected


def _base_replay_kwargs(cfg: dict, initial_cash: float) -> dict:
    return {
        "strategy": "heuristics_only",
        "initial_cash": initial_cash,
        "max_positions": _config_int(cfg, "max_positions", 20),
        "partial_profit_pct": _config_float(cfg, "partial_profit", 0.20),
        "penny_pct": _config_float(cfg, "penny_pct", 0.20),
        "max_pair_correlation": _config_float(cfg, "max_correlation", 0.80),
        "weak_theme_min_positions": _config_int(cfg, "weak_theme_min_positions", 2),
        "weak_theme_return_threshold": _config_float(cfg, "weak_theme_return_threshold", -0.03),
        "weak_theme_penalty_mult": _config_float(cfg, "weak_theme_penalty_mult", 0.50),
        "avoid_earnings_days": _config_int(cfg, "avoid_earnings", 3),
        "vol_lookback": _config_int(cfg, "vol_lookback", 20),
    }


def _write_config_key(key: str, value: str, path: str = "broker.config") -> None:
    """Update one broker.config key in-place and collapse duplicate entries."""
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    key_pattern = re.compile(rf"^(\s*{re.escape(key)}\s*=\s*)([^#\r\n]*)(.*)$")
    new_lines: list[str] = []
    replaced = False

    for line in lines:
        match = key_pattern.match(line)
        if not match:
            new_lines.append(line)
            continue

        if replaced:
            continue

        prefix, old_value, suffix = match.groups()
        comment_spacing = old_value[len(old_value.rstrip()):]
        if suffix and not comment_spacing:
            comment_spacing = " "
        new_lines.append(f"{prefix}{value}{comment_spacing}{suffix}\n")
        replaced = True

    if not replaced:
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines[-1] += "\n"
        new_lines.append(f"{key:<22} = {value}\n")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    logger.info("broker.config updated: %s = %s", key, value)


def _best_checkpoint(save_dir: str = "models") -> str | None:
    from pipeline.checkpoints import resolve_checkpoint_path

    return resolve_checkpoint_path(save_dir=save_dir)


def _split_replay_holdout(
    df_features: pd.DataFrame,
    replay_years: int,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """
    Split the trailing replay window into an optimization slice and a later
    holdout slice. Falls back to a shared in-sample window when history is too
    short to support a meaningful split.
    """
    dates = sorted(df_features.index.get_level_values("date").unique())
    cutoff = pd.Timestamp(dates[-1]) - pd.DateOffset(years=replay_years)
    df_window = df_features[df_features.index.get_level_values("date") >= cutoff]

    replay_dates = sorted(df_window.index.get_level_values("date").unique())
    if len(replay_dates) < (_MIN_HOLDOUT_DAYS * 2):
        logger.warning(
            "AutoTuner: only %d replay dates available; using a single in-sample window.",
            len(replay_dates),
        )
        return df_window, df_window, False

    holdout_days = int(round(len(replay_dates) * _HOLDOUT_FRACTION))
    holdout_days = max(_MIN_HOLDOUT_DAYS, holdout_days)
    holdout_days = min(_MAX_HOLDOUT_DAYS, holdout_days)
    holdout_days = min(holdout_days, len(replay_dates) - _MIN_HOLDOUT_DAYS)
    if holdout_days < _MIN_HOLDOUT_DAYS:
        logger.warning(
            "AutoTuner: replay split became too short after constraints; using a single in-sample window."
        )
        return df_window, df_window, False

    split_date = replay_dates[-holdout_days]
    df_search = df_window[df_window.index.get_level_values("date") < split_date]
    df_holdout = df_window[df_window.index.get_level_values("date") >= split_date]

    search_dates = sorted(df_search.index.get_level_values("date").unique())
    holdout_dates = sorted(df_holdout.index.get_level_values("date").unique())
    if not search_dates or not holdout_dates:
        logger.warning(
            "AutoTuner: failed to build a clean holdout split; using a single in-sample window."
        )
        return df_window, df_window, False

    logger.info(
        "AutoTuner: optimization window %s -> %s (%d days) | holdout %s -> %s (%d days)",
        search_dates[0].date(),
        search_dates[-1].date(),
        len(search_dates),
        holdout_dates[0].date(),
        holdout_dates[-1].date(),
        len(holdout_dates),
    )
    return df_search, df_holdout, True


def tune_parameters(
    df_features: pd.DataFrame,
    price_lookup: pd.DataFrame,
    initial_cash: float = 10_000.0,
    replay_years: int = 2,
    config_path: str = "broker.config",
) -> dict:
    """
    Run a parameter search over the trailing replay window.
    Winners are ranked on a holdout slice when enough history is available.
    """
    from broker.replay import run_replay
    from pipeline.benchmark import compute_metrics

    cfg = _read_config(config_path)
    search_mode = _config_str(cfg, "autotune_search_mode", "fast")
    max_combinations = _config_int(
        cfg, "autotune_max_combinations", _FAST_SEARCH_MAX_COMBINATIONS,
    )
    holdout_finalists = _config_int(
        cfg, "autotune_holdout_finalists", _FAST_HOLDOUT_FINALISTS,
    )
    param_grid = _build_param_grid(
        cfg,
        search_mode=search_mode,
        max_combinations=max_combinations,
    )

    logger.info(
        "AutoTuner: starting parameter optimisation (%s mode, %d/%d combinations)...",
        search_mode,
        len(param_grid),
        len(_PARAM_GRID),
    )

    shared_kwargs = _base_replay_kwargs(cfg, initial_cash)
    df_search, df_holdout, has_holdout = _split_replay_holdout(
        df_features,
        replay_years=replay_years,
    )
    results: list[dict] = []

    for i, params in enumerate(param_grid):
        try:
            rets, _ = run_replay(
                df_search,
                price_lookup,
                **shared_kwargs,
                **params,
                label=f"search_{i}",
            )
            m = compute_metrics(rets, label=str(params))
            results.append({
                **params,
                "search_total_return": m["total_return"],
                "search_ann_return": m["ann_return"],
                "search_sharpe": m["sharpe"],
                "search_max_drawdown": m["max_drawdown"],
                "holdout_total_return": np.nan,
                "holdout_ann_return": np.nan,
                "holdout_sharpe": np.nan,
                "holdout_max_drawdown": np.nan,
            })
        except Exception as exc:
            logger.warning("AutoTuner: param combo %d failed: %s", i, exc)

    if not results:
        raise RuntimeError("AutoTuner: no parameter combinations completed successfully.")

    if has_holdout:
        finalists = sorted(
            results,
            key=lambda row: (
                row["search_sharpe"],
                row["search_ann_return"],
                row["search_max_drawdown"],
            ),
            reverse=True,
        )[: min(max(1, holdout_finalists), len(results))]

        for finalist in finalists:
            rets, _ = run_replay(
                df_holdout,
                price_lookup,
                **shared_kwargs,
                min_score=finalist["min_score"],
                stop_loss_floor=finalist["stop_loss_floor"],
                take_profit=finalist["take_profit"],
                max_sector_pct=finalist["max_sector_pct"],
                max_position_pct=finalist["max_position_pct"],
                cash_floor=finalist["cash_floor"],
                max_gross_exposure=finalist["max_gross_exposure"],
                target_volatility=finalist["target_volatility"],
                label=(
                    "holdout_"
                    f"{finalist['min_score']:.2f}_"
                    f"{finalist['stop_loss_floor']:.2f}_"
                    f"{finalist['take_profit']:.2f}_"
                    f"{finalist['max_sector_pct']:.2f}_"
                    f"{finalist['max_position_pct']:.2f}_"
                    f"{finalist['cash_floor']:.2f}_"
                    f"{finalist['max_gross_exposure']:.2f}_"
                    f"{finalist['target_volatility']:.2f}"
                ),
            )
            m = compute_metrics(rets, label=str(finalist))
            finalist["holdout_total_return"] = m["total_return"]
            finalist["holdout_ann_return"] = m["ann_return"]
            finalist["holdout_sharpe"] = m["sharpe"]
            finalist["holdout_max_drawdown"] = m["max_drawdown"]

        best_row = max(
            finalists,
            key=lambda row: (
                row["holdout_sharpe"],
                row["holdout_ann_return"],
                row["holdout_max_drawdown"],
            ),
        )
        best_metric_name = "holdout_sharpe"
    else:
        best_row = max(
            results,
            key=lambda row: (
                row["search_sharpe"],
                row["search_ann_return"],
                row["search_max_drawdown"],
            ),
        )
        best_metric_name = "search_sharpe"

    best_params = {
        "min_score": best_row["min_score"],
        "stop_loss_floor": best_row["stop_loss_floor"],
        "take_profit": best_row["take_profit"],
        "max_sector_pct": best_row["max_sector_pct"],
        "max_position_pct": best_row["max_position_pct"],
        "cash_floor": best_row["cash_floor"],
        "max_gross_exposure": best_row["max_gross_exposure"],
        "target_volatility": best_row["target_volatility"],
    }
    best_metric = float(best_row[best_metric_name])

    logger.info(
        "AutoTuner: best params - min_score=%.2f  stop=%.2f  tp=%.2f  max_sector=%.2f  "
        "max_pos=%.2f  cash_floor=%.2f  gross=%.2f  target_vol=%.2f  %s=%.3f",
        best_params["min_score"],
        best_params["stop_loss_floor"],
        best_params["take_profit"],
        best_params["max_sector_pct"],
        best_params["max_position_pct"],
        best_params["cash_floor"],
        best_params["max_gross_exposure"],
        best_params["target_volatility"],
        best_metric_name,
        best_metric,
    )

    _write_config_key("min_score", f"{best_params['min_score']:.2f}", config_path)
    _write_config_key("stop_loss", f"{best_params['stop_loss_floor']:.2f}", config_path)
    _write_config_key("take_profit", f"{best_params['take_profit']:.2f}", config_path)
    _write_config_key("max_sector", f"{best_params['max_sector_pct']:.2f}", config_path)
    _write_config_key("max_position_pct", f"{best_params['max_position_pct']:.2f}", config_path)
    _write_config_key("cash_floor", f"{best_params['cash_floor']:.2f}", config_path)
    _write_config_key("max_gross_exposure", f"{best_params['max_gross_exposure']:.2f}", config_path)
    _write_config_key("target_volatility", f"{best_params['target_volatility']:.2f}", config_path)

    Path("plots").mkdir(exist_ok=True)
    sort_col = "holdout_sharpe" if has_holdout else "search_sharpe"
    pd.DataFrame(results).sort_values(sort_col, ascending=False, na_position="last").to_csv(
        "plots/param_tuning.csv",
        index=False,
    )
    logger.info("AutoTuner: full parameter results saved -> plots/param_tuning.csv")

    return best_params


def tune_rl_mode(
    df_features: pd.DataFrame,
    price_lookup: pd.DataFrame,
    initial_cash: float = 10_000.0,
    replay_years: int = 2,
    save_dir: str = "models",
    config_path: str = "broker.config",
) -> bool:
    """
    Run the RL ablation gate. If screener_rl beats heuristics_only by the
    required margin, sets rl_enabled = true in broker.config.
    """
    from broker.replay import run_ablation, _check_ablation_gate

    checkpoint = _best_checkpoint(save_dir)
    if checkpoint is None:
        logger.warning("AutoTuner: no checkpoint found - keeping rl_enabled = false")
        _write_config_key("rl_enabled", "false", config_path)
        return False

    logger.info("AutoTuner: running RL ablation gate with checkpoint %s...", checkpoint)

    cfg = _read_config(config_path)
    df_search, df_holdout, has_holdout = _split_replay_holdout(
        df_features,
        replay_years=replay_years,
    )
    df_gate = df_holdout if has_holdout else df_search
    if has_holdout:
        logger.info("AutoTuner: RL gate will run on the holdout window.")
    else:
        logger.info("AutoTuner: RL gate falling back to the full tuning window.")

    try:
        report_df = run_ablation(
            df_gate,
            price_lookup,
            checkpoint_path=checkpoint,
            initial_cash=initial_cash,
            replay_years=replay_years,
            max_positions=_config_int(cfg, "max_positions", 20),
            min_score=_config_float(cfg, "min_score", 0.60),
            stop_loss_floor=_config_float(cfg, "stop_loss", 0.07),
            take_profit=_config_float(cfg, "take_profit", 0.45),
            partial_profit_pct=_config_float(cfg, "partial_profit", 0.20),
            penny_pct=_config_float(cfg, "penny_pct", 0.20),
            max_sector_pct=_config_float(cfg, "max_sector", 0.40),
            max_pair_correlation=_config_float(cfg, "max_correlation", 0.80),
            weak_theme_min_positions=_config_int(cfg, "weak_theme_min_positions", 2),
            weak_theme_return_threshold=_config_float(cfg, "weak_theme_return_threshold", -0.03),
            weak_theme_penalty_mult=_config_float(cfg, "weak_theme_penalty_mult", 0.50),
            avoid_earnings_days=_config_int(cfg, "avoid_earnings", 3),
            max_position_pct=_config_float(cfg, "max_position_pct", 0.10),
            max_gross_exposure=_config_float(cfg, "max_gross_exposure", 0.95),
            cash_floor=_config_float(cfg, "cash_floor", 0.05),
            target_volatility=_config_float(cfg, "target_volatility", 0.15),
            vol_lookback=_config_int(cfg, "vol_lookback", 20),
            rl_phase=_config_int(cfg, "rl_phase", 1),
            rl_exit_threshold=_config_float(cfg, "rl_exit_threshold", 0.30),
            rl_conviction_drop=_config_float(cfg, "rl_conviction_drop", 0.20),
            rl_min_score=_config_float(cfg, "rl_min_score", 0.0),
        )
        gate = _check_ablation_gate(report_df)
    except Exception as exc:
        logger.error("AutoTuner: ablation failed - keeping rl_enabled = false: %s", exc)
        _write_config_key("rl_enabled", "false", config_path)
        return False

    if gate == "PASSED":
        _write_config_key("rl_enabled", "true", config_path)
        _write_config_key("rl_checkpoint_path", checkpoint, config_path)
        logger.info("AutoTuner: RL ENABLED - ablation gate passed.")
        return True

    _write_config_key("rl_enabled", "false", config_path)
    logger.info("AutoTuner: RL DISABLED - ablation gate failed.")
    return False


def run_autotuner(
    initial_cash: float = 10_000.0,
    replay_years: int = 2,
    save_dir: str = "models",
    config_path: str = "broker.config",
) -> None:
    """
    Full auto-tune pass: parameter optimization then RL gate decision.
    Called by the scheduler after each weekly fine-tune.
    """
    _setup_autotuner_logging()
    logger.info("=" * 60)
    logger.info("AutoTuner run started: %s", datetime.now().strftime("%Y-%m-%d %H:%M"))
    logger.info("=" * 60)

    try:
        from broker.replay import _build_price_lookup
        from pipeline.data import load_master

        cfg = _read_config(config_path)
        top_n = _config_int(cfg, "top_n", 500)
        logger.info("AutoTuner: loading market data...")
        df_features = load_master(top_n=top_n)
        price_lookup = _build_price_lookup()

        tune_parameters(
            df_features,
            price_lookup,
            initial_cash=initial_cash,
            replay_years=replay_years,
            config_path=config_path,
        )

        tune_rl_mode(
            df_features,
            price_lookup,
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
    if not any(
        isinstance(h, logging.FileHandler) and "autotuner" in h.baseFilename
        for h in logger.handlers
    ):
        logger.addHandler(handler)
