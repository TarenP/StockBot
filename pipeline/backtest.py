"""
Backtesting and performance metrics.
Compares the trained policy against:
  - SPY (auto-fetched — mandatory benchmark)
  - Equal-weight portfolio
"""

import logging
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline.environment import PortfolioEnv
from pipeline.features    import FEATURE_COLS
from pipeline.model       import PortfolioTransformer
from pipeline.benchmark   import (
    align_return_series, fetch_spy_returns, compute_metrics, benchmark_vs_spy,
    print_benchmark_report, plot_benchmark,
)

logger = logging.getLogger(__name__)


# ── Backtest runner ───────────────────────────────────────────────────────────

def run_backtest(
    model: PortfolioTransformer,
    df_test: pd.DataFrame,
    asset_list: list[str],
    device: torch.device,
    spy_rets: np.ndarray | None = None,   # auto-fetched if None
    save_plot: str = "plots/backtest.png",
    ckpt_n_features: int | None = None,   # from checkpoint model_cfg
) -> dict:
    """
    Run the policy on df_test and compute full metrics vs SPY and equal-weight.
    SPY is fetched automatically if not provided.

    ckpt_n_features: if the checkpoint was trained with fewer features than the
    current FEATURE_COLS, pass the checkpoint's n_features here so the env and
    model see the same feature slice.
    """
    # ── Resolve feature columns to match checkpoint ───────────────────────────
    if ckpt_n_features is not None and ckpt_n_features != len(FEATURE_COLS):
        feature_cols = FEATURE_COLS[:ckpt_n_features]
        logger.info(
            "Backtest: checkpoint trained with %d features; current FEATURE_COLS "
            "has %d. Slicing to first %d features.",
            ckpt_n_features, len(FEATURE_COLS), ckpt_n_features,
        )
    else:
        feature_cols = [col for col in FEATURE_COLS if col in df_test.columns]

    # ── Run policy ────────────────────────────────────────────────────────────
    env    = PortfolioEnv(df_test, asset_list, feature_cols=feature_cols)
    obs, _ = env.reset()
    done   = False
    policy_rets = []

    model.eval()
    with torch.no_grad():
        while not done:
            obs_t   = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            weights = model.get_weights(obs_t).squeeze(0).cpu().numpy()
            obs, _, terminated, truncated, info = env.step(weights)
            policy_rets.append(info["port_ret"])
            done = terminated or truncated
    policy_rets = np.array(policy_rets)
    return_dates = pd.DatetimeIndex(env.dates[env.lookback:env.lookback + len(policy_rets)])

    # ── Equal-weight baseline ─────────────────────────────────────────────────
    ew_env = PortfolioEnv(df_test, asset_list, feature_cols=feature_cols)
    obs, _ = ew_env.reset()
    done   = False
    ew_rets = []
    ew_w    = np.ones(len(asset_list) + 1) / (len(asset_list) + 1)
    while not done:
        obs, _, terminated, truncated, info = ew_env.step(ew_w)
        ew_rets.append(info["port_ret"])
        done = terminated or truncated
    ew_rets = np.array(ew_rets)

    # ── Fetch SPY (mandatory) ─────────────────────────────────────────────────
    spy_series = None
    if spy_rets is not None:
        spy_rets = np.asarray(spy_rets, dtype=float)
        spy_series = pd.Series(
            spy_rets[:len(return_dates)],
            index=return_dates[:len(spy_rets)],
            name="benchmark",
        )

    if spy_rets is None:
        dates = sorted(df_test.index.get_level_values("date").unique())
        if len(dates) >= 2:
            start = (pd.Timestamp(dates[0]) - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
            end   = (pd.Timestamp(dates[-1]) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
            fetched_spy = fetch_spy_returns(start=start, end=end)
            if not fetched_spy.empty and len(fetched_spy) >= 2:
                spy_series = fetched_spy
            else:
                print("  Warning: could not fetch SPY — benchmark comparison unavailable")

    aligned = align_return_series(
        policy_rets,
        return_dates,
        benchmark_rets=spy_series,
        extra_series={"equal_weight": ew_rets},
    )
    policy_aligned = aligned["portfolio"].to_numpy(dtype=float)
    ew_aligned = (
        aligned["equal_weight"].to_numpy(dtype=float)
        if "equal_weight" in aligned.columns else None
    )
    spy_aligned = (
        aligned["benchmark"].to_numpy(dtype=float)
        if "benchmark" in aligned.columns else None
    )

    # ── Metrics ───────────────────────────────────────────────────────────────
    results = {
        "policy":       compute_metrics(policy_aligned, "Policy"),
        "equal_weight": compute_metrics(ew_aligned,     "Equal-Weight"),
    }
    if spy_aligned is not None:
        n = min(len(spy_aligned), len(policy_aligned))
        if n >= 2:
            results["spy"]    = compute_metrics(spy_aligned[:n], "SPY")
            results["vs_spy"] = benchmark_vs_spy(policy_aligned[:n], spy_aligned[:n])

    # ── Print report ──────────────────────────────────────────────────────────
    print_benchmark_report(
        policy_aligned,
        spy_aligned,
        ew_rets=ew_aligned,
        label="RL Policy",
    )

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_benchmark(
        policy_aligned,
        spy_aligned,
        ew_rets=ew_aligned,
        save_path=save_plot,
        label="RL Policy",
    )

    return results


# ── Load checkpoint ───────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: torch.device) -> tuple[PortfolioTransformer, int]:
    """
    Load a PortfolioTransformer checkpoint.
    Returns (model, n_features) so callers can slice feature columns correctly.
    """
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = ckpt.get("model_cfg", {})
    model = PortfolioTransformer(**model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    n_features = model_cfg.get("n_features", len(FEATURE_COLS))
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"  Fold={ckpt['fold']} | Steps={ckpt['steps']} | Val Sharpe={ckpt['val_sharpe']:.3f}")
    print(f"  n_features={n_features} (current FEATURE_COLS={len(FEATURE_COLS)})")
    return model, n_features
