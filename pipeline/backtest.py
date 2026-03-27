"""
Backtesting and performance metrics.
Compares the trained policy against:
  - SPY (auto-fetched — mandatory benchmark)
  - Equal-weight portfolio
"""

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline.environment import PortfolioEnv
from pipeline.model       import PortfolioTransformer
from pipeline.benchmark   import (
    fetch_spy_returns, compute_metrics, benchmark_vs_spy,
    print_benchmark_report, plot_benchmark,
)


# ── Backtest runner ───────────────────────────────────────────────────────────

def run_backtest(
    model: PortfolioTransformer,
    df_test: pd.DataFrame,
    asset_list: list[str],
    device: torch.device,
    spy_rets: np.ndarray | None = None,   # auto-fetched if None
    save_plot: str = "plots/backtest.png",
) -> dict:
    """
    Run the policy on df_test and compute full metrics vs SPY and equal-weight.
    SPY is fetched automatically if not provided.
    """
    # ── Run policy ────────────────────────────────────────────────────────────
    env    = PortfolioEnv(df_test, asset_list)
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

    # ── Equal-weight baseline ─────────────────────────────────────────────────
    ew_env = PortfolioEnv(df_test, asset_list)
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
    if spy_rets is None:
        dates = sorted(df_test.index.get_level_values("date").unique())
        if len(dates) >= 2:
            start = (pd.Timestamp(dates[0]) - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
            end   = (pd.Timestamp(dates[-1]) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
            spy_series = fetch_spy_returns(start=start, end=end)
            if not spy_series.empty and len(spy_series) >= 2:
                # Align SPY to the same number of trading days as policy
                spy_rets = spy_series.values[:len(policy_rets)]
            else:
                print("  Warning: could not fetch SPY — benchmark comparison unavailable")

    # ── Metrics ───────────────────────────────────────────────────────────────
    results = {
        "policy":       compute_metrics(policy_rets, "Policy"),
        "equal_weight": compute_metrics(ew_rets,     "Equal-Weight"),
    }
    if spy_rets is not None:
        n = min(len(spy_rets), len(policy_rets))
        if n >= 2:
            results["spy"]    = compute_metrics(spy_rets[:n], "SPY")
            results["vs_spy"] = benchmark_vs_spy(policy_rets[:n], spy_rets[:n])

    # ── Print report ──────────────────────────────────────────────────────────
    print_benchmark_report(
        policy_rets,
        spy_rets if spy_rets is not None else np.zeros_like(policy_rets),
        ew_rets=ew_rets,
        label="RL Policy",
    )

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_benchmark(
        policy_rets,
        spy_rets if spy_rets is not None else np.zeros_like(policy_rets),
        ew_rets=ew_rets,
        save_path=save_plot,
        label="RL Policy",
    )

    return results


# ── Load checkpoint ───────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: torch.device) -> PortfolioTransformer:
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = PortfolioTransformer(**ckpt["model_cfg"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"  Fold={ckpt['fold']} | Steps={ckpt['steps']} | Val Sharpe={ckpt['val_sharpe']:.3f}")
    return model
