"""
Backtesting and performance metrics.
Compares the trained policy against:
  - Buy-and-hold SPY (benchmark)
  - Equal-weight portfolio
"""

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from pipeline.environment import PortfolioEnv
from pipeline.model import PortfolioTransformer


# ── Metrics ──────────────────────────────────────────────────────────────────

def sharpe_ratio(rets: np.ndarray, periods: int = 252) -> float:
    if len(rets) < 2:
        return 0.0
    return float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(periods))


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd   = (equity - peak) / (peak + 1e-9)
    return float(dd.min())


def calmar_ratio(rets: np.ndarray, periods: int = 252) -> float:
    equity = np.cumprod(1 + rets)
    mdd    = abs(max_drawdown(equity))
    ann_r  = float(np.prod(1 + rets) ** (periods / len(rets)) - 1)
    return ann_r / (mdd + 1e-9)


def sortino_ratio(rets: np.ndarray, periods: int = 252) -> float:
    downside = rets[rets < 0]
    if len(downside) < 2:
        return 0.0
    return float(rets.mean() / (downside.std() + 1e-9) * np.sqrt(periods))


def compute_metrics(rets: np.ndarray, label: str = "") -> dict:
    equity = np.cumprod(1 + rets)
    return {
        "label":        label,
        "total_return": float(equity[-1] - 1),
        "ann_return":   float(np.prod(1 + rets) ** (252 / len(rets)) - 1),
        "sharpe":       sharpe_ratio(rets),
        "sortino":      sortino_ratio(rets),
        "max_drawdown": max_drawdown(equity),
        "calmar":       calmar_ratio(rets),
        "win_rate":     float((rets > 0).mean()),
    }


# ── Backtest runner ───────────────────────────────────────────────────────────

def run_backtest(
    model: PortfolioTransformer,
    df_test: pd.DataFrame,
    asset_list: list[str],
    device: torch.device,
    spy_rets: np.ndarray = None,
    save_plot: str = "plots/backtest.png",
) -> dict:
    """
    Run the policy on df_test and compute full metrics.
    Optionally compare against SPY and equal-weight.
    """
    env  = PortfolioEnv(df_test, asset_list)
    obs, _ = env.reset()
    done = False

    policy_rets   = []
    dates_visited = []
    weight_history = []

    model.eval()
    with torch.no_grad():
        while not done:
            obs_t   = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            weights = model.get_weights(obs_t).squeeze(0).cpu().numpy()
            obs, _, terminated, truncated, info = env.step(weights)
            policy_rets.append(info["port_ret"])
            weight_history.append(weights)
            done = terminated or truncated

    policy_rets = np.array(policy_rets)

    # Equal-weight baseline
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

    results = {
        "policy":       compute_metrics(policy_rets, "Policy"),
        "equal_weight": compute_metrics(ew_rets,     "Equal-Weight"),
    }

    if spy_rets is not None:
        n = min(len(spy_rets), len(policy_rets))
        results["spy"] = compute_metrics(spy_rets[:n], "SPY")

    # ── Print summary ────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print(f"{'Metric':<20} {'Policy':>12} {'Equal-Weight':>14}" +
          (f" {'SPY':>10}" if spy_rets is not None else ""))
    print("-"*65)
    for key in ["total_return", "ann_return", "sharpe", "sortino", "max_drawdown", "calmar", "win_rate"]:
        row = f"{key:<20}"
        for strat in (["policy", "equal_weight"] + (["spy"] if spy_rets is not None else [])):
            val = results[strat][key]
            row += f" {val:>12.3f}" if "return" not in key else f" {val:>12.2%}"
        print(row)
    print("="*65)

    # ── Plot ─────────────────────────────────────────────────────────────────
    _plot_equity_curves(policy_rets, ew_rets, spy_rets, save_plot)

    return results


def _plot_equity_curves(policy_rets, ew_rets, spy_rets, save_path):
    import os
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})

    # Equity curves
    ax = axes[0]
    ax.plot(np.cumprod(1 + policy_rets), label="Policy",       color="#2196F3", linewidth=1.5)
    ax.plot(np.cumprod(1 + ew_rets),     label="Equal-Weight", color="#FF9800", linewidth=1.2, linestyle="--")
    if spy_rets is not None:
        n = min(len(spy_rets), len(policy_rets))
        ax.plot(np.cumprod(1 + spy_rets[:n]), label="SPY", color="#4CAF50", linewidth=1.2, linestyle=":")
    ax.set_title("Portfolio Equity Curves", fontsize=13)
    ax.set_ylabel("Portfolio Value")
    ax.legend()
    ax.grid(alpha=0.3)

    # Drawdown
    ax2 = axes[1]
    equity = np.cumprod(1 + policy_rets)
    peak   = np.maximum.accumulate(equity)
    dd     = (equity - peak) / (peak + 1e-9)
    ax2.fill_between(range(len(dd)), dd, 0, color="#F44336", alpha=0.4, label="Drawdown")
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Trading Days")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {save_path}")


# ── Load checkpoint ───────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: torch.device) -> PortfolioTransformer:
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = PortfolioTransformer(**ckpt["model_cfg"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"  Fold={ckpt['fold']} | Steps={ckpt['steps']} | Val Sharpe={ckpt['val_sharpe']:.3f}")
    return model
