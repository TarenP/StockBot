"""
Benchmark module — SPY comparison as a first-class metric.

Provides:
  - fetch_spy_returns(): get SPY daily returns for a date range
  - benchmark_vs_spy():  compute all relative metrics
  - full_metrics():      extended compute_metrics with SPY-relative stats
  - print_benchmark_report(): formatted comparison table
  - plot_benchmark():    equity curves + drawdown + rolling relative perf
"""

import logging
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


@contextmanager
def _quiet():
    with open(os.devnull, "w") as dn:
        old = sys.stderr; sys.stderr = dn
        try: yield
        finally: sys.stderr = old


# ── SPY data fetcher ──────────────────────────────────────────────────────────

def fetch_spy_returns(
    start: str | None = None,
    end:   str | None = None,
    n_days: int | None = None,
) -> pd.Series:
    """
    Fetch SPY daily returns from yfinance.

    Args:
        start:  start date string "YYYY-MM-DD"
        end:    end date string "YYYY-MM-DD"
        n_days: if start/end not given, fetch last n_days

    Returns:
        pd.Series of daily returns indexed by date.
    """
    import yfinance as yf

    if n_days and not start:
        end   = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
        start = (datetime.today() - timedelta(days=n_days + 10)).strftime("%Y-%m-%d")

    try:
        with _quiet():
            raw = yf.download("SPY", start=start, end=end,
                              auto_adjust=True, progress=False)
        if raw.empty:
            logger.warning("Could not fetch SPY data")
            return pd.Series(dtype=float)

        rets = raw["Close"].pct_change().dropna()
        rets.index = pd.to_datetime(rets.index).normalize()
        return rets
    except Exception as e:
        logger.warning(f"SPY fetch failed: {e}")
        return pd.Series(dtype=float)


# ── Core metrics ──────────────────────────────────────────────────────────────

def _sharpe(rets, periods=252):
    if len(rets) < 2: return 0.0
    return float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(periods))

def _sortino(rets, periods=252):
    down = rets[rets < 0]
    if len(down) < 2: return 0.0
    return float(rets.mean() / (down.std() + 1e-9) * np.sqrt(periods))

def _max_dd(rets):
    eq   = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(eq)
    return float(((eq - peak) / (peak + 1e-9)).min())

def _calmar(rets, periods=252):
    mdd   = abs(_max_dd(rets))
    ann_r = float(np.prod(1 + rets) ** (periods / len(rets)) - 1)
    return ann_r / (mdd + 1e-9)

def _ann_return(rets, periods=252):
    return float(np.prod(1 + rets) ** (periods / len(rets)) - 1)

def _volatility(rets, periods=252):
    return float(rets.std() * np.sqrt(periods))


def compute_metrics(rets: np.ndarray, label: str = "") -> dict:
    """Base metrics (no benchmark required)."""
    rets = np.asarray(rets)
    eq   = np.cumprod(1 + rets)
    return {
        "label":        label,
        "total_return": float(eq[-1] - 1),
        "ann_return":   _ann_return(rets),
        "volatility":   _volatility(rets),
        "sharpe":       _sharpe(rets),
        "sortino":      _sortino(rets),
        "max_drawdown": _max_dd(rets),
        "calmar":       _calmar(rets),
        "win_rate":     float((rets > 0).mean()),
    }


def benchmark_vs_spy(
    portfolio_rets: np.ndarray,
    spy_rets: np.ndarray,
    rf_daily: float = 0.05 / 252,
) -> dict:
    """
    Compute all SPY-relative metrics.
    Both arrays must be aligned (same dates, same length).
    """
    n = min(len(portfolio_rets), len(spy_rets))
    p = np.asarray(portfolio_rets[:n])
    s = np.asarray(spy_rets[:n])

    # Beta and alpha
    cov_matrix = np.cov(p, s)
    beta       = cov_matrix[0, 1] / (cov_matrix[1, 1] + 1e-9)
    alpha_daily = (p.mean() - rf_daily) - beta * (s.mean() - rf_daily)
    alpha_ann   = alpha_daily * 252

    # Information ratio
    active_rets = p - s
    ir = float(active_rets.mean() / (active_rets.std() + 1e-9) * np.sqrt(252))

    # Upside / downside capture
    up_mask   = s > 0
    down_mask = s < 0
    up_cap    = (p[up_mask].mean()   / (s[up_mask].mean()   + 1e-9)) if up_mask.any()   else np.nan
    down_cap  = (p[down_mask].mean() / (s[down_mask].mean() + 1e-9)) if down_mask.any() else np.nan

    # Tracking error
    tracking_error = float(active_rets.std() * np.sqrt(252))

    # Beats SPY?
    beats_total  = float(np.prod(1 + p)) > float(np.prod(1 + s))
    beats_sharpe = _sharpe(p) > _sharpe(s)

    return {
        "beta":             round(beta, 3),
        "alpha_ann":        round(alpha_ann, 4),
        "information_ratio":round(ir, 3),
        "upside_capture":   round(up_cap, 3)   if not np.isnan(up_cap)   else None,
        "downside_capture": round(down_cap, 3) if not np.isnan(down_cap) else None,
        "tracking_error":   round(tracking_error, 4),
        "beats_spy_return": beats_total,
        "beats_spy_sharpe": beats_sharpe,
        "active_return_ann":round(float(active_rets.mean() * 252), 4),
    }


def rolling_relative_performance(
    portfolio_rets: np.ndarray,
    spy_rets: np.ndarray,
    windows: list[int] = [63, 126, 252],   # ~3m, 6m, 12m
) -> dict[str, np.ndarray]:
    """Rolling outperformance vs SPY over multiple windows."""
    n = min(len(portfolio_rets), len(spy_rets))
    p = pd.Series(portfolio_rets[:n])
    s = pd.Series(spy_rets[:n])
    result = {}
    for w in windows:
        p_roll = (1 + p).rolling(w).apply(np.prod) - 1
        s_roll = (1 + s).rolling(w).apply(np.prod) - 1
        result[f"{w}d"] = (p_roll - s_roll).values
    return result


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_benchmark_report(
    portfolio_rets: np.ndarray,
    spy_rets: np.ndarray,
    ew_rets: np.ndarray | None = None,
    label: str = "Strategy",
):
    """Print a full benchmark comparison table."""
    n = min(len(portfolio_rets), len(spy_rets))
    p_metrics   = compute_metrics(portfolio_rets[:n], label)
    spy_metrics = compute_metrics(spy_rets[:n],       "SPY")
    rel         = benchmark_vs_spy(portfolio_rets, spy_rets)

    cols = ["Policy", "SPY"] + (["Equal-Weight"] if ew_rets is not None else [])
    all_m = [p_metrics, spy_metrics]
    if ew_rets is not None:
        all_m.append(compute_metrics(ew_rets[:n], "Equal-Weight"))

    print(f"\n{'='*72}")
    print(f"  Benchmark Report — {label} vs SPY")
    print(f"{'='*72}")
    header = f"  {'Metric':<22}"
    for c in cols:
        header += f" {c:>14}"
    print(header)
    print(f"  {'─'*68}")

    pct_keys = {"total_return", "ann_return", "volatility", "max_drawdown"}
    for key in ["total_return", "ann_return", "volatility", "sharpe", "sortino",
                "max_drawdown", "calmar", "win_rate"]:
        row = f"  {key:<22}"
        for m in all_m:
            val = m.get(key, 0)
            row += f" {val:>13.2%}" if key in pct_keys else f" {val:>14.3f}"
        print(row)

    print(f"\n  {'─'*68}")
    print(f"  SPY-Relative Metrics")
    print(f"  {'─'*68}")
    print(f"  {'Beta':<22} {rel['beta']:>14.3f}")
    print(f"  {'Alpha (ann)':<22} {rel['alpha_ann']:>13.2%}")
    print(f"  {'Information Ratio':<22} {rel['information_ratio']:>14.3f}")
    print(f"  {'Tracking Error':<22} {rel['tracking_error']:>13.2%}")
    if rel['upside_capture'] is not None:
        print(f"  {'Upside Capture':<22} {rel['upside_capture']:>14.3f}")
    if rel['downside_capture'] is not None:
        print(f"  {'Downside Capture':<22} {rel['downside_capture']:>14.3f}")
    print(f"  {'Beats SPY (return)':<22} {'YES' if rel['beats_spy_return'] else 'NO':>14}")
    print(f"  {'Beats SPY (Sharpe)':<22} {'YES' if rel['beats_spy_sharpe'] else 'NO':>14}")
    print(f"{'='*72}\n")


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_benchmark(
    portfolio_rets: np.ndarray,
    spy_rets: np.ndarray,
    ew_rets: np.ndarray | None = None,
    dates: list | None = None,
    save_path: str = "plots/benchmark.png",
    label: str = "Strategy",
):
    """
    4-panel chart:
      1. Equity curves (strategy vs SPY vs equal-weight)
      2. Strategy drawdown vs SPY drawdown
      3. Rolling 63-day relative performance vs SPY
      4. Rolling 252-day relative performance vs SPY
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    n = min(len(portfolio_rets), len(spy_rets))
    p = np.asarray(portfolio_rets[:n])
    s = np.asarray(spy_rets[:n])
    x = dates[:n] if dates else range(n)

    fig, axes = plt.subplots(4, 1, figsize=(14, 16),
                             gridspec_kw={"height_ratios": [3, 1.5, 1.5, 1.5]})
    fig.suptitle(f"{label} vs SPY Benchmark", fontsize=14, fontweight="bold")

    # ── 1. Equity curves ──────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(x, np.cumprod(1 + p), label=label,        color="#2196F3", lw=1.8)
    ax.plot(x, np.cumprod(1 + s), label="SPY",        color="#4CAF50", lw=1.4, ls="--")
    if ew_rets is not None:
        ew = np.asarray(ew_rets[:n])
        ax.plot(x, np.cumprod(1 + ew), label="Equal-Weight", color="#FF9800", lw=1.2, ls=":")
    ax.set_ylabel("Portfolio Value ($1 start)")
    ax.legend(); ax.grid(alpha=0.3)

    # ── 2. Drawdown comparison ────────────────────────────────────────────────
    ax2 = axes[1]
    def _dd_series(rets):
        eq   = np.cumprod(1 + rets)
        peak = np.maximum.accumulate(eq)
        return (eq - peak) / (peak + 1e-9)

    ax2.fill_between(x, _dd_series(p), 0, alpha=0.5, color="#F44336", label=f"{label} DD")
    ax2.plot(x, _dd_series(s), color="#4CAF50", lw=1.2, ls="--", label="SPY DD")
    ax2.set_ylabel("Drawdown")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    # ── 3. Rolling 63-day relative performance ────────────────────────────────
    ax3 = axes[2]
    roll = rolling_relative_performance(p, s, windows=[63])["63d"]
    ax3.bar(range(len(roll)), roll, color=np.where(roll >= 0, "#2196F3", "#F44336"), alpha=0.7)
    ax3.axhline(0, color="black", lw=0.8)
    ax3.set_ylabel("3-Month Relative Return")
    ax3.grid(alpha=0.3)

    # ── 4. Rolling 252-day relative performance ───────────────────────────────
    ax4 = axes[3]
    roll12 = rolling_relative_performance(p, s, windows=[252])["252d"]
    ax4.bar(range(len(roll12)), roll12,
            color=np.where(roll12 >= 0, "#2196F3", "#F44336"), alpha=0.7)
    ax4.axhline(0, color="black", lw=0.8)
    ax4.set_ylabel("12-Month Relative Return")
    ax4.set_xlabel("Trading Days")
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Benchmark plot saved → {save_path}")
