"""
Quick 1-year performance test.

Runs the broker replay over the last 12 months and prints a
side-by-side comparison vs SPY.

Usage:
    python test_performance.py
"""

import logging
import warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)  # suppress noise

import pandas as pd
import numpy as np

from pipeline.data import load_master
from broker.replay import _build_price_lookup, run_replay
from pipeline.benchmark import compute_metrics, benchmark_vs_spy, fetch_spy_returns

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading market data...")
df = load_master(top_n=750)

dates  = sorted(df.index.get_level_values("date").unique())
cutoff = pd.Timestamp(dates[-1]) - pd.DateOffset(years=1)
df_1yr = df[df.index.get_level_values("date") >= cutoff]

replay_dates = sorted(df_1yr.index.get_level_values("date").unique())
print(f"Replay period: {replay_dates[0].date()} to {replay_dates[-1].date()} "
      f"({len(replay_dates)} trading days)\n")

price_lookup = _build_price_lookup()

# ── Run replay ────────────────────────────────────────────────────────────────
print("Running broker replay (heuristics)...")
rets, trades = run_replay(
    df_1yr,
    price_lookup,
    strategy="heuristics_only",
    initial_cash=10_000.0,
    label="1yr-test",
)

# ── Fetch SPY ─────────────────────────────────────────────────────────────────
print("Fetching SPY returns...")
spy = fetch_spy_returns(
    start=replay_dates[0].strftime("%Y-%m-%d"),
    end=(replay_dates[-1] + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
)

if spy.empty:
    # Retry with a wider window
    print("  SPY fetch failed, retrying with wider window...")
    spy = fetch_spy_returns(n_days=400)

if spy.empty:
    print("  WARNING: Could not fetch SPY data — showing broker metrics only\n")
    spy_rets = np.zeros_like(rets)
else:
    spy_rets = spy.values[:len(rets)]
    if len(spy_rets) < len(rets):
        spy_rets = np.pad(spy_rets, (0, len(rets) - len(spy_rets)))
    print(f"  SPY: {float(np.prod(1 + spy_rets) - 1):+.2%} over period\n")

# ── Metrics ───────────────────────────────────────────────────────────────────
m   = compute_metrics(rets,     "Broker")
spy_m = compute_metrics(spy_rets, "SPY")
rel = benchmark_vs_spy(rets, spy_rets)

n_buys  = sum(1 for t in trades if t["action"] == "BUY")
n_sells = sum(1 for t in trades if t["action"] == "SELL")

print(f"\n{'='*60}")
print(f"  1-Year Backtest Results")
print(f"{'='*60}")
print(f"  {'Metric':<22} {'Broker':>12}  {'SPY':>10}")
print(f"  {'-'*48}")
for key, fmt in [
    ("total_return", ".2%"),
    ("ann_return",   ".2%"),
    ("sharpe",       ".3f"),
    ("sortino",      ".3f"),
    ("max_drawdown", ".2%"),
    ("win_rate",     ".2%"),
]:
    print(f"  {key:<22} {m[key]:>12{fmt}}  {spy_m[key]:>10{fmt}}")

print(f"  {'-'*48}")
print(f"  {'Beta':<22} {rel['beta']:>12.3f}")
print(f"  {'Alpha (ann)':<22} {rel['alpha_ann']:>12.2%}")
print(f"  {'Info Ratio':<22} {rel['information_ratio']:>12.3f}")
if rel['upside_capture']:
    print(f"  {'Upside Capture':<22} {rel['upside_capture']:>12.3f}")
if rel['downside_capture']:
    print(f"  {'Downside Capture':<22} {rel['downside_capture']:>12.3f}")
beats = "YES" if rel['beats_spy_return'] else "NO"
print(f"  {'Beats SPY':<22} {beats:>12}")
print(f"  {'-'*48}")
print(f"  Trades: {n_buys} buys, {n_sells} sells")
print(f"{'='*60}\n")
