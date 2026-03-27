"""
Trade journal — logs every decision, tracks equity curve vs SPY,
and produces full benchmark performance reports.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

JOURNAL_PATH  = Path("broker/state/journal.jsonl")
EQUITY_PATH   = Path("broker/state/equity_curve.csv")
SPY_PATH      = Path("broker/state/spy_curve.csv")


# ── Cycle logging ─────────────────────────────────────────────────────────────

def log_cycle(
    decisions: list,
    portfolio_equity: float,
    portfolio_cash: float,
    spy_price: float | None = None,
):
    """Append a cycle's decisions and equity snapshot to the journal."""
    JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "time":      datetime.now().isoformat(),
        "equity":    round(portfolio_equity, 2),
        "cash":      round(portfolio_cash, 2),
        "spy_price": round(spy_price, 4) if spy_price else None,
        "decisions": [
            {
                "action": d.action,
                "ticker": d.ticker,
                "shares": round(d.shares, 4),
                "price":  round(d.price, 4),
                "score":  round(d.score, 4),
                "reason": d.reason,
            }
            for d in decisions
        ],
    }
    with open(JOURNAL_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

    # Equity curve
    eq_row = pd.DataFrame([{
        "time":      entry["time"],
        "equity":    entry["equity"],
        "cash":      entry["cash"],
        "spy_price": entry["spy_price"],
    }])
    if EQUITY_PATH.exists():
        eq_row.to_csv(EQUITY_PATH, mode="a", header=False, index=False)
    else:
        eq_row.to_csv(EQUITY_PATH, index=False)


def _fetch_current_spy_price() -> float | None:
    """Fetch latest SPY close price."""
    try:
        import yfinance as yf
        import os, sys
        from contextlib import contextmanager

        @contextmanager
        def _quiet():
            with open(os.devnull, "w") as dn:
                old = sys.stderr; sys.stderr = dn
                try: yield
                finally: sys.stderr = old

        with _quiet():
            raw = yf.download("SPY", period="5d", auto_adjust=True, progress=False)
        if not raw.empty:
            return float(raw["Close"].iloc[-1])
    except Exception:
        pass
    return None


# ── Performance reporting ─────────────────────────────────────────────────────

def print_report(portfolio, show_benchmark: bool = True):
    """Print full performance report including SPY benchmark."""
    print(portfolio.summary())

    if not EQUITY_PATH.exists():
        return

    eq = pd.read_csv(EQUITY_PATH, parse_dates=["time"])
    if len(eq) < 2:
        return

    initial = eq["equity"].iloc[0]
    current = eq["equity"].iloc[-1]
    peak    = eq["equity"].max()
    dd      = (current - peak) / peak

    print(f"  Equity curve: {len(eq)} data points")
    print(f"  Peak equity:  ${peak:,.2f}")
    print(f"  Max drawdown: {dd:.2%}")
    print(f"  Total return: {(current/initial - 1):.2%}")

    if not show_benchmark:
        return

    # ── SPY comparison ────────────────────────────────────────────────────────
    spy_col = eq["spy_price"].dropna()
    if len(spy_col) < 2:
        print("  SPY benchmark: not enough data yet (will appear after 2+ cycles)\n")
        return

    portfolio_rets = eq["equity"].pct_change().dropna().values
    spy_rets       = spy_col.pct_change().dropna().values
    n              = min(len(portfolio_rets), len(spy_rets))

    if n < 2:
        return

    from pipeline.benchmark import benchmark_vs_spy, compute_metrics
    p_m   = compute_metrics(portfolio_rets[:n], "Broker")
    spy_m = compute_metrics(spy_rets[:n],       "SPY")
    rel   = benchmark_vs_spy(portfolio_rets[:n], spy_rets[:n])

    print(f"\n  {'─'*55}")
    print(f"  {'Metric':<22} {'Broker':>12}  {'SPY':>10}")
    print(f"  {'─'*55}")
    for key, fmt in [
        ("total_return", ".2%"), ("ann_return", ".2%"),
        ("sharpe", ".3f"),       ("sortino", ".3f"),
        ("max_drawdown", ".2%"), ("win_rate", ".2%"),
    ]:
        pv = p_m[key]; sv = spy_m[key]
        print(f"  {key:<22} {pv:>12{fmt}}  {sv:>10{fmt}}")

    print(f"  {'─'*55}")
    print(f"  {'Beta':<22} {rel['beta']:>12.3f}")
    print(f"  {'Alpha (ann)':<22} {rel['alpha_ann']:>12.2%}")
    print(f"  {'Info Ratio':<22} {rel['information_ratio']:>12.3f}")
    if rel['upside_capture']:
        print(f"  {'Upside Capture':<22} {rel['upside_capture']:>12.3f}")
    if rel['downside_capture']:
        print(f"  {'Downside Capture':<22} {rel['downside_capture']:>12.3f}")
    beats = "✓ YES" if rel['beats_spy_return'] else "✗ NO"
    print(f"  {'Beats SPY':<22} {beats:>12}")
    print(f"  {'─'*55}\n")


def print_recent_trades(n: int = 20):
    """Print the last N trades from the journal."""
    if not JOURNAL_PATH.exists():
        print("No trades yet.")
        return

    entries = []
    with open(JOURNAL_PATH) as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except Exception:
                pass

    trades = []
    for entry in entries:
        for d in entry.get("decisions", []):
            trades.append({**d, "time": entry["time"], "equity": entry["equity"]})

    if not trades:
        print("No trades recorded yet.")
        return

    recent = trades[-n:]
    print(f"\n{'='*75}")
    print(f"  Recent Trades (last {len(recent)})")
    print(f"{'='*75}")
    print(f"  {'Time':<20} {'Act':<14} {'Ticker':<8} {'Shares':>7}  {'Price':>8}  {'Score':>6}")
    print(f"  {'─'*72}")
    for t in recent:
        print(
            f"  {t['time'][:19]:<20} {t['action']:<14} {t['ticker']:<8} "
            f"{t['shares']:>7.2f}  ${t['price']:>7.3f}  {t['score']:>6.3f}"
        )
        if t.get("reason"):
            print(f"    └─ {t['reason'][:72]}")
    print(f"{'='*75}\n")


def daily_integrity_check(portfolio) -> dict:
    """Run a daily sanity check and return a report dict."""
    spy_price = _fetch_current_spy_price()

    report = {
        "date":        datetime.today().date().isoformat(),
        "equity":      round(portfolio.equity, 2),
        "cash":        round(portfolio.cash, 2),
        "n_positions": len(portfolio.positions),
        "n_options":   len(portfolio.options.positions),
        "total_return":round(portfolio.total_return, 4),
        "spy_price":   spy_price,
        "cash_pct":    round(portfolio.cash / (portfolio.equity + 1e-9), 4),
        "options_reserved": round(portfolio.options.total_reserved_cash, 2),
    }

    # SPY comparison if we have history
    if EQUITY_PATH.exists():
        eq = pd.read_csv(EQUITY_PATH, parse_dates=["time"])
        spy_col = eq["spy_price"].dropna()
        if len(spy_col) >= 2 and spy_price:
            spy_start = float(spy_col.iloc[0])
            spy_ret   = (spy_price / spy_start) - 1
            report["spy_return_since_start"] = round(spy_ret, 4)
            report["alpha_vs_spy"] = round(portfolio.total_return - spy_ret, 4)
            report["beating_spy"]  = portfolio.total_return > spy_ret

    return report
