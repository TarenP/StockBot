"""
Trade journal — logs every decision with full reasoning,
tracks daily equity curve, and prints performance reports.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

JOURNAL_PATH = Path("broker/state/journal.jsonl")
EQUITY_PATH  = Path("broker/state/equity_curve.csv")


def log_cycle(decisions: list, portfolio_equity: float, portfolio_cash: float):
    """Append a cycle's decisions to the journal."""
    JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "time":     datetime.now().isoformat(),
        "equity":   round(portfolio_equity, 2),
        "cash":     round(portfolio_cash, 2),
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

    # Append to equity curve
    eq_row = pd.DataFrame([{
        "time":   entry["time"],
        "equity": entry["equity"],
        "cash":   entry["cash"],
    }])
    if EQUITY_PATH.exists():
        eq_row.to_csv(EQUITY_PATH, mode="a", header=False, index=False)
    else:
        eq_row.to_csv(EQUITY_PATH, index=False)


def print_report(portfolio):
    """Print a full performance report."""
    print(portfolio.summary())

    if not EQUITY_PATH.exists():
        return

    eq = pd.read_csv(EQUITY_PATH, parse_dates=["time"])
    if len(eq) < 2:
        return

    initial = eq["equity"].iloc[0]
    current = eq["equity"].iloc[-1]
    peak    = eq["equity"].max()
    trough  = eq["equity"].min()
    dd      = (current - peak) / peak

    print(f"  Equity curve: {len(eq)} data points")
    print(f"  Peak equity:  ${peak:,.2f}")
    print(f"  Max drawdown: {dd:.2%}")
    print(f"  Total return: {(current/initial - 1):.2%}\n")


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
    print(f"  {'Time':<20} {'Act':<5} {'Ticker':<8} {'Shares':>7}  {'Price':>8}  {'Score':>6}")
    print(f"  {'─'*68}")
    for t in recent:
        print(
            f"  {t['time'][:19]:<20} {t['action']:<5} {t['ticker']:<8} "
            f"{t['shares']:>7.2f}  ${t['price']:>7.3f}  {t['score']:>6.3f}"
        )
        if t.get("reason"):
            print(f"    └─ {t['reason'][:70]}")
    print(f"{'='*75}\n")
