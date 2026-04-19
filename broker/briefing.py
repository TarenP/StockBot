"""
Daily Briefing
==============
Produces a clean, plain-English action list at the end of every Broker.py run.
This is the primary human-facing output — everything else is logged detail.

Format:
  - TODAY'S ACTIONS: what to do with your real portfolio right now
  - PAPER PORTFOLIO: how the simulated portfolio is performing vs SPY
  - POSITIONS: current holdings with P&L and stop/target levels
  - SHADOW RECOMMENDATION: what the evolutionary optimizer suggests
"""

from __future__ import annotations

import json
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd

JOURNAL_PATH = Path("broker/state/journal.jsonl")
EQUITY_PATH  = Path("broker/state/equity_curve.csv")
SHADOW_PATH  = Path("broker/state/shadows.json")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pct(v: float) -> str:
    return f"{v:+.1%}"


def _spy_return_since_start() -> float | None:
    """Compute SPY total return since the first equity curve entry."""
    try:
        eq = pd.read_csv(EQUITY_PATH, parse_dates=["time"])
        spy_col = eq["spy_price"].dropna()
        if len(spy_col) < 2:
            return None
        return float(spy_col.iloc[-1] / spy_col.iloc[0] - 1)
    except Exception:
        return None


def _load_shadow_recommendation() -> str | None:
    """Return a one-line advisory from the shadow population if available."""
    try:
        state = json.loads(SHADOW_PATH.read_text())
        population = state.get("population", [])
        validated  = [g for g in population if g.get("validated") and not g.get("is_baseline")]
        baseline   = next((g for g in population if g.get("is_baseline")), {})
        if not validated:
            return None

        best = max(validated, key=lambda g: float(g.get("sharpe", -99)))
        baseline_sharpe = float(baseline.get("sharpe", 0.0))
        best_sharpe     = float(best.get("sharpe", -99))

        if best_sharpe <= baseline_sharpe + 0.05:
            return None

        parts = []
        if abs(float(best.get("min_score", 0)) - float(baseline.get("min_score", 0))) > 0.01:
            parts.append(f"min_score -> {best['min_score']:.2f}")
        if abs(float(best.get("stop_loss", 0)) - float(baseline.get("stop_loss", 0))) > 0.005:
            parts.append(f"stop_loss -> {best['stop_loss']:.2f}")
        if abs(float(best.get("take_profit", 0)) - float(baseline.get("take_profit", 0))) > 0.01:
            parts.append(f"take_profit -> {best['take_profit']:.2f}")

        if not parts:
            return None

        return (
            f"Sharpe {best_sharpe:.3f} vs baseline {baseline_sharpe:.3f} | "
            + ", ".join(parts)
            + " | run with --approve-promotion to apply"
        )
    except Exception:
        return None


# ── Main briefing ─────────────────────────────────────────────────────────────

def print_daily_briefing(
    decisions: list,
    portfolio,
    executed: list,
) -> None:
    """
    Print the daily briefing to stdout. Called at the end of every broker cycle.

    Parameters
    ----------
    decisions : list[Decision]
        All decisions generated this cycle (including unexecuted ones).
    portfolio : Portfolio
        Current portfolio state after execution.
    executed : list[Decision]
        Decisions that were actually executed this cycle.
    """
    now = datetime.now().strftime("%A, %B %-d %Y")
    width = 62

    print(f"\n{'='*width}")
    print(f"  DAILY BRIEFING  —  {now}")
    print(f"{'='*width}")

    # ── Today's actions ───────────────────────────────────────────────────────
    sells   = [d for d in executed if d.action in ("SELL", "SELL_PARTIAL")]
    buys    = [d for d in executed if d.action == "BUY"]
    no_action = not sells and not buys

    print(f"\n  TODAY'S ACTIONS")
    print(f"  {'-'*58}")

    if no_action:
        print("  No trades today — portfolio unchanged.")
        print("  (Nothing cleared the signal threshold or triggered an exit)")
    else:
        for d in sells:
            pct_label = ""
            if d.action == "SELL_PARTIAL":
                pct_label = " (50% partial)"
            reason_short = d.reason.split("|")[0].strip()
            print(f"  SELL  {d.ticker:<6}  {d.shares:.2f} shares @ ${d.price:.2f}{pct_label}")
            print(f"        Reason: {reason_short}")

        for d in buys:
            reason_short = d.reason.split("|")[0].strip()
            print(f"  BUY   {d.ticker:<6}  {d.shares:.2f} shares @ ${d.price:.2f}")
            print(f"        Reason: {reason_short}")

    # ── Current positions ─────────────────────────────────────────────────────
    print(f"\n  CURRENT POSITIONS")
    print(f"  {'-'*58}")

    if not portfolio.positions:
        print("  No open positions.")
    else:
        print(f"  {'Ticker':<8} {'Shares':>7}  {'Price':>8}  {'Value':>9}  {'P&L':>8}  {'Since':>6}")
        print(f"  {'-'*56}")
        for ticker, pos in sorted(portfolio.positions.items()):
            price    = pos.get("last_price", 0.0)
            shares   = pos.get("shares", 0.0)
            cost     = pos.get("avg_cost", price)
            value    = shares * price
            pnl_pct  = (price - cost) / cost if cost > 0 else 0.0
            pnl_str  = _pct(pnl_pct)
            print(f"  {ticker:<8} {shares:>7.2f}  ${price:>7.2f}  ${value:>8,.0f}  {pnl_str:>8}")

    # ── Portfolio summary ─────────────────────────────────────────────────────
    print(f"\n  PAPER PORTFOLIO PERFORMANCE")
    print(f"  {'-'*58}")

    total_ret = portfolio.total_return
    equity    = portfolio.equity
    cash_pct  = portfolio.cash / equity if equity > 0 else 0

    print(f"  Equity:       ${equity:>10,.2f}")
    print(f"  Cash:         ${portfolio.cash:>10,.2f}  ({cash_pct:.0%} of portfolio)")
    print(f"  Total return: {_pct(total_ret):>10}")

    spy_ret = _spy_return_since_start()
    if spy_ret is not None:
        alpha = total_ret - spy_ret
        beats = "YES" if total_ret > spy_ret else "NO"
        print(f"  SPY return:   {_pct(spy_ret):>10}  (same period)")
        print(f"  Alpha vs SPY: {_pct(alpha):>10}  (beats SPY: {beats})")

    # ── Shadow recommendation ─────────────────────────────────────────────────
    rec = _load_shadow_recommendation()
    if rec:
        print(f"\n  PARAMETER RECOMMENDATION (advisory)")
        print(f"  {'-'*58}")
        print(f"  {rec}")

    print(f"\n{'='*width}\n")


def print_watchlist(decisions: list, portfolio) -> None:
    """
    Print stocks the system is watching but not yet buying.
    Useful for manual monitoring.
    """
    # Decisions that were generated but not executed (e.g. blocked by risk/sector)
    # We can't easily distinguish these here, so we show the top screener candidates
    # from the last journal entry instead.
    pass   # placeholder — extend if needed
