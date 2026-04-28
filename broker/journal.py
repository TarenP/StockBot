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
PORTFOLIO_HISTORY_PATH = Path("broker/state/portfolio_history.jsonl")
CAP_IMPACT_SUMMARY_PATH = Path("broker/state/cap_impact_summary.json")
PERFORMANCE_ATTRIBUTION_PATH = Path("broker/state/performance_attribution.json")
PARITY_REPORT_PATH = Path("broker/state/replay_live_parity.json")


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


def _max_drawdown_from_equity(equity: pd.Series) -> float:
    equity = pd.Series(equity, dtype=float).dropna()
    if equity.empty:
        return 0.0
    peaks = equity.cummax()
    drawdowns = (equity - peaks) / (peaks + 1e-9)
    return float(drawdowns.min())


def _align_equity_and_spy_returns(eq: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    frame = eq.copy()
    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce").dt.tz_convert(None)
    frame = frame.dropna(subset=["time"]).sort_values("time")
    frame = frame.set_index("time")

    equity = frame["equity"].astype(float)
    spy_prices = frame["spy_price"].dropna().astype(float)

    portfolio_rets = equity.pct_change(fill_method=None).dropna().rename("portfolio")
    spy_rets = spy_prices.pct_change(fill_method=None).dropna().rename("benchmark")

    aligned = pd.concat([portfolio_rets, spy_rets], axis=1, join="inner").dropna()
    if aligned.empty:
        return np.array([], dtype=float), np.array([], dtype=float)
    return (
        aligned["portfolio"].to_numpy(dtype=float),
        aligned["benchmark"].to_numpy(dtype=float),
    )


# ── Performance reporting ─────────────────────────────────────────────────────

def _load_json_file(path: Path) -> dict:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return {}


def _latest_jsonl_row(path: Path) -> dict:
    try:
        if not path.exists():
            return {}
        last = ""
        with path.open() as fh:
            for line in fh:
                if line.strip():
                    last = line
        return json.loads(last) if last else {}
    except Exception:
        return {}


def _latest_eod_status(eq: pd.DataFrame | None) -> dict:
    if eq is None or eq.empty:
        return {}

    frame = eq.copy()
    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce").dt.tz_convert(None)
    frame = frame.dropna(subset=["time"]).sort_values("time")
    if frame.empty:
        return {}

    frame["date"] = frame["time"].dt.date
    eod = (
        frame.groupby("date", sort=True)
        .tail(1)
        .sort_values("time")
        .reset_index(drop=True)
    )
    latest = eod.iloc[-1]
    previous = eod.iloc[-2] if len(eod) >= 2 else None
    eod_return = None
    if previous is not None:
        prev_equity = float(previous["equity"])
        if prev_equity > 0:
            eod_return = float(float(latest["equity"]) / prev_equity - 1.0)

    same_day = frame[frame["date"] == latest["date"]]
    intraday_return = None
    if len(same_day) >= 2:
        first_equity = float(same_day.iloc[0]["equity"])
        if first_equity > 0:
            intraday_return = float(float(same_day.iloc[-1]["equity"]) / first_equity - 1.0)

    return {
        "latest_time": latest["time"],
        "latest_date": latest["date"],
        "latest_equity": float(latest["equity"]),
        "latest_cash": float(latest.get("cash", 0.0)),
        "previous_eod_equity": (
            float(previous["equity"]) if previous is not None else None
        ),
        "eod_return": eod_return,
        "intraday_return": intraday_return,
        "n_eod_points": int(len(eod)),
    }


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):+.2%}"


def _fmt_weight(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2%}"


def _print_status_snapshot(portfolio, eq: pd.DataFrame | None) -> None:
    eod = _latest_eod_status(eq)
    mark_info = getattr(portfolio, "_last_mark_to_market", {}) or {}
    dividend_info = getattr(portfolio, "_last_dividend_accrual", {}) or {}
    history = _latest_jsonl_row(PORTFOLIO_HISTORY_PATH)
    cap_summary = _load_json_file(CAP_IMPACT_SUMMARY_PATH)
    attribution = _load_json_file(PERFORMANCE_ATTRIBUTION_PATH)
    parity = _load_json_file(PARITY_REPORT_PATH)

    print(f"\n  {'-'*55}")
    print("  Current Status")
    print(f"  {'-'*55}")

    if eod:
        print(f"  Latest snapshot: {eod['latest_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Latest equity:   ${eod['latest_equity']:,.2f}")
        print(f"  Latest cash:     ${eod['latest_cash']:,.2f}")
        print(f"  EOD return:      {_fmt_pct(eod['eod_return'])}")
        if eod["intraday_return"] is not None:
            print(f"  Today change:    {_fmt_pct(eod['intraday_return'])}")
    else:
        print(f"  Current equity:  ${portfolio.equity:,.2f}")
        print(f"  Current cash:    ${portfolio.cash:,.2f}")
        print("  EOD return:      n/a (no equity curve yet)")

    if mark_info:
        updated = mark_info.get("updated") or {}
        sources = mark_info.get("sources") or {}
        missing = mark_info.get("missing") or []
        latest_date = mark_info.get("latest_date") or "unknown"
        live_count = sum(1 for source in sources.values() if source == "live")
        cache_count = sum(1 for source in sources.values() if source == "cache")
        print(
            "  Prices updated:  "
            f"{latest_date} ({len(updated)}/{len(portfolio.positions)} holdings; "
            f"live={live_count}, cache={cache_count})"
        )
        if missing:
            print(f"  Missing prices:  {', '.join(missing[:8])}")
        if mark_info.get("error"):
            print(f"  Price mark note: {mark_info['error']}")
        if mark_info.get("errors"):
            error_keys = ", ".join(sorted(mark_info["errors"])[:5])
            print(f"  Price notes:     {len(mark_info['errors'])} issue(s): {error_keys}")
        print(f"  Marked equity:   ${portfolio.equity:,.2f}")
        print(f"  Marked return:   {_fmt_pct(portfolio.total_return)}")
        prev_eod = eod.get("previous_eod_equity") if eod else None
        if prev_eod:
            print(f"  Marked EOD ret:  {_fmt_pct(portfolio.equity / prev_eod - 1.0)}")

    if dividend_info:
        dividend_total = float(dividend_info.get("total", 0.0) or 0.0)
        credited = dividend_info.get("credited") or []
        if dividend_total > 0:
            print(
                "  Dividends:       "
                f"${dividend_total:,.2f} credited ({len(credited)} event(s))"
            )
        if dividend_info.get("errors"):
            error_keys = ", ".join(sorted(dividend_info["errors"])[:5])
            print(
                "  Dividend notes:  "
                f"{len(dividend_info['errors'])} issue(s): {error_keys}"
            )

    if history:
        print(
            "  Concentration:   "
            f"top1={_fmt_weight(history.get('top_1_concentration'))} "
            f"top3={_fmt_weight(history.get('top_3_concentration'))} "
            f"theme_bets={float(history.get('theme_effective_bet_count', 0.0)):.2f}"
        )
        print(f"  Low-price exp.:  {_fmt_weight(history.get('low_price_exposure'))}")

    if attribution:
        total_cost = float(attribution.get("total_execution_cost", 0.0) or 0.0)
        realized = float(attribution.get("realized_net_pnl", 0.0) or 0.0)
        unrealized = float(attribution.get("unrealized_pnl", 0.0) or 0.0)
        dividends = float(attribution.get("dividend_income", 0.0) or 0.0)
        print(
            "  P&L attribution: "
            f"realized=${realized:,.2f} unrealized=${unrealized:,.2f} "
            f"dividends=${dividends:,.2f} exec_cost=${total_cost:,.2f}"
        )

    if cap_summary:
        print(
            "  Cap history:     "
            f"{int(cap_summary.get('cycles', 0))} cycle(s), "
            f"{int(cap_summary.get('cycles_with_cap_interventions', 0))} with caps"
        )

    if parity:
        status = "OK" if parity.get("compatible") else "CHECK"
        print(f"  Replay parity:   {status}")

    print("  Command:         python Broker.py --status")


def print_report(portfolio, show_benchmark: bool = True):
    """Print full performance report including SPY benchmark."""
    print(portfolio.summary())

    if not EQUITY_PATH.exists():
        _print_status_snapshot(portfolio, None)
        return

    eq = pd.read_csv(EQUITY_PATH, parse_dates=["time"])
    if len(eq) < 2:
        _print_status_snapshot(portfolio, eq)
        return

    eq["time"] = pd.to_datetime(eq["time"], utc=True, errors="coerce").dt.tz_convert(None)
    eq = eq.dropna(subset=["time"]).sort_values("time")

    initial = float(eq["equity"].iloc[0])
    current = float(eq["equity"].iloc[-1])
    peak    = float(eq["equity"].max())
    dd      = _max_drawdown_from_equity(eq["equity"])

    print(f"  Equity curve: {len(eq)} data points")
    print(f"  Peak equity:  ${peak:,.2f}")
    print(f"  Max drawdown: {dd:.2%}")
    print(f"  Total return: {(current/initial - 1):.2%}")
    _print_status_snapshot(portfolio, eq)

    if not show_benchmark:
        return

    # ── SPY comparison ────────────────────────────────────────────────────────
    spy_col = eq["spy_price"].dropna()
    if len(spy_col) < 2:
        print("  SPY benchmark: not enough data yet (will appear after 2+ cycles)\n")
        return

    portfolio_rets, spy_rets = _align_equity_and_spy_returns(eq)
    n = min(len(portfolio_rets), len(spy_rets))

    if n < 2:
        return

    from pipeline.benchmark import (
        MIN_HISTORY_FOR_STABLE_METRICS,
        benchmark_vs_spy,
        compute_metrics,
        format_metric_cell,
        short_history_note,
    )
    p_m = compute_metrics(
        portfolio_rets[:n],
        "Broker",
        min_obs_for_annualized=MIN_HISTORY_FOR_STABLE_METRICS,
        min_obs_for_risk=MIN_HISTORY_FOR_STABLE_METRICS,
    )
    spy_m = compute_metrics(
        spy_rets[:n],
        "SPY",
        min_obs_for_annualized=MIN_HISTORY_FOR_STABLE_METRICS,
        min_obs_for_risk=MIN_HISTORY_FOR_STABLE_METRICS,
    )
    rel = benchmark_vs_spy(
        portfolio_rets[:n],
        spy_rets[:n],
        min_obs_for_relative=MIN_HISTORY_FOR_STABLE_METRICS,
    )

    print(f"\n  {'─'*55}")
    print(f"  {'Metric':<22} {'Broker':>12}  {'SPY':>10}")
    print(f"  {'─'*55}")
    for key, fmt in [
        ("total_return", ".2%"), ("ann_return", ".2%"),
        ("sharpe", ".3f"),       ("sortino", ".3f"),
        ("max_drawdown", ".2%"), ("win_rate", ".2%"),
    ]:
        pv = p_m[key]
        sv = spy_m[key]
        print(
            f"  {key:<22} "
            f"{format_metric_cell(pv, fmt, 12)}  "
            f"{format_metric_cell(sv, fmt, 10)}"
        )

    print(f"  {'─'*55}")
    print(f"  {'Beta':<22} {format_metric_cell(rel['beta'], '.3f', 12)}")
    print(f"  {'Alpha (ann)':<22} {format_metric_cell(rel['alpha_ann'], '.2%', 12)}")
    print(f"  {'Info Ratio':<22} {format_metric_cell(rel['information_ratio'], '.3f', 12)}")
    if rel["upside_capture"] is not None:
        print(f"  {'Upside Capture':<22} {format_metric_cell(rel['upside_capture'], '.3f', 12)}")
    if rel["downside_capture"] is not None:
        print(f"  {'Downside Capture':<22} {format_metric_cell(rel['downside_capture'], '.3f', 12)}")
    beats = "✓ YES" if rel['beats_spy_return'] else "✗ NO"
    print(f"  {'Beats SPY':<22} {beats:>12}")
    note = short_history_note(n, min_obs=MIN_HISTORY_FOR_STABLE_METRICS)
    if note:
        print("  " + "-" * 55)
        print(note)
        note = None
    if note:
        print(f"  {'â”€'*55}")
        print(note)
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


def plot_live_performance(save_path: str = "plots/live_performance.png"):
    """
    Generate a live performance chart from the equity curve CSV.
    Called automatically after every broker run.
    """
    if not EQUITY_PATH.exists():
        return

    eq = pd.read_csv(EQUITY_PATH, parse_dates=["time"])
    if len(eq) < 3:
        return   # not enough data yet

    import os
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    portfolio_rets = eq["equity"].pct_change().dropna().values
    initial        = eq["equity"].iloc[0]
    equity_pct     = ((eq["equity"] / initial) - 1) * 100
    x              = range(len(equity_pct))

    # SPY comparison if available
    spy_col = eq["spy_price"].dropna()
    has_spy = len(spy_col) >= 3

    fig, axes = plt.subplots(3 if has_spy else 2, 1, figsize=(14, 12 if has_spy else 8),
                             gridspec_kw={"height_ratios": ([3, 1.5, 1.5] if has_spy else [3, 1.5])})
    fig.suptitle("Live Portfolio Performance", fontsize=15, fontweight="bold", y=0.98)

    # ── Panel 1: Return % ─────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(x, equity_pct, color="#2196F3", lw=2.0, label="My Portfolio")

    if has_spy:
        spy_initial = eq.loc[eq["spy_price"].first_valid_index(), "spy_price"]
        spy_pct     = ((eq["spy_price"] / spy_initial) - 1) * 100
        ax.plot(x, spy_pct, color="#4CAF50", lw=1.5, ls="--", label="SPY")
        ax.annotate(f"  Portfolio: {equity_pct.iloc[-1]:+.1f}%",
                    xy=(len(x)-1, equity_pct.iloc[-1]), fontsize=9, color="#2196F3")
        ax.annotate(f"  SPY: {spy_pct.iloc[-1]:+.1f}%",
                    xy=(len(x)-1, spy_pct.iloc[-1]), fontsize=9, color="#4CAF50")
    else:
        ax.annotate(f"  {equity_pct.iloc[-1]:+.1f}%",
                    xy=(len(x)-1, equity_pct.iloc[-1]), fontsize=9, color="#2196F3")

    ax.axhline(0, color="gray", lw=0.8, ls=":", alpha=0.6)
    ax.set_title("Total Return  (% gain/loss since start — higher is better)",
                 fontsize=11, pad=8)
    ax.set_ylabel("Return (%)")
    ax.legend(loc="upper left"); ax.grid(alpha=0.3)

    # ── Panel 2: Drawdown ─────────────────────────────────────────────────────
    ax2 = axes[1]
    eq_vals = eq["equity"].values
    peak    = np.maximum.accumulate(eq_vals)
    dd      = ((eq_vals - peak) / (peak + 1e-9)) * 100
    ax2.fill_between(x, dd, 0, alpha=0.5, color="#F44336", label="Drawdown")
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_title("Drawdown  (how far below your peak — closer to 0% is better)",
                  fontsize=11, pad=8)
    ax2.set_ylabel("% below peak")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    # ── Panel 3: vs SPY (if available) ───────────────────────────────────────
    if has_spy:
        ax3 = axes[2]
        spy_rets = eq["spy_price"].pct_change().dropna().values
        n        = min(len(portfolio_rets), len(spy_rets))
        active   = (portfolio_rets[:n] - spy_rets[:n]) * 100
        colors   = ["#2196F3" if v >= 0 else "#F44336" for v in active]
        ax3.bar(range(len(active)), active, color=colors, alpha=0.7, width=1.0)
        ax3.axhline(0, color="black", lw=1.0)
        ax3.set_title("Daily Outperformance vs SPY  (blue = beat SPY, red = trailed SPY)",
                      fontsize=11, pad=8)
        ax3.set_ylabel("% vs SPY that day")
        ax3.set_xlabel("Trading days since start")
        ax3.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Performance chart updated -> {save_path}")
