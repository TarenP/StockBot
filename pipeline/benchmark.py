"""
Benchmark module.

Provides:
  - fetch_spy_returns(): get SPY daily returns for a date range
  - benchmark_vs_spy(): compute relative metrics
  - compute_metrics(): base performance metrics
  - print_benchmark_report(): formatted comparison table
  - plot_benchmark(): equity, drawdown, and relative-performance charts
"""

import logging
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
MIN_HISTORY_FOR_STABLE_METRICS = 20


@contextmanager
def _quiet():
    with open(os.devnull, "w") as dn:
        old = sys.stderr
        sys.stderr = dn
        try:
            yield
        finally:
            sys.stderr = old


def _console_safe(text: str) -> str:
    """
    Downgrade a few Unicode presentation characters when the active console
    encoding cannot represent them.
    """
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        text.encode(encoding)
        return text
    except Exception:
        return (
            text.replace("—", "-")
                .replace("–", "-")
                .replace("→", "->")
                .replace("•", "*")
                .replace("…", "...")
        )


def format_metric_cell(value: float | None, fmt: str, width: int) -> str:
    """Format numeric table cells, falling back to n/a when hidden."""
    if value is None:
        return f"{'n/a':>{width}}"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return f"{'n/a':>{width}}"
    if not np.isfinite(numeric):
        return f"{'n/a':>{width}}"
    return f"{numeric:>{width}{fmt}}"


def format_yes_no_cell(value: bool | None, width: int, yes: str = "YES", no: str = "NO") -> str:
    """Format boolean table cells, allowing n/a for hidden metrics."""
    if value is None:
        return f"{'n/a':>{width}}"
    return f"{yes if value else no:>{width}}"


def short_history_note(
    n_obs: int,
    min_obs: int = MIN_HISTORY_FOR_STABLE_METRICS,
) -> str | None:
    """Explain why sample-sensitive metrics are hidden."""
    if n_obs >= min_obs:
        return None
    noun = "observation" if n_obs == 1 else "observations"
    return (
        f"  Note: annualized, Sharpe-style, and benchmark-relative metrics are hidden "
        f"until {min_obs} daily return observations are available "
        f"(currently {n_obs} {noun})."
    )


def fetch_spy_returns(
    start: str | None = None,
    end: str | None = None,
    n_days: int | None = None,
    parquet_path: str = "MasterDS/stooq_panel.parquet",
) -> pd.Series:
    """
    Fetch SPY daily returns from yfinance.
    """
    import yfinance as yf

    if n_days and not start:
        end = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
        start = (datetime.today() - timedelta(days=n_days + 10)).strftime("%Y-%m-%d")

    try:
        with _quiet():
            raw = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
        if raw.empty:
            with _quiet():
                raw = yf.Ticker("SPY").history(start=start, end=end, auto_adjust=True)
        if raw.empty:
            logger.warning("Could not fetch SPY data from yfinance. Falling back to local parquet.")
            return _load_local_symbol_returns("SPY", start=start, end=end, parquet_path=parquet_path)

        rets = raw["Close"].pct_change().dropna()
        if isinstance(rets, pd.DataFrame):
            rets = rets.iloc[:, 0]
        rets.index = pd.to_datetime(rets.index).normalize()
        return rets
    except Exception as exc:
        logger.warning("SPY fetch failed: %s", exc)
        return _load_local_symbol_returns("SPY", start=start, end=end, parquet_path=parquet_path)


def _load_local_symbol_returns(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    parquet_path: str = "MasterDS/stooq_panel.parquet",
) -> pd.Series:
    """
    Load a benchmark symbol from the local price parquet when network fetches
    are unavailable. Supports both the repo's flat index + ticker-column shape
    and older MultiIndex layouts.
    """
    if not os.path.exists(parquet_path):
        return pd.Series(dtype=float)

    try:
        raw = pd.read_parquet(parquet_path)
    except Exception as exc:
        logger.warning("Local benchmark fallback failed to read %s: %s", parquet_path, exc)
        return pd.Series(dtype=float)

    symbol = str(symbol).upper()
    frame: pd.DataFrame | None = None
    close_col = None

    if isinstance(raw.index, pd.MultiIndex) and "ticker" in raw.index.names:
        try:
            frame = raw.xs(symbol, level="ticker").copy()
        except Exception:
            frame = None
    elif "ticker" in raw.columns:
        tickers = raw["ticker"].astype(str).str.upper()
        frame = raw.loc[tickers == symbol].copy()

    if frame is None or frame.empty:
        return pd.Series(dtype=float)

    if "close" in frame.columns:
        close_col = "close"
    elif "Close" in frame.columns:
        close_col = "Close"
    else:
        return pd.Series(dtype=float)

    if isinstance(frame.index, pd.MultiIndex):
        if "date" not in frame.index.names:
            return pd.Series(dtype=float)
        dates = pd.to_datetime(frame.index.get_level_values("date"), errors="coerce")
        series = pd.Series(frame[close_col].to_numpy(dtype=float), index=dates, name=symbol)
    elif "date" in frame.columns and not isinstance(frame.index, pd.DatetimeIndex):
        dates = pd.to_datetime(frame["date"], errors="coerce")
        series = pd.Series(frame[close_col].to_numpy(dtype=float), index=dates, name=symbol)
    else:
        dates = pd.to_datetime(frame.index, errors="coerce")
        series = pd.Series(frame[close_col].to_numpy(dtype=float), index=dates, name=symbol)

    if getattr(series.index, "tz", None) is not None:
        series.index = series.index.tz_convert(None)
    series.index = series.index.normalize()
    series = series[~series.index.isna()].sort_index()

    if start is not None:
        series = series[series.index >= pd.Timestamp(start).normalize()]
    if end is not None:
        series = series[series.index <= pd.Timestamp(end).normalize()]

    rets = series.pct_change().dropna()
    if not rets.empty:
        logger.info(
            "Loaded %s benchmark from local parquet fallback (%s to %s).",
            symbol,
            rets.index.min().date(),
            rets.index.max().date(),
        )
    return rets.astype(float)


def align_return_series(
    portfolio_rets: np.ndarray,
    portfolio_dates,
    benchmark_rets: pd.Series | None = None,
    extra_series: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """
    Align portfolio returns, optional benchmark returns, and optional extra
    series onto a shared normalized date index.

    When benchmark data is unavailable or has no overlap, the portfolio and
    extra series are still returned on their native dates.
    """
    portfolio_arr = np.asarray(portfolio_rets, dtype=float)
    portfolio_dates = list(portfolio_dates)
    n = min(len(portfolio_arr), len(portfolio_dates))

    columns = ["portfolio"]
    if extra_series:
        columns.extend(extra_series.keys())
    if benchmark_rets is not None:
        columns.append("benchmark")

    if n == 0:
        return pd.DataFrame(columns=columns)

    index = pd.to_datetime(pd.Index(portfolio_dates[:n]), errors="coerce").normalize()
    frames = [pd.Series(portfolio_arr[:n], index=index, name="portfolio")]

    if extra_series:
        for name, values in extra_series.items():
            arr = np.asarray(values, dtype=float)
            extra_n = min(len(arr), n)
            frames.append(pd.Series(arr[:extra_n], index=index[:extra_n], name=name))

    base = pd.concat(frames, axis=1).dropna()
    if benchmark_rets is None or getattr(benchmark_rets, "empty", True):
        return base

    benchmark = benchmark_rets.copy()
    if isinstance(benchmark, pd.DataFrame):
        benchmark = benchmark.iloc[:, 0]

    benchmark_index = pd.to_datetime(benchmark.index, errors="coerce")
    if getattr(benchmark_index, "tz", None) is not None:
        benchmark_index = benchmark_index.tz_convert(None)
    benchmark_index = benchmark_index.normalize()

    if benchmark_index.isna().all() or not benchmark_index.is_unique:
        bench_arr = np.asarray(benchmark, dtype=float)
        bench_n = min(len(bench_arr), len(base.index))
        benchmark = pd.Series(
            bench_arr[:bench_n],
            index=base.index[:bench_n],
            name="benchmark",
            dtype=float,
        )
    else:
        benchmark.index = benchmark_index
        benchmark = benchmark.rename("benchmark").astype(float)

    aligned = pd.concat([base, benchmark], axis=1, join="inner").dropna()
    return aligned if not aligned.empty else base


def _sharpe(rets, periods: int = 252) -> float:
    if len(rets) < 2:
        return 0.0
    return float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(periods))


def _sortino(rets, periods: int = 252) -> float:
    down = rets[rets < 0]
    if len(down) < 2:
        return 0.0
    return float(rets.mean() / (down.std() + 1e-9) * np.sqrt(periods))


def _max_dd(rets) -> float:
    eq = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(eq)
    return float(((eq - peak) / (peak + 1e-9)).min())


def _calmar(rets, periods: int = 252) -> float:
    if len(rets) == 0:
        return 0.0
    mdd = abs(_max_dd(rets))
    ann_r = float(np.prod(1 + rets) ** (periods / len(rets)) - 1)
    return ann_r / (mdd + 1e-9)


def _ann_return(rets, periods: int = 252) -> float:
    if len(rets) == 0:
        return 0.0
    return float(np.prod(1 + rets) ** (periods / len(rets)) - 1)


def _volatility(rets, periods: int = 252) -> float:
    if len(rets) == 0:
        return 0.0
    return float(rets.std() * np.sqrt(periods))


def compute_metrics(
    rets: np.ndarray,
    label: str = "",
    min_obs_for_annualized: int = 0,
    min_obs_for_risk: int = 0,
) -> dict:
    """Base metrics with no benchmark dependency."""
    rets = np.asarray(rets)
    if len(rets) == 0:
        return {
            "label": label,
            "n_obs": 0,
            "total_return": 0.0,
            "ann_return": 0.0,
            "volatility": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "win_rate": 0.0,
        }

    eq = np.cumprod(1 + rets)
    ann_return = _ann_return(rets) if len(rets) >= min_obs_for_annualized else None
    volatility = _volatility(rets) if len(rets) >= min_obs_for_risk else None
    sharpe = _sharpe(rets) if len(rets) >= min_obs_for_risk else None
    sortino = _sortino(rets) if len(rets) >= min_obs_for_risk else None
    calmar = (
        _calmar(rets)
        if len(rets) >= max(min_obs_for_annualized, min_obs_for_risk)
        else None
    )
    return {
        "label": label,
        "n_obs": int(len(rets)),
        "total_return": float(eq[-1] - 1),
        "ann_return": ann_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": _max_dd(rets),
        "calmar": calmar,
        "win_rate": float((rets > 0).mean()),
    }


def benchmark_vs_spy(
    portfolio_rets: np.ndarray,
    spy_rets: np.ndarray,
    rf_daily: float = 0.05 / 252,
    min_obs_for_relative: int = 0,
) -> dict:
    """Compute SPY-relative metrics for aligned return series."""
    n = min(len(portfolio_rets), len(spy_rets))
    p = np.asarray(portfolio_rets[:n])
    s = np.asarray(spy_rets[:n])
    beats_total = float(np.prod(1 + p)) > float(np.prod(1 + s)) if n else False

    if n < 2:
        return {
            "n_obs": int(n),
            "beta": None,
            "alpha_ann": None,
            "information_ratio": None,
            "upside_capture": None,
            "downside_capture": None,
            "tracking_error": None,
            "beats_spy_return": beats_total,
            "beats_spy_sharpe": None,
            "active_return_ann": None,
        }

    cov_matrix = np.cov(p, s)
    beta = cov_matrix[0, 1] / (cov_matrix[1, 1] + 1e-9)
    alpha_daily = (p.mean() - rf_daily) - beta * (s.mean() - rf_daily)
    alpha_ann = alpha_daily * 252

    active_rets = p - s
    ir = float(active_rets.mean() / (active_rets.std() + 1e-9) * np.sqrt(252))

    up_mask = s > 0
    down_mask = s < 0
    up_cap = (p[up_mask].mean() / (s[up_mask].mean() + 1e-9)) if up_mask.any() else np.nan
    down_cap = (p[down_mask].mean() / (s[down_mask].mean() + 1e-9)) if down_mask.any() else np.nan

    tracking_error = float(active_rets.std() * np.sqrt(252))
    beats_sharpe = _sharpe(p) > _sharpe(s)

    if n < min_obs_for_relative:
        return {
            "n_obs": int(n),
            "beta": None,
            "alpha_ann": None,
            "information_ratio": None,
            "upside_capture": None,
            "downside_capture": None,
            "tracking_error": None,
            "beats_spy_return": beats_total,
            "beats_spy_sharpe": None,
            "active_return_ann": None,
        }

    return {
        "n_obs": int(n),
        "beta": round(beta, 3),
        "alpha_ann": round(alpha_ann, 4),
        "information_ratio": round(ir, 3),
        "upside_capture": round(up_cap, 3) if not np.isnan(up_cap) else None,
        "downside_capture": round(down_cap, 3) if not np.isnan(down_cap) else None,
        "tracking_error": round(tracking_error, 4),
        "beats_spy_return": beats_total,
        "beats_spy_sharpe": beats_sharpe,
        "active_return_ann": round(float(active_rets.mean() * 252), 4),
    }


def rolling_relative_performance(
    portfolio_rets: np.ndarray,
    spy_rets: np.ndarray,
    windows: list[int] = [63, 126, 252],
) -> dict[str, np.ndarray]:
    """Rolling outperformance vs SPY over multiple windows."""
    n = min(len(portfolio_rets), len(spy_rets))
    p = pd.Series(portfolio_rets[:n])
    s = pd.Series(spy_rets[:n])
    result = {}
    for window in windows:
        p_roll = (1 + p).rolling(window).apply(np.prod) - 1
        s_roll = (1 + s).rolling(window).apply(np.prod) - 1
        result[f"{window}d"] = (p_roll - s_roll).values
    return result


def print_benchmark_report(
    portfolio_rets: np.ndarray,
    spy_rets: np.ndarray | None,
    ew_rets: np.ndarray | None = None,
    label: str = "Strategy",
):
    """Print a benchmark comparison table."""
    min_obs = MIN_HISTORY_FOR_STABLE_METRICS
    spy_available = spy_rets is not None and len(spy_rets) >= 2
    lengths = [len(portfolio_rets)]
    if spy_available:
        lengths.append(len(spy_rets))
    if ew_rets is not None:
        lengths.append(len(ew_rets))
    n = min(lengths)

    p_metrics = compute_metrics(
        portfolio_rets[:n],
        label,
        min_obs_for_annualized=min_obs,
        min_obs_for_risk=min_obs,
    )
    spy_metrics = (
        compute_metrics(
            spy_rets[:n],
            "SPY",
            min_obs_for_annualized=min_obs,
            min_obs_for_risk=min_obs,
        )
        if spy_available else None
    )
    ew_metrics = (
        compute_metrics(
            ew_rets[:n],
            "Equal-Weight",
            min_obs_for_annualized=min_obs,
            min_obs_for_risk=min_obs,
        )
        if ew_rets is not None else None
    )
    rel = (
        benchmark_vs_spy(
            portfolio_rets[:n],
            spy_rets[:n],
            min_obs_for_relative=min_obs,
        )
        if spy_available else None
    )

    cols = ["Policy"]
    all_metrics = [p_metrics]
    if spy_available:
        cols.append("SPY")
        all_metrics.append(spy_metrics)
    if ew_metrics is not None:
        cols.append("Equal-Weight")
        all_metrics.append(ew_metrics)

    line = "-" * 68
    print(_console_safe(f"\n{'='*72}"))
    print(_console_safe(f"  Benchmark Report - {label} vs SPY"))
    print(_console_safe(f"{'='*72}"))
    header = f"  {'Metric':<22}"
    for col in cols:
        header += f" {col:>14}"
    print(_console_safe(header))
    print(f"  {line}")

    pct_keys = {"total_return", "ann_return", "volatility", "max_drawdown", "win_rate"}
    for key in ["total_return", "ann_return", "volatility", "sharpe", "sortino",
                "max_drawdown", "calmar", "win_rate"]:
        row = f"  {key:<22}"
        for metrics in all_metrics:
            val = metrics.get(key, 0)
            width = 13 if key in pct_keys else 14
            fmt = ".2%" if key in pct_keys else ".3f"
            row += f" {format_metric_cell(val, fmt, width)}"
        print(_console_safe(row))

    print(f"\n  {line}")
    print(_console_safe("  SPY-Relative Metrics"))
    print(f"  {line}")
    if not spy_available:
        print(_console_safe("  SPY benchmark unavailable for this run. Relative metrics skipped."))
    else:
        print(_console_safe(f"  {'Beta':<22} {format_metric_cell(rel['beta'], '.3f', 14)}"))
        print(_console_safe(f"  {'Alpha (ann)':<22} {format_metric_cell(rel['alpha_ann'], '.2%', 13)}"))
        print(_console_safe(f"  {'Information Ratio':<22} {format_metric_cell(rel['information_ratio'], '.3f', 14)}"))
        print(_console_safe(f"  {'Tracking Error':<22} {format_metric_cell(rel['tracking_error'], '.2%', 13)}"))
        if rel["upside_capture"] is not None:
            print(_console_safe(f"  {'Upside Capture':<22} {format_metric_cell(rel['upside_capture'], '.3f', 14)}"))
        if rel["downside_capture"] is not None:
            print(_console_safe(f"  {'Downside Capture':<22} {format_metric_cell(rel['downside_capture'], '.3f', 14)}"))
        print(_console_safe(f"  {'Beats SPY (return)':<22} {format_yes_no_cell(rel['beats_spy_return'], 14)}"))
        print(_console_safe(f"  {'Beats SPY (Sharpe)':<22} {format_yes_no_cell(rel['beats_spy_sharpe'], 14)}"))
        note = short_history_note(n, min_obs=min_obs)
        if note:
            print(_console_safe(""))
            print(_console_safe(note))
    print(_console_safe(f"{'='*72}\n"))


def plot_benchmark(
    portfolio_rets: np.ndarray,
    spy_rets: np.ndarray | None,
    ew_rets: np.ndarray | None = None,
    dates: list | None = None,
    save_path: str = "plots/benchmark.png",
    label: str = "Strategy",
):
    """Create a 4-panel benchmark chart."""
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    spy_available = spy_rets is not None and len(spy_rets) >= 2
    lengths = [len(portfolio_rets)]
    if spy_available:
        lengths.append(len(spy_rets))
    if ew_rets is not None:
        lengths.append(len(ew_rets))
    n = min(lengths)

    p = np.asarray(portfolio_rets[:n])
    s = np.asarray(spy_rets[:n]) if spy_available else None
    x = list(range(n))

    p_equity = np.cumprod(1 + p)
    s_equity = np.cumprod(1 + s) if spy_available else None

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(14, 18),
        gridspec_kw={"height_ratios": [3, 1.5, 1.5, 1.5]},
    )
    title = f"{label} vs SPY - Performance Report" if spy_available else f"{label} - Performance Report"
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)

    ax = axes[0]
    ax.plot(x, (p_equity - 1) * 100, label=label, color="#2196F3", lw=2.0)
    if spy_available:
        ax.plot(x, (s_equity - 1) * 100, label="SPY (benchmark)", color="#4CAF50", lw=1.5, ls="--")
    if ew_rets is not None:
        ew = np.asarray(ew_rets[:n])
        ax.plot(
            list(range(len(ew))),
            (np.cumprod(1 + ew) - 1) * 100,
            label="Equal-weight baseline",
            color="#FF9800",
            lw=1.2,
            ls=":",
        )
    ax.annotate(f"  {label}: {(p_equity[-1] - 1) * 100:+.1f}%",
                xy=(n - 1, (p_equity[-1] - 1) * 100), fontsize=9, color="#2196F3")
    if spy_available:
        ax.annotate(f"  SPY: {(s_equity[-1] - 1) * 100:+.1f}%",
                    xy=(n - 1, (s_equity[-1] - 1) * 100), fontsize=9, color="#4CAF50")
    ax.set_title("Total Return (% gain/loss since start - higher is better)", fontsize=11, pad=8)
    ax.set_ylabel("Return (%)")
    ax.set_xlabel("Trading days since start")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    ax.axhline(0, color="gray", lw=0.8, ls=":", alpha=0.5)

    ax2 = axes[1]

    def _dd(rets):
        eq = np.cumprod(1 + rets)
        peak = np.maximum.accumulate(eq)
        return (eq - peak) / (peak + 1e-9)

    ax2.fill_between(x, _dd(p) * 100, 0, alpha=0.5, color="#F44336", label=f"{label} drawdown")
    if spy_available:
        ax2.plot(x, _dd(s) * 100, color="#4CAF50", lw=1.2, ls="--", label="SPY drawdown")
    ax2.set_title("Drawdown (how far below the peak - closer to 0% is better)", fontsize=11, pad=8)
    ax2.set_ylabel("% below peak")
    ax2.set_xlabel("Trading days since start")
    ax2.legend(fontsize=9, loc="lower left")
    ax2.grid(alpha=0.3)

    ax3 = axes[2]
    if spy_available:
        roll3 = rolling_relative_performance(p, s, windows=[63])["63d"]
        colors3 = ["#2196F3" if val >= 0 else "#F44336" for val in roll3]
        ax3.bar(x, roll3 * 100, color=colors3, alpha=0.7, width=1.0)
        ax3.axhline(0, color="black", lw=1.0)
        ax3.set_title("3-Month Rolling Outperformance vs SPY", fontsize=11, pad=8)
        ax3.set_ylabel("% ahead of SPY")
        ax3.set_xlabel("Trading days since start")
        ax3.grid(alpha=0.3)
    else:
        ax3.axis("off")
        ax3.text(0.5, 0.5, "SPY unavailable\n3-month relative chart skipped",
                 ha="center", va="center", fontsize=12, transform=ax3.transAxes)

    ax4 = axes[3]
    if spy_available:
        roll12 = rolling_relative_performance(p, s, windows=[252])["252d"]
        colors12 = ["#2196F3" if val >= 0 else "#F44336" for val in roll12]
        ax4.bar(x, roll12 * 100, color=colors12, alpha=0.7, width=1.0)
        ax4.axhline(0, color="black", lw=1.0)
        ax4.set_title("12-Month Rolling Outperformance vs SPY", fontsize=11, pad=8)
        ax4.set_ylabel("% ahead of SPY")
        ax4.set_xlabel("Trading days since start")
        ax4.grid(alpha=0.3)
    else:
        ax4.axis("off")
        ax4.text(0.5, 0.5, "SPY unavailable\n12-month relative chart skipped",
                 ha="center", va="center", fontsize=12, transform=ax4.transAxes)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Chart saved -> %s", save_path)
