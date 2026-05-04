"""
Broker Replay Backtest
======================
Runs the actual broker decision logic (BrokerBrain) over historical data,
producing a realistic performance record that reflects what the broker
would have done — not what the RL agent does in its idealized environment.

Key differences from the RL backtest:
  - Uses the same screening, research, sector allocation, stop-loss,
    take-profit, and position sizing logic as the live broker
  - Feeds historical data slice-by-slice with no lookahead
  - Applies execution cost (spread model) on every trade
  - Compares result to SPY and equal-weight on the same period
  - Runs a sensitivity sweep over key parameters

Usage:
    python Agent.py --mode replay
    python Agent.py --mode replay --replay_years 3
    python Agent.py --mode replay --sensitivity
"""

import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from types import MethodType

import numpy as np
import pandas as pd
from tqdm import tqdm

from broker.portfolio import CASH_YIELD_ANNUAL_RATE, DAYS_PER_YEAR
from pipeline.run_manifest import hash_config, hash_ticker_list, write_run_manifest

logger = logging.getLogger(__name__)
_REPLAY_MIN_PRICE = 0.01
_REPLAY_SPLIT_RATIO_THRESHOLD = 2.0
_REPLAY_CORP_ACTION_LOOKBACK = 20
_LAST_REPLAY_SCORE_AUDIT = pd.DataFrame()


# ── Historical research stub ──────────────────────────────────────────────────

def _historical_feature_score(report: dict) -> float:
    """
    Score historical replay reports built from z-scored cross-sectional
    features. The live analyst score expects raw indicator scales and will
    systematically under-score normalized replay features.
    """
    def _bounded(value: float, scale: float = 1.0) -> float:
        return float(np.tanh(float(value) / max(scale, 1e-6)))

    score = 0.5
    score += 0.12 * _bounded(report.get("ret_5d", 0.0), 1.0)
    score += 0.08 * _bounded(report.get("ret_20d", 0.0), 1.0)
    score += 0.10 * _bounded(report.get("macd_hist", 0.0), 1.0)
    score += 0.08 * _bounded(report.get("vol_ratio", 0.0), 1.0)
    score += 0.06 * _bounded(report.get("vol_zscore", 0.0), 1.0)
    score += 0.06 * _bounded(report.get("price_pos_52w", 0.0), 1.0)
    score += 0.12 * _bounded(report.get("sent_net", 0.0), 1.0)
    score += 0.10 * _bounded(report.get("sent_surprise", 0.0), 1.0)
    score += 0.06 * _bounded(report.get("sent_accel", 0.0), 1.0)
    score += 0.04 * _bounded(report.get("sent_trend", 0.0), 1.0)
    score += 0.04 * max(0.0, 1.0 - min(abs(float(report.get("rsi", 0.0))) / 3.0, 1.0))
    score += 0.04 * max(0.0, 1.0 - min(abs(float(report.get("bb_pct", 0.0))) / 3.0, 1.0))
    score += 0.04 * max(0.0, 1.0 - min(max(float(report.get("atr", 0.0)), 0.0) / 3.0, 1.0))
    return float(np.clip(score, 0.0, 1.0))


def _make_historical_research(
    df_features: pd.DataFrame,
    price_lookup: pd.DataFrame,
    as_of_date,
):
    """
    Returns a research() function that uses historical feature data
    instead of fetching live data from yfinance.
    No lookahead: only data up to and including as_of_date is visible.
    """
    from pipeline.features import FEATURE_COLS

    def historical_research(ticker: str, days: int = 90) -> dict | None:
        try:
            ticker = ticker.upper()
            # Get the ticker's feature history up to as_of_date
            ticker_data = df_features.xs(ticker, level="ticker")
            ticker_data = ticker_data[ticker_data.index <= as_of_date]

            if len(ticker_data) < 5:
                return None

            latest = ticker_data.iloc[-1]
            if pd.Timestamp(latest.name) != pd.Timestamp(as_of_date):
                return None

            quote = _get_historical_quote(price_lookup, ticker, as_of_date)
            if quote is None:
                return None
            if _has_recent_corporate_action(price_lookup, ticker, as_of_date):
                return None
            price, volume = quote

            feat_dict = {col: float(latest[col]) if col in latest.index and pd.notna(latest[col]) else 0.0
                         for col in FEATURE_COLS}

            sent_net = float(feat_dict.get("sent_net", 0.0))
            pos_score = float(np.clip(feat_dict.get("sent_pos_raw", 0.45), 0.0, 1.0))
            neg_score = float(np.clip(pos_score - sent_net, 0.0, 1.0))
            if sent_net > 1e-6:
                sent_label = "positive"
            elif sent_net < -1e-6:
                sent_label = "negative"
            else:
                sent_label = "neutral"

            # Reconstruct a minimal sentiment dict from features
            sent = {
                "sent_net": sent_net,
                "pos_score": pos_score,
                "neg_score": neg_score,
                "sentiment": sent_label,
                "headlines": [],
            }

            report = {
                "ticker":    ticker,
                "price":     price,
                "volume":    volume,
                "sentiment": sent,
                "headlines": [],
            }
            report.update(feat_dict)
            report["composite_score"] = _historical_feature_score(report)
            return report

        except (KeyError, Exception):
            return None

    return historical_research


def _get_historical_quote(
    price_lookup: pd.DataFrame,
    ticker: str,
    as_of_date,
) -> tuple[float, float] | None:
    """Require a real same-day quote so replay does not fabricate entries."""
    try:
        row = price_lookup.loc[(as_of_date, ticker), ["close", "volume"]]
    except Exception:
        return None

    try:
        price = float(row["close"])
        volume = float(row["volume"])
    except Exception:
        return None

    if not np.isfinite(price) or price < _REPLAY_MIN_PRICE:
        return None
    if not np.isfinite(volume) or volume <= 0:
        return None
    return price, volume


def _adjust_replay_close_series(
    close: pd.Series,
    split_ratio_threshold: float = _REPLAY_SPLIT_RATIO_THRESHOLD,
) -> pd.Series:
    """
    Back-adjust split-like jumps out of replay prices so portfolio PnL and
    execution do not inherit obvious corporate-action artifacts from the raw
    panel. Prior history is scaled into continuity with post-event prices.
    """
    arr = close.to_numpy(dtype=np.float64, copy=True)
    n = len(arr)
    if n <= 1:
        return close.astype(float)

    event_factors = np.ones(n, dtype=np.float64)
    prev = arr[:-1]
    curr = arr[1:]
    valid = np.isfinite(prev) & np.isfinite(curr) & (prev > 0) & (curr > 0)
    ratios = np.ones(n - 1, dtype=np.float64)
    ratios[valid] = curr[valid] / prev[valid]

    event_mask = valid & (
        (ratios >= split_ratio_threshold) |
        (ratios <= (1.0 / split_ratio_threshold))
    )
    event_factors[1:] = np.where(event_mask, ratios, 1.0)

    future_factor = np.ones(n, dtype=np.float64)
    running = 1.0
    for i in range(n - 1, -1, -1):
        future_factor[i] = running
        running *= event_factors[i]

    adjusted = arr * future_factor
    return pd.Series(adjusted, index=close.index, name=close.name)


def _has_recent_corporate_action(
    price_lookup: pd.DataFrame,
    ticker: str,
    as_of_date,
    lookback_sessions: int = _REPLAY_CORP_ACTION_LOOKBACK,
    split_ratio_threshold: float = _REPLAY_SPLIT_RATIO_THRESHOLD,
) -> bool:
    """
    Skip research on names that recently had split-scale raw price jumps.
    These events can leak distorted momentum into replay decisions when the
    underlying historical panel is not fully adjusted.
    """
    raw_col = "close_raw" if "close_raw" in price_lookup.columns else "close"
    try:
        hist = price_lookup.xs(ticker, level="ticker")[raw_col]
    except Exception:
        return False

    hist = hist[hist.index <= as_of_date].tail(lookback_sessions + 1)
    if len(hist) < 2:
        return False

    prev = hist.shift(1)
    ratios = hist / prev
    event_mask = (
        (ratios >= split_ratio_threshold) |
        (ratios <= (1.0 / split_ratio_threshold))
    )
    return bool(event_mask.fillna(False).any())


def _apply_execution_cost(price: float, shares: float, execution_spread: float) -> tuple[float, float]:
    trade_value = shares * price
    adj_value = trade_value * (1.0 - execution_spread)
    adj_shares = adj_value / price if price > 0 else 0.0
    return adj_value, adj_shares


def _apply_sell_execution_cost(price: float, execution_spread: float) -> float:
    return float(price) * (1.0 - float(execution_spread))


def _execute_replay_decisions(
    portfolio,
    decisions: list,
    execution_spread: float,
    trade_log: list,
    date,
    price_lookup: pd.DataFrame | None = None,
    decision_date=None,
) -> None:
    fill_date = pd.Timestamp(date)
    decision_date = fill_date if decision_date is None else pd.Timestamp(decision_date)

    for d in decisions:
        fill_price = float(d.price)
        if price_lookup is not None:
            quote = _get_historical_quote(price_lookup, d.ticker, fill_date)
            if quote is None:
                continue
            fill_price, _fill_volume = quote

        ok = False
        traded_shares = 0.0
        if d.action == "BUY":
            target_value = max(float(d.shares), 0.0) * max(float(d.price), 0.0)
            if fill_price <= 0 or target_value <= 0:
                continue
            fill_shares = target_value / fill_price
            execution_cost = fill_shares * fill_price * float(execution_spread)
            _adj_value, adj_shares = _apply_execution_cost(fill_price, fill_shares, execution_spread)
            ok = portfolio.buy(d.ticker, adj_shares, fill_price, d.reason)
            traded_shares = float(adj_shares)
            if ok and hasattr(d, "_rl_score_at_entry") and d.ticker in portfolio.positions:
                portfolio.positions[d.ticker]["rl_score_at_entry"] = d._rl_score_at_entry
        elif d.action == "SELL":
            if d.ticker in portfolio.positions:
                traded_shares = float(portfolio.positions[d.ticker]["shares"])
            execution_cost = traded_shares * fill_price * float(execution_spread)
            execution_price = _apply_sell_execution_cost(fill_price, execution_spread)
            ok = portfolio.sell_all(d.ticker, execution_price, d.reason)
            fill_price = execution_price
        elif d.action == "SELL_PARTIAL":
            if d.ticker in portfolio.positions:
                traded_shares = float(
                    min(float(d.shares), float(portfolio.positions[d.ticker]["shares"]))
                )
            execution_cost = traded_shares * fill_price * float(execution_spread)
            execution_price = _apply_sell_execution_cost(fill_price, execution_spread)
            ok = portfolio.sell(d.ticker, d.shares, execution_price, d.reason)
            fill_price = execution_price
            if ok and d.ticker in portfolio.positions:
                portfolio.positions[d.ticker]["partial_taken"] = True
        else:
            ok = True
            execution_cost = 0.0

        if ok and d.action in ("BUY", "SELL", "SELL_PARTIAL"):
            trade_log.append({
                "date": str(fill_date),
                "decision_date": str(decision_date),
                "fill_date": str(fill_date),
                "action": d.action,
                "ticker": d.ticker,
                "shares": traded_shares,
                "price": fill_price,
                "decision_price": float(d.price),
                "execution_cost": float(execution_cost),
                "execution_model": f"replay_spread:{float(execution_spread):.4f}",
                "score": getattr(d, "score", 0.0),
                "reason": d.reason,
            })


def _make_historical_price_fetcher(price_lookup: pd.DataFrame, as_of_date):
    def _get_current_prices(_self, tickers: list[str]) -> dict[str, float]:
        prices = {}
        for ticker in tickers:
            try:
                price = float(price_lookup.loc[(as_of_date, ticker), "close"])
                if price > 0:
                    prices[ticker] = price
            except Exception:
                continue
        return prices
    return _get_current_prices


def _build_price_lookup(parquet_path: str = "MasterDS/stooq_panel.parquet") -> pd.DataFrame:
    """Load replay prices and smooth obvious split-scale jumps in close data."""
    df = pd.read_parquet(parquet_path)
    df = df.reset_index()
    df["date"]   = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_convert(None).dt.normalize()
    df["ticker"] = df["ticker"].str.upper()
    df = df.set_index(["date", "ticker"])[["close", "volume"]].sort_index()
    df["close_raw"] = df["close"]

    adjusted_parts: list[pd.DataFrame] = []
    for ticker, grp in df.groupby(level="ticker", sort=False):
        grp = grp.copy()
        close_raw = grp["close_raw"].droplevel("ticker")
        grp["close"] = _adjust_replay_close_series(close_raw).to_numpy(dtype=np.float64)
        adjusted_parts.append(grp)

    return pd.concat(adjusted_parts).sort_index()


# ── Replay portfolio (in-memory, no disk I/O) ─────────────────────────────────

class ReplayPortfolio:
    """
    Lightweight in-memory portfolio for replay — same interface as Portfolio
    but doesn't persist to disk and doesn't load options book.
    """

    def __init__(self, initial_cash: float):
        self.initial_cash = initial_cash
        self.cash         = initial_cash
        self.positions    = {}
        self.trade_log    = []
        self.cash_yield_last_date = None

    def buy(self, ticker: str, shares: float, price: float, reason: str = "") -> bool:
        cost = shares * price
        if cost > self.cash:
            shares = self.cash / price
            cost   = shares * price
        if shares < 0.001:
            return False
        if ticker in self.positions:
            pos = self.positions[ticker]
            total = pos["shares"] + shares
            pos["avg_cost"] = (pos["shares"] * pos["avg_cost"] + cost) / total
            pos["shares"]   = total
            pos["last_price"] = price
            pos["peak_price"] = max(float(pos.get("peak_price", price)), float(price))
            pos.setdefault("weak_signal_streak", 0)
        else:
            self.positions[ticker] = {
                "shares": shares, "avg_cost": price,
                "last_price": price, "partial_taken": False,
                "peak_price": price, "weak_signal_streak": 0,
            }
        self.cash -= cost
        self.trade_log.append({"date": None, "action": "BUY", "ticker": ticker,
                                "shares": shares, "price": price})
        return True

    def sell(self, ticker: str, shares: float, price: float, reason: str = "") -> bool:
        if ticker not in self.positions:
            return False
        pos    = self.positions[ticker]
        shares = min(shares, pos["shares"])
        if shares < 0.001:
            return False
        self.cash       += shares * price
        pos["shares"]   -= shares
        if pos["shares"] < 0.001:
            del self.positions[ticker]
        self.trade_log.append({"date": None, "action": "SELL", "ticker": ticker,
                                "shares": shares, "price": price})
        return True

    def sell_all(self, ticker: str, price: float, reason: str = "") -> bool:
        if ticker not in self.positions:
            return False
        return self.sell(ticker, self.positions[ticker]["shares"], price, reason)

    def update_prices(self, prices: dict):
        for ticker, price in prices.items():
            if ticker in self.positions and price > 0:
                self.positions[ticker]["last_price"] = price

    def accrue_cash_yield(self, as_of_date, annual_rate: float = CASH_YIELD_ANNUAL_RATE) -> float:
        current_date = pd.Timestamp(as_of_date).date()
        if self.cash_yield_last_date is None:
            self.cash_yield_last_date = current_date
            return 0.0
        if current_date <= self.cash_yield_last_date:
            return 0.0

        last_date = self.cash_yield_last_date
        self.cash_yield_last_date = current_date
        if self.cash <= 0 or annual_rate <= 0:
            return 0.0

        days_elapsed = (current_date - last_date).days
        growth = (1.0 + annual_rate) ** (days_elapsed / DAYS_PER_YEAR)
        starting_cash = self.cash
        self.cash *= growth
        return self.cash - starting_cash

    @property
    def equity(self) -> float:
        return self.cash + sum(
            p["shares"] * p["last_price"] for p in self.positions.values()
        )

    @property
    def total_return(self) -> float:
        return (self.equity / self.initial_cash) - 1.0

    @property
    def position_values(self) -> dict:
        return {t: p["shares"] * p["last_price"] for t, p in self.positions.items()}

    # Stub options interface so BrokerBrain doesn't crash
    class _NoOpOptions:
        positions = {}
        total_reserved_cash = 0.0
        def check_expirations(self, *a, **kw): return []
        def summary_lines(self): return []

    options = _NoOpOptions()

    def save(self): pass   # no-op for replay


# ── Core replay loop ──────────────────────────────────────────────────────────

def run_replay(
    df_features: pd.DataFrame,
    price_lookup: pd.DataFrame,
    strategy: str = "heuristics_only",
    checkpoint_path: str | None = None,
    initial_cash: float = 10_000.0,
    rebalance_freq: int = 5,
    max_positions: int = 20,
    max_position_pct: float = 0.10,
    min_score: float = 0.60,
    stop_loss_floor: float = 0.07,
    take_profit: float = 1.00,
    trailing_stop_pct: float = 0.12,
    trailing_activation_pct: float = 0.18,
    signal_exit_score: float = 0.18,
    signal_exit_grace_cycles: int = 2,
    max_daily_loss: float = 0.03,
    max_drawdown: float = 0.15,
    max_gross_exposure: float = 0.95,
    cash_floor: float = 0.05,
    target_volatility: float = 0.15,
    vol_lookback: int = 20,
    partial_profit_pct: float = 0.35,
    penny_pct: float = 0.20,
    max_sector_pct: float = 0.40,
    max_pair_correlation: float = 0.80,
    correlation_lookback_days: int = 60,
    weak_theme_min_positions: int = 2,
    weak_theme_return_threshold: float = -0.03,
    weak_theme_penalty_mult: float = 0.50,
    weak_theme_cooldown_cycles: int = 0,
    weak_theme_cooldown_min_hits: int = 2,
    low_price_rank_policy: str = "late_cap",
    low_price_rank_penalty_mult: float = 0.70,
    low_price_high_rank_floor: float = 0.80,
    avoid_earnings_days: int = 3,
    execution_spread: float = 0.001,
    rl_phase: int = 1,
    rl_exit_threshold: float = 0.30,
    rl_conviction_drop: float = 0.20,
    rl_min_score: float = 0.0,
    dead_money_days: int = 0,
    dead_money_min_return: float = 0.02,
    label: str | None = None,
) -> tuple[np.ndarray, list]:
    """
    Run the broker decision logic over historical data.
    Delegates to _run_replay_v2 which uses the full BrokerBrain pipeline.
    """
    return _run_replay_v2(
        df_features=df_features,
        price_lookup=price_lookup,
        strategy=strategy,
        checkpoint_path=checkpoint_path,
        initial_cash=initial_cash,
        rebalance_freq=rebalance_freq,
        max_positions=max_positions,
        max_position_pct=max_position_pct,
        min_score=min_score,
        stop_loss_floor=stop_loss_floor,
        take_profit=take_profit,
        trailing_stop_pct=trailing_stop_pct,
        trailing_activation_pct=trailing_activation_pct,
        signal_exit_score=signal_exit_score,
        signal_exit_grace_cycles=signal_exit_grace_cycles,
        max_daily_loss=max_daily_loss,
        max_drawdown=max_drawdown,
        max_gross_exposure=max_gross_exposure,
        cash_floor=cash_floor,
        target_volatility=target_volatility,
        vol_lookback=vol_lookback,
        partial_profit_pct=partial_profit_pct,
        penny_pct=penny_pct,
        max_sector_pct=max_sector_pct,
        max_pair_correlation=max_pair_correlation,
        correlation_lookback_days=correlation_lookback_days,
        weak_theme_min_positions=weak_theme_min_positions,
        weak_theme_return_threshold=weak_theme_return_threshold,
        weak_theme_penalty_mult=weak_theme_penalty_mult,
        weak_theme_cooldown_cycles=weak_theme_cooldown_cycles,
        weak_theme_cooldown_min_hits=weak_theme_cooldown_min_hits,
        low_price_rank_policy=low_price_rank_policy,
        low_price_rank_penalty_mult=low_price_rank_penalty_mult,
        low_price_high_rank_floor=low_price_high_rank_floor,
        avoid_earnings_days=avoid_earnings_days,
        execution_spread=execution_spread,
        rl_phase=rl_phase,
        rl_exit_threshold=rl_exit_threshold,
        rl_conviction_drop=rl_conviction_drop,
        rl_min_score=rl_min_score,
        dead_money_days=dead_money_days,
        dead_money_min_return=dead_money_min_return,
        label=label,
    )


# ── Sensitivity sweep ─────────────────────────────────────────────────────────

def _run_replay_v2(
    df_features: pd.DataFrame,
    price_lookup: pd.DataFrame,
    strategy: str = "heuristics_only",
    checkpoint_path: str | None = None,
    initial_cash: float = 10_000.0,
    rebalance_freq: int = 5,
    max_positions: int = 20,
    max_position_pct: float = 0.10,
    min_score: float = 0.60,
    stop_loss_floor: float = 0.07,
    take_profit: float = 1.00,
    trailing_stop_pct: float = 0.12,
    trailing_activation_pct: float = 0.18,
    signal_exit_score: float = 0.18,
    signal_exit_grace_cycles: int = 2,
    max_daily_loss: float = 0.03,
    max_drawdown: float = 0.15,
    max_gross_exposure: float = 0.95,
    cash_floor: float = 0.05,
    target_volatility: float = 0.15,
    vol_lookback: int = 20,
    partial_profit_pct: float = 0.35,
    penny_pct: float = 0.20,
    max_sector_pct: float = 0.40,
    max_pair_correlation: float = 0.80,
    correlation_lookback_days: int = 60,
    weak_theme_min_positions: int = 2,
    weak_theme_return_threshold: float = -0.03,
    weak_theme_penalty_mult: float = 0.50,
    weak_theme_cooldown_cycles: int = 0,
    weak_theme_cooldown_min_hits: int = 2,
    low_price_rank_policy: str = "late_cap",
    low_price_rank_penalty_mult: float = 0.70,
    low_price_high_rank_floor: float = 0.80,
    avoid_earnings_days: int = 3,
    execution_spread: float = 0.001,
    rl_phase: int = 1,
    rl_exit_threshold: float = 0.30,
    rl_conviction_drop: float = 0.20,
    rl_min_score: float = 0.0,
    dead_money_days: int = 0,
    dead_money_min_return: float = 0.02,
    label: str | None = None,
) -> tuple[np.ndarray, list]:
    global _LAST_REPLAY_SCORE_AUDIT
    _LAST_REPLAY_SCORE_AUDIT = pd.DataFrame()
    if label is None:
        label = strategy

    import broker.brain as brain_module
    from broker.brain import BrokerBrain
    from broker.risk import PortfolioRiskEngine
    from broker.sectors import (
        compute_target_allocations,
        get_cached_sector_map,
        get_portfolio_sector_weights,
        score_sectors,
    )

    if strategy not in {"heuristics_only", "screener_heuristics", "screener_rl", "rl_weights"}:
        raise ValueError(f"Unknown replay strategy: {strategy}")

    dates = sorted(df_features.index.get_level_values("date").unique())
    portfolio = ReplayPortfolio(initial_cash)
    sector_map = get_cached_sector_map(
        df_features.index.get_level_values("ticker").unique().tolist()
    )

    # ── Pre-sort df_features by date for fast O(1) slicing ───────────────────
    # Instead of filtering the full DataFrame on every rebalance day
    # (O(N) per day × 150 days = very slow), we sort once and use
    # searchsorted to get the cutoff index in O(log N).
    df_sorted = df_features.sort_index(level="date")
    date_index_values = df_sorted.index.get_level_values("date")

    def _slice_up_to(date):
        """Return df_sorted rows with date <= date, using fast binary search."""
        pos = date_index_values.searchsorted(date, side="right")
        return df_sorted.iloc[:pos]

    equity_curve = [initial_cash]
    risk = PortfolioRiskEngine(
        max_daily_loss=max_daily_loss,
        max_drawdown=max_drawdown,
        max_gross_exposure=max_gross_exposure,
        cash_floor=cash_floor,
        target_volatility=target_volatility,
        vol_lookback=vol_lookback,
        equity_history_getter=lambda: equity_curve,
    )
    trade_log = []
    score_audit_parts: list[pd.DataFrame] = []
    pending_fills: dict[pd.Timestamp, list[tuple[pd.Timestamp, list]]] = {}

    original_research = getattr(brain_module, "research", None)
    original_get_next_earnings = getattr(brain_module, "_get_next_earnings_date", None)

    def _heuristic_screen(_self, features: pd.DataFrame, top_n: int = 100) -> list[str]:
        try:
            last_date = sorted(features.index.get_level_values("date").unique())[-1]
            snap = features.loc[last_date].copy()
            snap["_rank"] = (
                snap.get("ret_5d", 0) * 0.3
                + snap.get("vol_ratio", 0) * 0.2
                + snap.get("sent_net", 0) * 0.3
                + snap.get("macd_hist", 0) * 0.2
            )
            return snap.nlargest(top_n, "_rank").index.tolist()
        except Exception:
            return []

    def _no_options(_self, researched: list[dict], features: pd.DataFrame) -> list:
        return []

    def _make_static_sector_refresher(sectors: dict[str, str]):
        def _refresh(_self, _features: pd.DataFrame):
            _self._sector_map = sectors
            if getattr(_self, "_sector_cache_date", None) is None:
                _self._sector_cache_date = datetime.utcnow()
        return _refresh

    def _make_historical_stop_loss(features: pd.DataFrame):
        def _get_stop_loss_pct(_self, ticker: str, pos: dict) -> float:
            try:
                hist = features.xs(ticker, level="ticker")["ret_1d"].dropna().values[-14:]
                if len(hist) >= 5:
                    atr_pct = float(np.std(hist)) * _self.stop_loss_atr_mult
                    return float(np.clip(atr_pct, _self.stop_loss_pct_floor, _self.stop_loss_pct_ceil))
            except Exception:
                pass
            return _self.stop_loss_pct_floor
        return _get_stop_loss_pct

    def _run_legacy_rl_weights() -> tuple[np.ndarray, list]:
        pbar = tqdm(
            range(len(dates)),
            desc=f"Replay: {label}",
            unit="day",
            colour="blue",
            dynamic_ncols=True,
        )
        for i in pbar:
            date = dates[i]
            portfolio.accrue_cash_yield(date)
            if portfolio.positions:
                prices = {}
                for ticker in list(portfolio.positions.keys()):
                    try:
                        price = float(price_lookup.loc[(date, ticker), "close"])
                        if price > 0:
                            prices[ticker] = price
                    except Exception:
                        continue
                portfolio.update_prices(prices)

            if i % rebalance_freq == 0:
                df_slice = _slice_up_to(date)
                if not df_slice.empty:
                    brain_module.research = _make_historical_research(df_slice, price_lookup, date)
                    try:
                        snap = df_slice.loc[date].copy()
                    except KeyError:
                        snap = None

                    if snap is not None:
                        screener_candidates = None
                        try:
                            from pipeline.screener import run_screener
                            import torch

                            screener_df = run_screener(df_slice, device=torch.device("cpu"), top_n=50)
                            if not screener_df.empty and "ticker" in screener_df.columns:
                                screener_candidates = screener_df["ticker"].tolist()
                        except Exception as exc:
                            logger.warning(
                                "Screener unavailable on %s (%s) - falling back to rule-based shortlist",
                                date,
                                exc,
                            )

                        if screener_candidates is None:
                            snap["_score"] = (
                                snap.get("ret_5d", 0) * 0.3
                                + snap.get("vol_ratio", 0) * 0.2
                                + snap.get("sent_net", 0) * 0.3
                                + snap.get("macd_hist", 0) * 0.2
                            )
                            screener_candidates = snap.nlargest(50, "_score").index.tolist()

                        rl_weights = None
                        if checkpoint_path is not None:
                            try:
                                from pipeline.rl_inference import get_rl_targets

                                rl_weights = get_rl_targets(
                                    df_slice,
                                    screener_candidates,
                                    checkpoint_path,
                                    mode="weights",
                                )
                            except Exception as exc:
                                logger.warning(
                                    "RL weight inference failed on %s (%s) - falling back to no trades",
                                    date,
                                    exc,
                                )

                        sector_scores_map = score_sectors(df_slice, sector_map)
                        current_sw = get_portfolio_sector_weights(portfolio.positions, sector_map)
                        target_allocs = compute_target_allocations(
                            sector_scores_map,
                            current_sw,
                            max_single_sector=max_sector_pct,
                        )

                        n_slots = max_positions - len(portfolio.positions)
                        equity = portfolio.equity
                        penny_value = sum(
                            value
                            for ticker, value in portfolio.position_values.items()
                            if portfolio.positions[ticker]["last_price"] < 5.0
                        )
                        sector_spent: dict[str, float] = {}

                        for ticker in screener_candidates:
                            if n_slots <= 0 or ticker in portfolio.positions or ticker == "CASH":
                                continue

                            try:
                                price = float(price_lookup.loc[(date, ticker), "close"])
                                if price <= 0:
                                    continue
                            except Exception:
                                continue

                            weight = float(rl_weights.get(ticker, 0.0)) if rl_weights is not None else 0.0
                            if weight <= 0.0:
                                continue

                            is_penny = price < 5.0
                            sector = sector_map.get(ticker, "Unknown")
                            target_alloc = target_allocs.get(sector, 0.05)
                            current_sv = sum(
                                value
                                for held, value in portfolio.position_values.items()
                                if sector_map.get(held, "Unknown") == sector
                            ) + sector_spent.get(sector, 0.0)
                            sector_budget = equity * target_alloc - current_sv
                            if sector_budget <= equity * 0.01:
                                continue

                            alloc_value = min(weight * equity, sector_budget, portfolio.cash * 0.95)
                            if is_penny:
                                alloc_value = min(alloc_value, equity * penny_pct - penny_value)
                            if alloc_value < 1.0:
                                continue

                            shares = alloc_value / price
                            if shares < 0.001:
                                continue

                            portfolio.buy(ticker, shares, price, f"weight={weight:.4f}")
                            trade_log.append(
                                {
                                    "date": str(date),
                                    "action": "BUY",
                                    "ticker": ticker,
                                    "price": price,
                                    "score": weight,
                                    "sector": sector,
                                    "strategy": strategy,
                                }
                            )
                            sector_spent[sector] = sector_spent.get(sector, 0.0) + alloc_value
                            if is_penny:
                                penny_value += alloc_value
                            n_slots -= 1

            equity_curve.append(portfolio.equity)

            pbar.set_postfix(
                equity=f"${portfolio.equity:,.0f}",
                ret=f"{portfolio.total_return:+.1%}",
                pos=len(portfolio.positions),
            )

        returns = np.diff(equity_curve) / (np.array(equity_curve[:-1]) + 1e-9)
        return returns, trade_log

    if strategy == "rl_weights":
        try:
            return _run_legacy_rl_weights()
        finally:
            if original_research is not None:
                brain_module.research = original_research

    brain = BrokerBrain(
        portfolio=portfolio,
        max_positions=max_positions,
        max_position_pct=max_position_pct,
        stop_loss_pct_floor=stop_loss_floor,
        partial_profit_pct=partial_profit_pct,
        full_profit_pct=take_profit,
        trailing_stop_pct=trailing_stop_pct,
        trailing_activation_pct=trailing_activation_pct,
        signal_exit_score=signal_exit_score,
        signal_exit_grace_cycles=signal_exit_grace_cycles,
        min_score=min_score,
        penny_max_pct=penny_pct,
        max_sector_pct=max_sector_pct,
        max_pair_correlation=max_pair_correlation,
        correlation_lookback_days=correlation_lookback_days,
        weak_theme_min_positions=weak_theme_min_positions,
        weak_theme_return_threshold=weak_theme_return_threshold,
        weak_theme_penalty_mult=weak_theme_penalty_mult,
        weak_theme_cooldown_cycles=weak_theme_cooldown_cycles,
        weak_theme_cooldown_min_hits=weak_theme_cooldown_min_hits,
        low_price_rank_policy=low_price_rank_policy,
        low_price_rank_penalty_mult=low_price_rank_penalty_mult,
        low_price_high_rank_floor=low_price_high_rank_floor,
        avoid_earnings_days=avoid_earnings_days,
        device=None,
        rl_enabled=(strategy == "screener_rl"),
        rl_checkpoint_path=checkpoint_path,
        rl_phase=rl_phase,
        rl_exit_threshold=rl_exit_threshold,
        rl_conviction_drop=rl_conviction_drop,
        rl_min_score=rl_min_score,
        dead_money_days=dead_money_days,
        dead_money_min_return=dead_money_min_return,
    )
    brain._base_min_score = min_score
    brain._sector_map = sector_map.copy()
    brain._maybe_refresh_sector_map = MethodType(_make_static_sector_refresher(sector_map.copy()), brain)
    brain._evaluate_options = MethodType(_no_options, brain)
    if strategy == "heuristics_only":
        brain._screen_candidates = MethodType(_heuristic_screen, brain)

    pbar = tqdm(
        range(len(dates)),
        desc=f"Replay: {label}",
        unit="day",
        colour="blue",
        dynamic_ncols=True,
    )

    try:
        brain_module._get_next_earnings_date = lambda ticker: None

        for i in pbar:
            date = dates[i]
            portfolio.accrue_cash_yield(date)
            if portfolio.positions:
                prices = {}
                for ticker in list(portfolio.positions.keys()):
                    try:
                        price = float(price_lookup.loc[(date, ticker), "close"])
                        if price > 0:
                            prices[ticker] = price
                    except Exception:
                        continue
                portfolio.update_prices(prices)

            scheduled_fills = pending_fills.pop(pd.Timestamp(date), [])
            for decision_date, decisions in scheduled_fills:
                _execute_replay_decisions(
                    portfolio=portfolio,
                    decisions=decisions,
                    execution_spread=execution_spread,
                    trade_log=trade_log,
                    date=date,
                    price_lookup=price_lookup,
                    decision_date=decision_date,
                )

            if i % rebalance_freq == 0:
                df_slice = _slice_up_to(date)
                if not df_slice.empty:
                    brain_module.research = _make_historical_research(df_slice, price_lookup, date)
                    brain._get_current_prices = MethodType(_make_historical_price_fetcher(price_lookup, date), brain)
                    brain._get_stop_loss_pct = MethodType(_make_historical_stop_loss(df_slice), brain)
                    brain.min_score = brain._base_min_score

                    risk.set_market_regime(brain._current_market_regime(df_slice))
                    risk.start_session(portfolio.equity, session_key=pd.Timestamp(date).date())
                    health_status, _health_reason = risk.check_portfolio_health(portfolio)
                    if health_status == "halt":
                        brain.min_score = 999.0

                    decisions = brain.run_cycle(df_slice, screener_top_n=50, risk_engine=risk)
                    cycle_audit = getattr(brain, "_last_cycle_audit", pd.DataFrame())
                    if isinstance(cycle_audit, pd.DataFrame) and not cycle_audit.empty:
                        score_audit_parts.append(
                            cycle_audit.assign(cycle_date=pd.Timestamp(date).date().isoformat())
                        )
                    if decisions and i + 1 < len(dates):
                        fill_date = pd.Timestamp(dates[i + 1])
                        pending_fills.setdefault(fill_date, []).append(
                            (pd.Timestamp(date), decisions)
                        )

            equity_curve.append(portfolio.equity)

            pbar.set_postfix(
                equity=f"${portfolio.equity:,.0f}",
                ret=f"{portfolio.total_return:+.1%}",
                pos=len(portfolio.positions),
            )
    finally:
        if original_research is not None:
            brain_module.research = original_research
        if original_get_next_earnings is not None:
            brain_module._get_next_earnings_date = original_get_next_earnings
        if score_audit_parts:
            _LAST_REPLAY_SCORE_AUDIT = pd.concat(score_audit_parts, ignore_index=True)
        else:
            _LAST_REPLAY_SCORE_AUDIT = pd.DataFrame()

    daily_returns = np.diff(equity_curve) / (np.array(equity_curve[:-1]) + 1e-9)

    # ── P9: Replay invariant checks ───────────────────────────────────────────
    _check_replay_invariants(trade_log, dates)

    return daily_returns, trade_log


def _check_replay_invariants(trade_log: list, dates: list) -> dict[str, bool]:
    """
    Assert post-hoc invariants on a completed replay.

    Returns a dict of {invariant_name: passed} and logs warnings for failures.
    These are "trust but verify" checks — they catch impossible states that
    indicate bugs in the replay logic.
    """
    results: dict[str, bool] = {}
    date_set = {pd.Timestamp(d) for d in dates}

    # 1. No fill date before signal date
    fill_before_signal = [
        t for t in trade_log
        if t.get("fill_date") and t.get("decision_date")
        and pd.Timestamp(t["fill_date"]) < pd.Timestamp(t["decision_date"])
    ]
    results["no_fill_before_signal"] = len(fill_before_signal) == 0
    if fill_before_signal:
        logger.warning(
            "INVARIANT FAIL: %d trade(s) have fill_date before decision_date (lookahead): %s",
            len(fill_before_signal),
            [t["ticker"] for t in fill_before_signal[:3]],
        )

    # 2. No fill date outside the replay date range
    out_of_range = [
        t for t in trade_log
        if t.get("fill_date") and pd.Timestamp(t["fill_date"]) not in date_set
    ]
    results["fills_within_replay_dates"] = len(out_of_range) == 0
    if out_of_range:
        logger.warning(
            "INVARIANT FAIL: %d trade(s) filled on dates outside the replay window.",
            len(out_of_range),
        )

    # 3. No negative share counts in trade log
    negative_shares = [t for t in trade_log if float(t.get("shares", 0)) < 0]
    results["no_negative_shares"] = len(negative_shares) == 0
    if negative_shares:
        logger.warning(
            "INVARIANT FAIL: %d trade(s) have negative share counts.",
            len(negative_shares),
        )

    # 4. No zero-price fills
    zero_price = [t for t in trade_log if float(t.get("price", 1)) <= 0]
    results["no_zero_price_fills"] = len(zero_price) == 0
    if zero_price:
        logger.warning(
            "INVARIANT FAIL: %d trade(s) filled at zero or negative price.",
            len(zero_price),
        )

    # 5. All actions are known types
    known_actions = {"BUY", "SELL", "SELL_PARTIAL", "DIVIDEND"}
    unknown_actions = [t for t in trade_log if t.get("action") not in known_actions]
    results["all_actions_known"] = len(unknown_actions) == 0
    if unknown_actions:
        logger.warning(
            "INVARIANT FAIL: %d trade(s) have unknown action types: %s",
            len(unknown_actions),
            list({t.get("action") for t in unknown_actions}),
        )

    passed = sum(results.values())
    total = len(results)
    if passed == total:
        logger.info("Replay invariants: all %d checks passed.", total)
    else:
        logger.warning(
            "Replay invariants: %d/%d checks passed. See warnings above.",
            passed, total,
        )

    return results


run_replay = _run_replay_v2


def _resolve_replay_strategy(live_config: dict | None = None) -> str:
    import os
    from pipeline.screener import SCREENER_CKPT

    live_config = live_config or {}
    if live_config.get("rl_enabled", False):
        return "screener_rl"
    if os.path.exists(SCREENER_CKPT):
        return "screener_heuristics"
    return "heuristics_only"


def _replay_kwargs_from_live_config(live_config: dict | None = None) -> dict:
    live_config = live_config or {}
    return {
        "max_positions": int(live_config.get("max_positions", 20)),
        "max_position_pct": float(live_config.get("max_position_pct", 0.10)),
        "min_score": float(live_config.get("min_score", 0.60)),
        "stop_loss_floor": float(live_config.get("stop_loss", 0.07)),
        "take_profit": float(live_config.get("take_profit", 1.00)),
        "trailing_stop_pct": float(live_config.get("trailing_stop", 0.12)),
        "trailing_activation_pct": float(live_config.get("trailing_activation", 0.18)),
        "signal_exit_score": float(live_config.get("signal_exit_score", 0.18)),
        "signal_exit_grace_cycles": int(live_config.get("signal_exit_grace", 2)),
        "max_daily_loss": float(live_config.get("max_daily_loss", 0.03)),
        "max_drawdown": float(live_config.get("max_drawdown", 0.15)),
        "max_gross_exposure": float(live_config.get("max_gross_exposure", 0.95)),
        "cash_floor": float(live_config.get("cash_floor", 0.05)),
        "target_volatility": float(live_config.get("target_volatility", 0.15)),
        "vol_lookback": int(live_config.get("vol_lookback", 20)),
        "partial_profit_pct": float(live_config.get("partial_profit", 0.35)),
        "penny_pct": float(live_config.get("penny_pct", 0.20)),
        "max_sector_pct": float(live_config.get("max_sector", 0.40)),
        "max_pair_correlation": float(live_config.get("max_correlation", 0.80)),
        "weak_theme_min_positions": int(live_config.get("weak_theme_min_positions", 2)),
        "weak_theme_return_threshold": float(live_config.get("weak_theme_return_threshold", -0.03)),
        "weak_theme_penalty_mult": float(live_config.get("weak_theme_penalty_mult", 0.50)),
        "weak_theme_cooldown_cycles": int(live_config.get("weak_theme_cooldown_cycles", 0)),
        "weak_theme_cooldown_min_hits": int(live_config.get("weak_theme_cooldown_min_hits", 2)),
        "low_price_rank_policy": str(live_config.get("low_price_rank_policy", "late_cap")),
        "low_price_rank_penalty_mult": float(live_config.get("low_price_rank_penalty_mult", 0.70)),
        "low_price_high_rank_floor": float(live_config.get("low_price_high_rank_floor", 0.80)),
        "avoid_earnings_days": int(live_config.get("avoid_earnings", 3)),
        "execution_spread": float(live_config.get("execution_spread", 0.001)),
        "rl_phase": int(live_config.get("rl_phase", 1)),
        "rl_exit_threshold": float(live_config.get("rl_exit_threshold", 0.30)),
        "rl_conviction_drop": float(live_config.get("rl_conviction_drop", 0.20)),
        "rl_min_score": float(live_config.get("rl_min_score", 0.0)),
        "dead_money_days": int(live_config.get("dead_money_days", 0)),
        "dead_money_min_return": float(live_config.get("dead_money_min_return", 0.02)),
    }


def run_sensitivity(
    df_features: pd.DataFrame,
    price_lookup: pd.DataFrame,
    initial_cash: float = 10_000.0,
    live_config: dict | None = None,
    strategy: str | None = None,
    checkpoint_path: str | None = None,
) -> pd.DataFrame:
    """
    Run the replay across a grid of parameter perturbations around the
    current live config. Shows whether results are robust or knife-edge.
    """
    from pipeline.benchmark import compute_metrics
    from broker.paper_diagnostics import summarize_low_price_signal_suppression
    global _LAST_REPLAY_SCORE_AUDIT

    base = _replay_kwargs_from_live_config(live_config)
    strategy = strategy or _resolve_replay_strategy(live_config)

    scenarios = [{**base, "label": "current_config (base)"}]

    def add(label: str, **overrides) -> None:
        scenarios.append({**base, **overrides, "label": label})

    add("stop=7%", stop_loss_floor=0.07)
    add("stop=10%", stop_loss_floor=0.10)
    add("tp=50%", take_profit=0.50)
    add("tp=65%", take_profit=0.65)
    add("trail=10%", trailing_stop_pct=0.10)
    add("trail=15%", trailing_stop_pct=0.15)
    add("signal_exit=0.15", signal_exit_score=0.15)
    add("signal_exit=0.25", signal_exit_score=0.25)
    add("max_pos=15%", max_position_pct=0.15)
    add("max_pos=20%", max_position_pct=0.20)
    add("cash_floor=1%", cash_floor=0.01)
    add("cash_floor=3%", cash_floor=0.03)
    add("gross=99%", max_gross_exposure=0.99)
    add("target_vol=22%", target_volatility=0.22)
    add("target_vol=30%", target_volatility=0.30)
    add("sector=45%", max_sector_pct=0.45)
    add("weak_sleeve=50%", weak_theme_penalty_mult=0.50)
    add("weak_sleeve=25%", weak_theme_penalty_mult=0.25)
    add("weak_sleeve=block", weak_theme_penalty_mult=0.0)
    add("weak_sleeve=cooldown2", weak_theme_cooldown_cycles=2)
    add("low_price=pre_penalty", low_price_rank_policy="pre_penalty")
    add("low_price=exclude_high_rank", low_price_rank_policy="exclude_high_rank")
    add("earnings=4d", avoid_earnings_days=4)
    add("spread=5bps", execution_spread=0.0005)
    add("spread=20bps", execution_spread=0.0020)

    if strategy == "screener_rl":
        add("rl_min=5%", rl_min_score=0.05)
        add("rl_min=10%", rl_min_score=0.10)
        add("rl_min=20%", rl_min_score=0.20)
        add("rl_phase=2", rl_phase=2)
        add("rl_phase2 exit=15%", rl_phase=2, rl_exit_threshold=0.15)
        add("rl_phase2 drop=25%", rl_phase=2, rl_conviction_drop=0.25)
    else:
        add("min_score=0.55", min_score=0.55)
        add("min_score=0.65", min_score=0.65)

    rows = []
    seen = set()
    for scenario in scenarios:
        params = dict(scenario)
        label = params.pop("label")
        key = tuple(sorted(params.items()))
        if key in seen:
            continue
        seen.add(key)

        _LAST_REPLAY_SCORE_AUDIT = pd.DataFrame()
        rets, trade_log = run_replay(
            df_features,
            price_lookup,
            strategy=strategy,
            checkpoint_path=checkpoint_path,
            initial_cash=initial_cash,
            label=label,
            **params,
        )
        m = compute_metrics(rets, label)
        policy_metrics = _summarize_replay_control_metrics(
            trade_log,
            globals().get("_LAST_REPLAY_SCORE_AUDIT", pd.DataFrame()),
            summarize_low_price_signal_suppression,
        )
        rows.append({
            "params":       label,
            "total_return": m["total_return"],
            "ann_return":   m["ann_return"],
            "sharpe":       m["sharpe"],
            "max_drawdown": m["max_drawdown"],
            "win_rate":     m["win_rate"],
            **policy_metrics,
        })

    return pd.DataFrame(rows)


def _summarize_replay_control_metrics(
    trade_log: list[dict],
    score_audit: pd.DataFrame | None,
    low_price_summary_fn,
) -> dict:
    low_price = low_price_summary_fn(trade_log)
    weak_reentries = 0
    weak_reentry_themes: set[str] = set()
    for rec in trade_log or []:
        if str(rec.get("action", "")).upper() != "BUY":
            continue
        reason = str(rec.get("reason", "") or "")
        if "WeakSleeve=" not in reason:
            continue
        weak_reentries += 1
        match = re.search(r"(?:^|\|\s*)Theme=([^|]+)", reason)
        if match:
            weak_reentry_themes.add(match.group(1).split()[0].strip())

    metrics = {
        "weak_sleeve_reentry_count": weak_reentries,
        "weak_sleeve_reentry_theme_count": len(weak_reentry_themes),
        "tokenized_high_rank_low_price_count": int(
            low_price.get("tokenized_high_rank_low_price_entries", 0) or 0
        ),
        "high_rank_low_price_count": int(
            low_price.get("high_rank_low_price_entries", 0) or 0
        ),
        "low_price_tokenized_rate": low_price.get("tokenized_high_rank_low_price_rate"),
    }

    if not isinstance(score_audit, pd.DataFrame) or score_audit.empty:
        metrics.update({
            "avg_top_theme_concentration": np.nan,
            "max_top_theme_concentration": np.nan,
            "avg_low_price_exposure": np.nan,
            "max_low_price_exposure": np.nan,
            "weak_sleeve_selected_count": 0,
        })
        return metrics

    selected = score_audit[
        score_audit.get("candidate_status", pd.Series(index=score_audit.index)).eq("buy_selected")
    ].copy()
    if selected.empty:
        metrics.update({
            "avg_top_theme_concentration": 0.0,
            "max_top_theme_concentration": 0.0,
            "avg_low_price_exposure": 0.0,
            "max_low_price_exposure": 0.0,
            "weak_sleeve_selected_count": 0,
        })
        return metrics

    theme_concentrations = []
    low_price_exposures = []
    for _cycle_date, group in selected.groupby("cycle_date", dropna=False):
        weights = group.get("final_weight", pd.Series(index=group.index, dtype=float)).astype(float)
        if "theme_bucket" in group:
            theme_weights = group.assign(_weight=weights).groupby("theme_bucket")["_weight"].sum()
            theme_concentrations.append(float(theme_weights.max()) if not theme_weights.empty else 0.0)
        low_mask = group.get("low_price_bucket", pd.Series(index=group.index)).isin({"sub_5", "5_to_10"})
        low_price_exposures.append(float(weights[low_mask].sum()))

    weak_impact = selected.get(
        "weak_sleeve_cap_impact",
        pd.Series(0.0, index=selected.index, dtype=float),
    ).astype(float)
    metrics.update({
        "avg_top_theme_concentration": (
            float(np.mean(theme_concentrations)) if theme_concentrations else 0.0
        ),
        "max_top_theme_concentration": (
            float(np.max(theme_concentrations)) if theme_concentrations else 0.0
        ),
        "avg_low_price_exposure": (
            float(np.mean(low_price_exposures)) if low_price_exposures else 0.0
        ),
        "max_low_price_exposure": (
            float(np.max(low_price_exposures)) if low_price_exposures else 0.0
        ),
        "weak_sleeve_selected_count": int((weak_impact > 1e-9).sum()),
    })
    return metrics


# ── Friction regime reporting ─────────────────────────────────────────────────

# Three friction regimes for honest cost-model sensitivity reporting.
# Optimistic: tight spread, no gap risk.
# Base: realistic 10bps spread (default).
# Stressed: 30bps spread + 5% partial-fill penalty on each trade.

FRICTION_REGIMES = {
    "optimistic": {"execution_spread": 0.0002},
    "base":       {"execution_spread": 0.0010},
    "stressed":   {"execution_spread": 0.0030},
}


def run_friction_report(
    df_features: pd.DataFrame,
    price_lookup: pd.DataFrame,
    initial_cash: float = 10_000.0,
    live_config: dict | None = None,
    strategy: str | None = None,
    checkpoint_path: str | None = None,
) -> pd.DataFrame:
    """
    Run the replay under three friction regimes (optimistic / base / stressed)
    and return a DataFrame with gross and net returns for each.

    This makes it easy to see whether reported performance depends heavily on
    the cost model assumptions.

    Returns
    -------
    pd.DataFrame with columns:
        regime, execution_spread, total_return, ann_return, sharpe,
        max_drawdown, win_rate
    """
    from pipeline.benchmark import compute_metrics

    base_kwargs = _replay_kwargs_from_live_config(live_config)
    resolved_strategy = strategy or _resolve_replay_strategy(live_config)

    rows = []
    for regime_name, overrides in FRICTION_REGIMES.items():
        kwargs = {**base_kwargs, **overrides}
        rets, _ = run_replay(
            df_features,
            price_lookup,
            strategy=resolved_strategy,
            checkpoint_path=checkpoint_path,
            initial_cash=initial_cash,
            label=f"friction_{regime_name}",
            **kwargs,
        )
        m = compute_metrics(rets, label=regime_name)
        rows.append({
            "regime":           regime_name,
            "execution_spread": overrides["execution_spread"],
            "total_return":     m["total_return"],
            "ann_return":       m["ann_return"],
            "sharpe":           m["sharpe"],
            "max_drawdown":     m["max_drawdown"],
            "win_rate":         m["win_rate"],
        })

    df = pd.DataFrame(rows)

    # Log a warning if stressed Sharpe is more than 0.3 below base Sharpe —
    # that indicates the strategy is sensitive to execution cost assumptions.
    if len(df) >= 2:
        base_sharpe = float(df.loc[df["regime"] == "base", "sharpe"].iloc[0]) if "base" in df["regime"].values else 0.0
        stressed_sharpe = float(df.loc[df["regime"] == "stressed", "sharpe"].iloc[0]) if "stressed" in df["regime"].values else 0.0
        if base_sharpe - stressed_sharpe > 0.30:
            logger.warning(
                "FRICTION SENSITIVITY: Sharpe drops from %.3f (base) to %.3f (stressed). "
                "Strategy may be sensitive to execution cost assumptions.",
                base_sharpe, stressed_sharpe,
            )

    return df


# ── Full replay report ────────────────────────────────────────────────────────

def run_full_replay(
    df_features: pd.DataFrame,
    initial_cash: float = 10_000.0,
    replay_years: int = 3,
    run_sensitivity_sweep: bool = False,
    save_plot: str = "plots/replay.png",
    live_config: dict | None = None,
    checkpoint_path: str | None = None,
):
    """
    Run the full broker replay and print a side-by-side report vs SPY.
    """
    from pipeline.benchmark import (
        align_return_series,
        compute_trade_friction_metrics,
        fetch_spy_benchmark_data,
        fetch_spy_returns,
        plot_benchmark,
        print_benchmark_report,
        print_trade_friction_report,
    )
    from broker.sectors import get_cached_sector_map

    # Restrict to replay window
    dates  = sorted(df_features.index.get_level_values("date").unique())
    cutoff = pd.Timestamp(dates[-1]) - pd.DateOffset(years=replay_years)
    df_replay = df_features[df_features.index.get_level_values("date") >= cutoff]

    replay_dates = sorted(df_replay.index.get_level_values("date").unique())
    logger.info(
        f"Replay period: {replay_dates[0].date()} → {replay_dates[-1].date()} "
        f"({replay_years} years, {len(replay_dates)} trading days)"
    )

    # Load raw prices
    logger.info("Loading raw prices for replay...")
    price_lookup = _build_price_lookup()

    live_config = live_config or {}
    strategy = _resolve_replay_strategy(live_config)
    replay_kwargs = _replay_kwargs_from_live_config(live_config)

    # Run broker replay
    logger.info("Running broker replay...")
    broker_rets, trade_log = run_replay(
        df_replay,
        price_lookup,
        strategy=strategy,
        checkpoint_path=checkpoint_path,
        initial_cash=initial_cash,
        label="Broker",
        **replay_kwargs,
    )

    # Fetch SPY for same period
    benchmark_bundle = fetch_spy_benchmark_data(
        start=(replay_dates[0] - pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
        end=(replay_dates[-1] + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
    )
    spy_series = benchmark_bundle["returns"]
    if getattr(spy_series, "empty", True) or benchmark_bundle.get("status") != "present":
        compat_spy_series = fetch_spy_returns(
            start=(replay_dates[0] - pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
            end=(replay_dates[-1] + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
        )
        if compat_spy_series is not None and len(compat_spy_series) > 0:
            spy_series = compat_spy_series
            benchmark_bundle = {
                "returns": spy_series,
                "status": "present",
                "source": "compat_fetch_spy_returns",
            }

    # Equal-weight baseline (buy-and-hold all tickers equally)
    logger.info("Computing equal-weight baseline...")
    ew_rets = _equal_weight_returns(df_replay, price_lookup, replay_dates)
    aligned = align_return_series(
        broker_rets,
        replay_dates,
        benchmark_rets=spy_series,
        extra_series={"equal_weight": ew_rets},
    )
    broker_aligned = aligned["portfolio"].to_numpy(dtype=float)
    ew_aligned = (
        aligned["equal_weight"].to_numpy(dtype=float)
        if "equal_weight" in aligned.columns else None
    )
    spy_rets = (
        aligned["benchmark"].to_numpy(dtype=float)
        if "benchmark" in aligned.columns else None
    )
    if spy_rets is None:
        logger.warning("SPY benchmark unavailable for this replay run; skipping relative comparisons.")

    # Print report
    try:
        print_benchmark_report(
            broker_aligned,
            spy_rets,
            ew_rets=ew_aligned,
            label="Broker Replay",
            benchmark_status=str(benchmark_bundle.get("status", "unknown")),
        )
    except TypeError:
        print_benchmark_report(
            broker_aligned,
            spy_rets,
            ew_rets=ew_aligned,
            label="Broker Replay",
        )
    friction = compute_trade_friction_metrics(
        trade_log,
        broker_aligned,
        initial_cash=initial_cash,
        execution_spread=float(replay_kwargs.get("execution_spread", 0.0)),
    )
    spy_total_return = float(np.prod(1.0 + spy_rets) - 1.0) if spy_rets is not None else None
    print_trade_friction_report(friction, spy_total_return=spy_total_return)

    # ── Friction regime summary (P6: always-on) ───────────────────────────────
    try:
        friction_df = run_friction_report(
            df_replay,
            price_lookup,
            initial_cash=initial_cash,
            live_config=live_config,
            strategy=strategy,
            checkpoint_path=checkpoint_path,
        )
        print(f"\n{'='*72}")
        print("  Friction Sensitivity (optimistic / base / stressed execution cost)")
        print(f"{'='*72}")
        print(f"  {'Regime':<12} {'Spread':>8}  {'Return':>8}  {'Sharpe':>8}  {'MaxDD':>8}")
        print(f"  {'-'*52}")
        for _, row in friction_df.iterrows():
            print(
                f"  {row['regime']:<12} {row['execution_spread']:>7.2%}  "
                f"{row['total_return']:>7.2%}  {row['sharpe']:>8.3f}  "
                f"{row['max_drawdown']:>7.2%}"
            )
        print(f"{'='*72}\n")
        # Store in manifest
        friction["regime_sensitivity"] = friction_df.to_dict(orient="records")
    except Exception as exc:
        logger.warning("Friction regime report failed (continuing): %s", exc)

    # Plot
    plot_benchmark(
        broker_aligned,
        spy_rets,
        ew_rets=ew_aligned,
        save_path=save_plot,
        label="Broker Replay",
    )

    # Trade summary
    n_trades = len(trade_log)
    buys     = sum(1 for t in trade_log if t["action"] == "BUY")
    sells    = n_trades - buys
    logger.info(f"Trades: {n_trades} total ({buys} buys, {sells} sells)")
    _print_trade_log_report(trade_log)
    if isinstance(_LAST_REPLAY_SCORE_AUDIT, pd.DataFrame) and not _LAST_REPLAY_SCORE_AUDIT.empty:
        score_audit_path = Path(save_plot).with_name(f"{Path(save_plot).stem}_score_audit.csv")
        score_audit_path.parent.mkdir(parents=True, exist_ok=True)
        _LAST_REPLAY_SCORE_AUDIT.to_csv(score_audit_path, index=False)
        logger.info("Score audit saved -> %s", score_audit_path)

    # Rolling-window robustness
    rolling_df, rolling_summary = _rolling_window_validation(aligned)
    if not rolling_df.empty:
        rolling_path = Path(save_plot).with_name(f"{Path(save_plot).stem}_rolling.csv")
        rolling_summary_path = Path(save_plot).with_name(
            f"{Path(save_plot).stem}_rolling_summary.csv"
        )
        rolling_path.parent.mkdir(parents=True, exist_ok=True)
        rolling_df.to_csv(rolling_path, index=False)
        rolling_summary.to_csv(rolling_summary_path, index=False)
        logger.info("Rolling validation saved -> %s", rolling_path)
        logger.info("Rolling validation summary saved -> %s", rolling_summary_path)
    _print_rolling_window_report(rolling_summary)

    # Closed-trade attribution
    sector_map = get_cached_sector_map(
        df_replay.index.get_level_values("ticker").unique().tolist()
    )
    attribution_df = _build_trade_attribution(trade_log, sector_map=sector_map)
    if not attribution_df.empty:
        attribution_path = Path(save_plot).with_name(
            f"{Path(save_plot).stem}_trade_attribution.csv"
        )
        attribution_path.parent.mkdir(parents=True, exist_ok=True)
        attribution_df.to_csv(attribution_path, index=False)
        logger.info("Trade attribution saved -> %s", attribution_path)
    _print_trade_attribution_report(attribution_df)

    # Sensitivity sweep
    if run_sensitivity_sweep:
        logger.info("\nRunning sensitivity sweep...")
        sens_df = run_sensitivity(
            df_replay,
            price_lookup,
            initial_cash=initial_cash,
            live_config=live_config,
            strategy=strategy,
            checkpoint_path=checkpoint_path,
        )
        print(f"\n{'='*72}")
        print("  Sensitivity Analysis — does performance hold across parameter changes?")
        print(f"{'='*72}")
        print(sens_df.to_string(
            index=False,
            formatters={
                "total_return": "{:.2%}".format,
                "ann_return":   "{:.2%}".format,
                "sharpe":       "{:.3f}".format,
                "max_drawdown": "{:.2%}".format,
                "win_rate":     "{:.2%}".format,
            }
        ))
        print(f"{'='*72}\n")
        sens_path = Path(save_plot).with_name(f"{Path(save_plot).stem}_sensitivity.csv")
        sens_df.to_csv(sens_path, index=False)
        logger.info("Sensitivity sweep saved -> %s", sens_path)

        # Flag if results are knife-edge
        sharpes = sens_df["sharpe"].values
        if sharpes.std() > 0.3:
            logger.warning(
                "HIGH SENSITIVITY: Sharpe std across parameter grid = "
                f"{sharpes.std():.2f}. Results may be overfit to current parameters."
            )
        else:
            logger.info(
                f"ROBUST: Sharpe std across parameter grid = {sharpes.std():.2f}. "
                "Results appear stable."
            )

    try:
        from pipeline.run_manifest import get_code_version as _get_code_version
        manifest_path = Path(save_plot).with_name(f"{Path(save_plot).stem}_manifest.json")
        cfg = live_config or {}
        payload = {
            "mode": "replay",
            "config_hash": hash_config(cfg),
            "code_version": _get_code_version(),
            "checkpoint_path": checkpoint_path,
            "snapshot_path": str(cfg.get("universe_snapshot_path", "")),
            "watchlist_included": bool(cfg.get("include_watchlist_in_universe", False)),
            "resolved_universe_size": int(df_replay.index.get_level_values("ticker").nunique()),
            "resolved_universe_hash": hash_ticker_list(
                df_replay.index.get_level_values("ticker").unique().tolist()
            ),
            "replay_window": {
                "start": replay_dates[0].date().isoformat(),
                "end": replay_dates[-1].date().isoformat(),
                "years": replay_years,
                "trading_days": len(replay_dates),
            },
            "benchmark": {
                "status": benchmark_bundle.get("status"),
                "source": benchmark_bundle.get("source"),
                "available": spy_rets is not None,
                "aligned_observations": int(len(spy_rets)) if spy_rets is not None else 0,
            },
            "friction": friction,
        }
        write_run_manifest("replay", payload, output_path=manifest_path)
        logger.info("Replay run manifest saved -> %s", manifest_path)
    except Exception as exc:
        logger.warning("Could not save replay run manifest: %s", exc)

    return broker_rets, trade_log


def _check_ablation_gate(report_df: pd.DataFrame) -> str:
    """
    Returns "PASSED" or "FAILED".
    Gate conditions:
      - screener_rl Sharpe >= heuristics_only Sharpe + 0.10
      - screener_rl max_drawdown <= heuristics_only max_drawdown + 0.05
    """
    rl_row   = report_df[report_df["strategy"] == "screener_rl"].iloc[0]
    base_row = report_df[report_df["strategy"] == "heuristics_only"].iloc[0]

    rl_sharpe   = float(rl_row["sharpe"])
    base_sharpe = float(base_row["sharpe"])
    rl_dd       = float(rl_row["max_drawdown"])
    base_dd     = float(base_row["max_drawdown"])

    sharpe_ok   = rl_sharpe >= base_sharpe + 0.10
    drawdown_ok = rl_dd <= base_dd + 0.05

    if not sharpe_ok:
        print(
            f"⚠️  ABLATION GATE FAILED: screener_rl Sharpe ({rl_sharpe:.4f}) "
            f"did not exceed heuristics_only Sharpe ({base_sharpe:.4f}) + 0.10 "
            f"(required >= {base_sharpe + 0.10:.4f})"
        )
    if not drawdown_ok:
        print(
            f"⚠️  ABLATION GATE FAILED: screener_rl max_drawdown ({rl_dd:.4f}) "
            f"exceeded heuristics_only max_drawdown ({base_dd:.4f}) + 0.05 "
            f"(required <= {base_dd + 0.05:.4f})"
        )

    result = "PASSED" if (sharpe_ok and drawdown_ok) else "FAILED"
    logger.info("Ablation gate: %s", result)
    return result


def _equal_weight_returns(
    df_features: pd.DataFrame,
    price_lookup: pd.DataFrame,
    dates: list,
) -> np.ndarray:
    """Buy-and-hold equal weight across all tickers in the universe."""
    tickers = df_features.index.get_level_values("ticker").unique().tolist()
    rets    = [0.0]
    for i in range(1, len(dates)):
        d0, d1 = dates[i - 1], dates[i]
        day_rets = []
        for ticker in tickers:
            try:
                p0 = float(price_lookup.loc[(d0, ticker), "close"])
                p1 = float(price_lookup.loc[(d1, ticker), "close"])
                if p0 > 0:
                    day_rets.append((p1 / p0) - 1.0)
            except (KeyError, Exception):
                pass
        rets.append(float(np.mean(day_rets)) if day_rets else 0.0)
    return np.array(rets, dtype=float)


def _print_trade_log_report(trade_log: list[dict]) -> None:
    print(f"\n{'='*72}")
    print("  Replay Trades")
    print(f"{'='*72}")
    if not trade_log:
        print("  No trades were executed.")
        print(f"{'='*72}\n")
        return

    df_trades = pd.DataFrame(trade_log).copy()
    preferred_cols = [
        "decision_date",
        "fill_date",
        "action",
        "ticker",
        "price",
        "decision_price",
        "score",
        "reason",
    ]
    df_trades = df_trades[[c for c in preferred_cols if c in df_trades.columns]]

    formatters = {}
    if "price" in df_trades.columns:
        formatters["price"] = lambda x: f"${float(x):.2f}" if pd.notna(x) else ""
    if "score" in df_trades.columns:
        formatters["score"] = lambda x: f"{float(x):.3f}" if pd.notna(x) else ""

    print(df_trades.to_string(index=False, formatters=formatters, justify="left"))
    print(f"{'='*72}\n")


def _rolling_window_validation(
    aligned: pd.DataFrame,
    windows: tuple[int, ...] = (63, 126, 252),
    step: int = 21,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate rolling subperiod robustness on already-aligned return series."""
    from pipeline.benchmark import benchmark_vs_spy, compute_metrics

    if aligned.empty:
        return pd.DataFrame(), pd.DataFrame()

    rows: list[dict] = []
    has_benchmark = "benchmark" in aligned.columns
    has_equal_weight = "equal_weight" in aligned.columns

    for window in windows:
        if len(aligned) < window:
            continue

        end_indices = list(range(window - 1, len(aligned), step))
        if end_indices[-1] != len(aligned) - 1:
            end_indices.append(len(aligned) - 1)

        for end_idx in end_indices:
            window_df = aligned.iloc[end_idx - window + 1:end_idx + 1]
            portfolio_rets = window_df["portfolio"].to_numpy(dtype=float)
            policy_metrics = compute_metrics(portfolio_rets, "Policy")

            row = {
                "window_days": int(window),
                "start_date": pd.Timestamp(window_df.index[0]).date().isoformat(),
                "end_date": pd.Timestamp(window_df.index[-1]).date().isoformat(),
                "n_obs": int(len(window_df)),
                "policy_total_return": policy_metrics["total_return"],
                "policy_ann_return": policy_metrics["ann_return"],
                "policy_sharpe": policy_metrics["sharpe"],
                "policy_max_drawdown": policy_metrics["max_drawdown"],
            }

            if has_benchmark:
                benchmark_rets = window_df["benchmark"].to_numpy(dtype=float)
                benchmark_metrics = compute_metrics(benchmark_rets, "SPY")
                rel = benchmark_vs_spy(portfolio_rets, benchmark_rets)
                row.update({
                    "spy_total_return": benchmark_metrics["total_return"],
                    "spy_ann_return": benchmark_metrics["ann_return"],
                    "excess_return_vs_spy": (
                        policy_metrics["total_return"] - benchmark_metrics["total_return"]
                    ),
                    "beats_spy_return": bool(rel["beats_spy_return"]),
                })

            if has_equal_weight:
                ew_rets = window_df["equal_weight"].to_numpy(dtype=float)
                ew_metrics = compute_metrics(ew_rets, "Equal-Weight")
                row.update({
                    "equal_weight_total_return": ew_metrics["total_return"],
                    "excess_return_vs_equal_weight": (
                        policy_metrics["total_return"] - ew_metrics["total_return"]
                    ),
                    "beats_equal_weight_return": (
                        policy_metrics["total_return"] > ew_metrics["total_return"]
                    ),
                })

            rows.append(row)

    rolling_df = pd.DataFrame(rows)
    if rolling_df.empty:
        return rolling_df, pd.DataFrame()

    summary_rows: list[dict] = []
    for window_days, group in rolling_df.groupby("window_days", sort=True):
        summary = {
            "window_days": int(window_days),
            "n_windows": int(len(group)),
            "median_policy_return": float(group["policy_total_return"].median()),
            "median_policy_drawdown": float(group["policy_max_drawdown"].median()),
        }
        if "excess_return_vs_spy" in group.columns:
            summary["median_excess_vs_spy"] = float(group["excess_return_vs_spy"].median())
            summary["beat_rate_vs_spy"] = float(group["beats_spy_return"].mean())
        if "excess_return_vs_equal_weight" in group.columns:
            summary["median_excess_vs_equal_weight"] = float(
                group["excess_return_vs_equal_weight"].median()
            )
            summary["beat_rate_vs_equal_weight"] = float(
                group["beats_equal_weight_return"].mean()
            )
        summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows).sort_values("window_days").reset_index(drop=True)
    return rolling_df, summary_df


def _print_rolling_window_report(summary_df: pd.DataFrame) -> None:
    print(f"\n{'='*72}")
    print("  Rolling Window Validation")
    print(f"{'='*72}")
    if summary_df.empty:
        print("  No rolling-window summary available.")
        print(f"{'='*72}\n")
        return

    cols = ["window_days", "n_windows", "median_policy_return", "median_policy_drawdown"]
    if "median_excess_vs_spy" in summary_df.columns:
        cols.extend(["median_excess_vs_spy", "beat_rate_vs_spy"])
    if "median_excess_vs_equal_weight" in summary_df.columns:
        cols.extend(["median_excess_vs_equal_weight", "beat_rate_vs_equal_weight"])

    pretty = summary_df[cols].copy()
    print(
        pretty.to_string(
            index=False,
            formatters={
                "median_policy_return": "{:.2%}".format,
                "median_policy_drawdown": "{:.2%}".format,
                "median_excess_vs_spy": "{:.2%}".format,
                "beat_rate_vs_spy": "{:.2%}".format,
                "median_excess_vs_equal_weight": "{:.2%}".format,
                "beat_rate_vs_equal_weight": "{:.2%}".format,
            },
            justify="left",
        )
    )
    print(f"{'='*72}\n")


def _normalize_exit_reason(reason: str | None) -> str:
    reason = str(reason or "").strip()
    if not reason:
        return "Unknown"
    return reason.split("(", 1)[0].strip()


def _build_trade_attribution(
    trade_log: list[dict],
    sector_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Reconstruct closed-trade attribution from replay fills using FIFO lots."""
    if not trade_log:
        return pd.DataFrame()

    sector_map = {str(k).upper(): str(v) for k, v in (sector_map or {}).items()}
    df_trades = pd.DataFrame(trade_log).copy()
    if df_trades.empty:
        return pd.DataFrame()

    df_trades["fill_date"] = pd.to_datetime(df_trades.get("fill_date", df_trades.get("date")))
    df_trades["ticker"] = df_trades["ticker"].astype(str).str.upper()
    df_trades["shares"] = pd.to_numeric(df_trades.get("shares"), errors="coerce").fillna(0.0)
    df_trades["price"] = pd.to_numeric(df_trades.get("price"), errors="coerce")
    df_trades["score"] = pd.to_numeric(df_trades.get("score"), errors="coerce")
    df_trades = df_trades.sort_values(["fill_date", "ticker", "action"]).reset_index(drop=True)

    open_lots: dict[str, list[dict]] = {}
    rows: list[dict] = []

    for rec in df_trades.to_dict("records"):
        ticker = rec["ticker"]
        action = str(rec.get("action", "")).upper()
        fill_date = pd.Timestamp(rec["fill_date"])
        price = float(rec.get("price", 0.0) or 0.0)
        shares = float(rec.get("shares", 0.0) or 0.0)
        score = rec.get("score")
        reason = str(rec.get("reason", "") or "")

        if action == "BUY":
            if shares <= 0 or price <= 0:
                continue
            open_lots.setdefault(ticker, []).append({
                "shares": shares,
                "entry_date": fill_date,
                "entry_price": price,
                "entry_score": float(score) if pd.notna(score) else np.nan,
                "entry_reason": reason,
                "sector": sector_map.get(ticker, "Unknown"),
            })
            continue

        if action not in {"SELL", "SELL_PARTIAL"}:
            continue

        lots = open_lots.get(ticker, [])
        if not lots:
            continue

        remaining = shares
        if remaining <= 0 and action == "SELL":
            remaining = float(sum(lot["shares"] for lot in lots))
        if remaining <= 0:
            continue

        while remaining > 1e-9 and lots:
            lot = lots[0]
            used_shares = min(remaining, float(lot["shares"]))
            pnl = used_shares * (price - float(lot["entry_price"]))
            entry_price = float(lot["entry_price"])
            rows.append({
                "ticker": ticker,
                "sector": lot["sector"],
                "entry_date": pd.Timestamp(lot["entry_date"]).date().isoformat(),
                "exit_date": fill_date.date().isoformat(),
                "holding_days": int((fill_date - pd.Timestamp(lot["entry_date"])).days),
                "shares": float(used_shares),
                "entry_price": entry_price,
                "exit_price": price,
                "gross_pnl": float(pnl),
                "return_pct": float((price / entry_price) - 1.0) if entry_price > 0 else np.nan,
                "entry_score": lot["entry_score"],
                "exit_score": float(score) if pd.notna(score) else np.nan,
                "entry_reason": lot["entry_reason"],
                "exit_reason": reason,
                "exit_reason_bucket": _normalize_exit_reason(reason),
            })

            lot["shares"] -= used_shares
            remaining -= used_shares
            if lot["shares"] <= 1e-9:
                lots.pop(0)

    return pd.DataFrame(rows)


def _print_trade_attribution_report(attribution_df: pd.DataFrame) -> None:
    print(f"\n{'='*72}")
    print("  Closed Trade Attribution")
    print(f"{'='*72}")
    if attribution_df.empty:
        print("  No closed trades yet, so attribution is unavailable.")
        print(f"{'='*72}\n")
        return

    realized_pnl = float(attribution_df["gross_pnl"].sum())
    win_rate = float((attribution_df["gross_pnl"] > 0).mean())
    avg_holding_days = float(attribution_df["holding_days"].mean())
    winners = attribution_df.loc[attribution_df["gross_pnl"] > 0, "gross_pnl"].sum()
    losers = attribution_df.loc[attribution_df["gross_pnl"] < 0, "gross_pnl"].sum()
    profit_factor = float(winners / abs(losers)) if losers < 0 else np.inf

    print(
        "  closed_trades={:d}  realized_pnl=${:,.2f}  win_rate={:.2%}  "
        "avg_holding_days={:.1f}  profit_factor={:.2f}".format(
            int(len(attribution_df)),
            realized_pnl,
            win_rate,
            avg_holding_days,
            profit_factor,
        )
    )

    sector_summary = (
        attribution_df.groupby("sector", dropna=False)
        .agg(
            trades=("gross_pnl", "size"),
            total_pnl=("gross_pnl", "sum"),
            win_rate=("gross_pnl", lambda s: float((s > 0).mean())),
        )
        .sort_values("total_pnl", ascending=False)
        .reset_index()
    )
    reason_summary = (
        attribution_df.groupby("exit_reason_bucket", dropna=False)
        .agg(
            trades=("gross_pnl", "size"),
            total_pnl=("gross_pnl", "sum"),
            win_rate=("gross_pnl", lambda s: float((s > 0).mean())),
        )
        .sort_values(["trades", "total_pnl"], ascending=[False, False])
        .reset_index()
    )

    print("\n  Sector Summary")
    print(
        sector_summary.head(8).to_string(
            index=False,
            formatters={
                "total_pnl": "${:,.2f}".format,
                "win_rate": "{:.2%}".format,
            },
            justify="left",
        )
    )

    print("\n  Exit Reason Summary")
    print(
        reason_summary.head(8).to_string(
            index=False,
            formatters={
                "total_pnl": "${:,.2f}".format,
                "win_rate": "{:.2%}".format,
            },
            justify="left",
        )
    )

    top_cols = ["ticker", "sector", "gross_pnl", "return_pct", "holding_days", "exit_reason_bucket"]
    print("\n  Top Winners")
    print(
        attribution_df.nlargest(5, "gross_pnl")[top_cols].to_string(
            index=False,
            formatters={
                "gross_pnl": "${:,.2f}".format,
                "return_pct": "{:.2%}".format,
            },
            justify="left",
        )
    )

    print("\n  Top Losers")
    print(
        attribution_df.nsmallest(5, "gross_pnl")[top_cols].to_string(
            index=False,
            formatters={
                "gross_pnl": "${:,.2f}".format,
                "return_pct": "{:.2%}".format,
            },
            justify="left",
        )
    )
    print(f"{'='*72}\n")


# ── Ablation report ───────────────────────────────────────────────────────────

def run_ablation(
    df_features: pd.DataFrame,
    price_lookup: pd.DataFrame,
    checkpoint_path: str | None = None,
    initial_cash: float = 10_000.0,
    replay_years: int = 3,
    max_positions: int = 20,
    max_position_pct: float = 0.10,
    min_score: float = 0.60,
    stop_loss_floor: float = 0.07,
    take_profit: float = 1.00,
    trailing_stop_pct: float = 0.12,
    trailing_activation_pct: float = 0.18,
    signal_exit_score: float = 0.18,
    signal_exit_grace_cycles: int = 2,
    max_gross_exposure: float = 0.95,
    cash_floor: float = 0.05,
    target_volatility: float = 0.15,
    vol_lookback: int = 20,
    partial_profit_pct: float = 0.35,
    penny_pct: float = 0.20,
    max_sector_pct: float = 0.40,
    max_pair_correlation: float = 0.80,
    weak_theme_min_positions: int = 2,
    weak_theme_return_threshold: float = -0.03,
    weak_theme_penalty_mult: float = 0.50,
    weak_theme_cooldown_cycles: int = 0,
    weak_theme_cooldown_min_hits: int = 2,
    low_price_rank_policy: str = "late_cap",
    low_price_rank_penalty_mult: float = 0.70,
    low_price_high_rank_floor: float = 0.80,
    avoid_earnings_days: int = 3,
    rl_phase: int = 1,
    rl_exit_threshold: float = 0.30,
    rl_conviction_drop: float = 0.20,
    rl_min_score: float = 0.0,
    save_report: str = "plots/ablation_report.csv",
    save_plot: str = "plots/ablation.png",
) -> pd.DataFrame:
    """
    Run all four strategy variants over the same historical period and
    produce a side-by-side AblationReport DataFrame.

    Parameters
    ----------
    df_features : pd.DataFrame
        MultiIndex [date, ticker] feature DataFrame.
    price_lookup : pd.DataFrame
        Raw close prices indexed by [date, ticker].
    checkpoint_path : str | None
        Path to a PortfolioTransformer .pt checkpoint.
        Required for "screener_rl" and "rl_weights" variants.
    initial_cash : float
        Starting cash for every variant.
    replay_years : int
        How many trailing years of df_features to use.
    save_report : str
        CSV output path for the AblationReport.
    save_plot : str
        PNG output path for the Sharpe bar chart.

    Returns
    -------
    pd.DataFrame
        AblationReport with columns:
        strategy, total_return, ann_return, sharpe, max_drawdown,
        win_rate, spy_alpha, n_trades
    """
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pipeline.benchmark import compute_metrics, fetch_spy_returns

    # ── Restrict to replay window (same pattern as run_full_replay) ───────────
    dates  = sorted(df_features.index.get_level_values("date").unique())
    cutoff = pd.Timestamp(dates[-1]) - pd.DateOffset(years=replay_years)
    df_replay = df_features[df_features.index.get_level_values("date") >= cutoff]

    replay_dates = sorted(df_replay.index.get_level_values("date").unique())
    logger.info(
        "Ablation period: %s → %s (%d years, %d trading days)",
        replay_dates[0].date(), replay_dates[-1].date(),
        replay_years, len(replay_dates),
    )

    # ── Fetch SPY returns for the same window ─────────────────────────────────
    spy_series = fetch_spy_returns(
        start=(replay_dates[0] - pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
        end=(replay_dates[-1] + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
    )
    if not spy_series.empty:
        spy_ann_return = float(
            (1 + spy_series.values).prod() ** (252 / max(len(spy_series), 1)) - 1
        )
    else:
        spy_ann_return = 0.0
    logger.info("SPY annualised return over ablation window: %.4f", spy_ann_return)

    # ── Four strategy variants ────────────────────────────────────────────────
    variants = [
        ("heuristics_only",    None),
        ("screener_heuristics", None),
        ("screener_rl",        checkpoint_path),
        ("rl_weights",         checkpoint_path),
    ]

    rows = []
    for strategy, ckpt in variants:
        logger.info("Running ablation variant: %s", strategy)
        rets, trade_log = run_replay(
            df_replay,
            price_lookup,
            strategy=strategy,
            checkpoint_path=ckpt,
            initial_cash=initial_cash,
            max_positions=max_positions,
            max_position_pct=max_position_pct,
            min_score=min_score,
            stop_loss_floor=stop_loss_floor,
            take_profit=take_profit,
            trailing_stop_pct=trailing_stop_pct,
            trailing_activation_pct=trailing_activation_pct,
            signal_exit_score=signal_exit_score,
            signal_exit_grace_cycles=signal_exit_grace_cycles,
            max_gross_exposure=max_gross_exposure,
            cash_floor=cash_floor,
            target_volatility=target_volatility,
            vol_lookback=vol_lookback,
            partial_profit_pct=partial_profit_pct,
            penny_pct=penny_pct,
            max_sector_pct=max_sector_pct,
            max_pair_correlation=max_pair_correlation,
            weak_theme_min_positions=weak_theme_min_positions,
            weak_theme_return_threshold=weak_theme_return_threshold,
            weak_theme_penalty_mult=weak_theme_penalty_mult,
            weak_theme_cooldown_cycles=weak_theme_cooldown_cycles,
            weak_theme_cooldown_min_hits=weak_theme_cooldown_min_hits,
            low_price_rank_policy=low_price_rank_policy,
            low_price_rank_penalty_mult=low_price_rank_penalty_mult,
            low_price_high_rank_floor=low_price_high_rank_floor,
            avoid_earnings_days=avoid_earnings_days,
            rl_phase=rl_phase,
            rl_exit_threshold=rl_exit_threshold,
            rl_conviction_drop=rl_conviction_drop,
            rl_min_score=rl_min_score,
            label=strategy,
        )

        m = compute_metrics(rets, label=strategy)
        n_trades = sum(1 for t in trade_log if t["action"] == "BUY")

        rows.append({
            "strategy":     strategy,
            "total_return": m["total_return"],
            "ann_return":   m["ann_return"],
            "sharpe":       m["sharpe"],
            "max_drawdown": m["max_drawdown"],
            "win_rate":     m["win_rate"],
            "spy_alpha":    m["ann_return"] - spy_ann_return,
            "n_trades":     n_trades,
        })

    report_df = pd.DataFrame(rows, columns=[
        "strategy", "total_return", "ann_return", "sharpe",
        "max_drawdown", "win_rate", "spy_alpha", "n_trades",
    ])

    # ── Ablation gate ─────────────────────────────────────────────────────────
    gate_result = _check_ablation_gate(report_df)
    logger.info("Ablation gate result: %s", gate_result)

    # ── Save CSV report ───────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(save_report) if os.path.dirname(save_report) else ".", exist_ok=True)
    report_df.to_csv(save_report, index=False)
    logger.info("Ablation report saved -> %s", save_report)

    # ── Save Sharpe bar chart ─────────────────────────────────────────────────
    os.makedirs(os.path.dirname(save_plot) if os.path.dirname(save_plot) else ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    strategies = report_df["strategy"].tolist()
    sharpes    = report_df["sharpe"].tolist()
    colors     = ["#4CAF50" if s >= 0 else "#F44336" for s in sharpes]
    bars = ax.bar(strategies, sharpes, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_title("Ablation Study — Sharpe Ratio by Strategy", fontsize=13, fontweight="bold")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_xlabel("Strategy")
    for bar, val in zip(bars, sharpes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9,
        )
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(save_plot, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Ablation chart saved -> %s", save_plot)

    return report_df
