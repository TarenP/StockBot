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
from datetime import datetime, timedelta
from types import MethodType

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


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
            feat_dict = {col: float(latest[col]) if col in latest.index and pd.notna(latest[col]) else 0.0
                         for col in FEATURE_COLS}

            # Reconstruct a minimal sentiment dict from features
            sent = {
                "sent_net":  feat_dict.get("sent_net", 0.0),
                "pos_score": feat_dict.get("sent_pos_raw", 0.45),
                "neg_score": feat_dict.get("sent_pos_raw", 0.45),
                "sentiment": "positive" if feat_dict.get("sent_net", 0) > 0 else "negative",
                "headlines": [],
            }

            report = {
                "ticker":    ticker,
                "price":     _get_historical_price(price_lookup, ticker, as_of_date),
                "volume":    _get_historical_volume(price_lookup, ticker, as_of_date),
                "sentiment": sent,
                "headlines": [],
            }
            report.update(feat_dict)
            report["composite_score"] = _historical_feature_score(report)
            return report

        except (KeyError, Exception):
            return None

    return historical_research


def _get_historical_price(price_lookup: pd.DataFrame, ticker: str, as_of_date) -> float:
    """Get the most recent close price for a ticker as of a given date."""
    try:
        return float(price_lookup.loc[(as_of_date, ticker), "close"])
    except Exception:
        return 100.0


def _get_historical_volume(price_lookup: pd.DataFrame, ticker: str, as_of_date) -> float:
    try:
        return float(price_lookup.loc[(as_of_date, ticker), "volume"])
    except Exception:
        return 1_000_000.0


def _apply_execution_cost(price: float, shares: float, execution_spread: float) -> tuple[float, float]:
    trade_value = shares * price
    adj_value = trade_value * (1.0 - execution_spread)
    adj_shares = adj_value / price if price > 0 else 0.0
    return adj_value, adj_shares


def _execute_replay_decisions(
    portfolio,
    decisions: list,
    execution_spread: float,
    trade_log: list,
    date,
) -> None:
    for d in decisions:
        ok = False
        if d.action == "BUY":
            _adj_value, adj_shares = _apply_execution_cost(d.price, d.shares, execution_spread)
            ok = portfolio.buy(d.ticker, adj_shares, d.price, d.reason)
            if ok and hasattr(d, "_rl_score_at_entry") and d.ticker in portfolio.positions:
                portfolio.positions[d.ticker]["rl_score_at_entry"] = d._rl_score_at_entry
        elif d.action == "SELL":
            ok = portfolio.sell_all(d.ticker, d.price, d.reason)
        elif d.action == "SELL_PARTIAL":
            ok = portfolio.sell(d.ticker, d.shares, d.price, d.reason)
        else:
            ok = True

        if ok and d.action in ("BUY", "SELL", "SELL_PARTIAL"):
            trade_log.append({
                "date": str(date),
                "action": d.action,
                "ticker": d.ticker,
                "price": d.price,
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
    """Load raw close prices indexed by [date, ticker] for the replay."""
    df = pd.read_parquet(parquet_path)
    df = df.reset_index()
    df["date"]   = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_convert(None).dt.normalize()
    df["ticker"] = df["ticker"].str.upper()
    df = df.set_index(["date", "ticker"])[["close", "volume"]].sort_index()
    return df


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
        else:
            self.positions[ticker] = {
                "shares": shares, "avg_cost": price,
                "last_price": price, "partial_taken": False,
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
    rebalance_freq: int = 5,          # run broker logic every N trading days
    max_positions: int = 20,
    min_score: float = 0.60,
    stop_loss_floor: float = 0.07,
    take_profit: float = 0.45,
    penny_pct: float = 0.20,
    max_sector_pct: float = 0.40,
    max_pair_correlation: float = 0.80,
    correlation_lookback_days: int = 60,
    execution_spread: float = 0.001,  # 10 bps flat for replay simplicity
    label: str | None = None,
) -> tuple[np.ndarray, list]:
    """
    Run the broker decision logic over historical data.

    Parameters
    ----------
    strategy : str
        One of "heuristics_only", "screener_heuristics", "screener_rl", "rl_weights".
    checkpoint_path : str | None
        Path to a PortfolioTransformer .pt checkpoint. Required for
        "screener_rl" and "rl_weights" strategies.
    label : str | None
        Display label for the progress bar. Defaults to the strategy name.

    Returns
    -------
    (daily_returns, trade_log)
    """
    if label is None:
        label = strategy

    import broker.brain as brain_module
    from broker.brain import BrokerBrain, Decision
    from broker.sectors import score_sectors, compute_target_allocations, get_portfolio_sector_weights
    from broker.risk import PortfolioRiskEngine

    dates = sorted(df_features.index.get_level_values("date").unique())
    portfolio = ReplayPortfolio(initial_cash)
    risk      = PortfolioRiskEngine(max_daily_loss=0.05, max_drawdown=0.20)

    equity_curve = [initial_cash]
    trade_log    = []

    # Patch research() to use historical data
    original_research = brain_module.research if hasattr(brain_module, "research") else None

    # Build a simple sector map from the static map in sectors.py
    from broker.sectors import _STATIC_SECTOR_MAP
    sector_map = dict(_STATIC_SECTOR_MAP)

    pbar = tqdm(range(len(dates)), desc=f"Replay: {label}", unit="day",
                colour="blue", dynamic_ncols=True)

    for i in pbar:
        date = dates[i]

        # Update prices for held positions
        if portfolio.positions:
            prices = {}
            for ticker in list(portfolio.positions.keys()):
                try:
                    p = float(price_lookup.loc[(date, ticker), "close"])
                    if p > 0:
                        prices[ticker] = p
                except (KeyError, Exception):
                    pass
            portfolio.update_prices(prices)

        # Record daily equity
        equity_curve.append(portfolio.equity)

        # Run broker logic every rebalance_freq days
        if i % rebalance_freq != 0:
            continue

        # Slice features up to this date (no lookahead)
        df_slice = df_features[df_features.index.get_level_values("date") <= date]
        if df_slice.empty:
            continue

        # ── Exit checks ───────────────────────────────────────────────────────
        for ticker in list(portfolio.positions.keys()):
            pos     = portfolio.positions[ticker]
            price   = pos["last_price"]
            cost    = pos["avg_cost"]
            if cost <= 0:
                continue
            pnl_pct = (price - cost) / cost

            # ATR-based stop: use ret_1d std as vol proxy
            try:
                hist = df_slice.xs(ticker, level="ticker")["ret_1d"].dropna().values[-14:]
                atr_pct = float(np.std(hist)) * 2.5 if len(hist) >= 5 else stop_loss_floor
                stop = np.clip(atr_pct, stop_loss_floor, 0.25)
            except Exception:
                stop = stop_loss_floor

            if pnl_pct <= -stop:
                portfolio.sell_all(ticker, price, "stop-loss")
                trade_log.append({"date": str(date), "action": "SELL", "ticker": ticker,
                                   "reason": f"stop-loss {pnl_pct:.1%}"})
                continue

            if pnl_pct >= take_profit:
                portfolio.sell_all(ticker, price, "take-profit")
                trade_log.append({"date": str(date), "action": "SELL", "ticker": ticker,
                                   "reason": f"take-profit {pnl_pct:.1%}"})
                continue

            if pnl_pct >= 0.20 and not pos.get("partial_taken"):
                portfolio.sell(ticker, pos["shares"] * 0.5, price, "partial take-profit")
                pos["partial_taken"] = True
                trade_log.append({"date": str(date), "action": "SELL_PARTIAL",
                                   "ticker": ticker, "reason": "partial +20%"})

        # ── Screen candidates ─────────────────────────────────────────────────
        try:
            snap = df_slice.loc[date].copy()
        except KeyError:
            continue

        # ── Strategy: build candidate shortlist ───────────────────────────────
        if strategy == "heuristics_only":
            # Rule-based _score for shortlisting
            snap["_score"] = (
                snap.get("ret_5d",    0) * 0.3 +
                snap.get("vol_ratio", 0) * 0.2 +
                snap.get("sent_net",  0) * 0.3 +
                snap.get("macd_hist", 0) * 0.2
            )
            candidates = snap.nlargest(50, "_score").index.tolist()
            rl_scores  = None
            rl_weights = None
        else:
            # screener_heuristics / screener_rl / rl_weights — use TickerScorer
            screener_candidates = None
            try:
                from pipeline.screener import run_screener
                import torch
                screener_df = run_screener(
                    df_slice,
                    device=torch.device("cpu"),
                    top_n=50,
                )
                if not screener_df.empty and "ticker" in screener_df.columns:
                    screener_candidates = screener_df["ticker"].tolist()
            except Exception as exc:
                logger.warning(
                    "Screener unavailable on %s (%s) — falling back to rule-based shortlist",
                    date, exc,
                )

            if screener_candidates is None:
                # Fallback to rule-based
                snap["_score"] = (
                    snap.get("ret_5d",    0) * 0.3 +
                    snap.get("vol_ratio", 0) * 0.2 +
                    snap.get("sent_net",  0) * 0.3 +
                    snap.get("macd_hist", 0) * 0.2
                )
                screener_candidates = snap.nlargest(50, "_score").index.tolist()

            candidates = screener_candidates
            rl_scores  = None
            rl_weights = None

            # ── RL scoring for screener_rl / rl_weights ───────────────────────
            if strategy in ("screener_rl", "rl_weights") and checkpoint_path is not None:
                try:
                    from pipeline.rl_inference import get_rl_targets
                    rl_mode = "rank" if strategy == "screener_rl" else "weights"
                    rl_result = get_rl_targets(
                        df_slice, candidates, checkpoint_path, mode=rl_mode
                    )
                    if strategy == "screener_rl":
                        rl_scores = rl_result   # pd.Series[ticker → rl_score]
                    else:
                        rl_weights = rl_result  # pd.Series[ticker|CASH → rl_weight]
                except Exception as exc:
                    logger.warning(
                        "RL inference failed on %s (%s) — falling back to heuristic ranking",
                        date, exc,
                    )

            # Sort candidates by rl_score when screener_rl
            if strategy == "screener_rl" and rl_scores is not None:
                candidates = sorted(
                    candidates,
                    key=lambda t: float(rl_scores.get(t, 0.0)),
                    reverse=True,
                )

        # ── Sector scoring ────────────────────────────────────────────────────
        sector_scores_map = score_sectors(df_slice, sector_map)
        current_sw = get_portfolio_sector_weights(portfolio.positions, sector_map)
        target_allocs = compute_target_allocations(
            sector_scores_map, current_sw, max_single_sector=max_sector_pct
        )

        # ── Buy decisions ─────────────────────────────────────────────────────
        n_slots = max_positions - len(portfolio.positions)
        equity  = portfolio.equity
        penny_value = sum(
            v for t, v in portfolio.position_values.items()
            if portfolio.positions[t]["last_price"] < 5.0
        )
        sector_spent: dict[str, float] = {}

        for ticker in candidates:
            if n_slots <= 0:
                break
            if ticker in portfolio.positions:
                continue
            if ticker == "CASH":
                continue

            # Get price
            try:
                price = float(price_lookup.loc[(date, ticker), "close"])
                if price <= 0:
                    continue
            except (KeyError, Exception):
                continue

            # ── Compute composite_score (used for heuristics_only and screener_heuristics) ──
            try:
                feats = df_slice.loc[date, ticker]
                sent_net = float(feats.get("sent_net", 0.0))
                mom5     = float(feats.get("ret_5d",   0.0))
                rsi      = float(feats.get("rsi",      50.0))
                macd     = float(feats.get("macd_hist",0.0))
                vol      = float(feats.get("vol_ratio",1.0))
                surprise = float(feats.get("sent_surprise", 0.0))

                composite_score = (
                    np.clip(0.5 + mom5 * 5, 0, 1) * 0.15 +
                    np.clip(0.5 + mom5 * 2, 0, 1) * 0.10 +
                    np.clip(1.0 - abs(rsi - 52.5) / 52.5, 0, 1) * 0.10 +
                    np.clip(0.5 + macd * 10, 0, 1) * 0.10 +
                    np.clip(vol / 3.0, 0, 1) * 0.10 +
                    np.clip(0.5 + sent_net, 0, 1) * 0.20 +
                    np.clip(0.5 + surprise * 2, 0, 1) * 0.20 +
                    0.05   # bb_pct neutral
                )
            except (KeyError, Exception):
                composite_score = 0.0

            # ── Determine effective score and alloc_value per strategy ─────────
            is_penny = price < 5.0
            sector   = sector_map.get(ticker, "Unknown")

            if is_penny:
                penny_budget = equity * penny_pct - penny_value
                if penny_budget <= 0:
                    continue

            target_alloc = target_allocs.get(sector, 0.05)
            current_sv   = sum(
                v for t, v in portfolio.position_values.items()
                if sector_map.get(t, "Unknown") == sector
            ) + sector_spent.get(sector, 0.0)
            sector_budget = equity * target_alloc - current_sv
            if sector_budget <= equity * 0.01:
                continue

            if strategy == "rl_weights" and rl_weights is not None:
                # Size directly from rl_weight × equity
                weight = float(rl_weights.get(ticker, 0.0))
                if weight <= 0.0:
                    continue
                alloc_value = weight * equity
                alloc_value = min(alloc_value, sector_budget, portfolio.cash * 0.95)
                if is_penny:
                    alloc_value = min(alloc_value, equity * penny_pct - penny_value)
                score = weight  # for logging
            elif strategy == "screener_rl" and rl_scores is not None:
                # Rank by rl_score, apply min_score threshold to rl_score
                score = float(rl_scores.get(ticker, 0.0))
                if score < min_score:
                    continue
                conviction  = (score - min_score) / (1.0 - min_score)
                alloc_pct   = np.clip(0.10 * conviction, 0.01, 0.10)
                alloc_value = min(equity * alloc_pct, sector_budget,
                                  portfolio.cash * 0.95)
                if is_penny:
                    alloc_value = min(alloc_value, equity * penny_pct - penny_value)
            else:
                # heuristics_only or screener_heuristics — rank by composite_score
                score = composite_score
                if score < min_score:
                    continue
                conviction  = (score - min_score) / (1.0 - min_score)
                alloc_pct   = np.clip(0.10 * conviction, 0.01, 0.10)
                alloc_value = min(equity * alloc_pct, sector_budget,
                                  portfolio.cash * 0.95)
                if is_penny:
                    alloc_value = min(alloc_value, equity * penny_pct - penny_value)

            # Apply execution cost (same across all strategies)
            alloc_value *= (1.0 - execution_spread)
            shares = alloc_value / price
            if shares < 0.001 or alloc_value < 1.0:
                continue

            portfolio.buy(ticker, shares, price, f"score={score:.4f}")
            trade_log.append({"date": str(date), "action": "BUY", "ticker": ticker,
                               "price": price, "score": score, "sector": sector,
                               "strategy": strategy})
            sector_spent[sector] = sector_spent.get(sector, 0.0) + alloc_value
            if is_penny:
                penny_value += alloc_value
            n_slots -= 1

        pbar.set_postfix(
            equity=f"${portfolio.equity:,.0f}",
            ret=f"{portfolio.total_return:+.1%}",
            pos=len(portfolio.positions),
        )

    # Final equity
    equity_curve.append(portfolio.equity)
    daily_returns = np.diff(equity_curve) / (np.array(equity_curve[:-1]) + 1e-9)
    return daily_returns, trade_log


# ── Sensitivity sweep ─────────────────────────────────────────────────────────

def _run_replay_v2(
    df_features: pd.DataFrame,
    price_lookup: pd.DataFrame,
    strategy: str = "heuristics_only",
    checkpoint_path: str | None = None,
    initial_cash: float = 10_000.0,
    rebalance_freq: int = 5,
    max_positions: int = 20,
    min_score: float = 0.60,
    stop_loss_floor: float = 0.07,
    take_profit: float = 0.45,
    penny_pct: float = 0.20,
    max_sector_pct: float = 0.40,
    max_pair_correlation: float = 0.80,
    correlation_lookback_days: int = 60,
    execution_spread: float = 0.001,
    label: str | None = None,
) -> tuple[np.ndarray, list]:
    if label is None:
        label = strategy

    import broker.brain as brain_module
    from broker.brain import BrokerBrain
    from broker.risk import PortfolioRiskEngine
    from broker.sectors import (
        _STATIC_SECTOR_MAP,
        compute_target_allocations,
        get_portfolio_sector_weights,
        score_sectors,
    )

    if strategy not in {"heuristics_only", "screener_heuristics", "screener_rl", "rl_weights"}:
        raise ValueError(f"Unknown replay strategy: {strategy}")

    dates = sorted(df_features.index.get_level_values("date").unique())
    portfolio = ReplayPortfolio(initial_cash)
    risk = PortfolioRiskEngine(max_daily_loss=0.05, max_drawdown=0.20)
    sector_map = dict(_STATIC_SECTOR_MAP)

    equity_curve = [initial_cash]
    trade_log = []

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

            equity_curve.append(portfolio.equity)
            if i % rebalance_freq != 0:
                continue

            df_slice = df_features[df_features.index.get_level_values("date") <= date]
            if df_slice.empty:
                continue

            brain_module.research = _make_historical_research(df_slice, price_lookup, date)
            try:
                snap = df_slice.loc[date].copy()
            except KeyError:
                continue

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

            pbar.set_postfix(
                equity=f"${portfolio.equity:,.0f}",
                ret=f"{portfolio.total_return:+.1%}",
                pos=len(portfolio.positions),
            )

        equity_curve.append(portfolio.equity)
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
        stop_loss_pct_floor=stop_loss_floor,
        full_profit_pct=take_profit,
        min_score=min_score,
        penny_max_pct=penny_pct,
        max_sector_pct=max_sector_pct,
        max_pair_correlation=max_pair_correlation,
        correlation_lookback_days=correlation_lookback_days,
        device=None,
        rl_enabled=(strategy == "screener_rl"),
        rl_checkpoint_path=checkpoint_path,
        rl_phase=2,
        # Replay should mirror live RL defaults: top-k ranking unless the
        # caller explicitly configures an RL floor elsewhere.
        rl_min_score=0.0,
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

            equity_curve.append(portfolio.equity)
            if i % rebalance_freq != 0:
                continue

            df_slice = df_features[df_features.index.get_level_values("date") <= date]
            if df_slice.empty:
                continue

            brain_module.research = _make_historical_research(df_slice, price_lookup, date)
            brain._get_current_prices = MethodType(_make_historical_price_fetcher(price_lookup, date), brain)
            brain._get_stop_loss_pct = MethodType(_make_historical_stop_loss(df_slice), brain)
            brain.min_score = brain._base_min_score

            risk.start_session(portfolio.equity)
            health_status, _health_reason = risk.check_portfolio_health(portfolio)
            if health_status == "halt":
                brain.min_score = 999.0

            decisions = brain.run_cycle(df_slice, screener_top_n=50, risk_engine=risk)
            _execute_replay_decisions(
                portfolio=portfolio,
                decisions=decisions,
                execution_spread=execution_spread,
                trade_log=trade_log,
                date=date,
            )

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

    equity_curve.append(portfolio.equity)
    daily_returns = np.diff(equity_curve) / (np.array(equity_curve[:-1]) + 1e-9)
    return daily_returns, trade_log


run_replay = _run_replay_v2


def run_sensitivity(
    df_features: pd.DataFrame,
    price_lookup: pd.DataFrame,
    initial_cash: float = 10_000.0,
) -> pd.DataFrame:
    """
    Run the replay across a grid of parameter perturbations.
    Shows whether results are robust or knife-edge.
    """
    from pipeline.benchmark import compute_metrics, fetch_spy_returns

    base = dict(min_score=0.60, stop_loss_floor=0.07,
                take_profit=0.45, penny_pct=0.20, execution_spread=0.001)

    grid = [
        # Vary min_score
        {**base, "min_score": 0.50, "label": "min_score=0.50"},
        {**base, "min_score": 0.55, "label": "min_score=0.55"},
        {**base, "min_score": 0.60, "label": "min_score=0.60 (base)"},
        {**base, "min_score": 0.65, "label": "min_score=0.65"},
        {**base, "min_score": 0.70, "label": "min_score=0.70"},
        # Vary stop-loss
        {**base, "stop_loss_floor": 0.05, "label": "stop=5%"},
        {**base, "stop_loss_floor": 0.10, "label": "stop=10%"},
        {**base, "stop_loss_floor": 0.15, "label": "stop=15%"},
        # Vary execution cost
        {**base, "execution_spread": 0.0005, "label": "spread=5bps"},
        {**base, "execution_spread": 0.002,  "label": "spread=20bps"},
        {**base, "execution_spread": 0.005,  "label": "spread=50bps"},
        # Vary take-profit
        {**base, "take_profit": 0.30, "label": "tp=30%"},
        {**base, "take_profit": 0.60, "label": "tp=60%"},
    ]

    rows = []
    for params in grid:
        label = params.pop("label")
        rets, _ = run_replay(df_features, price_lookup,
                             initial_cash=initial_cash, label=label, **params)
        m = compute_metrics(rets, label)
        rows.append({
            "params":       label,
            "total_return": m["total_return"],
            "ann_return":   m["ann_return"],
            "sharpe":       m["sharpe"],
            "max_drawdown": m["max_drawdown"],
            "win_rate":     m["win_rate"],
        })

    return pd.DataFrame(rows)


# ── Full replay report ────────────────────────────────────────────────────────

def run_full_replay(
    df_features: pd.DataFrame,
    initial_cash: float = 10_000.0,
    replay_years: int = 3,
    run_sensitivity_sweep: bool = False,
    save_plot: str = "plots/replay.png",
):
    """
    Run the full broker replay and print a side-by-side report vs SPY.
    """
    from pipeline.benchmark import (
        fetch_spy_returns, compute_metrics, benchmark_vs_spy,
        print_benchmark_report, plot_benchmark,
    )

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

    # Run broker replay
    logger.info("Running broker replay...")
    broker_rets, trade_log = run_replay(
        df_replay, price_lookup, initial_cash=initial_cash, label="Broker"
    )

    # Fetch SPY for same period
    spy_series = fetch_spy_returns(
        start=replay_dates[0].strftime("%Y-%m-%d"),
        end=(replay_dates[-1] + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
    )
    spy_rets = spy_series.values[:len(broker_rets)] if not spy_series.empty else None

    # Equal-weight baseline (buy-and-hold all tickers equally)
    logger.info("Computing equal-weight baseline...")
    ew_rets = _equal_weight_returns(df_replay, price_lookup, replay_dates)

    # Print report
    print_benchmark_report(
        broker_rets,
        spy_rets if spy_rets is not None else np.zeros_like(broker_rets),
        ew_rets=ew_rets,
        label="Broker Replay",
    )

    # Plot
    plot_benchmark(
        broker_rets,
        spy_rets if spy_rets is not None else np.zeros_like(broker_rets),
        ew_rets=ew_rets,
        save_path=save_plot,
        label="Broker Replay",
    )

    # Trade summary
    n_trades = len(trade_log)
    buys     = sum(1 for t in trade_log if t["action"] == "BUY")
    sells    = n_trades - buys
    logger.info(f"Trades: {n_trades} total ({buys} buys, {sells} sells)")
    _print_trade_log_report(trade_log)

    # Sensitivity sweep
    if run_sensitivity_sweep:
        logger.info("\nRunning sensitivity sweep...")
        sens_df = run_sensitivity(df_replay, price_lookup, initial_cash)
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
    rets    = []
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
    return np.array(rets)


def _print_trade_log_report(trade_log: list[dict]) -> None:
    print(f"\n{'='*72}")
    print("  Replay Trades")
    print(f"{'='*72}")
    if not trade_log:
        print("  No trades were executed.")
        print(f"{'='*72}\n")
        return

    df_trades = pd.DataFrame(trade_log).copy()
    preferred_cols = ["date", "action", "ticker", "price", "score", "reason"]
    df_trades = df_trades[[c for c in preferred_cols if c in df_trades.columns]]

    formatters = {}
    if "price" in df_trades.columns:
        formatters["price"] = lambda x: f"${float(x):.2f}" if pd.notna(x) else ""
    if "score" in df_trades.columns:
        formatters["score"] = lambda x: f"{float(x):.3f}" if pd.notna(x) else ""

    print(df_trades.to_string(index=False, formatters=formatters, justify="left"))
    print(f"{'='*72}\n")


# ── Ablation report ───────────────────────────────────────────────────────────

def run_ablation(
    df_features: pd.DataFrame,
    price_lookup: pd.DataFrame,
    checkpoint_path: str | None = None,
    initial_cash: float = 10_000.0,
    replay_years: int = 3,
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
        start=replay_dates[0].strftime("%Y-%m-%d"),
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
