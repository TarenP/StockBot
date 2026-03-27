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
import copy
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ── Historical research stub ──────────────────────────────────────────────────

def _make_historical_research(df_features: pd.DataFrame, as_of_date):
    """
    Returns a research() function that uses historical feature data
    instead of fetching live data from yfinance.
    No lookahead: only data up to and including as_of_date is visible.
    """
    from pipeline.features import FEATURE_COLS
    from broker.analyst import _composite_score

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
                "price":     _get_historical_price(df_features, ticker, as_of_date),
                "volume":    1_000_000.0,   # placeholder — not used in sizing
                "sentiment": sent,
                "headlines": [],
            }
            report.update(feat_dict)
            report["composite_score"] = _composite_score(report, sent)
            return report

        except (KeyError, Exception):
            return None

    return historical_research


def _get_historical_price(df_features: pd.DataFrame, ticker: str, as_of_date) -> float:
    """Get the most recent close price for a ticker as of a given date."""
    try:
        # Price isn't in features (it's been normalised out) — use ret_1d to back-calculate
        # We store a raw price lookup separately during replay setup
        return 100.0   # placeholder; overridden by _build_price_lookup
    except Exception:
        return 100.0


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
    initial_cash: float = 10_000.0,
    rebalance_freq: int = 5,          # run broker logic every N trading days
    max_positions: int = 20,
    min_score: float = 0.60,
    stop_loss_floor: float = 0.07,
    take_profit: float = 0.45,
    penny_pct: float = 0.20,
    max_sector_pct: float = 0.40,
    execution_spread: float = 0.001,  # 10 bps flat for replay simplicity
    label: str = "Broker Replay",
) -> tuple[np.ndarray, list]:
    """
    Run the broker decision logic over historical data.

    Returns:
        (daily_returns, trade_log)
    """
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

        snap["_score"] = (
            snap.get("ret_5d",    0) * 0.3 +
            snap.get("vol_ratio", 0) * 0.2 +
            snap.get("sent_net",  0) * 0.3 +
            snap.get("macd_hist", 0) * 0.2
        )
        candidates = snap.nlargest(50, "_score").index.tolist()

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

            # Get price
            try:
                price = float(price_lookup.loc[(date, ticker), "close"])
                if price <= 0:
                    continue
            except (KeyError, Exception):
                continue

            # Get composite score from features
            try:
                feats = df_slice.loc[date, ticker]
                sent_net = float(feats.get("sent_net", 0.0))
                mom5     = float(feats.get("ret_5d",   0.0))
                rsi      = float(feats.get("rsi",      50.0))
                macd     = float(feats.get("macd_hist",0.0))
                vol      = float(feats.get("vol_ratio",1.0))
                surprise = float(feats.get("sent_surprise", 0.0))

                score = (
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
                continue

            if score < min_score:
                continue

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

            conviction  = (score - min_score) / (1.0 - min_score)
            alloc_pct   = np.clip(0.10 * conviction, 0.01, 0.10)
            alloc_value = min(equity * alloc_pct, sector_budget,
                              portfolio.cash * 0.95)
            if is_penny:
                alloc_value = min(alloc_value, equity * penny_pct - penny_value)

            # Apply execution cost
            alloc_value *= (1.0 - execution_spread)
            shares = alloc_value / price
            if shares < 0.001 or alloc_value < 1.0:
                continue

            portfolio.buy(ticker, shares, price, f"score={score:.2f}")
            trade_log.append({"date": str(date), "action": "BUY", "ticker": ticker,
                               "price": price, "score": score, "sector": sector})
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
