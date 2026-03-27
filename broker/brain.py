"""
Broker decision engine.

Decision process each cycle:
  1. Validate + update prices (cross-checks suspicious moves)
  2. Volatility-adjusted stop-loss and partial take-profit
  3. Re-research held positions for signal deterioration
  4. Score sectors dynamically — broker decides its own allocations
  5. Screen full universe for candidates
  6. Skip earnings-window stocks (configurable)
  7. Deep-research top candidates
  8. Sector-aware position sizing with diversification penalty
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from broker.analyst   import research, fetch_ticker_data
from broker.portfolio import Portfolio
from broker.sectors   import (
    get_sectors_bulk, score_sectors, compute_target_allocations,
    get_portfolio_sector_weights,
)
from broker.validator import validate_portfolio_prices

logger = logging.getLogger(__name__)


@dataclass
class Decision:
    action:  str          # "BUY", "SELL", "SELL_PARTIAL", "HOLD"
    ticker:  str
    shares:  float
    price:   float
    score:   float
    reason:  str


class BrokerBrain:
    def __init__(
        self,
        portfolio:            Portfolio,
        max_positions:        int   = 20,
        max_position_pct:     float = 0.10,   # max 10% of equity per position
        stop_loss_atr_mult:   float = 2.5,    # stop = entry - 2.5 × ATR
        stop_loss_pct_floor:  float = 0.07,   # minimum stop regardless of ATR
        stop_loss_pct_ceil:   float = 0.25,   # maximum stop regardless of ATR
        partial_profit_pct:   float = 0.20,   # sell half at +20%
        full_profit_pct:      float = 0.45,   # sell rest at +45%
        min_score:            float = 0.60,
        penny_max_pct:        float = 0.20,
        penny_threshold:      float = 5.0,
        max_sector_pct:       float = 0.40,   # hard cap per sector
        avoid_earnings_days:  int   = 3,       # skip stocks within N days of earnings
        device=None,
    ):
        self.portfolio            = portfolio
        self.max_positions        = max_positions
        self.max_position_pct     = max_position_pct
        self.stop_loss_atr_mult   = stop_loss_atr_mult
        self.stop_loss_pct_floor  = stop_loss_pct_floor
        self.stop_loss_pct_ceil   = stop_loss_pct_ceil
        self.partial_profit_pct   = partial_profit_pct
        self.full_profit_pct      = full_profit_pct
        self.min_score            = min_score
        self.penny_max_pct        = penny_max_pct
        self.penny_threshold      = penny_threshold
        self.max_sector_pct       = max_sector_pct
        self.avoid_earnings_days  = avoid_earnings_days
        self.device               = device

        # Cache sector map across cycles (refreshed weekly)
        self._sector_map:   dict[str, str]   = {}
        self._sector_cache_date: datetime | None = None

    # ── Main decision cycle ───────────────────────────────────────────────────

    def run_cycle(
        self,
        df_features: pd.DataFrame,
        screener_top_n: int = 100,
        risk_engine=None,   # PortfolioRiskEngine instance
    ) -> list[Decision]:
        decisions = []

        # ── 1. Refresh sector map (weekly) ────────────────────────────────────
        self._maybe_refresh_sector_map(df_features)

        # ── 2. Validate + update prices ───────────────────────────────────────
        held_tickers = list(self.portfolio.positions.keys())
        if held_tickers:
            raw_prices = self._get_current_prices(held_tickers)
            clean_prices = validate_portfolio_prices(
                self.portfolio.positions, raw_prices
            )
            self.portfolio.update_prices(clean_prices)

        # ── 3. Exit decisions ─────────────────────────────────────────────────
        for ticker in list(self.portfolio.positions.keys()):
            pos   = self.portfolio.positions[ticker]
            price = pos["last_price"]
            cost  = pos["avg_cost"]
            if cost <= 0:
                continue

            pnl_pct = (price - cost) / cost

            # Compute volatility-adjusted stop for this position
            stop_pct = self._get_stop_loss_pct(ticker, pos)

            # Full stop-loss
            if pnl_pct <= -stop_pct:
                decisions.append(Decision(
                    action="SELL", ticker=ticker,
                    shares=pos["shares"], price=price, score=0.0,
                    reason=f"Stop-loss ({pnl_pct:.1%} vs -{stop_pct:.1%} ATR-adjusted)",
                ))
                continue

            # Partial take-profit at +20% — sell half, let rest run
            if pnl_pct >= self.partial_profit_pct and not pos.get("partial_taken"):
                half_shares = pos["shares"] * 0.5
                decisions.append(Decision(
                    action="SELL_PARTIAL", ticker=ticker,
                    shares=half_shares, price=price, score=0.8,
                    reason=f"Partial take-profit ({pnl_pct:.1%}), selling 50%",
                ))
                pos["partial_taken"] = True
                continue

            # Full take-profit at +45%
            if pnl_pct >= self.full_profit_pct:
                decisions.append(Decision(
                    action="SELL", ticker=ticker,
                    shares=pos["shares"], price=price, score=1.0,
                    reason=f"Full take-profit ({pnl_pct:.1%})",
                ))
                continue

            # Signal deterioration check
            report = research(ticker)
            if report and report["composite_score"] < 0.35:
                decisions.append(Decision(
                    action="SELL", ticker=ticker,
                    shares=pos["shares"], price=price,
                    score=report["composite_score"],
                    reason=f"Signal deteriorated (score={report['composite_score']:.2f})",
                ))

        # ── 4. Sector scoring — broker decides allocations ────────────────────
        sector_scores = score_sectors(df_features, self._sector_map)
        current_sector_weights = get_portfolio_sector_weights(
            self.portfolio.positions, self._sector_map
        )
        target_sector_allocs = compute_target_allocations(
            sector_scores,
            current_sector_weights,
            max_single_sector=self.max_sector_pct,
        )

        # ── 5. Screen for candidates ──────────────────────────────────────────
        candidates = self._screen_candidates(df_features, top_n=screener_top_n)

        # ── 6. Buy decisions ──────────────────────────────────────────────────
        sells_pending = sum(1 for d in decisions if d.action in ("SELL",))
        n_slots = self.max_positions - (
            len(self.portfolio.positions) - sells_pending
        )
        n_slots = max(0, n_slots)

        if n_slots > 0 and candidates:
            researched = []
            for ticker in candidates[:min(n_slots * 3, 40)]:
                if ticker in self.portfolio.positions:
                    continue

                # Skip if earnings are imminent
                if self._near_earnings(ticker):
                    logger.debug(f"Skipping {ticker} — near earnings window")
                    continue

                report = research(ticker)
                if report and report["composite_score"] >= self.min_score:
                    report["sector"] = self._sector_map.get(ticker.upper(), "Unknown")
                    researched.append(report)

            researched.sort(key=lambda r: r["composite_score"], reverse=True)

            equity      = self.portfolio.equity
            penny_value = sum(
                v for t, v in self.portfolio.position_values.items()
                if self.portfolio.positions[t]["last_price"] < self.penny_threshold
            )

            # Track sector spend this cycle to respect targets
            sector_spent: dict[str, float] = {}

            for report in researched[:n_slots]:
                ticker   = report["ticker"]
                price    = report["price"]
                score    = report["composite_score"]
                sector   = report.get("sector", "Unknown")
                is_penny = price < self.penny_threshold

                # ── Penny cap ─────────────────────────────────────────────────
                if is_penny:
                    penny_budget = equity * self.penny_max_pct - penny_value
                    if penny_budget <= 0:
                        logger.debug(f"Penny cap reached, skipping {ticker}")
                        continue

                # ── Sector budget check ───────────────────────────────────────
                target_alloc = target_sector_allocs.get(sector, 0.05)
                current_sector_val = sum(
                    v for t, v in self.portfolio.position_values.items()
                    if self._sector_map.get(t.upper(), "Unknown") == sector
                ) + sector_spent.get(sector, 0.0)
                sector_budget = equity * target_alloc - current_sector_val

                if sector_budget <= equity * 0.01:
                    logger.debug(
                        f"Sector budget exhausted for {sector} "
                        f"(target={target_alloc:.1%}), skipping {ticker}"
                    )
                    continue

                # ── Position sizing ───────────────────────────────────────────
                # Base size from conviction
                conviction  = (score - self.min_score) / (1.0 - self.min_score)
                alloc_pct   = self.max_position_pct * conviction
                alloc_pct   = np.clip(alloc_pct, 0.01, self.max_position_pct)
                alloc_value = equity * alloc_pct

                # Volatility scaling
                if risk_engine is not None:
                    alloc_value = risk_engine.vol_scale_allocation(alloc_value)

                # Constrain by sector budget
                alloc_value = min(alloc_value, sector_budget)

                # Constrain by penny budget
                if is_penny:
                    alloc_value = min(alloc_value, penny_budget)

                # Never spend more than 95% of remaining cash
                alloc_value = min(alloc_value, self.portfolio.cash * 0.95)

                # Pre-trade risk check
                if risk_engine is not None:
                    allowed, reason = risk_engine.check_pre_trade(alloc_value, self.portfolio)
                    if not allowed:
                        logger.debug(f"Pre-trade check blocked {ticker}: {reason}")
                        continue

                shares = alloc_value / price if price > 0 else 0
                if shares < 0.001 or alloc_value < 1.0:
                    continue

                # Build reason
                sent_label = report.get("sentiment", {})
                if isinstance(sent_label, dict):
                    sent_label = sent_label.get("sentiment", "neutral")
                earnings_note = ""
                next_earnings = _get_next_earnings_date(ticker)
                if next_earnings:
                    days_to = (next_earnings - datetime.today().date()).days
                    earnings_note = f" | Earnings in {days_to}d"

                reason = (
                    f"Score={score:.2f} | Sector={sector} "
                    f"(target={target_alloc:.0%}) | "
                    f"Sentiment={sent_label}{earnings_note} | "
                    f"{'PENNY ' if is_penny else ''}"
                    f"{report.get('headlines', [''])[0][:50]}"
                )

                decisions.append(Decision(
                    action="BUY", ticker=ticker,
                    shares=shares, price=price,
                    score=score, reason=reason,
                ))

                sector_spent[sector] = sector_spent.get(sector, 0.0) + alloc_value
                if is_penny:
                    penny_value += alloc_value

        # ── 7. Options decisions ──────────────────────────────────────────────
        options_decisions = self._evaluate_options(
            researched if n_slots > 0 and candidates else [],
            df_features,
        )
        decisions.extend(options_decisions)

        return decisions

    # ── Volatility-adjusted stop-loss ─────────────────────────────────────────

    def _get_stop_loss_pct(self, ticker: str, pos: dict) -> float:
        """
        Compute ATR-based stop-loss for a position.
        Falls back to floor if ATR unavailable.
        """
        try:
            data = fetch_ticker_data(ticker, days=30)
            if data is None or len(data) < 14:
                return self.stop_loss_pct_floor

            # ATR calculation
            h = data["high"].values
            l = data["low"].values
            c = data["close"].values
            prev_c = np.roll(c, 1); prev_c[0] = c[0]
            tr  = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
            atr = float(np.mean(tr[-14:]))

            entry_price = pos.get("avg_cost", c[-1])
            if entry_price <= 0:
                return self.stop_loss_pct_floor

            stop_pct = (self.stop_loss_atr_mult * atr) / entry_price
            return float(np.clip(stop_pct, self.stop_loss_pct_floor, self.stop_loss_pct_ceil))

        except Exception:
            return self.stop_loss_pct_floor

    # ── Earnings awareness ────────────────────────────────────────────────────

    def _near_earnings(self, ticker: str) -> bool:
        """Return True if earnings are within avoid_earnings_days."""
        if self.avoid_earnings_days <= 0:
            return False
        next_date = _get_next_earnings_date(ticker)
        if next_date is None:
            return False
        days_away = (next_date - datetime.today().date()).days
        return 0 <= days_away <= self.avoid_earnings_days

    # ── Sector map refresh ────────────────────────────────────────────────────

    def _maybe_refresh_sector_map(self, df_features: pd.DataFrame):
        """Refresh sector map weekly or on first run."""
        now = datetime.now()
        if (
            self._sector_cache_date is None
            or (now - self._sector_cache_date).days >= 7
        ):
            tickers = df_features.index.get_level_values("ticker").unique().tolist()
            # Also include held positions
            tickers += list(self.portfolio.positions.keys())
            tickers  = list(set(tickers))
            logger.info(f"Refreshing sector map for {len(tickers)} tickers...")
            self._sector_map = get_sectors_bulk(tickers)
            self._sector_cache_date = now

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _screen_candidates(
        self, df_features: pd.DataFrame, top_n: int = 100
    ) -> list[str]:
        import os
        from pipeline.screener import SCREENER_CKPT

        if os.path.exists(SCREENER_CKPT) and self.device is not None:
            try:
                from pipeline.screener import run_screener
                results = run_screener(df_features, device=self.device, top_n=top_n)
                return results["ticker"].tolist()
            except Exception as e:
                logger.warning(f"Screener failed, using rule-based fallback: {e}")

        try:
            dates     = sorted(df_features.index.get_level_values("date").unique())
            last_date = dates[-1]
            snap      = df_features.loc[last_date].copy()
            snap["_rank"] = (
                snap.get("ret_5d",    0) * 0.3 +
                snap.get("vol_ratio", 0) * 0.2 +
                snap.get("sent_net",  0) * 0.3 +
                snap.get("macd_hist", 0) * 0.2
            )
            return snap.nlargest(top_n, "_rank").index.tolist()
        except Exception:
            return []

    def _get_current_prices(self, tickers: list[str]) -> dict[str, float]:
        prices = {}
        for ticker in tickers:
            data = fetch_ticker_data(ticker, days=5)
            if data is not None and not data.empty:
                prices[ticker] = float(data["close"].iloc[-1])
        return prices

    def _evaluate_options(
        self,
        researched: list[dict],
        df_features: pd.DataFrame,
    ) -> list[Decision]:
        """
        Evaluate options opportunities for top-scored stocks.
        Also checks existing option positions for expiry/close signals.
        """
        from broker.options import analyse_options, OptionsBook

        decisions = []
        equity    = self.portfolio.equity

        # ── Check existing option positions ───────────────────────────────────
        current_prices = self._get_current_prices(
            list({c.ticker for c in self.portfolio.options.positions.values()})
        )
        expired_keys = self.portfolio.options.check_expirations(
            current_prices, self.portfolio
        )
        if expired_keys:
            logger.info(f"  {len(expired_keys)} option(s) expired/assigned")

        # Close options where P&L > 50% of max profit (lock in gains)
        for key, contract in list(self.portfolio.options.positions.items()):
            spot = current_prices.get(contract.ticker, 0.0)
            if spot <= 0:
                continue
            pnl = contract.pnl(spot)
            max_profit = abs(contract.total_cost)
            if max_profit > 0 and pnl / max_profit >= 0.50:
                decisions.append(Decision(
                    action="CLOSE_OPTION", ticker=contract.ticker,
                    shares=0, price=spot, score=0.9,
                    reason=f"Option P&L at {pnl/max_profit:.0%} of max — closing",
                ))

        # ── Open new option positions for top candidates ───────────────────────
        # Only use up to 10% of equity for options total
        options_budget = equity * 0.10
        current_options_value = sum(
            abs(c.total_cost) for c in self.portfolio.options.positions.values()
            if c.position == "long"
        )
        remaining_options_budget = max(0, options_budget - current_options_value)

        if remaining_options_budget < 100:
            return decisions

        # Consider top 5 researched stocks for options
        for report in (researched or [])[:5]:
            ticker = report["ticker"]
            score  = report["composite_score"]
            price  = report["price"]

            # Skip penny stocks for options (usually no liquid chain)
            if price < 5.0:
                continue

            # Skip if already have an option on this ticker
            if any(c.ticker == ticker for c in self.portfolio.options.positions.values()):
                continue

            sent = report.get("sentiment", {})
            if isinstance(sent, dict):
                sent_net = sent.get("sent_net", 0.0)
            else:
                sent_net = 0.0

            atr_pct = abs(report.get("atr", 0.02))
            per_trade_budget = min(remaining_options_budget * 0.3, equity * 0.03)

            contracts = analyse_options(
                ticker       = ticker,
                current_price= price,
                signal_score = score,
                sentiment_net= sent_net,
                atr_pct      = atr_pct,
                budget       = per_trade_budget,
            )

            if contracts:
                for contract in contracts:
                    decisions.append(Decision(
                        action="OPEN_OPTION", ticker=ticker,
                        shares=contract.contracts, price=contract.premium_paid,
                        score=score,
                        reason=f"{contract.strategy} | DTE={contract.days_to_expiry}",
                    ))
                    # Store contract on decision for execution
                    decisions[-1]._option_contract = contract

        return decisions


# ── Earnings date helper ──────────────────────────────────────────────────────

def _get_next_earnings_date(ticker: str):
    """
    Fetch next earnings date from yfinance.
    Returns a date object or None.
    """
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
            cal = yf.Ticker(ticker).calendar

        if cal is None or cal.empty:
            return None

        # calendar returns a DataFrame with 'Earnings Date' column
        if "Earnings Date" in cal.columns:
            dates = pd.to_datetime(cal["Earnings Date"], errors="coerce").dropna()
            future = [d.date() for d in dates if d.date() >= datetime.today().date()]
            return min(future) if future else None

        # Some versions return a dict
        if isinstance(cal, dict):
            ed = cal.get("Earnings Date")
            if ed:
                d = pd.to_datetime(ed, errors="coerce")
                if hasattr(d, "date"):
                    return d.date()
                if hasattr(d, "__iter__"):
                    dates = [pd.to_datetime(x).date() for x in d
                             if pd.to_datetime(x).date() >= datetime.today().date()]
                    return min(dates) if dates else None

    except Exception:
        pass
    return None
