"""
Portfolio state manager.
Tracks cash, stock positions, option positions, cost basis, P&L.
Persists to disk so state survives restarts.
"""

import json
import logging
from datetime import date, datetime
from pathlib import Path

from broker.exposure import (
    effective_bet_count,
    exposure_weights,
    portfolio_low_price_values,
    portfolio_theme_values,
)

logger = logging.getLogger(__name__)

STATE_PATH = Path("broker/state/portfolio.json")
CASH_YIELD_ANNUAL_RATE = 0.03
DAYS_PER_YEAR = 365.25


class Portfolio:
    def __init__(self, initial_cash: float = 10_000.0):
        self.initial_cash = initial_cash
        self.cash         = initial_cash
        self.positions    = {}   # ticker -> {shares, avg_cost, last_price}
        self.trade_log    = []
        self.cash_yield_last_date: date | None = None
        # Options book — lazy import to avoid circular deps
        from broker.options import OptionsBook
        self.options = OptionsBook()
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self):
        if STATE_PATH.exists():
            try:
                data = json.loads(STATE_PATH.read_text())
                self.cash         = float(data.get("cash", self.initial_cash))
                self.positions    = data.get("positions", {})
                self.trade_log    = data.get("trade_log", [])
                self.initial_cash = float(data.get("initial_cash", self.initial_cash))
                self.cash_yield_last_date = (
                    self._coerce_date(data.get("last_cash_yield_date"))
                    or self._coerce_date(data.get("last_saved"))
                )

                # Validate state consistency
                if self.cash < 0:
                    logger.warning("Portfolio cash is negative — resetting to 0")
                    self.cash = 0.0
                # Remove any positions with invalid data
                bad = [t for t, p in self.positions.items()
                       if not isinstance(p, dict) or p.get("shares", 0) <= 0]
                for t in bad:
                    logger.warning("Removing invalid position: %s", t)
                    del self.positions[t]

                # Migrate legacy positions: if rl_score_at_entry exists but
                # rl_rank_pct_at_entry does not, use the raw score as a proxy.
                # This ensures conviction-drop exits work for existing positions
                # without requiring them to be closed and reopened.
                migrated = 0
                for pos in self.positions.values():
                    if (
                        "rl_score_at_entry" in pos
                        and "rl_rank_pct_at_entry" not in pos
                    ):
                        pos["rl_rank_pct_at_entry"] = float(pos["rl_score_at_entry"])
                        migrated += 1
                if migrated:
                    logger.info(
                        "Migrated %d position(s): set rl_rank_pct_at_entry from "
                        "legacy rl_score_at_entry (approximate — will be accurate "
                        "on next buy).",
                        migrated,
                    )

                logger.info(f"Portfolio loaded. Cash: ${self.cash:,.2f} | "
                            f"Positions: {len(self.positions)}")
            except Exception as e:
                logger.warning(f"Could not load portfolio state: {e}. Starting fresh.")

    def save(self):
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps({
            "cash":         self.cash,
            "positions":    self.positions,
            "trade_log":    self.trade_log[-500:],   # keep last 500 trades
            "initial_cash": self.initial_cash,
            "last_cash_yield_date": (
                self.cash_yield_last_date.isoformat()
                if self.cash_yield_last_date is not None else None
            ),
            "last_saved":   datetime.now().isoformat(),
        }, indent=2))

    @staticmethod
    def _coerce_date(value) -> date | None:
        if value is None:
            return None
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        if isinstance(value, datetime):
            return value.date()
        try:
            return datetime.fromisoformat(str(value)).date()
        except Exception:
            return None

    def accrue_cash_yield(
        self,
        as_of: date | datetime | str | None = None,
        annual_rate: float = CASH_YIELD_ANNUAL_RATE,
    ) -> float:
        """
        Compound idle cash at a conservative annual rate using elapsed calendar days.
        """
        as_of_date = self._coerce_date(as_of) or date.today()
        if self.cash_yield_last_date is None:
            self.cash_yield_last_date = as_of_date
            return 0.0

        last_date = self.cash_yield_last_date
        if as_of_date <= last_date:
            return 0.0

        self.cash_yield_last_date = as_of_date
        if self.cash <= 0 or annual_rate <= 0:
            return 0.0

        days_elapsed = (as_of_date - last_date).days
        growth = (1.0 + annual_rate) ** (days_elapsed / DAYS_PER_YEAR)
        starting_cash = self.cash
        self.cash *= growth
        return self.cash - starting_cash

    # ── Trade execution ───────────────────────────────────────────────────────

    def buy(self, ticker: str, shares: float, price: float, reason: str = "") -> bool:
        cost = shares * price
        if cost > self.cash:
            shares = self.cash / price   # buy as many as we can afford
            cost   = shares * price
        if shares < 0.001:
            return False

        if ticker in self.positions:
            pos = self.positions[ticker]
            total_shares = pos["shares"] + shares
            pos["avg_cost"] = (pos["shares"] * pos["avg_cost"] + cost) / total_shares
            pos["shares"]   = total_shares
            pos["last_price"] = price
            pos["peak_price"] = max(float(pos.get("peak_price", price)), float(price))
            pos.setdefault("weak_signal_streak", 0)
        else:
            self.positions[ticker] = {
                "shares":        shares,
                "avg_cost":      price,
                "last_price":    price,
                "partial_taken": False,   # tracks whether partial profit was taken
                "peak_price":    price,
                "weak_signal_streak": 0,
            }

        self.cash -= cost
        self._log("BUY", ticker, shares, price, reason)
        return True

    def sell(self, ticker: str, shares: float, price: float, reason: str = "") -> bool:
        if ticker not in self.positions:
            return False
        pos = self.positions[ticker]
        shares = min(shares, pos["shares"])
        if shares < 0.001:
            return False

        proceeds = shares * price
        self.cash += proceeds
        pos["shares"] -= shares

        if pos["shares"] < 0.001:
            del self.positions[ticker]

        self._log("SELL", ticker, shares, price, reason)
        return True

    def sell_all(self, ticker: str, price: float, reason: str = "") -> bool:
        if ticker not in self.positions:
            return False
        return self.sell(ticker, self.positions[ticker]["shares"], price, reason)

    def update_prices(self, prices: dict[str, float]):
        """Update last_price for all held positions."""
        for ticker, price in prices.items():
            if ticker in self.positions and price > 0:
                self.positions[ticker]["last_price"] = price

    # ── Metrics ───────────────────────────────────────────────────────────────

    @property
    def equity(self) -> float:
        options_value = sum(
            c.current_value(
                self.positions.get(c.ticker, {}).get("last_price", 0.0)
            )
            for c in self.options.positions.values()
        )
        return self.cash + sum(
            p["shares"] * p["last_price"] for p in self.positions.values()
        ) + options_value

    @property
    def total_return(self) -> float:
        return (self.equity / self.initial_cash) - 1.0

    @property
    def position_values(self) -> dict[str, float]:
        return {t: p["shares"] * p["last_price"] for t, p in self.positions.items()}

    def unrealised_pnl(self, ticker: str) -> float:
        if ticker not in self.positions:
            return 0.0
        pos = self.positions[ticker]
        return (pos["last_price"] - pos["avg_cost"]) * pos["shares"]

    def _log(self, action, ticker, shares, price, reason):
        entry = {
            "time":   datetime.now().isoformat(),
            "action": action,
            "ticker": ticker,
            "shares": round(shares, 4),
            "price":  round(price, 4),
            "value":  round(shares * price, 2),
            "reason": reason,
            "equity": round(self.equity, 2),
        }
        self.trade_log.append(entry)
        logger.info(
            f"  {action:4s} {shares:.2f}x {ticker:6s} @ ${price:.4f} "
            f"= ${shares*price:.2f}  |  {reason}"
        )

    def summary(self) -> str:
        lines = [
            f"\n{'='*55}",
            f"  Portfolio Summary",
            f"{'='*55}",
            f"  Cash:          ${self.cash:>12,.2f}",
            f"  Equity:        ${self.equity:>12,.2f}",
            f"  Total Return:  {self.total_return:>+11.2%}",
            f"  Positions:     {len(self.positions)} stocks, "
            f"{len(self.options.positions)} options",
            f"{'─'*55}",
            f"  Stock Holdings ({len(self.positions)})",
        ]
        if self.positions:
            lines.append(f"  {'Ticker':<8} {'Shares':>8}  {'Price':>8}  {'Value':>10}  {'P&L':>10}")
            lines.append(f"  {'─'*52}")
            for ticker, pos in sorted(self.positions.items()):
                pnl = self.unrealised_pnl(ticker)
                lines.append(
                    f"  {ticker:<8} {pos['shares']:>8.2f}  "
                    f"${pos['last_price']:>7.3f}  "
                    f"${pos['shares']*pos['last_price']:>9.2f}  "
                    f"{pnl:>+9.2f}"
                )
            all_marked_at_cost = all(
                abs(float(pos.get("last_price", 0.0)) - float(pos.get("avg_cost", 0.0))) < 1e-6
                for pos in self.positions.values()
            )
            if all_marked_at_cost:
                lines.append(
                    "  Note: holdings are still marked at entry prices; unrealised P&L updates after the next price refresh."
                )
            try:
                from broker.sectors import get_cached_sector_map

                sector_map = get_cached_sector_map(list(self.positions.keys()))
                theme_weights = exposure_weights(
                    portfolio_theme_values(self.positions, sector_map)
                )
                low_price_weights = exposure_weights(
                    portfolio_low_price_values(self.positions)
                )
                if theme_weights:
                    lines.append(f"  {'-'*52}")
                    lines.append(
                        f"  Effective theme bets: {effective_bet_count(theme_weights):.2f}"
                    )
                    top_themes = sorted(
                        theme_weights.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )[:3]
                    theme_text = ", ".join(
                        f"{bucket} {weight:.0%}" for bucket, weight in top_themes
                    )
                    lines.append(f"  Top themes: {theme_text}")
                low_price_share = (
                    low_price_weights.get("sub_5", 0.0)
                    + low_price_weights.get("5_to_10", 0.0)
                )
                lines.append(f"  Sub-$10 exposure: {low_price_share:.0%}")
            except Exception:
                pass
        else:
            lines.append("  No stock positions")
        lines += self.options.summary_lines()
        lines.append(f"{'='*55}\n")
        return "\n".join(lines)
