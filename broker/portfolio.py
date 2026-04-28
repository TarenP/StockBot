"""
Portfolio state manager.
Tracks cash, stock positions, option positions, cost basis, P&L.
Persists to disk so state survives restarts.
"""

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

from broker.exposure import (
    effective_bet_count,
    exposure_weights,
    portfolio_low_price_values,
    portfolio_theme_values,
)

logger = logging.getLogger(__name__)

STATE_PATH = Path("broker/state/portfolio.json")
HISTORY_PATH = Path("broker/state/portfolio_history.jsonl")
PRICE_CACHE_PATH = Path("MasterDS/stooq_panel.parquet")
CASH_YIELD_ANNUAL_RATE = 0.03
DAYS_PER_YEAR = 365.25
DIVIDEND_LOOKBACK_DAYS = 370


def _positive_float(value) -> float | None:
    try:
        price = float(value)
    except Exception:
        return None
    if price > 0:
        return price
    return None


def _mapping_value(mapping, key: str):
    try:
        if hasattr(mapping, "get"):
            value = mapping.get(key)
            if value is not None:
                return value
    except Exception:
        pass
    try:
        return mapping[key]
    except Exception:
        return None


def _configure_yfinance_cache(yf_module) -> None:
    try:
        cache_dir = Path("broker/state/yfinance_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(yf_module, "set_tz_cache_location"):
            yf_module.set_tz_cache_location(str(cache_dir))
    except Exception:
        pass


def fetch_latest_market_price(ticker: str) -> dict:
    """
    Fetch a current-ish market price for one ticker.

    Prefers intraday bars so status/full-cycle marks can move during market
    hours. Falls back to yfinance quote fields, then daily bars.
    """
    result = {
        "price": None,
        "date": None,
        "source": None,
        "error": None,
    }
    ticker = str(ticker).upper()
    try:
        import pandas as pd
        import yfinance as yf

        _configure_yfinance_cache(yf)
        ticker_obj = yf.Ticker(ticker)

        for period, interval, source in [
            ("1d", "1m", "intraday"),
            ("5d", "5m", "intraday"),
        ]:
            try:
                hist = ticker_obj.history(
                    period=period,
                    interval=interval,
                    auto_adjust=True,
                    prepost=False,
                )
                if hist is not None and not hist.empty and "Close" in hist.columns:
                    close = hist["Close"].dropna()
                    if not close.empty:
                        price = _positive_float(close.iloc[-1])
                        if price is not None:
                            result.update(
                                {
                                    "price": price,
                                    "date": pd.Timestamp(close.index[-1]).date().isoformat(),
                                    "source": source,
                                }
                            )
                            return result
            except Exception:
                pass

        fast_info = getattr(ticker_obj, "fast_info", {}) or {}
        for key in [
            "last_price",
            "lastPrice",
            "regular_market_price",
            "regularMarketPrice",
            "current_price",
            "currentPrice",
            "previous_close",
            "previousClose",
        ]:
            price = _positive_float(_mapping_value(fast_info, key))
            if price is not None:
                result.update(
                    {
                        "price": price,
                        "date": date.today().isoformat(),
                        "source": f"quote:{key}",
                    }
                )
                return result

        daily = ticker_obj.history(period="10d", interval="1d", auto_adjust=True)
        if daily is not None and not daily.empty and "Close" in daily.columns:
            close = daily["Close"].dropna()
            if not close.empty:
                price = _positive_float(close.iloc[-1])
                if price is not None:
                    result.update(
                        {
                            "price": price,
                            "date": pd.Timestamp(close.index[-1]).date().isoformat(),
                            "source": "daily",
                        }
                    )
                    return result
    except Exception as exc:
        result["error"] = str(exc)
    return result


class Portfolio:
    def __init__(self, initial_cash: float = 10_000.0):
        self.initial_cash = initial_cash
        self.cash         = initial_cash
        self.positions    = {}   # ticker -> {shares, avg_cost, last_price}
        self.trade_log    = []
        self.cash_yield_last_date: date | None = None
        self.dividend_last_ex_date = {}
        self.dividend_cash_total = 0.0
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
                self.dividend_last_ex_date = dict(data.get("dividend_last_ex_date", {}) or {})
                self.dividend_cash_total = float(data.get("dividend_cash_total", 0.0) or 0.0)

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
            "dividend_last_ex_date": getattr(self, "dividend_last_ex_date", {}),
            "dividend_cash_total": round(float(getattr(self, "dividend_cash_total", 0.0)), 2),
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

    def _trade_record_date(self, record: dict) -> date | None:
        raw = record.get("fill_date") or record.get("date") or record.get("time")
        return self._coerce_date(raw)

    def _first_buy_date(self, ticker: str) -> date | None:
        ticker = str(ticker).upper()
        dates = [
            rec_date
            for rec in self.trade_log
            if str(rec.get("ticker", "")).upper() == ticker
            and str(rec.get("action", "")).upper() == "BUY"
            for rec_date in [self._trade_record_date(rec)]
            if rec_date is not None
        ]
        return min(dates) if dates else None

    def _shares_held_before_ex_date(self, ticker: str, ex_date: date) -> float:
        ticker = str(ticker).upper()
        shares = 0.0
        saw_trade = False
        records = sorted(
            self.trade_log,
            key=lambda rec: self._trade_record_date(rec) or date.min,
        )
        for rec in records:
            if str(rec.get("ticker", "")).upper() != ticker:
                continue
            rec_date = self._trade_record_date(rec)
            if rec_date is None or rec_date >= ex_date:
                continue
            action = str(rec.get("action", "")).upper()
            rec_shares = float(rec.get("shares", 0.0) or 0.0)
            if action == "BUY":
                shares += rec_shares
                saw_trade = True
            elif action in {"SELL", "SELL_PARTIAL"}:
                shares -= rec_shares
                saw_trade = True

        if saw_trade:
            return max(0.0, shares)
        return 0.0

    @staticmethod
    def _fetch_yfinance_dividends(ticker: str, start: date, end: date):
        import yfinance as yf

        _configure_yfinance_cache(yf)
        return yf.Ticker(ticker).dividends

    @staticmethod
    def _normalise_dividend_events(raw) -> list[tuple[date, float]]:
        if raw is None:
            return []

        import pandas as pd

        events: list[tuple[date, float]] = []
        if isinstance(raw, pd.Series):
            series = raw.dropna()
            for idx, value in series.items():
                try:
                    ex_date = pd.Timestamp(idx).date()
                    amount = float(value)
                except Exception:
                    continue
                if amount > 0:
                    events.append((ex_date, amount))
            return events

        frame = pd.DataFrame(raw).copy()
        if frame.empty:
            return []
        div_col = next(
            (
                col
                for col in frame.columns
                if str(col).lower() in {"dividend", "dividends", "amount"}
            ),
            None,
        )
        if div_col is None:
            return []
        if "ex_date" in frame.columns:
            dates = frame["ex_date"]
        elif "date" in frame.columns:
            dates = frame["date"]
        else:
            dates = frame.index
        for raw_date, value in zip(dates, frame[div_col]):
            try:
                ex_date = pd.Timestamp(raw_date).date()
                amount = float(value)
            except Exception:
                continue
            if amount > 0:
                events.append((ex_date, amount))
        return events

    def accrue_dividends(
        self,
        as_of: date | datetime | str | None = None,
        *,
        fetcher=None,
        lookback_days: int = DIVIDEND_LOOKBACK_DAYS,
    ) -> dict:
        """
        Credit cash dividends for held shares when dividend data is available.

        Dividends are credited on ex-date using shares held before that date.
        The last processed ex-date is stored per ticker so repeated status
        checks do not double-count the same dividend.
        """
        as_of_date = self._coerce_date(as_of) or date.today()
        result = {
            "as_of": as_of_date.isoformat(),
            "credited": [],
            "total": 0.0,
            "by_ticker": {},
            "errors": {},
            "state_changed": False,
        }
        if not self.positions:
            return result

        if not hasattr(self, "dividend_last_ex_date"):
            self.dividend_last_ex_date = {}
        if not hasattr(self, "dividend_cash_total"):
            self.dividend_cash_total = 0.0

        fetcher = fetcher or self._fetch_yfinance_dividends
        start_floor = as_of_date - timedelta(days=max(1, int(lookback_days)))

        for ticker in sorted(str(t).upper() for t in self.positions):
            first_buy = self._first_buy_date(ticker)
            if first_buy is None:
                continue
            last_ex = self._coerce_date(self.dividend_last_ex_date.get(ticker))
            start_date = max(
                date_value
                for date_value in [start_floor, first_buy, last_ex or start_floor]
                if date_value is not None
            )
            try:
                raw_events = fetcher(ticker, start_date, as_of_date)
                events = self._normalise_dividend_events(raw_events)
            except Exception as exc:
                result["errors"][ticker] = str(exc)
                continue

            for ex_date, dividend_per_share in sorted(events, key=lambda item: item[0]):
                if ex_date > as_of_date:
                    continue
                if last_ex is not None and ex_date <= last_ex:
                    continue
                if ex_date < first_buy:
                    continue

                shares = self._shares_held_before_ex_date(ticker, ex_date)
                if shares <= 0:
                    self.dividend_last_ex_date[ticker] = ex_date.isoformat()
                    result["state_changed"] = True
                    continue

                cash_credit = float(shares * dividend_per_share)
                if cash_credit <= 0:
                    continue

                self.cash += cash_credit
                self.dividend_cash_total += cash_credit
                self.dividend_last_ex_date[ticker] = ex_date.isoformat()
                result["state_changed"] = True
                result["total"] += cash_credit
                result["by_ticker"][ticker] = result["by_ticker"].get(ticker, 0.0) + cash_credit
                credit = {
                    "ticker": ticker,
                    "ex_date": ex_date.isoformat(),
                    "shares": float(shares),
                    "dividend_per_share": float(dividend_per_share),
                    "cash": cash_credit,
                }
                result["credited"].append(credit)
                self._log(
                    "DIVIDEND",
                    ticker,
                    shares,
                    dividend_per_share,
                    f"Dividend cash credit | ex_date={ex_date.isoformat()}",
                    net_cash_flow=cash_credit,
                    extra={
                        "ex_date": ex_date.isoformat(),
                        "dividend_per_share": round(float(dividend_per_share), 6),
                        "dividend_cash": round(cash_credit, 2),
                    },
                )

        result["total"] = round(float(result["total"]), 2)
        result["by_ticker"] = {
            ticker: round(float(value), 2)
            for ticker, value in result["by_ticker"].items()
        }
        return result

    # ── Trade execution ───────────────────────────────────────────────────────

    def buy(
        self,
        ticker: str,
        shares: float,
        price: float,
        reason: str = "",
        *,
        execution_cost: float = 0.0,
        decision_price: float | None = None,
        execution_model: str | None = None,
    ) -> bool:
        cost = shares * price
        execution_cost = max(0.0, float(execution_cost or 0.0))
        total_cost = cost + execution_cost
        if total_cost > self.cash and total_cost > 0:
            scale = self.cash / total_cost
            shares *= scale
            cost *= scale
            execution_cost *= scale
            total_cost = cost + execution_cost
        if shares < 0.001:
            return False

        if ticker in self.positions:
            pos = self.positions[ticker]
            total_shares = pos["shares"] + shares
            pos["avg_cost"] = (
                pos["shares"] * pos["avg_cost"] + cost + execution_cost
            ) / total_shares
            pos["shares"]   = total_shares
            pos["last_price"] = price
            pos["peak_price"] = max(float(pos.get("peak_price", price)), float(price))
            pos.setdefault("weak_signal_streak", 0)
        else:
            self.positions[ticker] = {
                "shares":        shares,
                "avg_cost":      (cost + execution_cost) / shares,
                "last_price":    price,
                "partial_taken": False,   # tracks whether partial profit was taken
                "peak_price":    price,
                "weak_signal_streak": 0,
            }

        self.cash -= total_cost
        self._log(
            "BUY",
            ticker,
            shares,
            price,
            reason,
            execution_cost=execution_cost,
            decision_price=decision_price,
            execution_model=execution_model,
            net_cash_flow=-total_cost,
        )
        return True

    def sell(
        self,
        ticker: str,
        shares: float,
        price: float,
        reason: str = "",
        *,
        execution_cost: float = 0.0,
        decision_price: float | None = None,
        execution_model: str | None = None,
    ) -> bool:
        if ticker not in self.positions:
            return False
        pos = self.positions[ticker]
        shares = min(shares, pos["shares"])
        if shares < 0.001:
            return False

        gross_proceeds = shares * price
        execution_cost = min(max(0.0, float(execution_cost or 0.0)), gross_proceeds)
        proceeds = gross_proceeds - execution_cost
        realized_pnl = (price - float(pos.get("avg_cost", price))) * shares - execution_cost
        self.cash += proceeds
        pos["shares"] -= shares

        if pos["shares"] < 0.001:
            del self.positions[ticker]

        self._log(
            "SELL",
            ticker,
            shares,
            price,
            reason,
            execution_cost=execution_cost,
            decision_price=decision_price,
            execution_model=execution_model,
            realized_pnl=realized_pnl,
            net_cash_flow=proceeds,
        )
        return True

    def sell_all(self, ticker: str, price: float, reason: str = "", **kwargs) -> bool:
        if ticker not in self.positions:
            return False
        return self.sell(ticker, self.positions[ticker]["shares"], price, reason, **kwargs)

    def update_prices(self, prices: dict[str, float]):
        """Update last_price for all held positions."""
        for ticker, price in prices.items():
            if ticker in self.positions and price > 0:
                self.positions[ticker]["last_price"] = price

    def mark_to_latest_cached_prices(
        self,
        price_path: Path | str = PRICE_CACHE_PATH,
    ) -> dict:
        """
        Mark held positions to the latest locally cached close prices.

        This is intended for status/reporting. It updates the in-memory
        portfolio object but does not save to disk unless the caller chooses to.
        """
        price_path = Path(price_path)
        result = {
            "price_path": str(price_path),
            "updated": {},
            "missing": [],
            "latest_date": None,
        }
        if not self.positions or not price_path.exists():
            result["missing"] = sorted(self.positions.keys())
            return result

        try:
            import pandas as pd

            prices = pd.read_parquet(price_path)
            if not isinstance(prices.index, pd.MultiIndex):
                prices = prices.reset_index().set_index(["date", "ticker"])
            frame = prices.reset_index()
            frame["date"] = pd.to_datetime(
                frame["date"],
                utc=True,
                errors="coerce",
            ).dt.tz_convert(None).dt.normalize()
            frame["ticker"] = frame["ticker"].astype(str).str.upper()
            frame = frame.dropna(subset=["date"])
            frame = frame[frame["ticker"].isin([t.upper() for t in self.positions])]
            frame = frame[pd.to_numeric(frame["close"], errors="coerce") > 0]
            if frame.empty:
                result["missing"] = sorted(self.positions.keys())
                return result

            latest_rows = (
                frame.sort_values(["ticker", "date"])
                .groupby("ticker", as_index=False)
                .tail(1)
            )
            updates = {
                str(row["ticker"]): float(row["close"])
                for _, row in latest_rows.iterrows()
            }
            self.update_prices(updates)
            result["updated"] = updates
            result["missing"] = sorted(
                ticker
                for ticker in self.positions
                if ticker.upper() not in updates
            )
            latest_date = latest_rows["date"].max()
            if pd.notna(latest_date):
                result["latest_date"] = pd.Timestamp(latest_date).date().isoformat()
            return result
        except Exception as exc:
            logger.warning("Could not mark portfolio to cached prices: %s", exc)
            result["error"] = str(exc)
            result["missing"] = sorted(self.positions.keys())
            return result

    def refresh_latest_holding_prices(
        self,
        price_path: Path | str = PRICE_CACHE_PATH,
    ) -> dict:
        """
        Fetch latest prices for current holdings and update `last_price`.

        This is the status-mode mark-to-market path: it touches only existing
        positions, never opens or closes trades, and falls back to the local
        parquet cache for any ticker the live fetch misses.
        """
        tickers = sorted(str(ticker).upper() for ticker in self.positions)
        result = {
            "updated": {},
            "sources": {},
            "missing": [],
            "latest_date": None,
            "errors": {},
        }
        if not tickers:
            return result

        import pandas as pd

        live_prices: dict[str, float] = {}
        live_dates: dict[str, str] = {}
        live_source_details: dict[str, str] = {}
        for ticker in tickers:
            try:
                quote = fetch_latest_market_price(ticker)
                price = _positive_float(quote.get("price"))
                if price is not None:
                    live_prices[ticker] = price
                    live_dates[ticker] = quote.get("date") or date.today().isoformat()
                    live_source_details[ticker] = quote.get("source") or "quote"
                    continue
                if quote.get("error"):
                    result["errors"][ticker] = quote["error"]

                from broker.analyst import fetch_ticker_data

                data = fetch_ticker_data(ticker, days=10)
                if data is None or data.empty:
                    continue
                latest = data.dropna(subset=["close"]).iloc[-1]
                price = _positive_float(latest["close"])
                if price is None:
                    continue
                live_prices[ticker] = price
                live_dates[ticker] = pd.Timestamp(latest.name).date().isoformat()
                live_source_details[ticker] = "daily_fetch"
            except Exception as exc:
                result["errors"][ticker] = str(exc)

        if live_prices:
            try:
                from broker.validator import validate_portfolio_prices

                live_prices = validate_portfolio_prices(self.positions, live_prices)
            except Exception as exc:
                result["errors"]["validation"] = str(exc)

        self.update_prices(live_prices)
        for ticker, price in live_prices.items():
            result["updated"][ticker] = float(price)
            result["sources"][ticker] = "live"
        if live_source_details:
            result["source_details"] = live_source_details

        missing_live = [ticker for ticker in tickers if ticker not in live_prices]
        if missing_live:
            cache_mark = self.mark_to_latest_cached_prices(price_path)
            cache_updates = cache_mark.get("updated") or {}
            for ticker in missing_live:
                if ticker in cache_updates:
                    result["updated"][ticker] = float(cache_updates[ticker])
                    result["sources"][ticker] = "cache"
            if live_prices:
                self.update_prices(live_prices)
            if cache_mark.get("latest_date") and not result["latest_date"]:
                result["latest_date"] = cache_mark["latest_date"]
            if cache_mark.get("error"):
                result["errors"]["cache"] = cache_mark["error"]

        all_dates = [date for date in live_dates.values() if date]
        if all_dates:
            result["latest_date"] = max(all_dates)
        result["missing"] = [
            ticker for ticker in tickers if ticker not in result["updated"]
        ]
        return result

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

    def build_snapshot(
        self,
        allocation_summary: dict | None = None,
        spy_price: float | None = None,
        extra: dict | None = None,
    ) -> dict:
        equity = float(self.equity)
        position_values = self.position_values
        position_weights = {
            ticker: float(value / equity) if equity > 0 else 0.0
            for ticker, value in position_values.items()
        }
        top_positions = [
            {"ticker": ticker, "weight": float(weight)}
            for ticker, weight in sorted(
                position_weights.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:5]
        ]

        sector_map = {}
        try:
            from broker.sectors import get_cached_sector_map

            sector_map = get_cached_sector_map(list(self.positions.keys()))
        except Exception:
            sector_map = {}

        theme_weights = exposure_weights(portfolio_theme_values(self.positions, sector_map))
        low_price_weights = exposure_weights(portfolio_low_price_values(self.positions))
        total_execution_cost = sum(
            float(t.get("execution_cost", 0.0) or 0.0)
            for t in self.trade_log
        )

        snapshot = {
            "time": datetime.now().isoformat(),
            "equity": round(equity, 2),
            "cash": round(float(self.cash), 2),
            "cash_weight": float(self.cash / equity) if equity > 0 else 0.0,
            "n_positions": len(self.positions),
            "position_weights": position_weights,
            "top_positions": top_positions,
            "top_1_concentration": top_positions[0]["weight"] if top_positions else 0.0,
            "top_3_concentration": float(
                sum(item["weight"] for item in top_positions[:3])
            ),
            "theme_exposure": theme_weights,
            "theme_effective_bet_count": effective_bet_count(theme_weights),
            "low_price_exposure": (
                low_price_weights.get("sub_5", 0.0)
                + low_price_weights.get("5_to_10", 0.0)
            ),
            "dividend_cash_total": round(float(self.dividend_cash_total), 2),
            "total_execution_cost": total_execution_cost,
            "spy_price": spy_price,
            "allocation_summary": allocation_summary or {},
        }
        if extra:
            snapshot["extra"] = extra
        return snapshot

    def record_snapshot(
        self,
        allocation_summary: dict | None = None,
        spy_price: float | None = None,
        extra: dict | None = None,
        path: Path | str = HISTORY_PATH,
    ) -> dict:
        snapshot = self.build_snapshot(
            allocation_summary=allocation_summary,
            spy_price=spy_price,
            extra=extra,
        )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            fh.write(json.dumps(snapshot) + "\n")
        return snapshot

    def _log(
        self,
        action,
        ticker,
        shares,
        price,
        reason,
        *,
        execution_cost: float = 0.0,
        decision_price: float | None = None,
        execution_model: str | None = None,
        realized_pnl: float | None = None,
        net_cash_flow: float | None = None,
        extra: dict | None = None,
    ):
        entry = {
            "time":   datetime.now().isoformat(),
            "action": action,
            "ticker": ticker,
            "shares": round(shares, 4),
            "price":  round(price, 4),
            "value":  round(shares * price, 2),
            "decision_price": (
                round(float(decision_price), 4)
                if decision_price is not None else round(price, 4)
            ),
            "execution_cost": round(float(execution_cost or 0.0), 4),
            "execution_model": execution_model or "none",
            "net_cash_flow": round(float(net_cash_flow), 2) if net_cash_flow is not None else None,
            "realized_pnl": round(float(realized_pnl), 2) if realized_pnl is not None else None,
            "reason": reason,
            "equity": round(self.equity, 2),
        }
        if extra:
            entry.update(extra)
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
