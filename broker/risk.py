"""
Portfolio-level risk engine.

Enforces:
  - Daily loss limit
  - Drawdown circuit breaker
  - Gross exposure limit
  - Cash floor
  - Volatility scaling
  - Execution cost estimation
"""

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EQUITY_PATH = Path("broker/state/equity_curve.csv")


class PortfolioRiskEngine:
    def __init__(
        self,
        max_daily_loss: float = 0.03,
        max_drawdown: float = 0.15,
        max_gross_exposure: float = 0.95,
        cash_floor: float = 0.05,
        target_volatility: float = 0.15,
        vol_lookback: int = 20,
        equity_curve_path: Path | str | None = EQUITY_PATH,
        equity_history_getter: Callable[[], Sequence[float]] | None = None,
    ):
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown
        self.max_gross_exposure = max_gross_exposure
        self.cash_floor = cash_floor
        self.target_volatility = target_volatility
        self.vol_lookback = vol_lookback
        self.equity_curve_path = (
            Path(equity_curve_path) if equity_curve_path is not None else None
        )
        self.equity_history_getter = equity_history_getter

        self._session_start_equity: float | None = None
        self._peak_equity: float | None = None
        self._session_key = None
        self._market_regime: int | None = None

    def start_session(self, equity: float, session_key=None):
        """Call at the start of each trading cycle."""
        if session_key is not None and session_key != self._session_key:
            self._session_start_equity = equity
            self._session_key = session_key
        elif self._session_start_equity is None:
            self._session_start_equity = equity
        if self._peak_equity is None or equity > self._peak_equity:
            self._peak_equity = equity

    def update_peak(self, equity: float):
        if self._peak_equity is None or equity > self._peak_equity:
            self._peak_equity = equity

    def set_market_regime(self, market_regime: int | None):
        if market_regime in {0, 1, 2, 3}:
            self._market_regime = int(market_regime)
        else:
            self._market_regime = None

    def _effective_cash_floor(self) -> float:
        return float(np.clip(float(self.cash_floor), 0.0, 0.15))

    def _effective_max_gross_exposure(self) -> float:
        return float(np.clip(float(self.max_gross_exposure), 0.75, 1.0))

    def _effective_target_volatility(self) -> float:
        return float(np.clip(float(self.target_volatility), 0.05, 0.40))

    def check_portfolio_health(self, portfolio) -> tuple[str, str]:
        """
        Returns (status, reason) where status is "ok", "warning", or "halt".
        """
        equity = portfolio.equity
        self.update_peak(equity)

        if self._session_start_equity and self._session_start_equity > 0:
            daily_loss = (equity - self._session_start_equity) / self._session_start_equity
            if daily_loss <= -self.max_daily_loss:
                return "halt", (
                    f"Daily loss limit hit: {daily_loss:.1%} "
                    f"(limit: -{self.max_daily_loss:.0%}). No new entries."
                )

        if self._peak_equity and self._peak_equity > 0:
            drawdown = (equity - self._peak_equity) / self._peak_equity
            if drawdown <= -self.max_drawdown:
                return "halt", (
                    f"Drawdown circuit breaker: {drawdown:.1%} from peak "
                    f"(limit: -{self.max_drawdown:.0%}). No new entries."
                )
            if drawdown <= -self.max_drawdown * 0.7:
                return "warning", f"Approaching drawdown limit: {drawdown:.1%} from peak"

        position_value = sum(
            p["shares"] * p["last_price"] for p in portfolio.positions.values()
        )
        gross_exposure = position_value / (equity + 1e-9)
        effective_gross = self._effective_max_gross_exposure()
        if gross_exposure >= effective_gross:
            return "warning", f"Gross exposure at {gross_exposure:.0%} - near limit"

        cash_pct = portfolio.cash / (equity + 1e-9)
        effective_cash_floor = self._effective_cash_floor()
        if cash_pct < effective_cash_floor:
            return "warning", (
                f"Cash below floor: {cash_pct:.1%} "
                f"(floor: {effective_cash_floor:.0%})"
            )

        return "ok", ""

    def check_pre_trade(self, alloc_value: float, portfolio) -> tuple[bool, str]:
        """Returns (allowed, reason). Called before every BUY."""
        equity = portfolio.equity
        effective_cash_floor = self._effective_cash_floor()
        effective_gross = self._effective_max_gross_exposure()

        cash_after = portfolio.cash - alloc_value
        if cash_after < equity * effective_cash_floor:
            max_spend = portfolio.cash - equity * effective_cash_floor
            if max_spend < alloc_value * 0.5:
                return False, f"Would breach cash floor ({effective_cash_floor:.0%})"
            return True, "Capped to preserve cash floor"

        position_value = sum(
            p["shares"] * p["last_price"] for p in portfolio.positions.values()
        )
        new_exposure = (position_value + alloc_value) / (equity + 1e-9)
        if new_exposure > effective_gross:
            return False, f"Would exceed gross exposure limit ({effective_gross:.0%})"

        return True, ""

    def vol_scale_allocation(self, alloc_value: float) -> float:
        """Scale position size down when recent portfolio volatility is elevated."""
        realized_vol = self._get_realized_vol()
        if realized_vol <= 0:
            return alloc_value

        scalar = min(1.0, self._effective_target_volatility() / realized_vol)
        if scalar < 0.9:
            logger.debug(
                "  Vol scaling: %.2fx (realized vol=%.1f%%)",
                scalar,
                realized_vol * 100,
            )
        return alloc_value * scalar

    def _get_realized_vol(self) -> float:
        """Compute annualized realized vol from equity history."""
        try:
            if self.equity_history_getter is not None:
                history = np.asarray(list(self.equity_history_getter()), dtype=float)
                if len(history) < self.vol_lookback + 1:
                    return 0.0
                rets = pd.Series(history).pct_change().dropna().values[-self.vol_lookback:]
                return float(np.std(rets) * np.sqrt(252))

            if self.equity_curve_path is None or not self.equity_curve_path.exists():
                return 0.0

            eq = pd.read_csv(self.equity_curve_path, parse_dates=["time"])
            if len(eq) < self.vol_lookback + 1:
                return 0.0
            rets = eq["equity"].pct_change().dropna().values[-self.vol_lookback:]
            return float(np.std(rets) * np.sqrt(252))
        except Exception:
            return 0.0

    @staticmethod
    def estimate_execution_cost(
        price: float,
        shares: float,
        avg_daily_volume: float = 1_000_000,
    ) -> float:
        """Estimate total execution cost as a fraction of trade value."""
        if price < 1.0:
            spread_pct = 0.05
        elif price < 5.0:
            spread_pct = 0.02
        elif price < 20.0:
            spread_pct = 0.005
        else:
            spread_pct = 0.001

        trade_value = shares * price
        participation = trade_value / (avg_daily_volume * price + 1e-9)
        impact_pct = 0.1 * np.sqrt(participation)

        return spread_pct + impact_pct

    @staticmethod
    def apply_execution_cost(alloc_value: float, price: float, is_penny: bool) -> float:
        """Return allocation reduced by an estimated spread."""
        spread = 0.02 if is_penny else (0.005 if price < 20 else 0.001)
        return alloc_value * (1.0 - spread)


def validate_startup(
    portfolio,
    parquet_path: str = "MasterDS/stooq_panel.parquet",
    sentiment_path: str = "Sentiment/analyst_ratings_with_sentiment.csv",
    max_price_staleness_days: int = 3,
    max_sentiment_staleness_days: int = 7,
) -> list[str]:
    """
    Run pre-flight checks before the broker starts trading.
    Returns a list of blocking errors.
    """
    errors = []
    warnings = []
    today = datetime.today().date()

    try:
        df = pd.read_parquet(parquet_path)
        last_date = pd.to_datetime(df.index).max().date()
        stale = (today - last_date).days
        if stale > max_price_staleness_days:
            errors.append(
                f"PRICE DATA STALE: last date is {last_date} ({stale} days ago). "
                "Run: python Agent.py --mode update"
            )
        else:
            logger.info("  Price data: current (last: %s)", last_date)
    except Exception as exc:
        errors.append(f"PRICE DATA UNREADABLE: {exc}")

    try:
        sent = pd.read_csv(sentiment_path, usecols=["date"])
        parsed = pd.to_datetime(sent["date"], utc=False, errors="coerce")
        parsed = parsed.apply(
            lambda x: x.tz_localize(None) if x is not pd.NaT and x.tzinfo else x
        )
        last_sent = parsed.max()
        if pd.isna(last_sent):
            warnings.append("SENTIMENT DATE UNREADABLE")
        else:
            last_sent = last_sent.date()
            stale = (today - last_sent).days
            if stale > max_sentiment_staleness_days:
                warnings.append(
                    f"SENTIMENT DATA STALE: last date is {last_sent} ({stale} days ago). "
                    "Run: python Agent.py --mode update"
                )
            else:
                logger.info("  Sentiment data: current (last: %s)", last_sent)
    except Exception as exc:
        warnings.append(f"SENTIMENT DATA UNREADABLE: {exc}")

    if portfolio.cash < 0:
        errors.append(f"NEGATIVE CASH: ${portfolio.cash:.2f} - portfolio state is corrupt")

    total_pos_value = sum(
        p["shares"] * p["last_price"] for p in portfolio.positions.values()
    )
    if total_pos_value > portfolio.equity * 1.05:
        errors.append(
            f"POSITION VALUE (${total_pos_value:,.0f}) EXCEEDS EQUITY "
            f"(${portfolio.equity:,.0f}) - accounting error"
        )

    for ticker, pos in portfolio.positions.items():
        if pos.get("shares", 0) < 0:
            errors.append(f"NEGATIVE SHARES in {ticker}: {pos['shares']}")
        if pos.get("last_price", 0) <= 0:
            warnings.append(f"ZERO/NEGATIVE PRICE for {ticker}: {pos.get('last_price')}")

    reserved = portfolio.options.total_reserved_cash
    if reserved > portfolio.cash + 1.0:
        errors.append(
            f"OPTIONS CASH RESERVATION (${reserved:,.0f}) EXCEEDS CASH "
            f"(${portfolio.cash:,.0f}) - accounting error"
        )

    for warning in warnings:
        logger.warning("  STARTUP WARNING: %s", warning)

    if errors:
        for error in errors:
            logger.error("  STARTUP ERROR: %s", error)
    else:
        logger.info("  Startup validation passed.")

    return errors
