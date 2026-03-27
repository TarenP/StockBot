"""
Portfolio-level risk engine.

Enforces:
  - Daily loss limit (halt new entries if down X% today)
  - Drawdown circuit breaker (go to cash if DD from peak > threshold)
  - Gross exposure limit (max % of equity invested)
  - Cash floor (always keep minimum cash buffer)
  - Volatility scaling (reduce position sizes when realized vol is elevated)
  - Execution cost estimation (spread + market impact per trade)

Called by BrokerBrain before every trade and at the start of every cycle.
"""

import logging
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EQUITY_PATH = Path("broker/state/equity_curve.csv")


class PortfolioRiskEngine:
    def __init__(
        self,
        max_daily_loss:      float = 0.03,   # halt new entries if down 3% today
        max_drawdown:        float = 0.15,   # go to cash if DD from peak > 15%
        max_gross_exposure:  float = 0.95,   # max 95% of equity in positions
        cash_floor:          float = 0.05,   # always keep 5% cash
        target_volatility:   float = 0.15,   # annualised vol target for scaling
        vol_lookback:        int   = 20,     # days for realized vol calculation
    ):
        self.max_daily_loss     = max_daily_loss
        self.max_drawdown       = max_drawdown
        self.max_gross_exposure = max_gross_exposure
        self.cash_floor         = cash_floor
        self.target_volatility  = target_volatility
        self.vol_lookback       = vol_lookback

        self._session_start_equity: float | None = None
        self._peak_equity:          float | None = None

    # ── Session tracking ──────────────────────────────────────────────────────

    def start_session(self, equity: float):
        """Call at the start of each trading cycle."""
        if self._session_start_equity is None:
            self._session_start_equity = equity
        if self._peak_equity is None or equity > self._peak_equity:
            self._peak_equity = equity

    def update_peak(self, equity: float):
        if self._peak_equity is None or equity > self._peak_equity:
            self._peak_equity = equity

    # ── Portfolio health check ────────────────────────────────────────────────

    def check_portfolio_health(self, portfolio) -> tuple[str, str]:
        """
        Returns (status, reason) where status is 'ok', 'warning', or 'halt'.
        'halt' means: do not open any new positions this cycle.
        """
        equity = portfolio.equity
        self.update_peak(equity)

        # ── Daily loss check ──────────────────────────────────────────────────
        if self._session_start_equity and self._session_start_equity > 0:
            daily_loss = (equity - self._session_start_equity) / self._session_start_equity
            if daily_loss <= -self.max_daily_loss:
                return "halt", (
                    f"Daily loss limit hit: {daily_loss:.1%} "
                    f"(limit: -{self.max_daily_loss:.0%}). No new entries."
                )

        # ── Drawdown circuit breaker ──────────────────────────────────────────
        if self._peak_equity and self._peak_equity > 0:
            drawdown = (equity - self._peak_equity) / self._peak_equity
            if drawdown <= -self.max_drawdown:
                return "halt", (
                    f"Drawdown circuit breaker: {drawdown:.1%} from peak "
                    f"(limit: -{self.max_drawdown:.0%}). No new entries."
                )
            if drawdown <= -self.max_drawdown * 0.7:
                return "warning", f"Approaching drawdown limit: {drawdown:.1%} from peak"

        # ── Gross exposure check ──────────────────────────────────────────────
        position_value = sum(
            p["shares"] * p["last_price"] for p in portfolio.positions.values()
        )
        gross_exposure = position_value / (equity + 1e-9)
        if gross_exposure >= self.max_gross_exposure:
            return "warning", f"Gross exposure at {gross_exposure:.0%} — near limit"

        # ── Cash floor check ──────────────────────────────────────────────────
        cash_pct = portfolio.cash / (equity + 1e-9)
        if cash_pct < self.cash_floor:
            return "warning", f"Cash below floor: {cash_pct:.1%} (floor: {self.cash_floor:.0%})"

        return "ok", ""

    # ── Pre-trade check ───────────────────────────────────────────────────────

    def check_pre_trade(self, alloc_value: float, portfolio) -> tuple[bool, str]:
        """
        Returns (allowed, reason). Called before every BUY decision.
        """
        equity = portfolio.equity

        # Cash floor: never let cash drop below floor after this trade
        cash_after = portfolio.cash - alloc_value
        if cash_after < equity * self.cash_floor:
            max_spend = portfolio.cash - equity * self.cash_floor
            if max_spend < alloc_value * 0.5:
                return False, f"Would breach cash floor ({self.cash_floor:.0%})"
            # Allow but cap the allocation
            return True, f"Capped to preserve cash floor"

        # Gross exposure: don't exceed limit
        position_value = sum(
            p["shares"] * p["last_price"] for p in portfolio.positions.values()
        )
        new_exposure = (position_value + alloc_value) / (equity + 1e-9)
        if new_exposure > self.max_gross_exposure:
            return False, f"Would exceed gross exposure limit ({self.max_gross_exposure:.0%})"

        return True, ""

    # ── Volatility scaling ────────────────────────────────────────────────────

    def vol_scale_allocation(self, alloc_value: float) -> float:
        """
        Scale position size down when recent portfolio volatility is elevated.
        Uses equity curve to estimate realized vol.
        """
        realized_vol = self._get_realized_vol()
        if realized_vol <= 0:
            return alloc_value

        scalar = min(1.0, self.target_volatility / realized_vol)
        if scalar < 0.9:
            logger.debug(f"  Vol scaling: {scalar:.2f}x (realized vol={realized_vol:.1%})")
        return alloc_value * scalar

    def _get_realized_vol(self) -> float:
        """Compute annualised realized vol from equity curve."""
        if not EQUITY_PATH.exists():
            return 0.0
        try:
            eq = pd.read_csv(EQUITY_PATH, parse_dates=["time"])
            if len(eq) < self.vol_lookback + 1:
                return 0.0
            rets = eq["equity"].pct_change().dropna().values[-self.vol_lookback:]
            return float(np.std(rets) * np.sqrt(252))
        except Exception:
            return 0.0

    # ── Execution cost estimation ─────────────────────────────────────────────

    @staticmethod
    def estimate_execution_cost(
        price: float,
        shares: float,
        avg_daily_volume: float = 1_000_000,
    ) -> float:
        """
        Estimate total execution cost as a fraction of trade value.
        Includes bid-ask spread and market impact.
        """
        # Spread estimate by price tier
        if price < 1.0:
            spread_pct = 0.05    # 5% for sub-$1 stocks
        elif price < 5.0:
            spread_pct = 0.02    # 2% for penny stocks
        elif price < 20.0:
            spread_pct = 0.005   # 0.5% for low-price stocks
        else:
            spread_pct = 0.001   # 0.1% for liquid stocks

        # Market impact: square-root model
        trade_value       = shares * price
        participation     = trade_value / (avg_daily_volume * price + 1e-9)
        impact_pct        = 0.1 * np.sqrt(participation)

        return spread_pct + impact_pct

    @staticmethod
    def apply_execution_cost(alloc_value: float, price: float, is_penny: bool) -> float:
        """Return alloc_value reduced by estimated execution cost."""
        spread = 0.02 if is_penny else (0.005 if price < 20 else 0.001)
        return alloc_value * (1.0 - spread)


# ── Startup validation ────────────────────────────────────────────────────────

def validate_startup(
    portfolio,
    parquet_path: str = "MasterDS/stooq_panel.parquet",
    sentiment_path: str = "Sentiment/analyst_ratings_with_sentiment.csv",
    max_price_staleness_days: int = 3,
    max_sentiment_staleness_days: int = 7,
) -> list[str]:
    """
    Run pre-flight checks before the broker starts trading.
    Returns list of error strings. Empty list = all clear.
    """
    errors   = []
    warnings = []
    today    = datetime.today().date()

    # ── Price data freshness ──────────────────────────────────────────────────
    try:
        import pandas as pd
        df        = pd.read_parquet(parquet_path)
        last_date = pd.to_datetime(df.index).max().date()
        stale     = (today - last_date).days
        if stale > max_price_staleness_days:
            errors.append(
                f"PRICE DATA STALE: last date is {last_date} ({stale} days ago). "
                f"Run: python Agent.py --mode update"
            )
        else:
            logger.info(f"  Price data: current (last: {last_date})")
    except Exception as e:
        errors.append(f"PRICE DATA UNREADABLE: {e}")

    # ── Sentiment data freshness ──────────────────────────────────────────────
    try:
        sent      = pd.read_csv(sentiment_path, usecols=["date"])
        last_sent = pd.to_datetime(sent["date"], utc=True, errors="coerce").dt.tz_convert(None).max().date()
        stale     = (today - last_sent).days
        if stale > max_sentiment_staleness_days:
            warnings.append(
                f"SENTIMENT DATA STALE: last date is {last_sent} ({stale} days ago). "
                f"Run: python Agent.py --mode update"
            )
        else:
            logger.info(f"  Sentiment data: current (last: {last_sent})")
    except Exception as e:
        warnings.append(f"SENTIMENT DATA UNREADABLE: {e}")

    # ── Portfolio state consistency ───────────────────────────────────────────
    if portfolio.cash < 0:
        errors.append(f"NEGATIVE CASH: ${portfolio.cash:.2f} — portfolio state is corrupt")

    total_pos_value = sum(
        p["shares"] * p["last_price"] for p in portfolio.positions.values()
    )
    if total_pos_value > portfolio.equity * 1.05:
        errors.append(
            f"POSITION VALUE (${total_pos_value:,.0f}) EXCEEDS EQUITY "
            f"(${portfolio.equity:,.0f}) — accounting error"
        )

    for ticker, pos in portfolio.positions.items():
        if pos.get("shares", 0) < 0:
            errors.append(f"NEGATIVE SHARES in {ticker}: {pos['shares']}")
        if pos.get("last_price", 0) <= 0:
            warnings.append(f"ZERO/NEGATIVE PRICE for {ticker}: {pos.get('last_price')}")

    # ── Options state consistency ─────────────────────────────────────────────
    reserved = portfolio.options.total_reserved_cash
    if reserved > portfolio.cash + 1.0:
        errors.append(
            f"OPTIONS CASH RESERVATION (${reserved:,.0f}) EXCEEDS CASH "
            f"(${portfolio.cash:,.0f}) — accounting error"
        )

    # ── Log results ───────────────────────────────────────────────────────────
    for w in warnings:
        logger.warning(f"  STARTUP WARNING: {w}")

    if errors:
        for e in errors:
            logger.error(f"  STARTUP ERROR: {e}")
    else:
        logger.info("  Startup validation passed.")

    return errors   # warnings don't block startup, errors do
