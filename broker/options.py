"""
Options trading module.

Supports the following strategies (all defined-risk, no naked selling):
  - Long Call:    bullish directional bet, capped loss = premium paid
  - Long Put:     bearish / hedge, capped loss = premium paid
  - Covered Call: own stock + sell call = income generation
  - Cash-Secured Put: sell put with cash reserved = buy stock cheaper
  - Bull Call Spread: buy call + sell higher call = cheaper bullish bet
  - Bear Put Spread:  buy put + sell lower put = cheaper bearish bet

Data source: yfinance options chain (free, no API key)

Greeks computed via Black-Scholes (scipy).
"""

import logging
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Literal

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)


@contextmanager
def _quiet():
    with open(os.devnull, "w") as dn:
        old = sys.stderr; sys.stderr = dn
        try: yield
        finally: sys.stderr = old


# ── Black-Scholes Greeks ──────────────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    from math import erf, sqrt
    return 0.5 * (1 + erf(x / sqrt(2)))


def black_scholes_greeks(
    S: float,      # current stock price
    K: float,      # strike price
    T: float,      # time to expiry in years
    r: float,      # risk-free rate (use 0.05 as default)
    sigma: float,  # implied volatility (annualised)
    option_type: Literal["call", "put"] = "call",
) -> dict:
    """
    Compute Black-Scholes price and Greeks.
    Returns dict with: price, delta, gamma, theta, vega, iv (=sigma).
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {"price": 0.0, "delta": 0.0, "gamma": 0.0,
                "theta": 0.0, "vega": 0.0, "iv": sigma}

    from math import log, sqrt, exp

    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    if option_type == "call":
        price = S * _norm_cdf(d1) - K * exp(-r * T) * _norm_cdf(d2)
        delta = _norm_cdf(d1)
    else:
        price = K * exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)
        delta = _norm_cdf(d1) - 1.0

    from math import pi
    gamma = _norm_cdf(d1) / (S * sigma * sqrt(T))   # approx
    vega  = S * sqrt(T) * _norm_cdf(d1) * 0.01      # per 1% IV move
    theta = (
        -(S * sigma * _norm_cdf(d1)) / (2 * sqrt(T))
        - r * K * exp(-r * T) * (_norm_cdf(d2) if option_type == "call" else _norm_cdf(-d2))
    ) / 365   # per day

    return {
        "price": round(price, 4),
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta": round(theta, 4),
        "vega":  round(vega,  4),
        "iv":    round(sigma, 4),
    }


# ── Option contract dataclass ─────────────────────────────────────────────────

@dataclass
class OptionContract:
    ticker:       str
    option_type:  Literal["call", "put"]
    strike:       float
    expiry:       date
    contracts:    int          # number of contracts (1 contract = 100 shares)
    premium_paid: float        # per share (multiply by 100 for total cost)
    position:     Literal["long", "short"] = "long"
    strategy:     str          = ""
    greeks:       dict         = field(default_factory=dict)
    open_date:    str          = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def total_cost(self) -> float:
        """Total premium paid/received (positive = paid, negative = received)."""
        mult = 1 if self.position == "long" else -1
        return mult * self.premium_paid * self.contracts * 100

    @property
    def days_to_expiry(self) -> int:
        return (self.expiry - datetime.today().date()).days

    @property
    def is_expired(self) -> bool:
        return self.days_to_expiry <= 0

    @property
    def contract_key(self) -> str:
        return f"{self.ticker}_{self.option_type}_{self.strike}_{self.expiry}"

    def current_value(self, current_price: float, risk_free_rate: float = 0.05) -> float:
        """Estimate current option value using Black-Scholes."""
        T = max(self.days_to_expiry / 365, 0.001)
        iv = self.greeks.get("iv", 0.30)
        bs = black_scholes_greeks(current_price, self.strike, T,
                                  risk_free_rate, iv, self.option_type)
        return bs["price"] * self.contracts * 100

    def pnl(self, current_price: float) -> float:
        """Unrealised P&L."""
        current_val = self.current_value(current_price)
        if self.position == "long":
            return current_val - abs(self.total_cost)
        else:
            return abs(self.total_cost) - current_val


# ── Options chain fetcher ─────────────────────────────────────────────────────

def fetch_options_chain(
    ticker: str,
    min_dte: int = 14,
    max_dte: int = 60,
) -> dict[str, pd.DataFrame] | None:
    """
    Fetch options chain for a ticker.
    Returns dict with 'calls' and 'puts' DataFrames for the best expiry,
    or None if unavailable.
    """
    try:
        with _quiet():
            t = yf.Ticker(ticker)
            expiries = t.options

        if not expiries:
            return None

        # Find expiry in target DTE window
        today = datetime.today().date()
        best_expiry = None
        for exp_str in expiries:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte = (exp_date - today).days
            if min_dte <= dte <= max_dte:
                best_expiry = exp_str
                break

        if best_expiry is None:
            # Fall back to nearest expiry beyond min_dte
            for exp_str in expiries:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                if (exp_date - today).days >= min_dte:
                    best_expiry = exp_str
                    break

        if best_expiry is None:
            return None

        with _quiet():
            chain = t.option_chain(best_expiry)

        return {
            "calls":  chain.calls,
            "puts":   chain.puts,
            "expiry": best_expiry,
        }

    except Exception as e:
        logger.debug(f"Options chain fetch failed for {ticker}: {e}")
        return None


# ── Options strategy analyser ─────────────────────────────────────────────────

def analyse_options(
    ticker: str,
    current_price: float,
    signal_score: float,       # 0-1 composite score from analyst
    sentiment_net: float,      # pos - neg sentiment
    atr_pct: float,            # ATR as % of price (volatility proxy)
    budget: float,             # max dollars to spend on this trade
    risk_free_rate: float = 0.05,
    min_dte: int = 14,
    max_dte: int = 60,
) -> list[OptionContract] | None:
    """
    Analyse options for a ticker and recommend the best strategy.

    Strategy selection logic:
      score > 0.75 + positive sentiment → Long Call (strong bullish)
      score > 0.65 + moderate sentiment → Bull Call Spread (moderate bullish, cheaper)
      score < 0.35 + negative sentiment → Long Put (bearish hedge)
      already own stock + score declining → Covered Call (income while holding)
      score > 0.60 + want to own cheaper → Cash-Secured Put

    Returns list of OptionContract objects to execute, or None.
    """
    chain = fetch_options_chain(ticker, min_dte=min_dte, max_dte=max_dte)
    if chain is None:
        logger.debug(f"No options chain for {ticker}")
        return None

    calls   = chain["calls"]
    puts    = chain["puts"]
    expiry  = datetime.strptime(chain["expiry"], "%Y-%m-%d").date()
    dte     = (expiry - datetime.today().date()).days
    T       = dte / 365

    # Estimate IV from ATR (rough proxy: annualised vol ≈ ATR% × √252)
    iv = min(max(atr_pct * (252 ** 0.5), 0.10), 2.0)

    # ── Filter to liquid, near-the-money options ──────────────────────────────
    atm_range = current_price * 0.15   # within 15% of current price

    liquid_calls = calls[
        (calls["strike"].between(current_price * 0.90, current_price * 1.20)) &
        (calls["volume"].fillna(0) > 0) &
        (calls["ask"].fillna(0) > 0)
    ].copy()

    liquid_puts = puts[
        (puts["strike"].between(current_price * 0.80, current_price * 1.10)) &
        (puts["volume"].fillna(0) > 0) &
        (puts["ask"].fillna(0) > 0)
    ].copy()

    if liquid_calls.empty and liquid_puts.empty:
        logger.debug(f"No liquid options for {ticker}")
        return None

    # ── Strategy selection ────────────────────────────────────────────────────

    # Strong bullish → Long Call (slightly OTM for leverage)
    if signal_score >= 0.75 and sentiment_net > 0.1:
        return _long_call(ticker, liquid_calls, current_price, expiry,
                          budget, iv, T, risk_free_rate, "Strong bullish signal")

    # Moderate bullish → Bull Call Spread (defined risk, cheaper)
    elif signal_score >= 0.65 and sentiment_net > 0:
        return _bull_call_spread(ticker, liquid_calls, current_price, expiry,
                                 budget, iv, T, risk_free_rate, "Moderate bullish signal")

    # Bearish → Long Put
    elif signal_score < 0.35 and sentiment_net < -0.1:
        return _long_put(ticker, liquid_puts, current_price, expiry,
                         budget, iv, T, risk_free_rate, "Bearish signal")

    # Neutral bullish → Cash-Secured Put (get paid to potentially buy cheaper)
    elif signal_score >= 0.60:
        return _cash_secured_put(ticker, liquid_puts, current_price, expiry,
                                 budget, iv, T, risk_free_rate, "Neutral-bullish, income strategy")

    return None


# ── Strategy builders ─────────────────────────────────────────────────────────

def _long_call(ticker, calls, spot, expiry, budget, iv, T, r, reason) -> list[OptionContract] | None:
    """Buy a slightly OTM call."""
    target_strike = spot * 1.05   # 5% OTM
    row = _nearest_strike(calls, target_strike)
    if row is None:
        return None

    strike  = float(row["strike"])
    premium = float(row["ask"])
    n_contracts = max(1, int(budget / (premium * 100)))
    if premium * 100 > budget:
        return None

    greeks = black_scholes_greeks(spot, strike, T, r, iv, "call")
    return [OptionContract(
        ticker=ticker, option_type="call", strike=strike,
        expiry=expiry, contracts=n_contracts, premium_paid=premium,
        position="long", strategy=f"Long Call | {reason}", greeks=greeks,
    )]


def _bull_call_spread(ticker, calls, spot, expiry, budget, iv, T, r, reason) -> list[OptionContract] | None:
    """Buy ATM call + sell OTM call. Cheaper than outright call."""
    buy_strike  = spot * 1.00   # ATM
    sell_strike = spot * 1.10   # 10% OTM

    buy_row  = _nearest_strike(calls, buy_strike)
    sell_row = _nearest_strike(calls, sell_strike)
    if buy_row is None or sell_row is None:
        return None

    buy_premium  = float(buy_row["ask"])
    sell_premium = float(sell_row["bid"])
    net_debit    = buy_premium - sell_premium

    if net_debit <= 0 or net_debit * 100 > budget:
        return None

    n_contracts = max(1, int(budget / (net_debit * 100)))
    buy_greeks  = black_scholes_greeks(spot, float(buy_row["strike"]),  T, r, iv, "call")
    sell_greeks = black_scholes_greeks(spot, float(sell_row["strike"]), T, r, iv, "call")

    return [
        OptionContract(
            ticker=ticker, option_type="call",
            strike=float(buy_row["strike"]), expiry=expiry,
            contracts=n_contracts, premium_paid=buy_premium,
            position="long", strategy=f"Bull Call Spread (buy leg) | {reason}",
            greeks=buy_greeks,
        ),
        OptionContract(
            ticker=ticker, option_type="call",
            strike=float(sell_row["strike"]), expiry=expiry,
            contracts=n_contracts, premium_paid=sell_premium,
            position="short", strategy=f"Bull Call Spread (sell leg) | {reason}",
            greeks=sell_greeks,
        ),
    ]


def _long_put(ticker, puts, spot, expiry, budget, iv, T, r, reason) -> list[OptionContract] | None:
    """Buy a slightly OTM put as a bearish bet or hedge."""
    target_strike = spot * 0.95   # 5% OTM
    row = _nearest_strike(puts, target_strike)
    if row is None:
        return None

    strike  = float(row["strike"])
    premium = float(row["ask"])
    if premium * 100 > budget:
        return None

    n_contracts = max(1, int(budget / (premium * 100)))
    greeks = black_scholes_greeks(spot, strike, T, r, iv, "put")
    return [OptionContract(
        ticker=ticker, option_type="put", strike=strike,
        expiry=expiry, contracts=n_contracts, premium_paid=premium,
        position="long", strategy=f"Long Put | {reason}", greeks=greeks,
    )]


def _cash_secured_put(ticker, puts, spot, expiry, budget, iv, T, r, reason) -> list[OptionContract] | None:
    """
    Sell an OTM put. Collect premium; if assigned, buy stock at strike.
    Requires cash = strike × 100 × contracts reserved.
    """
    target_strike = spot * 0.92   # 8% OTM — willing to buy at this price
    row = _nearest_strike(puts, target_strike)
    if row is None:
        return None

    strike  = float(row["strike"])
    premium = float(row["bid"])   # we receive the bid
    if premium <= 0:
        return None

    # Cash required = strike × 100 per contract
    cash_per_contract = strike * 100
    n_contracts = max(1, int(budget / cash_per_contract))

    greeks = black_scholes_greeks(spot, strike, T, r, iv, "put")
    return [OptionContract(
        ticker=ticker, option_type="put", strike=strike,
        expiry=expiry, contracts=n_contracts, premium_paid=premium,
        position="short", strategy=f"Cash-Secured Put | {reason}", greeks=greeks,
    )]


def _nearest_strike(df: pd.DataFrame, target: float) -> pd.Series | None:
    if df.empty:
        return None
    idx = (df["strike"] - target).abs().idxmin()
    return df.loc[idx]


# ── Options portfolio manager ─────────────────────────────────────────────────

class OptionsBook:
    """
    Tracks all open option positions.
    Integrated into Portfolio for unified P&L reporting.
    """

    def __init__(self):
        self.positions: dict[str, OptionContract] = {}   # key → contract

    def open(self, contract: OptionContract, cash_available: float) -> tuple[bool, float]:
        """
        Open an option position.
        Returns (success, cash_spent).
        """
        cost = abs(contract.total_cost)
        if contract.position == "long" and cost > cash_available:
            return False, 0.0

        self.positions[contract.contract_key] = contract
        cash_delta = -cost if contract.position == "long" else cost  # short = receive premium
        logger.info(
            f"  OPTION {contract.position.upper()} {contract.option_type.upper()} "
            f"{contract.ticker} ${contract.strike} exp {contract.expiry} "
            f"× {contract.contracts} contracts @ ${contract.premium_paid:.2f} "
            f"| {contract.strategy}"
        )
        return True, cash_delta

    def close(self, key: str, current_price: float) -> tuple[bool, float]:
        """Close an option position. Returns (success, cash_received)."""
        if key not in self.positions:
            return False, 0.0
        contract = self.positions.pop(key)
        current_val = contract.current_value(current_price)
        if contract.position == "long":
            cash_back = current_val
        else:
            cash_back = abs(contract.total_cost) - current_val
        logger.info(
            f"  CLOSE OPTION {contract.ticker} {contract.option_type} "
            f"${contract.strike} | P&L: ${contract.pnl(current_price):+.2f}"
        )
        return True, cash_back

    def expire_worthless(self, key: str) -> float:
        """Mark expired worthless option. Returns loss amount."""
        if key not in self.positions:
            return 0.0
        contract = self.positions.pop(key)
        loss = abs(contract.total_cost) if contract.position == "long" else 0.0
        logger.info(
            f"  EXPIRED WORTHLESS: {contract.ticker} {contract.option_type} "
            f"${contract.strike} | Loss: ${loss:.2f}"
        )
        return loss

    def check_expirations(self, current_prices: dict[str, float]) -> list[str]:
        """
        Check all positions for expiry. Returns list of keys that expired.
        Handles assignment for short puts (stock purchase) and short calls.
        """
        expired = []
        for key, contract in list(self.positions.items()):
            if not contract.is_expired:
                continue

            spot = current_prices.get(contract.ticker, 0.0)
            if spot <= 0:
                expired.append(key)
                continue

            if contract.position == "long":
                # Long call: exercise if ITM
                if contract.option_type == "call" and spot > contract.strike:
                    intrinsic = (spot - contract.strike) * contract.contracts * 100
                    logger.info(
                        f"  EXERCISED: Long call {contract.ticker} ${contract.strike} "
                        f"| Intrinsic value: ${intrinsic:.2f}"
                    )
                # Long put: exercise if ITM
                elif contract.option_type == "put" and spot < contract.strike:
                    intrinsic = (contract.strike - spot) * contract.contracts * 100
                    logger.info(
                        f"  EXERCISED: Long put {contract.ticker} ${contract.strike} "
                        f"| Intrinsic value: ${intrinsic:.2f}"
                    )
                else:
                    self.expire_worthless(key)

            elif contract.position == "short":
                # Short put assigned: must buy stock at strike
                if contract.option_type == "put" and spot < contract.strike:
                    logger.info(
                        f"  ASSIGNED: Short put {contract.ticker} ${contract.strike} "
                        f"— buying {contract.contracts * 100} shares at ${contract.strike}"
                    )
                else:
                    # Expires worthless — keep premium (already received)
                    logger.info(
                        f"  EXPIRED: Short {contract.option_type} {contract.ticker} "
                        f"${contract.strike} — premium kept"
                    )

            expired.append(key)

        for key in expired:
            self.positions.pop(key, None)

        return expired

    @property
    def total_value(self) -> float:
        """Estimated total market value of all option positions."""
        return sum(
            c.current_value(0.0) for c in self.positions.values()
        )

    def summary_lines(self) -> list[str]:
        if not self.positions:
            return []
        lines = [
            f"  {'─'*52}",
            f"  Options ({len(self.positions)} positions):",
            f"  {'Contract':<30} {'DTE':>4}  {'Delta':>7}  {'Theta':>7}",
            f"  {'─'*52}",
        ]
        for key, c in self.positions.items():
            delta = c.greeks.get("delta", 0.0)
            theta = c.greeks.get("theta", 0.0)
            label = f"{c.position.upper()} {c.option_type.upper()} {c.ticker} ${c.strike}"
            lines.append(
                f"  {label:<30} {c.days_to_expiry:>4}d  "
                f"{delta:>+7.3f}  {theta:>+7.3f}/d"
            )
        return lines
