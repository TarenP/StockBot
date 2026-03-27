"""
Options trading module.

Supports the following strategies (all defined-risk, no naked selling):
  - Long Call:        bullish directional bet, capped loss = premium paid
  - Long Put:         bearish / hedge, capped loss = premium paid
  - Cash-Secured Put: sell put with FULL cash reserved at strike price
  - Bull Call Spread: buy call + sell higher call = cheaper bullish bet

Data source: yfinance options chain (free, no API key)
Greeks computed via Black-Scholes. IV sourced from market chain, not ATR proxy.

ACCOUNTING RULES (audit-compliant):
  - Long options:        cash reduced by premium × 100 × contracts on open
  - Short puts (CSP):    cash reduced by (strike × 100 × contracts) - premium received
                         i.e. full cash reservation at strike price
  - Assignment (short):  cash deducted at strike, shares added to portfolio
  - Exercise (long):     cash deducted at strike, shares added to portfolio
  - Expiry worthless:    long = lose premium, short = keep premium (already received)
"""

import logging
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, date
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
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: Literal["call", "put"] = "call",
) -> dict:
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

    gamma = _norm_cdf(d1) / (S * sigma * sqrt(T))
    vega  = S * sqrt(T) * _norm_cdf(d1) * 0.01
    theta = (
        -(S * sigma * _norm_cdf(d1)) / (2 * sqrt(T))
        - r * K * exp(-r * T) * (_norm_cdf(d2) if option_type == "call" else _norm_cdf(-d2))
    ) / 365

    return {
        "price": round(price, 4), "delta": round(delta, 4),
        "gamma": round(gamma, 6), "theta": round(theta, 4),
        "vega":  round(vega,  4), "iv":    round(sigma, 4),
    }


# ── Option contract ───────────────────────────────────────────────────────────

@dataclass
class OptionContract:
    ticker:           str
    option_type:      Literal["call", "put"]
    strike:           float
    expiry:           date
    contracts:        int
    premium_paid:     float        # per share (×100 for total)
    position:         Literal["long", "short"] = "long"
    strategy:         str          = ""
    greeks:           dict         = field(default_factory=dict)
    open_date:        str          = field(default_factory=lambda: datetime.now().isoformat())
    # For short puts: cash reserved at strike price (not just premium)
    cash_reserved:    float        = 0.0

    @property
    def total_cost(self) -> float:
        """Net cash impact at open (positive = cash out, negative = cash in)."""
        if self.position == "long":
            return self.premium_paid * self.contracts * 100
        else:
            # Short: received premium minus reserved cash
            # cash_reserved is stored separately; total_cost = premium received
            return -(self.premium_paid * self.contracts * 100)

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
        """Current mark-to-market value using Black-Scholes with stored IV."""
        T  = max(self.days_to_expiry / 365, 0.001)
        iv = self.greeks.get("iv", 0.30)
        bs = black_scholes_greeks(current_price, self.strike, T,
                                  risk_free_rate, iv, self.option_type)
        val = bs["price"] * self.contracts * 100
        return val if self.position == "long" else -val

    def pnl(self, current_price: float) -> float:
        """Unrealised P&L vs open cost."""
        if self.position == "long":
            return self.current_value(current_price) - (self.premium_paid * self.contracts * 100)
        else:
            # Short: profit = premium received - current cost to close
            premium_received = self.premium_paid * self.contracts * 100
            cost_to_close    = abs(self.current_value(current_price))
            return premium_received - cost_to_close

    @property
    def max_profit(self) -> float:
        if self.position == "long":
            return float("inf")   # unlimited for calls
        else:
            return self.premium_paid * self.contracts * 100   # premium received

    @property
    def max_loss(self) -> float:
        if self.position == "long":
            return self.premium_paid * self.contracts * 100
        else:
            # Short put: max loss = (strike - premium) × 100 × contracts
            return (self.strike - self.premium_paid) * self.contracts * 100


# ── Options chain fetcher ─────────────────────────────────────────────────────

# Liquidity filters
MIN_OPTION_OI     = 100
MIN_OPTION_VOLUME = 10
MAX_SPREAD_PCT    = 0.15   # max bid-ask spread as % of mid


def fetch_options_chain(
    ticker: str, min_dte: int = 14, max_dte: int = 60,
) -> dict | None:
    try:
        with _quiet():
            t        = yf.Ticker(ticker)
            expiries = t.options
        if not expiries:
            return None

        today       = datetime.today().date()
        best_expiry = None
        for exp_str in expiries:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte      = (exp_date - today).days
            if min_dte <= dte <= max_dte:
                best_expiry = exp_str
                break
        if best_expiry is None:
            for exp_str in expiries:
                if (datetime.strptime(exp_str, "%Y-%m-%d").date() - today).days >= min_dte:
                    best_expiry = exp_str
                    break
        if best_expiry is None:
            return None

        with _quiet():
            chain = t.option_chain(best_expiry)

        return {"calls": chain.calls, "puts": chain.puts, "expiry": best_expiry}
    except Exception as e:
        logger.debug(f"Options chain fetch failed for {ticker}: {e}")
        return None


def _filter_liquid(df: pd.DataFrame) -> pd.DataFrame:
    """Apply liquidity filters to an options chain DataFrame."""
    df = df.copy()
    df = df[df["openInterest"].fillna(0) >= MIN_OPTION_OI]
    df = df[df["volume"].fillna(0)       >= MIN_OPTION_VOLUME]
    df = df[df["ask"].fillna(0) > 0]
    df = df[df["bid"].fillna(0) > 0]
    mid    = (df["ask"] + df["bid"]) / 2
    spread = (df["ask"] - df["bid"]) / (mid + 1e-9)
    df     = df[spread <= MAX_SPREAD_PCT]
    return df


def _get_market_iv(chain_df: pd.DataFrame, strike: float) -> float:
    """Extract implied volatility from the options chain near a given strike."""
    try:
        row = chain_df.iloc[(chain_df["strike"] - strike).abs().argsort()[:1]]
        iv  = float(row["impliedVolatility"].iloc[0])
        if pd.notna(iv) and 0.01 < iv < 5.0:
            return iv
    except Exception:
        pass
    return 0.30   # fallback


def _nearest_strike(df: pd.DataFrame, target: float) -> pd.Series | None:
    if df.empty:
        return None
    idx = (df["strike"] - target).abs().idxmin()
    return df.loc[idx]


# ── Strategy builders ─────────────────────────────────────────────────────────

def _long_call(ticker, calls, spot, expiry, budget, T, r, reason) -> list[OptionContract] | None:
    target = spot * 1.05
    row    = _nearest_strike(calls, target)
    if row is None:
        return None
    strike  = float(row["strike"])
    premium = float(row["ask"])
    if premium * 100 > budget:
        return None
    n_contracts = max(1, int(budget / (premium * 100)))
    iv     = _get_market_iv(calls, strike)
    greeks = black_scholes_greeks(spot, strike, T, r, iv, "call")
    return [OptionContract(
        ticker=ticker, option_type="call", strike=strike, expiry=expiry,
        contracts=n_contracts, premium_paid=premium,
        position="long", strategy=f"Long Call | {reason}", greeks=greeks,
    )]


def _bull_call_spread(ticker, calls, spot, expiry, budget, T, r, reason) -> list[OptionContract] | None:
    buy_row  = _nearest_strike(calls, spot * 1.00)
    sell_row = _nearest_strike(calls, spot * 1.10)
    if buy_row is None or sell_row is None:
        return None
    buy_premium  = float(buy_row["ask"])
    sell_premium = float(sell_row["bid"])
    net_debit    = buy_premium - sell_premium
    if net_debit <= 0 or net_debit * 100 > budget:
        return None
    n_contracts = max(1, int(budget / (net_debit * 100)))
    buy_iv  = _get_market_iv(calls, float(buy_row["strike"]))
    sell_iv = _get_market_iv(calls, float(sell_row["strike"]))
    return [
        OptionContract(
            ticker=ticker, option_type="call", strike=float(buy_row["strike"]),
            expiry=expiry, contracts=n_contracts, premium_paid=buy_premium,
            position="long", strategy=f"Bull Call Spread (buy) | {reason}",
            greeks=black_scholes_greeks(spot, float(buy_row["strike"]), T, r, buy_iv, "call"),
        ),
        OptionContract(
            ticker=ticker, option_type="call", strike=float(sell_row["strike"]),
            expiry=expiry, contracts=n_contracts, premium_paid=sell_premium,
            position="short", strategy=f"Bull Call Spread (sell) | {reason}",
            greeks=black_scholes_greeks(spot, float(sell_row["strike"]), T, r, sell_iv, "call"),
        ),
    ]


def _long_put(ticker, puts, spot, expiry, budget, T, r, reason) -> list[OptionContract] | None:
    row = _nearest_strike(puts, spot * 0.95)
    if row is None:
        return None
    strike  = float(row["strike"])
    premium = float(row["ask"])
    if premium * 100 > budget:
        return None
    n_contracts = max(1, int(budget / (premium * 100)))
    iv     = _get_market_iv(puts, strike)
    greeks = black_scholes_greeks(spot, strike, T, r, iv, "put")
    return [OptionContract(
        ticker=ticker, option_type="put", strike=strike, expiry=expiry,
        contracts=n_contracts, premium_paid=premium,
        position="long", strategy=f"Long Put | {reason}", greeks=greeks,
    )]


def _cash_secured_put(ticker, puts, spot, expiry, budget, T, r, reason) -> list[OptionContract] | None:
    """
    Sell OTM put. FULL cash reservation at strike price is enforced.
    budget here is the maximum cash to reserve (not just premium budget).
    """
    row = _nearest_strike(puts, spot * 0.92)
    if row is None:
        return None
    strike  = float(row["strike"])
    premium = float(row["bid"])
    if premium <= 0:
        return None

    # Cash required per contract = strike × 100 (full reservation)
    cash_per_contract = strike * 100
    n_contracts = max(1, int(budget / cash_per_contract))
    if cash_per_contract > budget:
        return None   # can't afford even one contract

    cash_reserved = cash_per_contract * n_contracts
    iv     = _get_market_iv(puts, strike)
    greeks = black_scholes_greeks(spot, strike, T, r, iv, "put")

    return [OptionContract(
        ticker=ticker, option_type="put", strike=strike, expiry=expiry,
        contracts=n_contracts, premium_paid=premium,
        position="short", strategy=f"Cash-Secured Put | {reason}",
        greeks=greeks, cash_reserved=cash_reserved,
    )]


# ── Strategy selector ─────────────────────────────────────────────────────────

def analyse_options(
    ticker: str, current_price: float, signal_score: float,
    sentiment_net: float, atr_pct: float, budget: float,
    risk_free_rate: float = 0.05, min_dte: int = 14, max_dte: int = 60,
) -> list[OptionContract] | None:
    chain = fetch_options_chain(ticker, min_dte=min_dte, max_dte=max_dte)
    if chain is None:
        return None

    expiry = datetime.strptime(chain["expiry"], "%Y-%m-%d").date()
    T      = max((expiry - datetime.today().date()).days / 365, 0.001)
    r      = risk_free_rate

    calls = _filter_liquid(chain["calls"])
    calls = calls[calls["strike"].between(current_price * 0.90, current_price * 1.20)]

    puts  = _filter_liquid(chain["puts"])
    puts  = puts[puts["strike"].between(current_price * 0.80, current_price * 1.10)]

    if calls.empty and puts.empty:
        return None

    if signal_score >= 0.75 and sentiment_net > 0.1:
        return _long_call(ticker, calls, current_price, expiry, budget, T, r, "Strong bullish")
    elif signal_score >= 0.65 and sentiment_net > 0:
        return _bull_call_spread(ticker, calls, current_price, expiry, budget, T, r, "Moderate bullish")
    elif signal_score < 0.35 and sentiment_net < -0.1:
        return _long_put(ticker, puts, current_price, expiry, budget, T, r, "Bearish")
    elif signal_score >= 0.60:
        return _cash_secured_put(ticker, puts, current_price, expiry, budget, T, r, "Neutral-bullish income")

    return None


# ── Options book ──────────────────────────────────────────────────────────────

class OptionsBook:
    def __init__(self):
        self.positions: dict[str, OptionContract] = {}

    def open(self, contract: OptionContract, cash_available: float) -> tuple[bool, float]:
        """
        Open an option position. Returns (success, cash_delta).
        cash_delta is negative (cash out) for longs and CSPs (reservation).
        """
        if contract.position == "long":
            cost = contract.premium_paid * contract.contracts * 100
            if cost > cash_available:
                return False, 0.0
            cash_delta = -cost

        elif contract.position == "short" and contract.option_type == "put":
            # Cash-secured put: reserve full strike value, receive premium
            cash_needed = contract.cash_reserved - (contract.premium_paid * contract.contracts * 100)
            if cash_needed > cash_available:
                logger.warning(
                    f"  Insufficient cash for CSP {contract.ticker} ${contract.strike}: "
                    f"need ${cash_needed:,.0f}, have ${cash_available:,.0f}"
                )
                return False, 0.0
            cash_delta = -cash_needed   # net cash out = reservation - premium received

        else:
            # Other short positions (spread sell leg): receive premium
            cash_delta = contract.premium_paid * contract.contracts * 100

        self.positions[contract.contract_key] = contract
        logger.info(
            f"  OPTION {contract.position.upper()} {contract.option_type.upper()} "
            f"{contract.ticker} ${contract.strike} exp {contract.expiry} "
            f"×{contract.contracts} @ ${contract.premium_paid:.2f} | {contract.strategy}"
        )
        return True, cash_delta

    def close(self, key: str, current_price: float) -> tuple[bool, float]:
        """Close a position. Returns (success, cash_received)."""
        if key not in self.positions:
            return False, 0.0
        contract = self.positions.pop(key)
        pnl      = contract.pnl(current_price)

        if contract.position == "long":
            # Sell the option: get current market value
            cash_back = max(0.0, contract.current_value(current_price))
        else:
            # Buy back short: return reserved cash minus cost to close
            cost_to_close = abs(contract.current_value(current_price))
            cash_back     = contract.cash_reserved - cost_to_close

        logger.info(
            f"  CLOSE OPTION {contract.ticker} {contract.option_type} "
            f"${contract.strike} | P&L: ${pnl:+.2f}"
        )
        return True, max(0.0, cash_back)

    def check_expirations(
        self,
        current_prices: dict[str, float],
        portfolio,   # Portfolio instance for assignment handling
    ) -> list[str]:
        """
        Handle all expired options. Correctly updates portfolio cash and positions.
        Returns list of expired contract keys.
        """
        expired = []
        for key, contract in list(self.positions.items()):
            if not contract.is_expired:
                continue

            spot = current_prices.get(contract.ticker, 0.0)
            expired.append(key)

            if spot <= 0:
                logger.warning(f"  No price for expired option {contract.ticker} — treating as worthless")
                if contract.position == "long":
                    logger.info(f"  EXPIRED WORTHLESS (no price): {key} | Loss: ${contract.max_loss:.2f}")
                else:
                    # Return reserved cash
                    portfolio.cash += contract.cash_reserved
                    logger.info(f"  EXPIRED (no price): short {key} — reserved cash returned")
                continue

            if contract.position == "long":
                if contract.option_type == "call" and spot > contract.strike:
                    # ITM long call: exercise — buy shares at strike
                    shares      = contract.contracts * 100
                    total_cost  = contract.strike * shares
                    if portfolio.cash >= total_cost:
                        portfolio.cash -= total_cost
                        portfolio.buy(contract.ticker, shares, contract.strike,
                                      f"Exercised long call ${contract.strike}")
                        logger.info(
                            f"  EXERCISED: Long call {contract.ticker} ${contract.strike} "
                            f"— bought {shares} shares @ ${contract.strike}"
                        )
                    else:
                        # Cash settle: take intrinsic value in cash
                        intrinsic = (spot - contract.strike) * shares
                        portfolio.cash += intrinsic
                        logger.info(
                            f"  CASH SETTLED: Long call {contract.ticker} "
                            f"— intrinsic ${intrinsic:.2f} (insufficient cash to exercise)"
                        )
                elif contract.option_type == "put" and spot < contract.strike:
                    # ITM long put: exercise — sell shares at strike (or cash settle)
                    shares = contract.contracts * 100
                    if contract.ticker in portfolio.positions:
                        portfolio.sell(contract.ticker, shares, contract.strike,
                                       f"Exercised long put ${contract.strike}")
                        logger.info(
                            f"  EXERCISED: Long put {contract.ticker} ${contract.strike} "
                            f"— sold {shares} shares @ ${contract.strike}"
                        )
                    else:
                        # Cash settle
                        intrinsic = (contract.strike - spot) * shares
                        portfolio.cash += intrinsic
                        logger.info(
                            f"  CASH SETTLED: Long put {contract.ticker} "
                            f"— intrinsic ${intrinsic:.2f}"
                        )
                else:
                    # OTM — expires worthless
                    logger.info(
                        f"  EXPIRED WORTHLESS: Long {contract.option_type} "
                        f"{contract.ticker} ${contract.strike} | Loss: ${contract.max_loss:.2f}"
                    )

            elif contract.position == "short":
                if contract.option_type == "put" and spot < contract.strike:
                    # ITM short put — assigned: must buy shares at strike
                    shares     = contract.contracts * 100
                    total_cost = contract.strike * shares
                    # Cash was already reserved — deduct from reserved amount
                    portfolio.cash -= (total_cost - contract.cash_reserved)
                    portfolio.buy(contract.ticker, shares, contract.strike,
                                  f"Assigned short put ${contract.strike}")
                    logger.info(
                        f"  ASSIGNED: Short put {contract.ticker} ${contract.strike} "
                        f"— bought {shares} shares @ ${contract.strike}"
                    )
                else:
                    # OTM — expires worthless, keep premium, return reserved cash
                    portfolio.cash += contract.cash_reserved
                    logger.info(
                        f"  EXPIRED: Short {contract.option_type} {contract.ticker} "
                        f"${contract.strike} — premium kept, reserved cash returned"
                    )

        for key in expired:
            self.positions.pop(key, None)

        return expired

    @property
    def total_delta(self) -> float:
        return sum(
            c.greeks.get("delta", 0) * c.contracts * 100 *
            (1 if c.position == "long" else -1)
            for c in self.positions.values()
        )

    @property
    def total_theta(self) -> float:
        return sum(
            c.greeks.get("theta", 0) * c.contracts * 100 *
            (1 if c.position == "long" else -1)
            for c in self.positions.values()
        )

    @property
    def total_reserved_cash(self) -> float:
        return sum(c.cash_reserved for c in self.positions.values())

    def summary_lines(self) -> list[str]:
        if not self.positions:
            return []
        lines = [
            f"  {'─'*60}",
            f"  Options ({len(self.positions)} positions) | "
            f"Δ={self.total_delta:+.2f}  θ={self.total_theta:+.2f}/d  "
            f"Cash reserved: ${self.total_reserved_cash:,.0f}",
            f"  {'Contract':<32} {'DTE':>4}  {'Delta':>7}  {'Theta':>7}  {'P&L':>8}",
            f"  {'─'*60}",
        ]
        for key, c in self.positions.items():
            delta = c.greeks.get("delta", 0.0)
            theta = c.greeks.get("theta", 0.0)
            label = f"{c.position.upper()} {c.option_type.upper()} {c.ticker} ${c.strike}"
            lines.append(
                f"  {label:<32} {c.days_to_expiry:>4}d  "
                f"{delta:>+7.3f}  {theta:>+7.3f}/d  "
                f"{'n/a':>8}"
            )
        return lines
