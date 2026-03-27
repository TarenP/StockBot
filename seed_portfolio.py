"""
Seed the portfolio with initial positions at today's last prices.

Usage:
    python seed_portfolio.py

Edit the POSITIONS dict below with your desired holdings.
Format: "TICKER": dollar_amount_to_invest

Run once, then delete or ignore this file.
The broker will manage everything from here.
"""

import json
import sys
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import yfinance as yf

# ── Configure your initial positions here ────────────────────────────────────
# Format: "TICKER": dollar amount to invest
# Leave CASH as whatever is left over from your starting balance

STARTING_CASH = 10_000.00   # must match broker.config cash value

POSITIONS = {
    "AMZN": 2500,
    "NVDA": 2500,
    "LLY": 2500
}

# ─────────────────────────────────────────────────────────────────────────────


@contextmanager
def _quiet():
    with open(os.devnull, "w") as dn:
        old = sys.stderr; sys.stderr = dn
        try: yield
        finally: sys.stderr = old


def fetch_price(ticker: str) -> float | None:
    try:
        with _quiet():
            raw = yf.download(ticker, period="5d", auto_adjust=True, progress=False)
        if not raw.empty:
            return float(raw["Close"].iloc[-1].iloc[0] if hasattr(raw["Close"].iloc[-1], 'iloc') else raw["Close"].iloc[-1])
    except Exception:
        pass
    return None


def seed():
    state_path = Path("broker/state/portfolio.json")
    state_path.parent.mkdir(parents=True, exist_ok=True)

    if state_path.exists():
        existing = json.loads(state_path.read_text())
        if existing.get("positions"):
            print(f"Portfolio already has {len(existing['positions'])} positions.")
            confirm = input("Overwrite? (yes/no): ").strip().lower()
            if confirm != "yes":
                print("Aborted.")
                return

    positions  = {}
    trade_log  = []
    cash_spent = 0.0

    print(f"\nStarting cash: ${STARTING_CASH:,.2f}")
    print(f"Seeding {len(POSITIONS)} positions...\n")

    for ticker, alloc in POSITIONS.items():
        ticker = ticker.upper()
        price  = fetch_price(ticker)
        if price is None:
            print(f"  {ticker:<8} — could not fetch price, skipping")
            continue

        shares = alloc / price
        cost   = shares * price

        if cash_spent + cost > STARTING_CASH:
            print(f"  {ticker:<8} — insufficient cash, skipping")
            continue

        positions[ticker] = {
            "shares":        round(shares, 4),
            "avg_cost":      round(price, 4),
            "last_price":    round(price, 4),
            "partial_taken": False,
        }
        trade_log.append({
            "time":   datetime.now().isoformat(),
            "action": "BUY",
            "ticker": ticker,
            "shares": round(shares, 4),
            "price":  round(price, 4),
            "value":  round(cost, 2),
            "reason": "Initial portfolio seed",
            "equity": round(STARTING_CASH, 2),
        })
        cash_spent += cost
        print(f"  {ticker:<8}  {shares:.4f} shares @ ${price:.2f}  = ${cost:.2f}")

    cash_remaining = STARTING_CASH - cash_spent
    equity         = cash_remaining + sum(
        p["shares"] * p["last_price"] for p in positions.values()
    )

    state = {
        "cash":         round(cash_remaining, 2),
        "positions":    positions,
        "trade_log":    trade_log,
        "initial_cash": STARTING_CASH,
        "last_saved":   datetime.now().isoformat(),
    }

    state_path.write_text(json.dumps(state, indent=2))

    print(f"\nDone.")
    print(f"  Positions: {len(positions)}")
    print(f"  Invested:  ${cash_spent:,.2f}")
    print(f"  Cash left: ${cash_remaining:,.2f}")
    print(f"  Equity:    ${equity:,.2f}")
    print(f"\nPortfolio saved to {state_path}")
    print("Run 'python Broker.py' to start managing it.")


if __name__ == "__main__":
    seed()
