"""
Autonomous broker — the main loop.

Runs continuously at a user-specified frequency, making buy/sell decisions
across penny stocks and regular stocks using technical + sentiment signals.

Usage:
    python Broker.py --cash 10000 --interval daily
    python Broker.py --cash 50000 --interval hourly --max_positions 30
    python Broker.py --status          # show portfolio without trading
    python Broker.py --trades          # show recent trade history
"""

import argparse
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import schedule
import torch

from broker.portfolio import Portfolio
from broker.brain     import BrokerBrain
from broker.journal   import log_cycle, print_report, print_recent_trades
from broker.universe  import refresh_universe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/broker.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

Path("logs").mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INTERVALS = {
    "hourly":  60,
    "2hour":   120,
    "4hour":   240,
    "daily":   1440,
    "weekly":  10080,
}


# ── Single trading cycle ──────────────────────────────────────────────────────

def run_cycle(portfolio: Portfolio, brain: BrokerBrain, top_n: int = 500):
    logger.info(f"{'─'*55}")
    logger.info(f"Trading cycle started | Equity: ${portfolio.equity:,.2f}")

    # ── Load feature data ─────────────────────────────────────────────────────
    try:
        from pipeline.data import load_master
        df = load_master(top_n=top_n, min_price=0.01, min_avg_volume=5_000)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # ── Run decision engine ───────────────────────────────────────────────────
    decisions = brain.run_cycle(df, screener_top_n=100)

    if not decisions:
        logger.info("No trades this cycle.")
    else:
        logger.info(f"{len(decisions)} decisions:")

    # ── Execute decisions ─────────────────────────────────────────────────────
    executed = []
    for d in decisions:
        if d.action == "BUY":
            ok = portfolio.buy(d.ticker, d.shares, d.price, d.reason)
        elif d.action == "SELL":
            ok = portfolio.sell_all(d.ticker, d.price, d.reason)
        elif d.action == "SELL_PARTIAL":
            ok = portfolio.sell(d.ticker, d.shares, d.price, d.reason)
        elif d.action == "OPEN_OPTION":
            contract = getattr(d, "_option_contract", None)
            if contract:
                ok, cash_delta = portfolio.options.open(contract, portfolio.cash)
                if ok:
                    portfolio.cash += cash_delta
            else:
                ok = False
        elif d.action == "CLOSE_OPTION":
            # Find matching option key
            matching = [
                k for k, c in portfolio.options.positions.items()
                if c.ticker == d.ticker
            ]
            ok = False
            for key in matching:
                success, cash_back = portfolio.options.close(key, d.price)
                if success:
                    portfolio.cash += cash_back
                    ok = True
        else:
            ok = True

        if ok:
            executed.append(d)

    portfolio.save()
    log_cycle(executed, portfolio.equity, portfolio.cash)

    logger.info(f"Cycle complete | Equity: ${portfolio.equity:,.2f} | "
                f"Cash: ${portfolio.cash:,.2f} | "
                f"Positions: {len(portfolio.positions)}")
    print(portfolio.summary())


# ── Universe refresh (weekly) ─────────────────────────────────────────────────

def run_universe_refresh():
    logger.info("Refreshing universe — discovering new stocks...")
    new = refresh_universe(max_new=200)
    if new:
        logger.info(f"Added {len(new)} new tickers: {new[:10]}...")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Autonomous Broker Agent")
    p.add_argument("--cash",          type=float, default=10_000,
                   help="Starting cash (only used on first run)")
    p.add_argument("--interval",      choices=list(INTERVALS.keys()), default="daily",
                   help="How often to trade (default: daily)")
    p.add_argument("--max_positions", type=int,   default=20,
                   help="Max simultaneous positions")
    p.add_argument("--stop_loss",     type=float, default=0.12,
                   help="Minimum stop-loss floor (default: 12%%, ATR-adjusted above this)")
    p.add_argument("--take_profit",   type=float, default=0.45,
                   help="Full take-profit threshold (default: 45%%, partial at 20%%)")
    p.add_argument("--min_score",     type=float, default=0.60,
                   help="Minimum buy signal score 0-1 (default: 0.60)")
    p.add_argument("--penny_pct",     type=float, default=0.20,
                   help="Max %% of portfolio in penny stocks (default: 20%%)")
    p.add_argument("--max_sector",    type=float, default=0.40,
                   help="Hard cap per sector (default: 40%%, broker self-adjusts below this)")
    p.add_argument("--avoid_earnings",type=int,   default=3,
                   help="Skip stocks within N days of earnings (default: 3, 0=disabled)")
    p.add_argument("--top_n",         type=int,   default=500,
                   help="Universe size for screening")
    p.add_argument("--no_options",    action="store_true",
                   help="Disable options trading (stocks only)")
    p.add_argument("--status",        action="store_true",
                   help="Show portfolio status and exit")
    p.add_argument("--trades",        action="store_true",
                   help="Show recent trades and exit")
    p.add_argument("--once",          action="store_true",
                   help="Run one cycle and exit (no loop)")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    portfolio = Portfolio(initial_cash=args.cash)
    brain     = BrokerBrain(
        portfolio           = portfolio,
        max_positions       = args.max_positions,
        stop_loss_pct_floor = args.stop_loss,
        full_profit_pct     = args.take_profit,
        min_score           = args.min_score,
        penny_max_pct       = args.penny_pct,
        max_sector_pct      = args.max_sector,
        avoid_earnings_days = args.avoid_earnings,
        device              = DEVICE,
    )

    # ── Status / trades only ──────────────────────────────────────────────────
    if args.status:
        print_report(portfolio)
        return

    if args.trades:
        print_recent_trades(n=30)
        return

    # ── Single run ────────────────────────────────────────────────────────────
    if args.once:
        run_cycle(portfolio, brain, top_n=args.top_n)
        return

    # ── Continuous loop ───────────────────────────────────────────────────────
    interval_mins = INTERVALS[args.interval]
    logger.info(f"Broker started | Interval: {args.interval} ({interval_mins}min) | "
                f"Device: {DEVICE}")
    logger.info(f"Settings: max_positions={args.max_positions} | "
                f"stop_loss_floor={args.stop_loss:.0%} (ATR-adjusted) | "
                f"take_profit={args.take_profit:.0%} (partial at 20%%) | "
                f"min_score={args.min_score} | penny_pct={args.penny_pct:.0%} | "
                f"max_sector={args.max_sector:.0%} | avoid_earnings={args.avoid_earnings}d")

    # Run immediately on start
    run_cycle(portfolio, brain, top_n=args.top_n)

    # Schedule recurring cycles
    schedule.every(interval_mins).minutes.do(
        run_cycle, portfolio=portfolio, brain=brain, top_n=args.top_n
    )

    # Discover new stocks every Sunday
    schedule.every().sunday.at("08:00").do(run_universe_refresh)

    logger.info(f"Next cycle in {interval_mins} minutes. Press Ctrl+C to stop.")

    try:
        while True:
            schedule.run_pending()
            time.sleep(30)
    except KeyboardInterrupt:
        portfolio.save()
        logger.info("Broker stopped. Portfolio saved.")
        print_report(portfolio)


if __name__ == "__main__":
    main()
