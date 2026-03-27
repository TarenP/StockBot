"""
Autonomous broker — one-shot mode.

Run this script whenever you want the broker to update.
It does one full cycle and exits:
  1. Validates data freshness and portfolio state
  2. Fetches latest prices + news
  3. Checks exits on all held positions
  4. Screens for new opportunities
  5. Executes buy/sell/options decisions
  6. Logs results and SPY benchmark
  7. Exits

Usage:
    python Broker.py                        # run a cycle with defaults
    python Broker.py --cash 10000           # set starting cash (first run only)
    python Broker.py --status               # show portfolio + SPY benchmark, no trading
    python Broker.py --trades               # show recent trade history, no trading
    python Broker.py --no_options           # stocks only

Schedule it however you like:
    Windows Task Scheduler  → run daily at 4:30pm ET
    Cron (Mac/Linux)        → 30 16 * * 1-5 python /path/to/Broker.py
    Manually                → just run it when you want
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch

# Create required directories before anything else
Path("logs").mkdir(exist_ok=True)
Path("plots").mkdir(exist_ok=True)
Path("broker/state").mkdir(parents=True, exist_ok=True)

from broker.portfolio import Portfolio
from broker.brain     import BrokerBrain
from broker.risk      import PortfolioRiskEngine, validate_startup
from broker.journal   import (
    log_cycle, print_report, print_recent_trades,
    daily_integrity_check, _fetch_current_spy_price,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/broker.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

Path("logs").mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ET     = ZoneInfo("America/New_York")

MARKET_OPEN  = (9, 30)
MARKET_CLOSE = (16, 0)


# ── Market hours check ────────────────────────────────────────────────────────

def _is_market_hours() -> bool:
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    t = (now.hour, now.minute)
    return MARKET_OPEN <= t < MARKET_CLOSE


# ── One-shot cycle ────────────────────────────────────────────────────────────

def run_cycle(
    portfolio: Portfolio,
    brain: BrokerBrain,
    risk: PortfolioRiskEngine,
    top_n: int = 500,
    enforce_market_hours: bool = True,
):
    now_et = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")
    logger.info(f"{'='*55}")
    logger.info(f"Broker cycle | {now_et} | Equity: ${portfolio.equity:,.2f}")
    logger.info(f"{'='*55}")

    # ── Auto-update prices + sentiment ────────────────────────────────────────
    logger.info("Fetching latest market data and news...")
    try:
        from pipeline.updater import update_parquet, _load_trained_universe
        # Use checkpoint universe if available, otherwise use top_n from load_master
        universe = _load_trained_universe("models")
        n_prices = update_parquet(universe=universe, save_dir="models")
        if n_prices:
            logger.info(f"  Prices: {n_prices} new rows added")
        else:
            logger.info("  Prices: already up to date")
    except Exception as e:
        logger.warning(f"  Price update failed (continuing with cached data): {e}")

    try:
        from pipeline.updater import _load_trained_universe
        from pipeline.sentiment import update_sentiment
        universe = _load_trained_universe("models")
        if not universe:
            # No checkpoint yet — sentiment update will happen after first train
            logger.info("  Sentiment: skipping (no checkpoint yet — run --mode train first)")
        else:
            n_sent = update_sentiment(universe, lookback_days=3)
            if n_sent:
                logger.info(f"  Sentiment: {n_sent} new headlines scored")
            else:
                logger.info("  Sentiment: already up to date")
    except Exception as e:
        logger.warning(f"  Sentiment update failed (continuing with cached data): {e}")

    # ── Market hours ──────────────────────────────────────────────────────────
    in_market = _is_market_hours()
    if enforce_market_hours and not in_market:
        now = datetime.now(ET)
        logger.warning(
            f"Outside market hours ({now.strftime('%H:%M ET, %A')}). "
            f"Price updates and exits will run, but no new entries."
        )
        brain.min_score = 999.0
    else:
        brain.min_score = brain._base_min_score

    # ── Risk engine ───────────────────────────────────────────────────────────
    risk.start_session(portfolio.equity)
    health_status, health_reason = risk.check_portfolio_health(portfolio)

    if health_status == "halt":
        logger.warning(f"RISK HALT: {health_reason}")
        brain.min_score = 999.0
    elif health_status == "warning":
        logger.warning(f"RISK WARNING: {health_reason}")

    # ── Fetch latest data ─────────────────────────────────────────────────────
    logger.info("Loading market data...")
    try:
        from pipeline.data import load_master
        df = load_master(top_n=top_n, min_price=0.01, min_avg_volume=5_000)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        brain.min_score = brain._base_min_score
        return

    # ── Run decisions ─────────────────────────────────────────────────────────
    decisions = brain.run_cycle(df, screener_top_n=100, risk_engine=risk)

    if not decisions:
        logger.info("No trades this cycle.")

    # ── Execute ───────────────────────────────────────────────────────────────
    executed = []
    for d in decisions:
        if d.action == "BUY":
            is_penny  = d.price < 5.0
            adj_value = risk.apply_execution_cost(d.shares * d.price, d.price, is_penny)
            adj_shares = adj_value / d.price if d.price > 0 else 0
            ok = portfolio.buy(d.ticker, adj_shares, d.price, d.reason)

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
            matching = [k for k, c in portfolio.options.positions.items()
                        if c.ticker == d.ticker]
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

    # ── SPY tracking + save ───────────────────────────────────────────────────
    spy_price = _fetch_current_spy_price()
    portfolio.save()
    log_cycle(executed, portfolio.equity, portfolio.cash, spy_price=spy_price)

    # ── Summary ───────────────────────────────────────────────────────────────
    report = daily_integrity_check(portfolio)
    if "beating_spy" in report:
        status = "✓ beating SPY" if report["beating_spy"] else "✗ trailing SPY"
        logger.info(
            f"Done | Equity: ${portfolio.equity:,.2f} | "
            f"Return: {portfolio.total_return:+.2%} | {status} "
            f"(alpha: {report.get('alpha_vs_spy', 0):+.2%})"
        )
    else:
        logger.info(
            f"Done | Equity: ${portfolio.equity:,.2f} | "
            f"Positions: {len(portfolio.positions)} stocks, "
            f"{len(portfolio.options.positions)} options"
        )

    print(portfolio.summary())

    # ── Auto-show full status + cycle summary ─────────────────────────────────
    print_report(portfolio, show_benchmark=True)

    # Update live performance chart
    from broker.journal import plot_live_performance
    plot_live_performance("plots/live_performance.png")

    # Show only what changed this cycle
    if executed:
        print(f"\n  {'─'*55}")
        print(f"  This cycle — {len(executed)} trade(s):")
        print(f"  {'─'*55}")
        for d in executed:
            if d.action == "BUY":
                print(f"  BOUGHT  {d.ticker:<8} {d.shares:.2f} shares @ ${d.price:.2f}")
            elif d.action in ("SELL", "SELL_PARTIAL"):
                label = "SOLD   " if d.action == "SELL" else "SOLD 50%"
                print(f"  {label} {d.ticker:<8} {d.shares:.2f} shares @ ${d.price:.2f}  ({d.reason.split('|')[0].strip()})")
            elif d.action == "OPEN_OPTION":
                print(f"  OPTION  {d.ticker:<8} {d.reason[:60]}")
            elif d.action == "CLOSE_OPTION":
                print(f"  CLOSED  {d.ticker:<8} option — {d.reason[:50]}")
        print(f"  {'─'*55}\n")
    else:
        print(f"\n  No trades this cycle.\n")

    brain.min_score = brain._base_min_score


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(config: dict = None):
    p = argparse.ArgumentParser(
        description="Autonomous broker — run once, exits when done.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    cfg = config or {}
    p.add_argument("--cash",           type=float, default=cfg.get("cash",           10_000))
    p.add_argument("--max_positions",  type=int,   default=cfg.get("max_positions",  10))
    p.add_argument("--stop_loss",      type=float, default=cfg.get("stop_loss",      0.08))
    p.add_argument("--take_profit",    type=float, default=cfg.get("take_profit",    0.35))
    p.add_argument("--partial_profit", type=float, default=cfg.get("partial_profit", 0.15))
    p.add_argument("--min_score",      type=float, default=cfg.get("min_score",      0.58))
    p.add_argument("--penny_pct",      type=float, default=cfg.get("penny_pct",      0.03))
    p.add_argument("--max_sector",     type=float, default=cfg.get("max_sector",     0.25))
    p.add_argument("--avoid_earnings", type=int,   default=cfg.get("avoid_earnings", 5))
    p.add_argument("--top_n",          type=int,   default=cfg.get("top_n",          500))
    p.add_argument("--max_daily_loss", type=float, default=cfg.get("max_daily_loss", 0.025))
    p.add_argument("--max_drawdown",   type=float, default=cfg.get("max_drawdown",   0.12))
    p.add_argument("--no_options",     action="store_true", default=cfg.get("no_options", True))
    p.add_argument("--no_market_hours",action="store_true", default=False)
    p.add_argument("--status",         action="store_true")
    p.add_argument("--trades",         action="store_true")
    return p.parse_args()


# ── Entry point ───────────────────────────────────────────────────────────────

def main(config: dict = None):
    args = parse_args(config)

    portfolio = Portfolio(initial_cash=args.cash)

    # Status / trades — no trading, just reporting
    if args.status:
        print_report(portfolio)
        return

    if args.trades:
        print_recent_trades(n=30)
        return

    # Startup validation before any trading
    errors = validate_startup(portfolio)
    if errors:
        logger.error("Startup validation failed — fix these before trading:")
        for e in errors:
            logger.error(f"  • {e}")
        return

    brain = BrokerBrain(
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
    brain._base_min_score = args.min_score

    risk = PortfolioRiskEngine(
        max_daily_loss = args.max_daily_loss,
        max_drawdown   = args.max_drawdown,
    )

    run_cycle(
        portfolio,
        brain,
        risk,
        top_n             = args.top_n,
        enforce_market_hours = not args.no_market_hours,
    )


if __name__ == "__main__":
    main()
