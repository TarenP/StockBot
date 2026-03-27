"""
Entry point for the autonomous broker.

Settings are loaded from broker.config (edit that file to set your defaults).
Any flag passed on the command line overrides the config file.

Usage:
    python Broker.py              # run a cycle with config defaults
    python Broker.py --status     # show portfolio + SPY benchmark
    python Broker.py --trades     # show recent trade history
"""

import sys
import argparse
from pathlib import Path


def _load_config(path: str = "broker.config") -> dict:
    """Parse broker.config into a dict of defaults."""
    defaults = {}
    config_path = Path(path)
    if not config_path.exists():
        return defaults
    for line in config_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, val = line.partition("=")
            key = key.strip().replace("-", "_")
            val = val.strip()
            # Type coercion
            if val.lower() == "true":
                defaults[key] = True
            elif val.lower() == "false":
                defaults[key] = False
            else:
                try:
                    defaults[key] = int(val)
                except ValueError:
                    try:
                        defaults[key] = float(val)
                    except ValueError:
                        defaults[key] = val
    return defaults


def main():
    # Load config file first, then let CLI args override
    config = _load_config()

    # Patch sys.argv defaults via argparse set_defaults
    from broker.broker import parse_args, run_cycle, Portfolio, BrokerBrain
    from broker.broker import PortfolioRiskEngine, validate_startup, DEVICE
    from broker.broker import print_report, print_recent_trades
    from broker.broker import _is_market_hours, _market_hours_warning
    from broker.journal import _fetch_current_spy_price, log_cycle, daily_integrity_check
    from broker.risk    import PortfolioRiskEngine, validate_startup

    import broker.broker as _bb
    args = parse_args(config)

    portfolio = Portfolio(initial_cash=args.cash)

    if args.status:
        print_report(portfolio)
        return

    if args.trades:
        print_recent_trades(n=30)
        return

    errors = validate_startup(portfolio)
    if errors:
        import logging
        logger = logging.getLogger(__name__)
        logger.error("Startup validation failed:")
        for e in errors:
            logger.error(f"  * {e}")
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
        top_n                = args.top_n,
        enforce_market_hours = not args.no_market_hours,
    )


if __name__ == "__main__":
    main()
