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

from broker.portfolio import CASH_YIELD_ANNUAL_RATE, Portfolio
from broker.brain     import BrokerBrain
from broker.risk      import PortfolioRiskEngine, validate_startup
from broker.paper_diagnostics import (
    CAP_IMPACT_SUMMARY_PATH,
    PARITY_REPORT_PATH,
    PERFORMANCE_ATTRIBUTION_PATH,
    build_replay_live_parity_report,
    summarize_cap_impact_history,
    summarize_performance_attribution,
    write_json,
)
from broker.journal   import (
    log_cycle, print_report, print_recent_trades,
    daily_integrity_check, _fetch_current_spy_price,
)
from pipeline.checkpoints import resolve_checkpoint_path
from pipeline.run_manifest import (
    get_code_version,
    hash_config,
    hash_ticker_list,
    summarize_price_sentiment_freshness,
    write_run_manifest,
)
from pipeline.universe_resolver import get_investable_universe_filters
from pipeline.universe_resolver import load_typed_config

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


def _today_et():
    return datetime.now(ET).date()


# ── Checkpoint resolver ───────────────────────────────────────────────────────

def _resolve_checkpoint(path: str | None) -> str | None:
    """
    Resolve 'auto' to the best available checkpoint in models/ by val_sharpe.
    Returns the path as-is for any other value.
    """
    return resolve_checkpoint_path(checkpoint_path=path, save_dir="models")


def _resolve_cycle_universe(
    config: dict | None = None,
    save_dir: str = "models",
    as_of_date=None,
) -> list[str]:
    from pipeline.updater import get_live_universe

    cfg = config if config is not None else load_typed_config()
    return get_live_universe(
        save_dir=save_dir,
        config=cfg,
        target_size=int(cfg.get("live_target_size", max(int(cfg.get("top_n", 500)) * 3, 1000))),
        as_of_date=as_of_date,
    )


# ── Market hours check ────────────────────────────────────────────────────────

def _is_market_hours() -> bool:
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    t = (now.hour, now.minute)
    return MARKET_OPEN <= t < MARKET_CLOSE


def _paper_execution_cost(
    price: float,
    shares: float,
    *,
    is_penny: bool = False,
    base_spread: float = 0.001,
) -> tuple[float, float, str]:
    """Estimate paper execution drag as an explicit cash cost."""
    price = max(float(price or 0.0), 0.0)
    shares = max(float(shares or 0.0), 0.0)
    if price <= 0 or shares <= 0:
        return 0.0, 0.0, "paper_spread:none"

    tiered_spread = 0.02 if is_penny else (0.005 if price < 20.0 else 0.001)
    spread = max(float(base_spread or 0.0), tiered_spread)
    return shares * price * spread, spread, f"paper_spread:{spread:.4f}"


# ── One-shot cycle ────────────────────────────────────────────────────────────

def run_cycle(
    portfolio: Portfolio,
    brain: BrokerBrain,
    risk: PortfolioRiskEngine,
    top_n: int = 500,
    enforce_market_hours: bool = True,
    maintenance_context: dict | None = None,
    config: dict | None = None,
    execution_spread: float | None = None,
):
    cfg = config or {}
    base_execution_spread = float(
        execution_spread if execution_spread is not None else cfg.get("execution_spread", 0.001)
    )
    run_manifest: dict[str, object] = {
        "mode": "live",
        "config_hash": hash_config(cfg),
        "code_version": get_code_version(),
        "snapshot_path": str(cfg.get("universe_snapshot_path", "")),
        "watchlist_included": bool(cfg.get("include_watchlist_in_universe", False)),
        "friction": {"execution_spread": base_execution_spread},
        "benchmark_status": {"symbol": "SPY", "available": None},  # updated after data load
    }
    now_et = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")
    logger.info(f"{'='*55}")
    logger.info(f"Broker cycle | {now_et} | Equity: ${portfolio.equity:,.2f}")
    logger.info(f"{'='*55}")
    today_iso = _today_et().isoformat()
    prices_fresh_from_maintenance = (
        maintenance_context is not None
        and maintenance_context.get("prices_updated") == today_iso
    )
    sentiment_fresh_from_maintenance = (
        maintenance_context is not None
        and maintenance_context.get("sentiment_updated") == today_iso
    )

    # ── Auto-update prices + sentiment ────────────────────────────────────────
    logger.info("Fetching latest market data and news...")
    try:
        from pipeline.updater import update_parquet
        if prices_fresh_from_maintenance:
            logger.info("  Prices: already refreshed during maintenance")
        else:
            universe = _resolve_cycle_universe(config=config, save_dir="models")
            n_prices = update_parquet(universe=universe, save_dir="models", config=config)
            if n_prices:
                logger.info(f"  Prices: {n_prices} new rows added")
            else:
                logger.info("  Prices: already up to date")
    except Exception as e:
        logger.warning(f"  Price update failed (continuing with cached data): {e}")

    try:
        from pipeline.sentiment import update_sentiment
        if sentiment_fresh_from_maintenance:
            logger.info("  Sentiment: already refreshed during maintenance")
            universe = []
        else:
            universe = _resolve_cycle_universe(config=config, save_dir="models")
        if not sentiment_fresh_from_maintenance and not universe:
            # No checkpoint yet — sentiment update will happen after first train
            logger.info("  Sentiment: skipping (no checkpoint yet — run --mode train first)")
        elif not sentiment_fresh_from_maintenance:
            n_sent = update_sentiment(universe, lookback_days=3, save_dir="models")
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

    # ── Fetch latest data ─────────────────────────────────────────────────────
    logger.info("Loading market data...")
    try:
        from pipeline.data import load_master
        investable_filters = get_investable_universe_filters(config)
        cycle_universe = _resolve_cycle_universe(config=config, save_dir="models")
        run_manifest["resolved_universe_size"] = len(cycle_universe)
        run_manifest["resolved_universe_hash"] = hash_ticker_list(cycle_universe)
        df = load_master(
            top_n=top_n,
            min_history_days=int(investable_filters["min_history_days"]),
            min_price=float(investable_filters["min_price"]),
            min_avg_volume=float(investable_filters["min_avg_volume"]),
            include_raw_cols=True,
            use_snapshot_fundamentals=True,
            config=config,
        )
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        brain.min_score = brain._base_min_score
        return
    freshness = summarize_price_sentiment_freshness(df, positions=portfolio.positions)
    run_manifest["freshness"] = freshness
    cfg = config or {}
    min_price_coverage = float(cfg.get("min_fresh_price_coverage", 0.90))
    min_sentiment_coverage = float(cfg.get("min_fresh_sentiment_coverage", 0.50))
    freshness_gate_failed = (
        freshness["fresh_price_coverage"] < min_price_coverage
        or freshness["stale_holdings_count"] > 0
        or (
            "sent_net" in df.columns
            and freshness["fresh_sentiment_coverage"] < min_sentiment_coverage
        )
    )
    run_manifest["freshness_gate"] = {
        "passed": not freshness_gate_failed,
        "min_fresh_price_coverage": min_price_coverage,
        "min_fresh_sentiment_coverage": min_sentiment_coverage,
    }
    if freshness_gate_failed:
        logger.warning(
            "Freshness gate triggered: price_coverage=%.2f sentiment_coverage=%.2f stale_holdings=%d. "
            "Blocking new entries for this cycle.",
            float(freshness["fresh_price_coverage"]),
            float(freshness["fresh_sentiment_coverage"]),
            int(freshness["stale_holdings_count"]),
        )
        brain.min_score = 999.0
    if hasattr(risk, "set_market_regime"):
        risk.set_market_regime(brain._current_market_regime(df))
    risk.start_session(portfolio.equity)
    health_status, health_reason = risk.check_portfolio_health(portfolio)

    if health_status == "halt":
        logger.warning(f"RISK HALT: {health_reason}")
        brain.min_score = 999.0
    elif health_status == "warning":
        logger.warning(f"RISK WARNING: {health_reason}")

    # ── Run decisions ─────────────────────────────────────────────────────────
    decisions = brain.run_cycle(df, screener_top_n=50, risk_engine=risk)

    if not decisions:
        logger.info("No trades this cycle.")

    # ── Execute ───────────────────────────────────────────────────────────────
    executed = []
    for d in decisions:
        if d.action == "BUY":
            is_penny  = d.price < 5.0
            execution_cost, _spread, execution_model = _paper_execution_cost(
                d.price,
                d.shares,
                is_penny=is_penny,
                base_spread=base_execution_spread,
            )
            ok = portfolio.buy(
                d.ticker,
                d.shares,
                d.price,
                d.reason,
                execution_cost=execution_cost,
                decision_price=d.price,
                execution_model=execution_model,
            )
            # Store rl_score_at_entry and rl_rank_pct_at_entry in position metadata.
            # rank_pct is the rank percentile within the shortlist at entry time —
            # used by _rl_exit_checks() for stable threshold comparisons.
            if ok and brain.rl_enabled and d.ticker in portfolio.positions:
                portfolio.positions[d.ticker]["rl_score_at_entry"] = d.score
                # Compute rank percentile from the brain's last rl_scores if available
                # (stored as a cycle-level attribute set during run_cycle)
                last_rl_scores = getattr(brain, "_last_rl_scores", None)
                if last_rl_scores is not None and not last_rl_scores.empty:
                    rank_pct = float(last_rl_scores.get(d.ticker, d.score))
                else:
                    rank_pct = float(d.score)  # fallback: use raw score as proxy
                portfolio.positions[d.ticker]["rl_rank_pct_at_entry"] = rank_pct

        elif d.action == "SELL":
            shares_to_sell = float(
                portfolio.positions.get(d.ticker, {}).get("shares", d.shares)
            )
            execution_cost, _spread, execution_model = _paper_execution_cost(
                d.price,
                shares_to_sell,
                is_penny=d.price < 5.0,
                base_spread=base_execution_spread,
            )
            ok = portfolio.sell_all(
                d.ticker,
                d.price,
                d.reason,
                execution_cost=execution_cost,
                decision_price=d.price,
                execution_model=execution_model,
            )

        elif d.action == "SELL_PARTIAL":
            shares_to_sell = min(
                float(d.shares),
                float(portfolio.positions.get(d.ticker, {}).get("shares", d.shares)),
            )
            execution_cost, _spread, execution_model = _paper_execution_cost(
                d.price,
                shares_to_sell,
                is_penny=d.price < 5.0,
                base_spread=base_execution_spread,
            )
            ok = portfolio.sell(
                d.ticker,
                d.shares,
                d.price,
                d.reason,
                execution_cost=execution_cost,
                decision_price=d.price,
                execution_model=execution_model,
            )
            if ok and d.ticker in portfolio.positions:
                portfolio.positions[d.ticker]["partial_taken"] = True

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

    try:
        portfolio._last_mark_to_market = portfolio.refresh_latest_holding_prices()
        mark = portfolio._last_mark_to_market or {}
        logger.info(
            "Marked holdings before reporting: %d/%d updated (latest=%s)",
            len(mark.get("updated") or {}),
            len(portfolio.positions),
            mark.get("latest_date") or "unknown",
        )
    except Exception as exc:
        logger.warning("Could not refresh holding prices before reporting: %s", exc)

    # ── SPY tracking + save ───────────────────────────────────────────────────
    spy_price = _fetch_current_spy_price()
    allocation_summary = getattr(brain, "_last_allocation_summary", {}) or {}
    try:
        snapshot = portfolio.record_snapshot(
            allocation_summary=allocation_summary,
            spy_price=spy_price,
            extra={
                "executed_count": len(executed),
                "config_hash": run_manifest["config_hash"],
            },
        )
        run_manifest["portfolio_snapshot"] = {
            "path": "broker/state/portfolio_history.jsonl",
            "equity": snapshot.get("equity"),
            "top_1_concentration": snapshot.get("top_1_concentration"),
            "top_3_concentration": snapshot.get("top_3_concentration"),
        }
        cap_summary = summarize_cap_impact_history()
        write_json(CAP_IMPACT_SUMMARY_PATH, cap_summary)
        attribution = summarize_performance_attribution(portfolio)
        write_json(PERFORMANCE_ATTRIBUTION_PATH, attribution)
        parity_report = build_replay_live_parity_report(cfg, brain=brain)
        write_json(PARITY_REPORT_PATH, parity_report)
        run_manifest["diagnostics"] = {
            "cap_impact_summary_path": str(CAP_IMPACT_SUMMARY_PATH),
            "performance_attribution_path": str(PERFORMANCE_ATTRIBUTION_PATH),
            "replay_live_parity_path": str(PARITY_REPORT_PATH),
            "replay_live_parity_compatible": parity_report.get("compatible"),
        }
    except Exception as exc:
        logger.warning("Could not write paper diagnostics: %s", exc)
    portfolio.save()
    log_cycle(executed, portfolio.equity, portfolio.cash, spy_price=spy_price)

    # ── Summary ───────────────────────────────────────────────────────────────
    report = daily_integrity_check(portfolio)
    if "beating_spy" in report:
        status = "beating SPY" if report["beating_spy"] else "trailing SPY"
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

    # Update live performance chart (silent)
    try:
        from broker.journal import plot_live_performance
        plot_live_performance("plots/live_performance.png")
    except Exception:
        pass

    try:
        manifest_path = write_run_manifest(
            "live_cycle",
            run_manifest,
            output_path=Path("broker/state") / "last_live_manifest.json",
        )
        logger.info("Live run manifest saved -> %s", manifest_path)
    except Exception as exc:
        logger.warning("Could not save live run manifest: %s", exc)

    # ── Daily briefing — the primary human-facing output ──────────────────────
    from broker.briefing import print_daily_briefing
    print_daily_briefing(decisions, portfolio, executed)

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
    p.add_argument("--max_position_pct", type=float, default=cfg.get("max_position_pct", 0.10))
    p.add_argument("--stop_loss",      type=float, default=cfg.get("stop_loss",      0.08))
    p.add_argument("--take_profit",    type=float, default=cfg.get("take_profit",    1.00))
    p.add_argument("--partial_profit", type=float, default=cfg.get("partial_profit", 0.35))
    p.add_argument("--trailing_stop",  type=float, default=cfg.get("trailing_stop",  0.12))
    p.add_argument("--trailing_activation", type=float, default=cfg.get("trailing_activation", 0.18))
    p.add_argument("--signal_exit_score", type=float, default=cfg.get("signal_exit_score", 0.18))
    p.add_argument("--signal_exit_grace", type=int, default=cfg.get("signal_exit_grace", 2))
    p.add_argument("--min_score",      type=float, default=cfg.get("min_score",      0.58))
    p.add_argument("--penny_pct",      type=float, default=cfg.get("penny_pct",      0.03))
    p.add_argument("--max_sector",     type=float, default=cfg.get("max_sector",     0.25))
    p.add_argument("--max_correlation",type=float, default=cfg.get("max_correlation", 0.80))
    p.add_argument("--theme_max_pct",  type=float, default=cfg.get("theme_max_pct",  1.00))
    p.add_argument("--low_price_max_pct", type=float, default=cfg.get("low_price_max_pct", 1.00))
    p.add_argument("--sentiment_policy", type=str, default=cfg.get("sentiment_policy", "informational"))
    p.add_argument("--sentiment_negative_weight_mult", type=float, default=cfg.get("sentiment_negative_weight_mult", 0.80))
    p.add_argument("--sentiment_veto_composite_floor", type=float, default=cfg.get("sentiment_veto_composite_floor", 0.50))
    p.add_argument("--avoid_earnings", type=int,   default=cfg.get("avoid_earnings", 5))
    p.add_argument("--top_n",          type=int,   default=cfg.get("top_n",          500))
    p.add_argument("--max_daily_loss", type=float, default=cfg.get("max_daily_loss", 0.025))
    p.add_argument("--max_drawdown",   type=float, default=cfg.get("max_drawdown",   0.12))
    p.add_argument("--max_gross_exposure", type=float, default=cfg.get("max_gross_exposure", 0.95))
    p.add_argument("--cash_floor",     type=float, default=cfg.get("cash_floor",     0.05))
    p.add_argument("--target_volatility", type=float, default=cfg.get("target_volatility", 0.15))
    p.add_argument("--execution_spread", type=float, default=cfg.get("execution_spread", 0.001))
    p.add_argument("--vol_lookback",   type=int,   default=cfg.get("vol_lookback",   20))
    p.add_argument("--no_options",        action="store_true", default=cfg.get("no_options", True))
    p.add_argument("--no_market_hours",   action="store_true", default=False)
    p.add_argument("--status",            action="store_true")
    p.add_argument("--trades",            action="store_true")
    p.add_argument("--rl_enabled",        action="store_true", default=cfg.get("rl_enabled", False))
    p.add_argument("--rl_checkpoint_path",type=str,   default=cfg.get("rl_checkpoint_path", "models/best_fold9.pt"))
    p.add_argument("--rl_phase",          type=int,   default=cfg.get("rl_phase",          1))
    p.add_argument("--rl_exit_threshold", type=float, default=cfg.get("rl_exit_threshold", 0.30))
    p.add_argument("--rl_conviction_drop",type=float, default=cfg.get("rl_conviction_drop", 0.20))
    p.add_argument("--rl_min_score",      type=float, default=cfg.get("rl_min_score",      0.0))
    p.add_argument("--dead_money_days",   type=int,   default=cfg.get("dead_money_days",   0))
    p.add_argument("--dead_money_min_return", type=float, default=cfg.get("dead_money_min_return", 0.02))
    return p.parse_args()


# ── Entry point ───────────────────────────────────────────────────────────────

def main(config: dict = None, maintenance_context: dict | None = None):
    args = parse_args(config)

    portfolio = Portfolio(initial_cash=args.cash)
    # Status / trades — no trading, just reporting
    if args.status:
        portfolio._last_mark_to_market = portfolio.refresh_latest_holding_prices()
        portfolio._last_dividend_accrual = portfolio.accrue_dividends(_today_et())
        portfolio.save()
        spy_price = _fetch_current_spy_price()
        portfolio.record_snapshot(
            spy_price=spy_price,
            extra={"status_refresh": True},
        )
        try:
            attribution = summarize_performance_attribution(portfolio)
            write_json(PERFORMANCE_ATTRIBUTION_PATH, attribution)
        except Exception as exc:
            logger.warning("Could not update status attribution: %s", exc)
        log_cycle([], portfolio.equity, portfolio.cash, spy_price=spy_price)
        print_report(portfolio)
        return

    if args.trades:
        print_recent_trades(n=30)
        return

    today = _today_et()
    cash_yield = portfolio.accrue_cash_yield(today)
    portfolio._last_dividend_accrual = portfolio.accrue_dividends(today)
    dividend_cash = float(portfolio._last_dividend_accrual.get("total", 0.0) or 0.0)
    if cash_yield > 0:
        logger.info(
            "Accrued cash yield: $%.2f at %.2f%% annualized",
            cash_yield,
            CASH_YIELD_ANNUAL_RATE * 100,
        )
    if dividend_cash > 0:
        logger.info("Credited dividends: $%.2f", dividend_cash)
    if cash_yield > 0 or portfolio._last_dividend_accrual.get("state_changed"):
        portfolio.save()

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
        max_position_pct    = args.max_position_pct,
        stop_loss_pct_floor = args.stop_loss,
        partial_profit_pct  = args.partial_profit,
        full_profit_pct     = args.take_profit,
        trailing_stop_pct   = args.trailing_stop,
        trailing_activation_pct = args.trailing_activation,
        signal_exit_score   = args.signal_exit_score,
        signal_exit_grace_cycles = args.signal_exit_grace,
        min_score           = args.min_score,
        penny_max_pct       = args.penny_pct,
        max_sector_pct      = args.max_sector,
        max_pair_correlation = args.max_correlation,
        theme_max_pct       = args.theme_max_pct,
        low_price_max_pct   = args.low_price_max_pct,
        sentiment_policy    = args.sentiment_policy,
        sentiment_negative_weight_mult = args.sentiment_negative_weight_mult,
        sentiment_veto_composite_floor = args.sentiment_veto_composite_floor,
        avoid_earnings_days = args.avoid_earnings,
        device              = DEVICE,
        rl_enabled          = args.rl_enabled,
        rl_checkpoint_path  = _resolve_checkpoint(args.rl_checkpoint_path),
        rl_phase            = args.rl_phase,
        rl_exit_threshold   = args.rl_exit_threshold,
        rl_conviction_drop  = args.rl_conviction_drop,
        rl_min_score        = args.rl_min_score,
        dead_money_days     = int(getattr(args, "dead_money_days", 0)),
        dead_money_min_return = float(getattr(args, "dead_money_min_return", 0.02)),
    )
    brain._base_min_score = args.min_score

    risk = PortfolioRiskEngine(
        max_daily_loss     = args.max_daily_loss,
        max_drawdown       = args.max_drawdown,
        max_gross_exposure = args.max_gross_exposure,
        cash_floor         = args.cash_floor,
        target_volatility  = args.target_volatility,
        vol_lookback       = args.vol_lookback,
    )

    run_cycle(
        portfolio,
        brain,
        risk,
        top_n             = args.top_n,
        enforce_market_hours = not args.no_market_hours,
        maintenance_context = maintenance_context,
        config = config,
        execution_spread = args.execution_spread,
    )


if __name__ == "__main__":
    main()
