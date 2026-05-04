import json
from datetime import date
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from broker.broker import _paper_execution_cost
from broker.paper_diagnostics import (
    build_replay_live_parity_report,
    summarize_price_sanity,
    summarize_cap_impact_history,
    summarize_low_price_signal_suppression,
    summarize_performance_attribution,
    summarize_redeployment_quality,
)
from broker.portfolio import Portfolio


class _EmptyOptions:
    positions = {}
    total_reserved_cash = 0.0

    @staticmethod
    def summary_lines() -> list[str]:
        return []


def _portfolio() -> Portfolio:
    portfolio = Portfolio.__new__(Portfolio)
    portfolio.initial_cash = 10_000.0
    portfolio.cash = 10_000.0
    portfolio.positions = {}
    portfolio.trade_log = []
    portfolio.cash_yield_last_date = None
    portfolio.dividend_last_ex_date = {}
    portfolio.dividend_cash_total = 0.0
    portfolio.options = _EmptyOptions()
    return portfolio


def test_portfolio_buy_records_execution_cost_and_snapshot():
    portfolio = _portfolio()

    ok = portfolio.buy(
        "AAA",
        shares=10.0,
        price=100.0,
        reason="test",
        execution_cost=5.0,
        decision_price=99.5,
        execution_model="paper_spread:0.0050",
    )

    assert ok is True
    assert np.isclose(portfolio.cash, 8_995.0)
    assert np.isclose(portfolio.positions["AAA"]["avg_cost"], 100.5)
    assert np.isclose(portfolio.equity, 9_995.0)
    assert portfolio.trade_log[-1]["execution_cost"] == 5.0
    assert portfolio.trade_log[-1]["decision_price"] == 99.5

    path = Path("tests/_tmp") / f"portfolio_history_{uuid4().hex}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        snapshot = portfolio.record_snapshot(
            allocation_summary={
                "cap_impact": {"theme_cap_impact": 0.05},
                "cap_intervention_counts": {"theme_cap_impact": 1},
            },
            path=path,
        )

        assert snapshot["top_1_concentration"] > 0.0
        assert snapshot["total_execution_cost"] == 5.0
        assert path.exists()
        saved = json.loads(path.read_text().strip())
        assert saved["allocation_summary"]["cap_impact"]["theme_cap_impact"] == 0.05
    finally:
        path.unlink(missing_ok=True)


def test_portfolio_status_marks_to_latest_cached_close():
    portfolio = _portfolio()
    portfolio.positions = {
        "AAA": {"shares": 2.0, "avg_cost": 10.0, "last_price": 10.0},
        "BBB": {"shares": 1.0, "avg_cost": 20.0, "last_price": 20.0},
    }
    price_path = Path("tests/_tmp") / f"price_cache_{uuid4().hex}.parquet"
    price_path.parent.mkdir(parents=True, exist_ok=True)
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02"), "AAA"),
            (pd.Timestamp("2024-01-03"), "AAA"),
            (pd.Timestamp("2024-01-02"), "BBB"),
        ],
        names=["date", "ticker"],
    )
    prices = pd.DataFrame({"close": [11.0, 12.5, 19.0], "volume": [1, 1, 1]}, index=idx)

    try:
        prices.to_parquet(price_path)
        mark = portfolio.mark_to_latest_cached_prices(price_path)

        assert mark["latest_date"] == "2024-01-03"
        assert mark["updated"] == {"AAA": 12.5, "BBB": 19.0}
        assert portfolio.positions["AAA"]["last_price"] == 12.5
        assert portfolio.positions["BBB"]["last_price"] == 19.0
    finally:
        price_path.unlink(missing_ok=True)


def test_portfolio_status_refreshes_live_prices_with_cache_fallback(monkeypatch):
    import broker.analyst as analyst_module
    import broker.portfolio as portfolio_module
    import broker.validator as validator_module

    portfolio = _portfolio()
    portfolio.positions = {
        "AAA": {"shares": 2.0, "avg_cost": 10.0, "last_price": 10.0},
        "BBB": {"shares": 1.0, "avg_cost": 20.0, "last_price": 20.0},
    }
    price_path = Path("tests/_tmp") / f"price_cache_{uuid4().hex}.parquet"
    price_path.parent.mkdir(parents=True, exist_ok=True)
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-03"), "BBB")],
        names=["date", "ticker"],
    )
    pd.DataFrame({"close": [19.0], "volume": [1]}, index=idx).to_parquet(price_path)

    def fake_quote(ticker: str):
        if ticker == "AAA":
            return {"price": 11.5, "date": "2024-01-04", "source": "intraday", "error": None}
        return {"price": None, "date": None, "source": None, "error": None}

    def fake_fetch(ticker: str, days: int = 90):
        return None

    try:
        monkeypatch.setattr(portfolio_module, "fetch_latest_market_price", fake_quote)
        monkeypatch.setattr(analyst_module, "fetch_ticker_data", fake_fetch)
        monkeypatch.setattr(
            validator_module,
            "validate_portfolio_prices",
            lambda positions, prices: prices,
        )

        mark = portfolio.refresh_latest_holding_prices(price_path)

        assert mark["latest_date"] == "2024-01-04"
        assert mark["updated"] == {"AAA": 11.5, "BBB": 19.0}
        assert mark["sources"] == {"AAA": "live", "BBB": "cache"}
        assert mark["missing"] == []
        assert portfolio.positions["AAA"]["last_price"] == 11.5
        assert portfolio.positions["BBB"]["last_price"] == 19.0
    finally:
        price_path.unlink(missing_ok=True)


def test_portfolio_accrues_dividend_cash_once():
    portfolio = _portfolio()
    portfolio.cash = 100.0
    portfolio.positions = {
        "AAA": {"shares": 10.0, "avg_cost": 10.0, "last_price": 10.0},
    }
    portfolio.trade_log = [
        {
            "time": "2024-01-01T10:00:00",
            "action": "BUY",
            "ticker": "AAA",
            "shares": 10.0,
            "price": 10.0,
        }
    ]
    dividends = pd.Series(
        [0.50],
        index=[pd.Timestamp("2024-01-10")],
        name="Dividends",
    )

    def fetcher(ticker, start, end):
        assert ticker == "AAA"
        return dividends

    accrual = portfolio.accrue_dividends(
        date(2024, 1, 15),
        fetcher=fetcher,
    )
    second = portfolio.accrue_dividends(
        date(2024, 1, 15),
        fetcher=fetcher,
    )

    assert accrual["total"] == 5.0
    assert accrual["by_ticker"] == {"AAA": 5.0}
    assert portfolio.cash == 105.0
    assert portfolio.dividend_cash_total == 5.0
    assert portfolio.dividend_last_ex_date["AAA"] == "2024-01-10"
    assert portfolio.trade_log[-1]["action"] == "DIVIDEND"
    assert portfolio.trade_log[-1]["net_cash_flow"] == 5.0
    assert second["total"] == 0.0
    assert sum(1 for rec in portfolio.trade_log if rec["action"] == "DIVIDEND") == 1

    attribution = summarize_performance_attribution(portfolio)
    assert attribution["dividend_income"] == 5.0
    assert attribution["total_pnl"] == 5.0


def test_cap_impact_history_summarizes_over_time():
    path = Path("tests/_tmp") / f"cap_history_{uuid4().hex}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "allocation_summary": {
                "cap_impact": {"theme_cap_impact": 0.05, "total_cap_impact": 0.08},
                "cap_intervention_counts": {"theme_cap_impact": 1},
                "top_cap_interventions": [
                    {"ticker": "AAA", "cap_class": "theme_cap", "impact": 0.05}
                ],
            }
        },
        {
            "allocation_summary": {
                "cap_impact": {"sector_cap_impact": 0.03, "total_cap_impact": 0.03},
                "cap_intervention_counts": {"sector_cap_impact": 1},
                "top_cap_interventions": [
                    {"ticker": "BBB", "cap_class": "sector_cap", "impact": 0.03}
                ],
            }
        },
    ]
    try:
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

        trade_log = [
            {
                "action": "BUY",
                "ticker": "AAA",
                "reason": "score=1 | downweight_reason=theme_cap | Theme=test_theme |",
            },
            {
                "action": "BUY",
                "ticker": "BBB",
                "reason": "score=1 | downweight_reason=low_price_or_penny_cap | Theme=test_theme |",
            },
            {
                "action": "BUY",
                "ticker": "CCC",
                "reason": "score=1 | downweight_reason=none |",
            },
        ]
        summary = summarize_cap_impact_history(path, trade_log=trade_log)

        assert summary["cycles"] == 2
        assert summary["cycles_with_cap_interventions"] == 2
        assert summary["total_impact"]["theme_cap_impact"] == 0.05
        assert summary["total_impact"]["sector_cap_impact"] == 0.03
        assert summary["intervention_counts"]["theme_cap_impact"] == 1
        assert summary["top_interventions"][0]["ticker"] == "AAA"
        assert summary["entry_cap_interventions"]["buy_trades_with_caps"] == 2
        assert summary["entry_cap_interventions"]["by_reason"]["theme_cap"] == 1
        assert (
            summary["entry_cap_interventions"]["intervention_counts"]["low_price_cap_impact"]
            == 1
        )
    finally:
        path.unlink(missing_ok=True)


def test_performance_attribution_includes_execution_cost():
    portfolio = _portfolio()
    assert portfolio.buy(
        "AAA",
        10.0,
        100.0,
        "entry | Theme=consumer_credit_finance |",
        execution_cost=1.0,
    )
    assert portfolio.sell_all("AAA", 110.0, "Trailing stop (test)", execution_cost=1.1)

    attribution = summarize_performance_attribution(portfolio)

    assert attribution["closed_trades"] == 1
    assert np.isclose(attribution["realized_gross_pnl"], 100.0)
    assert np.isclose(attribution["realized_net_pnl"], 97.9)
    assert np.isclose(attribution["total_execution_cost"], 2.1)
    assert attribution["win_rate"] == 1.0
    assert attribution["exit_reason_counts"] == {"trailing_stop": 1}
    assert attribution["by_theme"][0]["theme"] == "consumer_credit_finance"
    assert attribution["by_theme"][0]["closed_trades"] == 1
    assert attribution["by_theme"][0]["stop_out_rate"] == 0.0
    assert attribution["by_price_bucket"][0]["price_bucket"] == "over_10"


def test_performance_attribution_groups_open_and_closed_by_theme_and_price_bucket():
    portfolio = _portfolio()
    assert portfolio.buy(
        "UWMC",
        100.0,
        4.0,
        "entry | downweight_reason=low_price_or_penny_cap | Theme=consumer_credit_finance |",
        execution_cost=1.0,
    )
    assert portfolio.sell_all("UWMC", 3.5, "Stop-loss (-12.5% vs -8.0% ATR-adjusted)", execution_cost=0.5)
    assert portfolio.buy(
        "SSRM",
        10.0,
        30.0,
        "entry | Theme=precious_metals_miners |",
    )
    portfolio.positions["SSRM"]["last_price"] = 28.0

    attribution = summarize_performance_attribution(portfolio)
    by_theme = {row["theme"]: row for row in attribution["by_theme"]}
    by_bucket = {row["price_bucket"]: row for row in attribution["by_price_bucket"]}

    assert attribution["exit_reason_counts"] == {"stop_loss": 1}
    assert by_theme["consumer_credit_finance"]["closed_trades"] == 1
    assert by_theme["consumer_credit_finance"]["stop_out_rate"] == 1.0
    assert by_theme["precious_metals_miners"]["open_positions"] == 1
    assert np.isclose(by_theme["precious_metals_miners"]["unrealized_pnl"], -20.0)
    assert by_bucket["sub_5"]["closed_trades"] == 1
    assert by_bucket["over_10"]["open_positions"] == 1
    assert by_theme["precious_metals_miners"]["weak_open_positions"] == 1
    assert np.isclose(by_theme["precious_metals_miners"]["avg_open_return_pct"], -2.0 / 30.0)
    assert by_theme["precious_metals_miners"]["sample_size"]["open_positions"] == 1
    assert by_theme["precious_metals_miners"]["small_sample"] is True


def test_price_sanity_explains_post_split_bkng_scale():
    portfolio = _portfolio()
    assert portfolio.buy(
        "BKNG",
        1.0,
        172.0,
        "entry | Theme=sector_consumer_discretionary |",
    )
    portfolio.trade_log[-1]["time"] = "2026-04-30T16:39:32"
    portfolio.positions["BKNG"]["last_price"] = 166.0

    report = summarize_price_sanity(portfolio)

    assert report["warnings"] == []
    assert report["known_price_events"][0]["ticker"] == "BKNG"
    assert report["known_price_events"][0]["known_price_event"]["ratio"] == 25.0


def test_redeployment_quality_tracks_stop_loss_recycling():
    portfolio = _portfolio()
    assert portfolio.buy("HOOD", 10.0, 100.0, "entry | Theme=consumer_credit_finance |")
    portfolio.trade_log[-1]["time"] = "2026-04-27T10:00:00"
    assert portfolio.sell_all(
        "HOOD",
        80.0,
        "Stop-loss (-20.0% vs -10.0% ATR-adjusted)",
        execution_cost=1.0,
    )
    portfolio.trade_log[-1]["time"] = "2026-04-29T10:00:00"
    assert portfolio.buy(
        "STX",
        1.0,
        600.0,
        "entry | downweight_reason=cash_or_risk_cap | Theme=sector_technology |",
    )
    portfolio.trade_log[-1]["time"] = "2026-04-30T10:00:00"
    portfolio.positions["STX"]["last_price"] = 660.0

    report = summarize_redeployment_quality(portfolio)

    assert report["realized_loss_events"] == 1
    assert report["replacement_entries"] == 1
    assert report["sample_size"]["open_replacement_entries"] == 1
    assert report["small_sample"] is True
    assert report["replacement_entries_detail"][0]["source_exit_ticker"] == "HOOD"
    assert report["replacement_entries_detail"][0]["replacement_ticker"] == "STX"
    assert report["replacement_entries_detail"][0]["downweight_reason"] == "cash_or_risk_cap"
    assert np.isclose(report["avg_open_replacement_return_pct"], 0.10)


def test_low_price_signal_suppression_quantifies_tokenized_top_rank_names():
    trade_log = [
        {
            "action": "BUY",
            "ticker": "UWMC",
            "price": 4.0,
            "reason": (
                "rl_rank_pct=1.0000 | target_weight_pre_caps=0.2200 | "
                "final_weight=0.0300 | downweight_reason=low_price_or_penny_cap |"
            ),
        },
        {
            "action": "BUY",
            "ticker": "SNAP",
            "price": 6.0,
            "reason": (
                "rl_rank_pct=0.6000 | target_weight_pre_caps=0.1200 | "
                "final_weight=0.1200 | downweight_reason=none |"
            ),
        },
    ]

    report = summarize_low_price_signal_suppression(trade_log)

    assert report["low_price_buy_entries"] == 2
    assert report["high_rank_low_price_entries"] == 1
    assert report["tokenized_high_rank_low_price_entries"] == 1
    assert np.isclose(report["avg_tokenized_high_rank_suppression"], 0.19)
    assert report["examples"][0]["ticker"] == "UWMC"


def test_paper_execution_cost_uses_tiered_or_configured_spread():
    cost, spread, model = _paper_execution_cost(10.0, 100.0, base_spread=0.001)
    assert np.isclose(cost, 5.0)
    assert spread == 0.005
    assert model == "paper_spread:0.0050"

    cost, spread, _model = _paper_execution_cost(
        100.0,
        10.0,
        base_spread=0.01,
    )
    assert np.isclose(cost, 10.0)
    assert spread == 0.01


def test_replay_live_parity_report_flags_runtime_mismatch():
    class _Brain:
        max_positions = 7
        max_position_pct = 0.10
        _base_min_score = 0.60
        penny_max_pct = 0.20
        max_sector_pct = 0.40
        max_pair_correlation = 0.80
        avoid_earnings_days = 3
        rl_phase = 1
        rl_exit_threshold = 0.30
        rl_conviction_drop = 0.20
        rl_min_score = 0.0

    report = build_replay_live_parity_report({"max_positions": 10}, brain=_Brain())

    assert report["compatible"] is False
    assert report["runtime_parameter_mismatches"][0]["parameter"] == "max_positions"
