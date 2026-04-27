import json
from pathlib import Path
from uuid import uuid4

import numpy as np

from broker.broker import _paper_execution_cost
from broker.paper_diagnostics import (
    build_replay_live_parity_report,
    summarize_cap_impact_history,
    summarize_performance_attribution,
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

        summary = summarize_cap_impact_history(path)

        assert summary["cycles"] == 2
        assert summary["cycles_with_cap_interventions"] == 2
        assert summary["total_impact"]["theme_cap_impact"] == 0.05
        assert summary["total_impact"]["sector_cap_impact"] == 0.03
        assert summary["intervention_counts"]["theme_cap_impact"] == 1
        assert summary["top_interventions"][0]["ticker"] == "AAA"
    finally:
        path.unlink(missing_ok=True)


def test_performance_attribution_includes_execution_cost():
    portfolio = _portfolio()
    assert portfolio.buy("AAA", 10.0, 100.0, "entry", execution_cost=1.0)
    assert portfolio.sell_all("AAA", 110.0, "exit", execution_cost=1.1)

    attribution = summarize_performance_attribution(portfolio)

    assert attribution["closed_trades"] == 1
    assert np.isclose(attribution["realized_gross_pnl"], 100.0)
    assert np.isclose(attribution["realized_net_pnl"], 97.9)
    assert np.isclose(attribution["total_execution_cost"], 2.1)
    assert attribution["win_rate"] == 1.0


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
