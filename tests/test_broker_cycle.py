import json
from datetime import date
from pathlib import Path
from uuid import uuid4

import pandas as pd

import broker.broker as broker_module
import broker.journal as journal_module
from broker.portfolio import Portfolio
from pipeline import data as data_module
from pipeline import sentiment as sentiment_module
from pipeline import updater as updater_module


class _DummyOptions:
    positions = {}


class _DummyPortfolio:
    def __init__(self):
        self.cash = 10_000.0
        self.positions = {}
        self.options = _DummyOptions()

    @property
    def equity(self) -> float:
        return 10_000.0

    @property
    def total_return(self) -> float:
        return 0.0

    def save(self) -> None:
        pass

    def summary(self) -> str:
        return "summary"


class _DummyBrain:
    def __init__(self):
        self.min_score = 0.6
        self._base_min_score = 0.6

    def run_cycle(self, df, screener_top_n=50, risk_engine=None):
        return []


class _DummyRisk:
    def start_session(self, equity: float) -> None:
        self.started_equity = equity

    def check_portfolio_health(self, portfolio):
        return "ok", ""


def test_run_cycle_skips_duplicate_refresh_after_maintenance(monkeypatch):
    calls = {"prices": 0, "sentiment": 0}
    cycle_date = date(2026, 4, 15)

    monkeypatch.setattr(broker_module, "_is_market_hours", lambda: True)
    monkeypatch.setattr(broker_module, "_today_et", lambda: cycle_date)
    monkeypatch.setattr(updater_module, "_load_trained_universe", lambda save_dir: ["AAA"])
    monkeypatch.setattr(
        updater_module,
        "update_parquet",
        lambda **kwargs: calls.__setitem__("prices", calls["prices"] + 1) or 0,
    )
    monkeypatch.setattr(
        sentiment_module,
        "update_sentiment",
        lambda *args, **kwargs: calls.__setitem__("sentiment", calls["sentiment"] + 1) or 0,
    )
    monkeypatch.setattr(data_module, "load_master", lambda **kwargs: pd.DataFrame())
    monkeypatch.setattr(broker_module, "log_cycle", lambda *args, **kwargs: None)
    monkeypatch.setattr(broker_module, "daily_integrity_check", lambda portfolio: {})
    monkeypatch.setattr(broker_module, "print_report", lambda *args, **kwargs: None)
    monkeypatch.setattr(broker_module, "_fetch_current_spy_price", lambda: None)
    monkeypatch.setattr(journal_module, "plot_live_performance", lambda *args, **kwargs: None)

    broker_module.run_cycle(
        portfolio=_DummyPortfolio(),
        brain=_DummyBrain(),
        risk=_DummyRisk(),
        maintenance_context={
            "prices_updated": cycle_date.isoformat(),
            "sentiment_updated": cycle_date.isoformat(),
        },
    )

    assert calls["prices"] == 0
    assert calls["sentiment"] == 0


def test_run_cycle_prints_summary_only_via_report(monkeypatch):
    printed: list[str] = []

    monkeypatch.setattr(broker_module, "_is_market_hours", lambda: True)
    monkeypatch.setattr(updater_module, "_load_trained_universe", lambda save_dir: ["AAA"])
    monkeypatch.setattr(updater_module, "update_parquet", lambda **kwargs: 0)
    monkeypatch.setattr(sentiment_module, "update_sentiment", lambda *args, **kwargs: 0)
    monkeypatch.setattr(data_module, "load_master", lambda **kwargs: pd.DataFrame())
    monkeypatch.setattr(broker_module, "log_cycle", lambda *args, **kwargs: None)
    monkeypatch.setattr(broker_module, "daily_integrity_check", lambda portfolio: {})
    monkeypatch.setattr(broker_module, "print_report", lambda *args, **kwargs: None)
    monkeypatch.setattr(broker_module, "_fetch_current_spy_price", lambda: None)
    monkeypatch.setattr(journal_module, "plot_live_performance", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "builtins.print",
        lambda *args, **kwargs: printed.append(" ".join(map(str, args))),
    )

    broker_module.run_cycle(
        portfolio=_DummyPortfolio(),
        brain=_DummyBrain(),
        risk=_DummyRisk(),
    )

    assert "summary" not in printed


def test_run_cycle_loads_raw_cols_for_local_research_fallback(monkeypatch):
    load_kwargs = {}

    monkeypatch.setattr(broker_module, "_is_market_hours", lambda: True)
    monkeypatch.setattr(updater_module, "_load_trained_universe", lambda save_dir: ["AAA"])
    monkeypatch.setattr(updater_module, "update_parquet", lambda **kwargs: 0)
    monkeypatch.setattr(sentiment_module, "update_sentiment", lambda *args, **kwargs: 0)

    def _fake_load_master(**kwargs):
        load_kwargs.update(kwargs)
        return pd.DataFrame()

    monkeypatch.setattr(data_module, "load_master", _fake_load_master)
    monkeypatch.setattr(broker_module, "log_cycle", lambda *args, **kwargs: None)
    monkeypatch.setattr(broker_module, "daily_integrity_check", lambda portfolio: {})
    monkeypatch.setattr(broker_module, "print_report", lambda *args, **kwargs: None)
    monkeypatch.setattr(broker_module, "_fetch_current_spy_price", lambda: None)
    monkeypatch.setattr(journal_module, "plot_live_performance", lambda *args, **kwargs: None)

    broker_module.run_cycle(
        portfolio=_DummyPortfolio(),
        brain=_DummyBrain(),
        risk=_DummyRisk(),
    )

    assert load_kwargs["include_raw_cols"] is True


def test_portfolio_summary_always_includes_stock_holdings_section():
    class _EmptyOptions:
        positions = {}

        @staticmethod
        def summary_lines() -> list[str]:
            return []

    portfolio = Portfolio.__new__(Portfolio)
    portfolio.initial_cash = 10_000.0
    portfolio.cash = 10_000.0
    portfolio.positions = {}
    portfolio.options = _EmptyOptions()

    summary = Portfolio.summary(portfolio)

    assert "Stock Holdings (0)" in summary
    assert "No stock positions" in summary


def test_portfolio_summary_flags_positions_marked_at_entry_price():
    class _EmptyOptions:
        positions = {}

        @staticmethod
        def summary_lines() -> list[str]:
            return []

    portfolio = Portfolio.__new__(Portfolio)
    portfolio.initial_cash = 10_000.0
    portfolio.cash = 9_500.0
    portfolio.positions = {
        "ABC": {
            "shares": 5.0,
            "avg_cost": 100.0,
            "last_price": 100.0,
        }
    }
    portfolio.options = _EmptyOptions()

    summary = Portfolio.summary(portfolio)

    assert "marked at entry prices" in summary


def test_portfolio_cash_auto_accrues_at_three_percent(monkeypatch):
    state_path = Path("tests/_tmp") / f"portfolio_cash_yield_{uuid4().hex}.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "cash": 10_000.0,
                "positions": {},
                "trade_log": [],
                "initial_cash": 10_000.0,
                "last_saved": "2024-01-01T00:00:00",
            }
        )
    )
    monkeypatch.setattr("broker.portfolio.STATE_PATH", state_path)

    try:
        portfolio = Portfolio(initial_cash=10_000.0)
        accrued = portfolio.accrue_cash_yield(date(2025, 1, 1))

        assert 299.0 < accrued < 302.0
        assert 10_299.0 < portfolio.cash < 10_302.0
        assert portfolio.cash_yield_last_date == date(2025, 1, 1)
    finally:
        state_path.unlink(missing_ok=True)


def test_fresh_portfolio_does_not_accrue_cash_without_elapsed_days():
    class _EmptyOptions:
        positions = {}

    portfolio = Portfolio.__new__(Portfolio)
    portfolio.initial_cash = 10_000.0
    portfolio.cash = 10_000.0
    portfolio.positions = {}
    portfolio.trade_log = []
    portfolio.cash_yield_last_date = None
    portfolio.options = _EmptyOptions()

    accrued = portfolio.accrue_cash_yield(date(2025, 1, 1))

    assert accrued == 0.0
    assert portfolio.cash == 10_000.0
    assert portfolio.cash_yield_last_date == date(2025, 1, 1)
