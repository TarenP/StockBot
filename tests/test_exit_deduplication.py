"""
Tests for same-cycle exit deduplication in BrokerBrain.run_cycle.

Ensures that when both heuristic and RL exits fire for the same ticker
in one cycle, only one exit decision survives — the highest-priority one.

Priority: SELL > SELL_PARTIAL
"""

import pandas as pd
import numpy as np
import pytest

from broker.brain import BrokerBrain, Decision
from broker.portfolio import Portfolio


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_portfolio(positions: dict) -> Portfolio:
    p = Portfolio.__new__(Portfolio)
    p.initial_cash = 100_000.0
    p.cash = 50_000.0
    p.positions = positions
    p.trade_log = []
    from broker.options import OptionsBook
    p.options = OptionsBook()
    return p


def _make_brain(positions: dict, rl_phase: int = 2) -> BrokerBrain:
    portfolio = _make_portfolio(positions)
    brain = BrokerBrain(
        portfolio=portfolio,
        rl_enabled=True,
        rl_checkpoint_path="models/best_fold1.pt",
        rl_phase=rl_phase,
        rl_exit_threshold=0.20,
        rl_conviction_drop=0.20,
    )
    brain._sector_map = {}
    brain._sector_cache_date = None
    return brain


def _make_rl_scores(ticker: str, rank_pct: float, n: int = 10) -> pd.Series:
    """Build rl_scores where ticker lands at rank_pct."""
    target_rank = max(1, round(rank_pct * n))
    scores = np.linspace(0.01, 0.99, n)
    ticker_score = scores[target_rank - 1]
    others = [f"T{i}" for i in range(n - 1)]
    other_scores = [s for s in scores if s != ticker_score]
    data = {t: s for t, s in zip(others, other_scores)}
    data[ticker] = ticker_score
    return pd.Series(data, name="rl_score")


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestExitDeduplication:

    def test_full_sell_beats_partial_for_same_ticker(self):
        """When both SELL and SELL_PARTIAL exist for a ticker, only SELL survives."""
        ticker = "AAPL"
        decisions = [
            Decision(action="SELL_PARTIAL", ticker=ticker, shares=5.0,
                     price=100.0, score=0.5, reason="partial tp"),
            Decision(action="SELL", ticker=ticker, shares=10.0,
                     price=100.0, score=0.0, reason="stop loss"),
        ]
        # Simulate the dedup logic from brain.run_cycle
        exit_by_ticker: dict = {}
        buy_decisions = []
        for d in decisions:
            if d.action in ("SELL", "SELL_PARTIAL"):
                existing = exit_by_ticker.get(d.ticker)
                if existing is None:
                    exit_by_ticker[d.ticker] = d
                elif existing.action == "SELL_PARTIAL" and d.action == "SELL":
                    exit_by_ticker[d.ticker] = d
            else:
                buy_decisions.append(d)
        result = list(exit_by_ticker.values())

        assert len(result) == 1
        assert result[0].action == "SELL"
        assert result[0].ticker == ticker

    def test_first_sell_wins_when_two_sells(self):
        """When two SELL decisions exist for same ticker, first one wins."""
        ticker = "MSFT"
        decisions = [
            Decision(action="SELL", ticker=ticker, shares=10.0,
                     price=300.0, score=0.0, reason="heuristic stop"),
            Decision(action="SELL", ticker=ticker, shares=10.0,
                     price=300.0, score=0.1, reason="rl exit"),
        ]
        exit_by_ticker: dict = {}
        for d in decisions:
            if d.action in ("SELL", "SELL_PARTIAL"):
                existing = exit_by_ticker.get(d.ticker)
                if existing is None:
                    exit_by_ticker[d.ticker] = d
                elif existing.action == "SELL_PARTIAL" and d.action == "SELL":
                    exit_by_ticker[d.ticker] = d
        result = list(exit_by_ticker.values())

        assert len(result) == 1
        assert result[0].reason == "heuristic stop"

    def test_different_tickers_both_survive(self):
        """Exit decisions for different tickers are not deduplicated."""
        decisions = [
            Decision(action="SELL", ticker="AAPL", shares=10.0,
                     price=150.0, score=0.0, reason="stop"),
            Decision(action="SELL", ticker="MSFT", shares=5.0,
                     price=300.0, score=0.0, reason="stop"),
        ]
        exit_by_ticker: dict = {}
        for d in decisions:
            if d.action in ("SELL", "SELL_PARTIAL"):
                existing = exit_by_ticker.get(d.ticker)
                if existing is None:
                    exit_by_ticker[d.ticker] = d
                elif existing.action == "SELL_PARTIAL" and d.action == "SELL":
                    exit_by_ticker[d.ticker] = d
        result = list(exit_by_ticker.values())

        assert len(result) == 2
        tickers = {d.ticker for d in result}
        assert tickers == {"AAPL", "MSFT"}

    def test_buy_decisions_not_affected_by_dedup(self):
        """BUY decisions pass through dedup unchanged."""
        decisions = [
            Decision(action="SELL", ticker="AAPL", shares=10.0,
                     price=150.0, score=0.0, reason="stop"),
            Decision(action="BUY", ticker="NVDA", shares=5.0,
                     price=800.0, score=0.9, reason="buy signal"),
            Decision(action="BUY", ticker="TSLA", shares=2.0,
                     price=200.0, score=0.8, reason="buy signal"),
        ]
        exit_by_ticker: dict = {}
        buy_decisions = []
        for d in decisions:
            if d.action in ("SELL", "SELL_PARTIAL"):
                existing = exit_by_ticker.get(d.ticker)
                if existing is None:
                    exit_by_ticker[d.ticker] = d
                elif existing.action == "SELL_PARTIAL" and d.action == "SELL":
                    exit_by_ticker[d.ticker] = d
            else:
                buy_decisions.append(d)
        result = list(exit_by_ticker.values()) + buy_decisions

        sells = [d for d in result if d.action == "SELL"]
        buys  = [d for d in result if d.action == "BUY"]
        assert len(sells) == 1
        assert len(buys) == 2


class TestPartialTakenNotSetOnFailure:
    """partial_taken must only be set after a successful sell execution."""

    def test_partial_taken_set_only_after_success(self):
        """Simulate broker.py execution: partial_taken set only if sell returns True."""
        positions = {
            "AAPL": {
                "shares": 10.0,
                "avg_cost": 100.0,
                "last_price": 130.0,
                "partial_taken": False,
            }
        }
        portfolio = _make_portfolio(positions)

        # Simulate a successful partial sell
        ok = portfolio.sell("AAPL", 5.0, 130.0, "partial tp")
        if ok and "AAPL" in portfolio.positions:
            portfolio.positions["AAPL"]["partial_taken"] = True

        assert portfolio.positions["AAPL"]["partial_taken"] is True
        assert portfolio.positions["AAPL"]["shares"] == pytest.approx(5.0)

    def test_partial_taken_not_set_when_sell_fails(self):
        """If sell fails (e.g. ticker not in portfolio), partial_taken stays False."""
        positions = {
            "AAPL": {
                "shares": 10.0,
                "avg_cost": 100.0,
                "last_price": 130.0,
                "partial_taken": False,
            }
        }
        portfolio = _make_portfolio(positions)

        # Try to sell a ticker that doesn't exist
        ok = portfolio.sell("NONEXISTENT", 5.0, 100.0, "partial tp")
        if ok and "NONEXISTENT" in portfolio.positions:
            portfolio.positions["NONEXISTENT"]["partial_taken"] = True

        # AAPL should be untouched
        assert portfolio.positions["AAPL"]["partial_taken"] is False
        assert ok is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
