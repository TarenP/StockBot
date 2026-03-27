"""
Unit tests for BrokerBrain._rl_exit_checks (Phase 2).

The new signature is:
    _rl_exit_checks(held_tickers: list[str], rl_scores: pd.Series) -> list[Decision]

rl_scores is the cycle-level Series already computed by get_rl_targets — held
positions are evaluated in the same cross-sectional context as buy candidates,
not in isolation.  Tickers absent from rl_scores are skipped (deferred to
heuristic exits).

Tests:
  1. SELL generated when current_rl_score < rl_exit_threshold
  2. SELL_PARTIAL generated when conviction drop > rl_conviction_drop
  3. Ticker absent from rl_scores is skipped (no spurious exit)

Requirements: 4.1, 4.2, 4.4
"""

import logging
import unittest
import pandas as pd
import numpy as np
from datetime import date

from broker.brain import BrokerBrain, Decision
from broker.portfolio import Portfolio


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_portfolio_with_positions(positions: dict) -> Portfolio:
    p = Portfolio.__new__(Portfolio)
    p.initial_cash = 100_000.0
    p.cash = 50_000.0
    p.positions = positions
    p.trade_log = []
    from broker.options import OptionsBook
    p.options = OptionsBook()
    return p


def _make_brain(
    positions: dict,
    rl_exit_threshold: float = 0.30,
    rl_conviction_drop: float = 0.20,
    rl_phase: int = 2,
) -> BrokerBrain:
    portfolio = _make_portfolio_with_positions(positions)
    brain = BrokerBrain(
        portfolio=portfolio,
        rl_enabled=True,
        rl_checkpoint_path="models/best_fold9.pt",
        rl_phase=rl_phase,
        rl_exit_threshold=rl_exit_threshold,
        rl_conviction_drop=rl_conviction_drop,
    )
    brain._sector_map = {}
    brain._sector_cache_date = None
    return brain


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestRlExitChecks(unittest.TestCase):

    # ── Test 1: SELL when current_rl_score < rl_exit_threshold ───────────────

    def test_sell_generated_when_score_below_exit_threshold(self):
        """Requirement 4.1: SELL when current RL score < rl_exit_threshold."""
        ticker = "AAPL"
        positions = {
            ticker: {
                "shares": 10.0,
                "avg_cost": 150.0,
                "last_price": 155.0,
                "partial_taken": False,
                "rl_score_at_entry": 0.70,
            }
        }
        brain = _make_brain(positions, rl_exit_threshold=0.30)
        rl_scores = pd.Series({ticker: 0.20}, name="rl_score")  # below threshold

        decisions = brain._rl_exit_checks([ticker], rl_scores)

        self.assertEqual(len(decisions), 1)
        d = decisions[0]
        self.assertEqual(d.action, "SELL")
        self.assertEqual(d.ticker, ticker)
        self.assertEqual(d.shares, 10.0)
        self.assertAlmostEqual(d.price, 155.0)
        self.assertIn("rl_exit_threshold", d.reason)
        self.assertIn("rl_mode=true", d.reason)

    def test_no_sell_when_score_above_exit_threshold(self):
        """No exit when current RL score is above rl_exit_threshold."""
        ticker = "MSFT"
        positions = {
            ticker: {
                "shares": 5.0,
                "avg_cost": 300.0,
                "last_price": 310.0,
                "partial_taken": False,
                "rl_score_at_entry": 0.80,
            }
        }
        brain = _make_brain(positions, rl_exit_threshold=0.30, rl_conviction_drop=0.20)
        rl_scores = pd.Series({ticker: 0.75}, name="rl_score")  # above threshold, small drop

        decisions = brain._rl_exit_checks([ticker], rl_scores)

        self.assertEqual(decisions, [])

    # ── Test 2: SELL_PARTIAL when conviction drop exceeds threshold ───────────

    def test_sell_partial_generated_on_conviction_drop(self):
        """Requirement 4.2: SELL_PARTIAL when conviction drop > rl_conviction_drop."""
        ticker = "GOOG"
        shares = 8.0
        positions = {
            ticker: {
                "shares": shares,
                "avg_cost": 100.0,
                "last_price": 105.0,
                "partial_taken": False,
                "rl_score_at_entry": 0.80,
            }
        }
        # drop = 0.80 - 0.55 = 0.25 > rl_conviction_drop=0.20
        brain = _make_brain(positions, rl_exit_threshold=0.30, rl_conviction_drop=0.20)
        rl_scores = pd.Series({ticker: 0.55}, name="rl_score")

        decisions = brain._rl_exit_checks([ticker], rl_scores)

        self.assertEqual(len(decisions), 1)
        d = decisions[0]
        self.assertEqual(d.action, "SELL_PARTIAL")
        self.assertEqual(d.ticker, ticker)
        self.assertAlmostEqual(d.shares, shares * 0.5)
        self.assertAlmostEqual(d.price, 105.0)
        self.assertIn("rl_conviction_drop", d.reason)
        self.assertIn("rl_mode=true", d.reason)

    def test_sell_partial_not_generated_when_drop_below_threshold(self):
        """No SELL_PARTIAL when conviction drop is within the allowed threshold."""
        ticker = "TSLA"
        positions = {
            ticker: {
                "shares": 3.0,
                "avg_cost": 200.0,
                "last_price": 210.0,
                "partial_taken": False,
                "rl_score_at_entry": 0.70,
            }
        }
        # drop = 0.70 - 0.55 = 0.15 < rl_conviction_drop=0.20
        brain = _make_brain(positions, rl_exit_threshold=0.30, rl_conviction_drop=0.20)
        rl_scores = pd.Series({ticker: 0.55}, name="rl_score")

        decisions = brain._rl_exit_checks([ticker], rl_scores)

        self.assertEqual(decisions, [])

    def test_sell_partial_not_generated_without_entry_score(self):
        """Pre-RL positions (no rl_score_at_entry) skip conviction drop check."""
        ticker = "AMZN"
        positions = {
            ticker: {
                "shares": 4.0,
                "avg_cost": 180.0,
                "last_price": 185.0,
                "partial_taken": False,
                # no rl_score_at_entry
            }
        }
        brain = _make_brain(positions, rl_exit_threshold=0.30, rl_conviction_drop=0.20)
        rl_scores = pd.Series({ticker: 0.50}, name="rl_score")

        decisions = brain._rl_exit_checks([ticker], rl_scores)

        self.assertEqual(decisions, [])

    # ── Test 3: Ticker absent from rl_scores is skipped ──────────────────────

    def test_ticker_absent_from_rl_scores_is_skipped(self):
        """
        Requirement 4.4: A held ticker not present in the cycle rl_scores
        (e.g. fell off the shortlist) generates no exit decision.
        """
        ticker = "META"
        positions = {
            ticker: {
                "shares": 6.0,
                "avg_cost": 250.0,
                "last_price": 260.0,
                "partial_taken": False,
                "rl_score_at_entry": 0.75,
            }
        }
        brain = _make_brain(positions)
        # rl_scores does not contain META — it was not in the shortlist this cycle
        rl_scores = pd.Series({"OTHER": 0.50}, name="rl_score")

        decisions = brain._rl_exit_checks([ticker], rl_scores)

        self.assertEqual(decisions, [], "Ticker absent from rl_scores should be skipped")

    # ── Test 4: Mixed tickers — one present, one absent ───────────────────────

    def test_mixed_tickers_one_absent(self):
        """
        When one ticker is in rl_scores and triggers a SELL, and another is
        absent, only the present ticker generates a decision.
        """
        positions = {
            "GOOD": {
                "shares": 5.0,
                "avg_cost": 100.0,
                "last_price": 102.0,
                "partial_taken": False,
                "rl_score_at_entry": 0.70,
            },
            "ABSENT": {
                "shares": 3.0,
                "avg_cost": 200.0,
                "last_price": 205.0,
                "partial_taken": False,
                "rl_score_at_entry": 0.65,
            },
        }
        brain = _make_brain(positions, rl_exit_threshold=0.30)
        # GOOD is in rl_scores with a low score; ABSENT is not
        rl_scores = pd.Series({"GOOD": 0.15}, name="rl_score")

        decisions = brain._rl_exit_checks(["GOOD", "ABSENT"], rl_scores)

        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].ticker, "GOOD")
        self.assertEqual(decisions[0].action, "SELL")

    # ── Test 5: SELL takes precedence over SELL_PARTIAL ──────────────────────

    def test_sell_takes_precedence_over_conviction_drop(self):
        """When both conditions are met, SELL (full exit) is generated."""
        ticker = "NVDA"
        positions = {
            ticker: {
                "shares": 7.0,
                "avg_cost": 500.0,
                "last_price": 510.0,
                "partial_taken": False,
                "rl_score_at_entry": 0.90,
            }
        }
        # score 0.10 < threshold (0.30) AND drop (0.80) > conviction_drop (0.20)
        brain = _make_brain(positions, rl_exit_threshold=0.30, rl_conviction_drop=0.20)
        rl_scores = pd.Series({ticker: 0.10}, name="rl_score")

        decisions = brain._rl_exit_checks([ticker], rl_scores)

        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].action, "SELL")

    # ── Test 6: Reason field contains required log fields (Req 12.3) ──────────

    def test_rl_exit_reason_contains_required_log_fields(self):
        """Requirement 12.3: reason must contain entry score, current score, drop, threshold."""
        ticker = "AMD"
        positions = {
            ticker: {
                "shares": 10.0,
                "avg_cost": 120.0,
                "last_price": 125.0,
                "partial_taken": False,
                "rl_score_at_entry": 0.85,
            }
        }
        brain = _make_brain(positions, rl_exit_threshold=0.30, rl_conviction_drop=0.20)
        # Trigger SELL_PARTIAL: drop = 0.85 - 0.60 = 0.25 > 0.20, score 0.60 > threshold
        rl_scores = pd.Series({ticker: 0.60}, name="rl_score")

        decisions = brain._rl_exit_checks([ticker], rl_scores)

        self.assertEqual(len(decisions), 1)
        reason = decisions[0].reason
        self.assertIn("0.8500", reason)
        self.assertIn("0.6000", reason)
        self.assertIn("rl_conviction_drop", reason)


if __name__ == "__main__":
    unittest.main()
