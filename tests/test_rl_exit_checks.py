"""
Unit tests for BrokerBrain._rl_exit_checks (Phase 2).

The implementation uses rank percentiles within the current shortlist, not
raw RL scores. This is because mode="rank" scores sum to 1 across the
shortlist and shrink as the shortlist grows — comparing them to a fixed
absolute threshold produces exits that depend on shortlist size, not
conviction.

Rank percentile semantics:
  - With N tickers in rl_scores, the lowest-scoring ticker gets rank_pct ≈ 1/N
  - rl_exit_threshold=0.20 means "exit if in the bottom 20% of the shortlist"
  - rl_conviction_drop=0.20 means "exit 50% if rank dropped by 0.20 from entry"

To control rank percentiles in tests, we construct rl_scores Series with
enough tickers so the held ticker lands at a known percentile.

Helper: _scores_with_rank_pct(ticker, target_pct, n_total)
  Returns a Series of n_total tickers where `ticker` has rank_pct ≈ target_pct.
"""

import unittest
import pandas as pd
import numpy as np

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
    rl_exit_threshold: float = 0.20,
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


def _scores_with_rank_pct(
    ticker: str,
    target_rank_pct: float,
    n_total: int = 10,
) -> pd.Series:
    """
    Build a rl_scores Series of n_total tickers where `ticker` has
    rank_pct ≈ target_rank_pct.

    rank_pct = rank / n_total  (rank is 1-based, ascending)
    So target rank = round(target_rank_pct * n_total).
    We assign scores so that `ticker` lands at that rank.
    """
    target_rank = max(1, round(target_rank_pct * n_total))
    # Assign evenly spaced scores; ticker gets the score at target_rank
    scores = np.linspace(0.01, 0.99, n_total)
    ticker_score = scores[target_rank - 1]

    other_tickers = [f"T{i}" for i in range(n_total - 1)]
    other_scores = [s for s in scores if s != ticker_score]

    data = {t: s for t, s in zip(other_tickers, other_scores)}
    data[ticker] = ticker_score
    return pd.Series(data, name="rl_score")


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestRlExitChecks(unittest.TestCase):

    # ── Test 1: SELL when rank_pct < rl_exit_threshold ───────────────────────

    def test_sell_generated_when_rank_below_exit_threshold(self):
        """SELL when ticker's rank percentile is below rl_exit_threshold."""
        ticker = "AAPL"
        positions = {
            ticker: {
                "shares": 10.0,
                "avg_cost": 150.0,
                "last_price": 155.0,
                "partial_taken": False,
                "rl_rank_pct_at_entry": 0.80,
            }
        }
        brain = _make_brain(positions, rl_exit_threshold=0.20)
        # ticker at rank_pct ≈ 0.10 — below threshold of 0.20
        rl_scores = _scores_with_rank_pct(ticker, target_rank_pct=0.10, n_total=10)

        decisions = brain._rl_exit_checks([ticker], rl_scores)

        self.assertEqual(len(decisions), 1)
        d = decisions[0]
        self.assertEqual(d.action, "SELL")
        self.assertEqual(d.ticker, ticker)
        self.assertEqual(d.shares, 10.0)
        self.assertAlmostEqual(d.price, 155.0)
        self.assertIn("rl_exit_threshold", d.reason)
        self.assertIn("rl_mode=true", d.reason)

    def test_no_sell_when_rank_above_exit_threshold(self):
        """No exit when ticker's rank percentile is above rl_exit_threshold."""
        ticker = "MSFT"
        positions = {
            ticker: {
                "shares": 5.0,
                "avg_cost": 300.0,
                "last_price": 310.0,
                "partial_taken": False,
                "rl_rank_pct_at_entry": 0.80,
            }
        }
        brain = _make_brain(positions, rl_exit_threshold=0.20, rl_conviction_drop=0.20)
        # ticker at rank_pct ≈ 0.70 — above threshold, small drop from 0.80
        rl_scores = _scores_with_rank_pct(ticker, target_rank_pct=0.70, n_total=10)

        decisions = brain._rl_exit_checks([ticker], rl_scores)

        self.assertEqual(decisions, [])

    # ── Test 2: SELL_PARTIAL when rank drop exceeds rl_conviction_drop ────────

    def test_sell_partial_generated_on_rank_conviction_drop(self):
        """SELL_PARTIAL when rank_pct dropped by more than rl_conviction_drop."""
        ticker = "GOOG"
        shares = 8.0
        positions = {
            ticker: {
                "shares": shares,
                "avg_cost": 100.0,
                "last_price": 105.0,
                "partial_taken": False,
                "rl_rank_pct_at_entry": 0.80,  # entered at 80th percentile
            }
        }
        # Current rank_pct ≈ 0.50 — drop = 0.80 - 0.50 = 0.30 > rl_conviction_drop=0.20
        brain = _make_brain(positions, rl_exit_threshold=0.20, rl_conviction_drop=0.20)
        rl_scores = _scores_with_rank_pct(ticker, target_rank_pct=0.50, n_total=10)

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
        """No SELL_PARTIAL when rank drop is within the allowed threshold."""
        ticker = "TSLA"
        positions = {
            ticker: {
                "shares": 3.0,
                "avg_cost": 200.0,
                "last_price": 210.0,
                "partial_taken": False,
                "rl_rank_pct_at_entry": 0.70,
            }
        }
        # Current rank_pct ≈ 0.60 — drop = 0.70 - 0.60 = 0.10 < rl_conviction_drop=0.20
        brain = _make_brain(positions, rl_exit_threshold=0.20, rl_conviction_drop=0.20)
        rl_scores = _scores_with_rank_pct(ticker, target_rank_pct=0.60, n_total=10)

        decisions = brain._rl_exit_checks([ticker], rl_scores)

        self.assertEqual(decisions, [])

    def test_sell_partial_not_generated_without_entry_rank(self):
        """Pre-RL positions (no rank metadata) skip conviction drop check."""
        ticker = "AMZN"
        positions = {
            ticker: {
                "shares": 4.0,
                "avg_cost": 180.0,
                "last_price": 185.0,
                "partial_taken": False,
                # no rl_rank_pct_at_entry, no rl_score_at_entry
            }
        }
        brain = _make_brain(positions, rl_exit_threshold=0.20, rl_conviction_drop=0.20)
        # rank_pct ≈ 0.50 — above threshold, but no entry rank to compare
        rl_scores = _scores_with_rank_pct(ticker, target_rank_pct=0.50, n_total=10)

        decisions = brain._rl_exit_checks([ticker], rl_scores)

        self.assertEqual(decisions, [])

    # ── Test 3: Ticker absent from rl_scores is skipped ──────────────────────

    def test_ticker_absent_from_rl_scores_is_skipped(self):
        """A held ticker not in the cycle rl_scores generates no exit decision."""
        ticker = "META"
        positions = {
            ticker: {
                "shares": 6.0,
                "avg_cost": 250.0,
                "last_price": 260.0,
                "partial_taken": False,
                "rl_rank_pct_at_entry": 0.75,
            }
        }
        brain = _make_brain(positions)
        rl_scores = pd.Series({"OTHER": 0.50}, name="rl_score")

        decisions = brain._rl_exit_checks([ticker], rl_scores)

        self.assertEqual(decisions, [])

    # ── Test 4: Mixed tickers — one present, one absent ───────────────────────

    def test_mixed_tickers_one_absent(self):
        """Only the present ticker generates a decision."""
        positions = {
            "GOOD": {
                "shares": 5.0,
                "avg_cost": 100.0,
                "last_price": 102.0,
                "partial_taken": False,
                "rl_rank_pct_at_entry": 0.80,
            },
            "ABSENT": {
                "shares": 3.0,
                "avg_cost": 200.0,
                "last_price": 205.0,
                "partial_taken": False,
                "rl_rank_pct_at_entry": 0.65,
            },
        }
        brain = _make_brain(positions, rl_exit_threshold=0.20)
        # GOOD at rank_pct ≈ 0.10 (bottom 10%) — triggers SELL; ABSENT not in scores
        rl_scores = _scores_with_rank_pct("GOOD", target_rank_pct=0.10, n_total=10)

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
                "rl_rank_pct_at_entry": 0.90,
            }
        }
        # rank_pct ≈ 0.10 — below threshold (0.20) AND drop (0.80) > conviction_drop (0.20)
        brain = _make_brain(positions, rl_exit_threshold=0.20, rl_conviction_drop=0.20)
        rl_scores = _scores_with_rank_pct(ticker, target_rank_pct=0.10, n_total=10)

        decisions = brain._rl_exit_checks([ticker], rl_scores)

        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].action, "SELL")

    # ── Test 6: Reason field contains rank percentile fields ──────────────────

    def test_rl_exit_reason_contains_rank_fields(self):
        """Reason must contain rank_pct, entry_rank_pct, and threshold."""
        ticker = "AMD"
        positions = {
            ticker: {
                "shares": 10.0,
                "avg_cost": 120.0,
                "last_price": 125.0,
                "partial_taken": False,
                "rl_rank_pct_at_entry": 0.80,
            }
        }
        brain = _make_brain(positions, rl_exit_threshold=0.20, rl_conviction_drop=0.20)
        # rank_pct ≈ 0.50 — above threshold, drop = 0.80 - 0.50 = 0.30 > 0.20
        rl_scores = _scores_with_rank_pct(ticker, target_rank_pct=0.50, n_total=10)

        decisions = brain._rl_exit_checks([ticker], rl_scores)

        self.assertEqual(len(decisions), 1)
        reason = decisions[0].reason
        self.assertIn("rl_conviction_drop", reason)
        self.assertIn("rl_mode=true", reason)
        self.assertIn("shortlist_n=", reason)

    # ── Test 7: Legacy positions with only rl_score_at_entry ─────────────────

    def test_legacy_position_uses_raw_score_as_rank_proxy(self):
        """
        Positions with only rl_score_at_entry (no rl_rank_pct_at_entry) use
        the raw score as a rough rank proxy for backward compatibility.
        A large enough drop should still trigger SELL_PARTIAL.
        """
        ticker = "INTC"
        positions = {
            ticker: {
                "shares": 5.0,
                "avg_cost": 40.0,
                "last_price": 38.0,
                "partial_taken": False,
                "rl_score_at_entry": 0.80,  # legacy: raw score, no rank_pct
            }
        }
        brain = _make_brain(positions, rl_exit_threshold=0.20, rl_conviction_drop=0.20)
        # rank_pct ≈ 0.50 — above threshold, drop from proxy 0.80 = 0.30 > 0.20
        rl_scores = _scores_with_rank_pct(ticker, target_rank_pct=0.50, n_total=10)

        decisions = brain._rl_exit_checks([ticker], rl_scores)

        # Should trigger SELL_PARTIAL since drop from legacy proxy (0.80) is large
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].action, "SELL_PARTIAL")


if __name__ == "__main__":
    unittest.main()
