"""
Unit tests for BrokerBrain._rl_exit_checks (Phase 2).

Tests:
  1. SELL generated when current_rl_score < rl_exit_threshold
  2. SELL_PARTIAL generated when conviction drop > rl_conviction_drop
  3. Position retained and warning logged when get_rl_targets raises

Requirements: 4.1, 4.2, 4.4
"""

import logging
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import date

from broker.brain import BrokerBrain, Decision
from broker.portfolio import Portfolio


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_df_features(tickers: list[str]) -> pd.DataFrame:
    """Build a minimal MultiIndex df_features for testing."""
    dates = [date(2024, 1, i + 1) for i in range(25)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    rng = np.random.default_rng(0)
    data = rng.standard_normal((len(idx), 5))
    cols = ["ret_5d", "vol_ratio", "sent_net", "macd_hist", "rsi"]
    return pd.DataFrame(data, index=idx, columns=cols)


def _make_portfolio_with_positions(positions: dict) -> Portfolio:
    """
    Create a Portfolio with pre-seeded positions.
    Each entry in `positions` maps ticker → dict with at least
    {shares, avg_cost, last_price} and optionally rl_score_at_entry.
    """
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
        """
        Requirement 4.1: When current RL score drops below rl_exit_threshold,
        a SELL decision must be generated for the full position.
        """
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
        df = _make_df_features([ticker])

        # current score is 0.20 — below threshold of 0.30
        mock_series = pd.Series({ticker: 0.20}, name="rl_score")

        with patch("broker.brain.get_rl_targets", return_value=mock_series):
            decisions = brain._rl_exit_checks(df, [ticker])

        self.assertEqual(len(decisions), 1)
        d = decisions[0]
        self.assertEqual(d.action, "SELL")
        self.assertEqual(d.ticker, ticker)
        self.assertEqual(d.shares, 10.0)
        self.assertAlmostEqual(d.price, 155.0)
        self.assertIn("rl_exit_threshold", d.reason)
        self.assertIn("rl_mode=true", d.reason)

    def test_no_sell_when_score_above_exit_threshold(self):
        """No exit decision when current RL score is above rl_exit_threshold."""
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
        df = _make_df_features([ticker])

        # current score is 0.75 — above threshold, small drop (0.05 < 0.20)
        mock_series = pd.Series({ticker: 0.75}, name="rl_score")

        with patch("broker.brain.get_rl_targets", return_value=mock_series):
            decisions = brain._rl_exit_checks(df, [ticker])

        self.assertEqual(decisions, [])

    # ── Test 2: SELL_PARTIAL when conviction drop exceeds threshold ───────────

    def test_sell_partial_generated_on_conviction_drop(self):
        """
        Requirement 4.2: When (entry_rl_score - current_rl_score) > rl_conviction_drop,
        a SELL_PARTIAL decision for 50% of the position must be generated.
        """
        ticker = "GOOG"
        shares = 8.0
        positions = {
            ticker: {
                "shares": shares,
                "avg_cost": 100.0,
                "last_price": 105.0,
                "partial_taken": False,
                "rl_score_at_entry": 0.80,  # entry score
            }
        }
        # conviction drop = 0.80 - 0.55 = 0.25 > rl_conviction_drop=0.20
        brain = _make_brain(positions, rl_exit_threshold=0.30, rl_conviction_drop=0.20)
        df = _make_df_features([ticker])

        mock_series = pd.Series({ticker: 0.55}, name="rl_score")

        with patch("broker.brain.get_rl_targets", return_value=mock_series):
            decisions = brain._rl_exit_checks(df, [ticker])

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
        df = _make_df_features([ticker])

        mock_series = pd.Series({ticker: 0.55}, name="rl_score")

        with patch("broker.brain.get_rl_targets", return_value=mock_series):
            decisions = brain._rl_exit_checks(df, [ticker])

        self.assertEqual(decisions, [])

    def test_sell_partial_not_generated_without_entry_score(self):
        """
        Requirement 4.3: Positions opened before RL integration have no
        rl_score_at_entry. Conviction drop check is skipped; only the
        absolute threshold check applies.
        """
        ticker = "AMZN"
        positions = {
            ticker: {
                "shares": 4.0,
                "avg_cost": 180.0,
                "last_price": 185.0,
                "partial_taken": False,
                # no rl_score_at_entry — pre-RL position
            }
        }
        # current score 0.50 — above exit threshold (0.30), no entry score to compare
        brain = _make_brain(positions, rl_exit_threshold=0.30, rl_conviction_drop=0.20)
        df = _make_df_features([ticker])

        mock_series = pd.Series({ticker: 0.50}, name="rl_score")

        with patch("broker.brain.get_rl_targets", return_value=mock_series):
            decisions = brain._rl_exit_checks(df, [ticker])

        self.assertEqual(decisions, [])

    # ── Test 3: Position retained and warning logged on inference failure ─────

    def test_position_retained_on_inference_failure(self):
        """
        Requirement 4.4: When get_rl_targets raises an exception for a ticker,
        no exit decision is generated (position is retained).
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
        df = _make_df_features([ticker])

        with patch(
            "broker.brain.get_rl_targets",
            side_effect=RuntimeError("model inference failed"),
        ):
            decisions = brain._rl_exit_checks(df, [ticker])

        self.assertEqual(decisions, [], "No exit decision should be generated on failure")

    def test_warning_logged_on_inference_failure(self):
        """
        Requirement 4.4: A warning must be logged when inference fails for a ticker.
        """
        ticker = "NFLX"
        positions = {
            ticker: {
                "shares": 2.0,
                "avg_cost": 400.0,
                "last_price": 410.0,
                "partial_taken": False,
            }
        }
        brain = _make_brain(positions)
        df = _make_df_features([ticker])

        with patch(
            "broker.brain.get_rl_targets",
            side_effect=ValueError("checkpoint corrupt"),
        ):
            with self.assertLogs("broker.brain", level="WARNING") as log_ctx:
                decisions = brain._rl_exit_checks(df, [ticker])

        self.assertEqual(decisions, [])
        self.assertTrue(
            any("NFLX" in msg and "retaining position" in msg for msg in log_ctx.output),
            f"Expected warning about NFLX retention, got: {log_ctx.output}",
        )

    # ── Test 4: Multiple tickers — partial failure ────────────────────────────

    def test_mixed_tickers_partial_failure(self):
        """
        When one ticker fails inference and another triggers a SELL,
        only the successful ticker generates a decision.
        """
        positions = {
            "GOOD": {
                "shares": 5.0,
                "avg_cost": 100.0,
                "last_price": 102.0,
                "partial_taken": False,
                "rl_score_at_entry": 0.70,
            },
            "BAD": {
                "shares": 3.0,
                "avg_cost": 200.0,
                "last_price": 205.0,
                "partial_taken": False,
                "rl_score_at_entry": 0.65,
            },
        }
        brain = _make_brain(positions, rl_exit_threshold=0.30)
        df = _make_df_features(["GOOD", "BAD"])

        def _mock_get_rl_targets(df_features, asset_list, checkpoint_path, mode="rank", **kwargs):
            ticker = asset_list[0]
            if ticker == "BAD":
                raise RuntimeError("inference failed for BAD")
            # GOOD has score 0.15 — below threshold
            return pd.Series({ticker: 0.15}, name="rl_score")

        with patch("broker.brain.get_rl_targets", side_effect=_mock_get_rl_targets):
            decisions = brain._rl_exit_checks(df, ["GOOD", "BAD"])

        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].ticker, "GOOD")
        self.assertEqual(decisions[0].action, "SELL")

    # ── Test 5: SELL takes precedence over SELL_PARTIAL ──────────────────────

    def test_sell_takes_precedence_over_conviction_drop(self):
        """
        When both conditions are met (score below threshold AND conviction drop),
        SELL (full exit) is generated, not SELL_PARTIAL.
        """
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
        # current score 0.10 — below threshold (0.30) AND drop (0.80) > conviction_drop (0.20)
        brain = _make_brain(positions, rl_exit_threshold=0.30, rl_conviction_drop=0.20)
        df = _make_df_features([ticker])

        mock_series = pd.Series({ticker: 0.10}, name="rl_score")

        with patch("broker.brain.get_rl_targets", return_value=mock_series):
            decisions = brain._rl_exit_checks(df, [ticker])

        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].action, "SELL")

    # ── Test 6: RL exit log fields (Requirement 12.3) ─────────────────────────

    def test_rl_exit_reason_contains_required_log_fields(self):
        """
        Requirement 12.3: The reason field must contain entry rl_score,
        current rl_score, drop magnitude, and threshold crossed.
        """
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
        df = _make_df_features([ticker])

        # Trigger SELL_PARTIAL: drop = 0.85 - 0.60 = 0.25 > 0.20, score 0.60 > threshold
        mock_series = pd.Series({ticker: 0.60}, name="rl_score")

        with patch("broker.brain.get_rl_targets", return_value=mock_series):
            decisions = brain._rl_exit_checks(df, [ticker])

        self.assertEqual(len(decisions), 1)
        reason = decisions[0].reason
        # Must contain entry score, current score, drop, and threshold reference
        self.assertIn("0.8500", reason)   # entry_rl_score
        self.assertIn("0.6000", reason)   # current_rl_score
        self.assertIn("rl_conviction_drop", reason)


if __name__ == "__main__":
    unittest.main()
