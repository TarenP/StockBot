"""
Unit tests for Phase 1 RL ranking logic in BrokerBrain.run_cycle.

Tests:
  - Candidates are sorted by rl_score when rl_enabled=True
  - Composite_score sort is preserved when rl_enabled=False
  - min_score threshold is applied to rl_score (not composite_score) when rl_enabled=True

Requirements: 3.1, 3.3, 3.4
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import date

from broker.brain import BrokerBrain, Decision
from broker.portfolio import Portfolio


def _make_df_features(tickers: list[str]) -> pd.DataFrame:
    """Build a minimal MultiIndex df_features for testing."""
    dates = [date(2024, 1, i + 1) for i in range(25)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    rng = np.random.default_rng(42)
    data = rng.standard_normal((len(idx), 5))
    cols = ["ret_5d", "vol_ratio", "sent_net", "macd_hist", "rsi"]
    return pd.DataFrame(data, index=idx, columns=cols)


def _make_portfolio(cash: float = 100_000.0) -> Portfolio:
    """Create a portfolio with no positions."""
    p = Portfolio(initial_cash=cash)
    p.positions = {}
    p.cash = cash
    return p


def _make_brain(rl_enabled: bool, min_score: float = 0.0, rl_min_score: float = 0.0) -> BrokerBrain:
    portfolio = _make_portfolio()
    brain = BrokerBrain(
        portfolio=portfolio,
        max_positions=10,
        min_score=min_score,
        rl_enabled=rl_enabled,
        rl_checkpoint_path="models/best_fold9.pt",
        rl_phase=1,
        rl_min_score=rl_min_score,
    )
    brain._base_min_score = min_score
    brain._sector_map = {}
    brain._sector_cache_date = None
    return brain


def _research_factory(scores: dict[str, float]):
    """Return a research() mock that returns a report with the given composite_score."""
    def _research(ticker):
        if ticker not in scores:
            return None
        return {
            "ticker": ticker,
            "composite_score": scores[ticker],
            "price": 50.0,
            "sentiment": {"sentiment": "neutral", "sent_net": 0.0},
            "headlines": ["headline"],
            "atr": 0.02,
        }
    return _research


class TestPhase1RLRanking(unittest.TestCase):

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _run_cycle_patched(
        self,
        brain: BrokerBrain,
        df_features: pd.DataFrame,
        candidates: list[str],
        composite_scores: dict[str, float],
        rl_scores_series: pd.Series | None = None,
    ) -> list[Decision]:
        """
        Run brain.run_cycle with all external I/O mocked out.
        """
        from unittest.mock import PropertyMock
        with (
            patch("broker.brain.get_rl_targets", return_value=rl_scores_series) as mock_rl,
            patch("broker.brain.research", side_effect=_research_factory(composite_scores)),
            patch.object(brain, "_screen_candidates", return_value=candidates),
            patch.object(brain, "_maybe_refresh_sector_map"),
            patch.object(brain, "_get_current_prices", return_value={}),
            patch.object(brain, "_near_earnings", return_value=False),
            patch("broker.brain.validate_portfolio_prices", return_value={}),
            patch("broker.brain.score_sectors", return_value={}),
            patch("broker.brain.get_portfolio_sector_weights", return_value={}),
            patch("broker.brain.compute_target_allocations", return_value={"Unknown": 1.0}),
            patch("broker.brain._get_next_earnings_date", return_value=None),
            patch.object(type(brain.portfolio), "position_values",
                         new_callable=PropertyMock, return_value={}),
            patch.object(type(brain.portfolio), "equity",
                         new_callable=PropertyMock, return_value=brain.portfolio.cash),
        ):
            brain.portfolio.update_prices = MagicMock()
            decisions = brain.run_cycle(df_features, screener_top_n=100)

        return decisions

    # ── Test 1: RL mode sorts by rl_score ─────────────────────────────────────

    def test_rl_enabled_sorts_by_rl_score(self):
        """When rl_enabled=True, BUY decisions are ordered by rl_score descending."""
        tickers = ["AAA", "BBB", "CCC"]
        df = _make_df_features(tickers)
        brain = _make_brain(rl_enabled=True, min_score=0.0)

        # composite_score order: CCC > BBB > AAA
        composite_scores = {"AAA": 0.65, "BBB": 0.75, "CCC": 0.85}
        # rl_score order: AAA > CCC > BBB  (deliberately different from composite)
        rl_scores = pd.Series({"AAA": 0.90, "BBB": 0.50, "CCC": 0.70}, name="rl_score")

        with patch.object(brain, "_assert_model_available"):
            decisions = self._run_cycle_patched(
                brain, df, tickers, composite_scores, rl_scores
            )

        buy_tickers = [d.ticker for d in decisions if d.action == "BUY"]
        self.assertGreater(len(buy_tickers), 0, "Expected at least one BUY decision")

        # Verify ordering: AAA (0.90) before CCC (0.70) before BBB (0.50)
        if "AAA" in buy_tickers and "CCC" in buy_tickers:
            self.assertLess(
                buy_tickers.index("AAA"), buy_tickers.index("CCC"),
                "AAA (rl=0.90) should come before CCC (rl=0.70)"
            )
        if "CCC" in buy_tickers and "BBB" in buy_tickers:
            self.assertLess(
                buy_tickers.index("CCC"), buy_tickers.index("BBB"),
                "CCC (rl=0.70) should come before BBB (rl=0.50)"
            )

    # ── Test 2: Heuristic mode preserves composite_score sort ─────────────────

    def test_rl_disabled_sorts_by_composite_score(self):
        """When rl_enabled=False, BUY decisions are ordered by composite_score descending."""
        tickers = ["AAA", "BBB", "CCC"]
        df = _make_df_features(tickers)
        brain = _make_brain(rl_enabled=False, min_score=0.0)

        # composite_score order: CCC > BBB > AAA
        composite_scores = {"AAA": 0.65, "BBB": 0.75, "CCC": 0.85}

        decisions = self._run_cycle_patched(
            brain, df, tickers, composite_scores, rl_scores_series=None
        )

        buy_tickers = [d.ticker for d in decisions if d.action == "BUY"]
        self.assertGreater(len(buy_tickers), 0, "Expected at least one BUY decision")

        # Verify ordering: CCC (0.85) before BBB (0.75) before AAA (0.65)
        if "CCC" in buy_tickers and "BBB" in buy_tickers:
            self.assertLess(
                buy_tickers.index("CCC"), buy_tickers.index("BBB"),
                "CCC (composite=0.85) should come before BBB (composite=0.75)"
            )
        if "BBB" in buy_tickers and "AAA" in buy_tickers:
            self.assertLess(
                buy_tickers.index("BBB"), buy_tickers.index("AAA"),
                "BBB (composite=0.75) should come before AAA (composite=0.65)"
            )

    # ── Test 3: min_score applied to rl_score when RL enabled ─────────────────

    def test_rl_enabled_min_score_filters_on_rl_score(self):
        """
        When rl_enabled=True, rl_min_score threshold is applied to rl_score
        (not the heuristic min_score).
        A ticker with composite_score >= rl_min_score but rl_score < rl_min_score is excluded.
        A ticker with composite_score < rl_min_score but rl_score >= rl_min_score is included.
        """
        tickers = ["PASS_RL", "FAIL_RL"]
        df = _make_df_features(tickers)
        brain = _make_brain(rl_enabled=True, min_score=0.0, rl_min_score=0.60)

        # PASS_RL: composite below rl_min_score, rl_score above → should be included
        # FAIL_RL: composite above rl_min_score, rl_score below → should be excluded
        composite_scores = {
            "PASS_RL": 0.40,   # below rl_min_score
            "FAIL_RL": 0.80,   # above rl_min_score
        }
        rl_scores = pd.Series(
            {"PASS_RL": 0.75, "FAIL_RL": 0.30},  # PASS_RL above, FAIL_RL below
            name="rl_score",
        )

        with patch.object(brain, "_assert_model_available"):
            decisions = self._run_cycle_patched(
                brain, df, tickers, composite_scores, rl_scores
            )

        buy_tickers = [d.ticker for d in decisions if d.action == "BUY"]

        self.assertIn(
            "PASS_RL", buy_tickers,
            "PASS_RL has rl_score=0.75 >= rl_min_score=0.60 and should be included"
        )
        self.assertNotIn(
            "FAIL_RL", buy_tickers,
            "FAIL_RL has rl_score=0.30 < rl_min_score=0.60 and should be excluded"
        )

    # ── Test 4: RL disabled — get_rl_targets is never called ──────────────────

    def test_rl_disabled_never_calls_get_rl_targets(self):
        """When rl_enabled=False, get_rl_targets is not invoked."""
        tickers = ["AAA", "BBB"]
        df = _make_df_features(tickers)
        brain = _make_brain(rl_enabled=False, min_score=0.0)
        composite_scores = {"AAA": 0.70, "BBB": 0.80}

        with patch("broker.brain.get_rl_targets") as mock_rl:
            self._run_cycle_patched(brain, df, tickers, composite_scores)
            mock_rl.assert_not_called()

    # ── Test 5: RL reason field contains rl_score and composite_score ──────────

    def test_rl_enabled_reason_contains_rl_fields(self):
        """When rl_enabled=True, BUY decision reason includes rl_score, composite_score, rl_mode=true."""
        tickers = ["AAA"]
        df = _make_df_features(tickers)
        brain = _make_brain(rl_enabled=True, min_score=0.0)
        composite_scores = {"AAA": 0.70}
        rl_scores = pd.Series({"AAA": 0.85}, name="rl_score")

        with patch.object(brain, "_assert_model_available"):
            decisions = self._run_cycle_patched(
                brain, df, tickers, composite_scores, rl_scores
            )

        buy_decisions = [d for d in decisions if d.action == "BUY" and d.ticker == "AAA"]
        self.assertEqual(len(buy_decisions), 1)
        reason = buy_decisions[0].reason
        self.assertIn("rl_score=", reason)
        self.assertIn("composite_score=", reason)
        self.assertIn("rl_mode=true", reason)

    # ── Test 6: Abort cycle when _assert_model_available raises ───────────────

    def test_rl_enabled_aborts_on_model_unavailable(self):
        """When _assert_model_available raises RuntimeError, run_cycle returns []."""
        tickers = ["AAA"]
        df = _make_df_features(tickers)
        brain = _make_brain(rl_enabled=True, min_score=0.0)

        with patch.object(
            brain, "_assert_model_available",
            side_effect=RuntimeError("checkpoint not found")
        ):
            with (
                patch.object(brain, "_maybe_refresh_sector_map"),
                patch.object(brain, "_screen_candidates", return_value=tickers),
            ):
                result = brain.run_cycle(df)

        self.assertEqual(result, [])

    # ── Test 7: Abort cycle when get_rl_targets raises ────────────────────────

    def test_rl_enabled_aborts_on_inference_failure(self):
        """When get_rl_targets raises, run_cycle aborts (returns only prior decisions)."""
        from unittest.mock import PropertyMock
        tickers = ["AAA"]
        df = _make_df_features(tickers)
        brain = _make_brain(rl_enabled=True, min_score=0.0)

        with (
            patch.object(brain, "_assert_model_available"),
            patch("broker.brain.get_rl_targets", side_effect=RuntimeError("GPU OOM")),
            patch.object(brain, "_screen_candidates", return_value=tickers),
            patch.object(brain, "_maybe_refresh_sector_map"),
            patch.object(brain, "_get_current_prices", return_value={}),
            patch("broker.brain.validate_portfolio_prices", return_value={}),
            patch("broker.brain.score_sectors", return_value={}),
            patch("broker.brain.get_portfolio_sector_weights", return_value={}),
            patch("broker.brain.compute_target_allocations", return_value={}),
            patch.object(type(brain.portfolio), "position_values",
                         new_callable=PropertyMock, return_value={}),
            patch.object(type(brain.portfolio), "equity",
                         new_callable=PropertyMock, return_value=brain.portfolio.cash),
        ):
            brain.portfolio.update_prices = MagicMock()
            result = brain.run_cycle(df)

        # No BUY decisions should be generated after inference failure
        buy_decisions = [d for d in result if d.action == "BUY"]
        self.assertEqual(buy_decisions, [])


if __name__ == "__main__":
    unittest.main()
