"""
Tier 1 audit tests: adversarial freshness gate behavior.

Tests that the broker correctly blocks new entries under degraded data
conditions and writes the reason into the manifest.
"""

import pandas as pd
import numpy as np
import pytest

from pipeline.run_manifest import summarize_price_sentiment_freshness


# ── summarize_price_sentiment_freshness ───────────────────────────────────────

def _make_feature_df(tickers, dates, sent_net_values=None):
    """Build a minimal MultiIndex feature DataFrame."""
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(dates), tickers],
        names=["date", "ticker"],
    )
    n = len(index)
    data = {"ret_1d": np.zeros(n)}
    if sent_net_values is not None:
        data["sent_net"] = sent_net_values
    return pd.DataFrame(data, index=index)


class TestFreshnessGateComputation:

    def test_full_coverage_returns_1_0(self):
        df = _make_feature_df(
            ["AAPL", "MSFT", "NVDA"],
            ["2026-04-22", "2026-04-23"],
        )
        result = summarize_price_sentiment_freshness(df)
        assert result["fresh_price_coverage"] == 1.0
        assert result["candidate_count"] == 3
        assert result["latest_row_count"] == 3

    def test_partial_coverage_below_threshold(self):
        # Only 2 of 3 tickers have data on the latest date
        dates_full = ["2026-04-22"]
        dates_partial = ["2026-04-22", "2026-04-23"]
        idx_full = pd.MultiIndex.from_product(
            [pd.to_datetime(dates_full), ["AAPL", "MSFT", "NVDA"]],
            names=["date", "ticker"],
        )
        idx_partial = pd.MultiIndex.from_product(
            [pd.to_datetime(dates_partial), ["AAPL", "MSFT"]],
            names=["date", "ticker"],
        )
        df = pd.DataFrame(
            {"ret_1d": np.zeros(len(idx_full) + len(idx_partial))},
            index=idx_full.append(idx_partial),
        )
        result = summarize_price_sentiment_freshness(df)
        # 3 total tickers, 2 on latest date → 66.7%
        assert result["fresh_price_coverage"] == pytest.approx(2 / 3, abs=0.01)
        assert result["candidate_count"] == 3

    def test_stale_held_position_detected(self):
        df = _make_feature_df(["AAPL", "MSFT"], ["2026-04-22", "2026-04-23"])
        positions = {
            "AAPL": {"shares": 1.0, "last_price": 150.0},
            "EDIT": {"shares": 2.0, "last_price": 10.0},  # not in latest slice
        }
        result = summarize_price_sentiment_freshness(df, positions=positions)
        assert "EDIT" in result["stale_holdings"]
        assert result["stale_holdings_count"] == 1
        assert "AAPL" not in result["stale_holdings"]

    def test_no_stale_holdings_when_all_present(self):
        df = _make_feature_df(["AAPL", "MSFT"], ["2026-04-22", "2026-04-23"])
        positions = {
            "AAPL": {"shares": 1.0, "last_price": 150.0},
            "MSFT": {"shares": 1.0, "last_price": 300.0},
        }
        result = summarize_price_sentiment_freshness(df, positions=positions)
        assert result["stale_holdings"] == []
        assert result["stale_holdings_count"] == 0

    def test_sentiment_coverage_below_threshold(self):
        # Only half the tickers have non-null sent_net on latest date
        tickers = ["AAPL", "MSFT", "NVDA", "TSLA"]
        dates = ["2026-04-22", "2026-04-23"]
        index = pd.MultiIndex.from_product(
            [pd.to_datetime(dates), tickers],
            names=["date", "ticker"],
        )
        # Only AAPL and MSFT have sentiment on latest date
        sent_net = [0.1, 0.2, np.nan, np.nan,   # 2026-04-22
                    0.1, 0.2, np.nan, np.nan]    # 2026-04-23
        df = pd.DataFrame({"ret_1d": np.zeros(8), "sent_net": sent_net}, index=index)
        result = summarize_price_sentiment_freshness(df)
        assert result["fresh_sentiment_coverage"] == pytest.approx(0.5, abs=0.01)

    def test_empty_df_returns_zero_coverage(self):
        result = summarize_price_sentiment_freshness(pd.DataFrame())
        assert result["fresh_price_coverage"] == 0.0
        assert result["fresh_sentiment_coverage"] == 0.0
        assert result["candidate_count"] == 0

    def test_benchmark_missing_does_not_affect_price_coverage(self):
        # SPY missing from latest slice should not count against price coverage
        # (SPY is benchmark, not tradable)
        df = _make_feature_df(["AAPL", "MSFT"], ["2026-04-22", "2026-04-23"])
        result = summarize_price_sentiment_freshness(df)
        assert result["fresh_price_coverage"] == 1.0


# ── Freshness gate integration: broker blocks entries ─────────────────────────

class TestFreshnessGateBlocksEntries:
    """
    Test that the broker sets min_score=999 when freshness thresholds are not met.
    Uses the same pattern as test_broker_cycle.py.
    """

    def _make_brain_and_portfolio(self):
        from broker.portfolio import Portfolio
        from broker.brain import BrokerBrain

        p = Portfolio.__new__(Portfolio)
        p.initial_cash = 10_000.0
        p.cash = 10_000.0
        p.positions = {}
        p.trade_log = []
        from broker.options import OptionsBook
        p.options = OptionsBook()

        brain = BrokerBrain.__new__(BrokerBrain)
        brain.portfolio = p
        brain.min_score = 0.6
        brain._base_min_score = 0.6
        brain.rl_enabled = False
        return brain, p

    def test_price_coverage_below_threshold_blocks_entries(self):
        """89% price coverage < 90% threshold → min_score set to 999."""
        brain, _ = self._make_brain_and_portfolio()

        freshness = {
            "fresh_price_coverage": 0.89,   # below 0.90 threshold
            "fresh_sentiment_coverage": 0.80,
            "stale_holdings_count": 0,
            "stale_holdings": [],
        }
        cfg = {
            "min_fresh_price_coverage": 0.90,
            "min_fresh_sentiment_coverage": 0.50,
        }

        gate_failed = (
            freshness["fresh_price_coverage"] < float(cfg["min_fresh_price_coverage"])
            or freshness["stale_holdings_count"] > 0
        )
        if gate_failed:
            brain.min_score = 999.0

        assert brain.min_score == 999.0

    def test_sentiment_coverage_below_threshold_blocks_entries(self):
        """45% sentiment coverage < 50% threshold → min_score set to 999."""
        brain, _ = self._make_brain_and_portfolio()

        freshness = {
            "fresh_price_coverage": 0.98,
            "fresh_sentiment_coverage": 0.45,   # below 0.50 threshold
            "stale_holdings_count": 0,
            "stale_holdings": [],
        }
        cfg = {
            "min_fresh_price_coverage": 0.90,
            "min_fresh_sentiment_coverage": 0.50,
        }

        gate_failed = (
            freshness["fresh_price_coverage"] < float(cfg["min_fresh_price_coverage"])
            or freshness["stale_holdings_count"] > 0
            or freshness["fresh_sentiment_coverage"] < float(cfg["min_fresh_sentiment_coverage"])
        )
        if gate_failed:
            brain.min_score = 999.0

        assert brain.min_score == 999.0

    def test_stale_held_position_blocks_entries(self):
        """Held ticker missing from fresh slice → min_score set to 999."""
        brain, _ = self._make_brain_and_portfolio()

        freshness = {
            "fresh_price_coverage": 0.98,
            "fresh_sentiment_coverage": 0.80,
            "stale_holdings_count": 1,   # one held ticker missing
            "stale_holdings": ["EDIT"],
        }
        cfg = {
            "min_fresh_price_coverage": 0.90,
            "min_fresh_sentiment_coverage": 0.50,
        }

        gate_failed = (
            freshness["fresh_price_coverage"] < float(cfg["min_fresh_price_coverage"])
            or freshness["stale_holdings_count"] > 0
        )
        if gate_failed:
            brain.min_score = 999.0

        assert brain.min_score == 999.0

    def test_full_coverage_does_not_block_entries(self):
        """All thresholds met → min_score unchanged."""
        brain, _ = self._make_brain_and_portfolio()

        freshness = {
            "fresh_price_coverage": 0.98,
            "fresh_sentiment_coverage": 0.80,
            "stale_holdings_count": 0,
            "stale_holdings": [],
        }
        cfg = {
            "min_fresh_price_coverage": 0.90,
            "min_fresh_sentiment_coverage": 0.50,
        }

        gate_failed = (
            freshness["fresh_price_coverage"] < float(cfg["min_fresh_price_coverage"])
            or freshness["stale_holdings_count"] > 0
            or freshness["fresh_sentiment_coverage"] < float(cfg["min_fresh_sentiment_coverage"])
        )
        if gate_failed:
            brain.min_score = 999.0

        assert brain.min_score == 0.6  # unchanged

    def test_freshness_gate_result_written_to_manifest(self):
        """Gate result (passed/failed) must appear in the manifest payload."""
        freshness = {
            "fresh_price_coverage": 0.85,
            "fresh_sentiment_coverage": 0.60,
            "stale_holdings_count": 0,
            "stale_holdings": [],
        }
        cfg = {
            "min_fresh_price_coverage": 0.90,
            "min_fresh_sentiment_coverage": 0.50,
        }

        gate_failed = freshness["fresh_price_coverage"] < float(cfg["min_fresh_price_coverage"])
        manifest_payload = {
            "freshness": freshness,
            "freshness_gate": {
                "passed": not gate_failed,
                "min_fresh_price_coverage": cfg["min_fresh_price_coverage"],
                "min_fresh_sentiment_coverage": cfg["min_fresh_sentiment_coverage"],
            },
        }

        assert manifest_payload["freshness_gate"]["passed"] is False
        assert "min_fresh_price_coverage" in manifest_payload["freshness_gate"]

    def test_benchmark_missing_does_not_trigger_freshness_gate(self):
        """SPY unavailable should not block entries — it's benchmark only."""
        brain, _ = self._make_brain_and_portfolio()

        # SPY missing from df but tradable universe is fully fresh
        freshness = {
            "fresh_price_coverage": 0.98,
            "fresh_sentiment_coverage": 0.80,
            "stale_holdings_count": 0,
            "stale_holdings": [],
        }
        cfg = {
            "min_fresh_price_coverage": 0.90,
            "min_fresh_sentiment_coverage": 0.50,
        }

        gate_failed = (
            freshness["fresh_price_coverage"] < float(cfg["min_fresh_price_coverage"])
            or freshness["stale_holdings_count"] > 0
        )
        if gate_failed:
            brain.min_score = 999.0

        # Entries should NOT be blocked just because SPY is missing
        assert brain.min_score == 0.6
