"""
P9 audit tests: replay invariants.

Proves that the replay emits loud warnings when impossible states occur —
fill before signal, out-of-range dates, negative shares, zero prices.
"""

import pandas as pd
import pytest
import logging

from broker.replay import _check_replay_invariants


def _dates():
    return [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04")]


class TestReplayInvariants:

    def test_clean_trade_log_passes_all_invariants(self):
        trade_log = [
            {"action": "BUY", "ticker": "AAPL", "shares": 1.0, "price": 150.0,
             "decision_date": "2024-01-02", "fill_date": "2024-01-03"},
            {"action": "SELL", "ticker": "AAPL", "shares": 1.0, "price": 155.0,
             "decision_date": "2024-01-03", "fill_date": "2024-01-04"},
        ]
        results = _check_replay_invariants(trade_log, _dates())
        assert all(results.values()), f"Expected all to pass, got: {results}"

    def test_fill_before_signal_fails_invariant(self, caplog):
        trade_log = [
            {"action": "BUY", "ticker": "AAPL", "shares": 1.0, "price": 150.0,
             "decision_date": "2024-01-03",
             "fill_date": "2024-01-02"},  # fill BEFORE decision — lookahead!
        ]
        with caplog.at_level(logging.WARNING, logger="broker.replay"):
            results = _check_replay_invariants(trade_log, _dates())

        assert results["no_fill_before_signal"] is False
        assert any("INVARIANT FAIL" in r.message for r in caplog.records)

    def test_fill_outside_replay_dates_fails_invariant(self, caplog):
        trade_log = [
            {"action": "BUY", "ticker": "AAPL", "shares": 1.0, "price": 150.0,
             "decision_date": "2024-01-02",
             "fill_date": "2024-06-15"},  # way outside replay window
        ]
        with caplog.at_level(logging.WARNING, logger="broker.replay"):
            results = _check_replay_invariants(trade_log, _dates())

        assert results["fills_within_replay_dates"] is False
        assert any("INVARIANT FAIL" in r.message for r in caplog.records)

    def test_negative_shares_fails_invariant(self, caplog):
        trade_log = [
            {"action": "SELL", "ticker": "AAPL", "shares": -5.0, "price": 150.0,
             "decision_date": "2024-01-02", "fill_date": "2024-01-03"},
        ]
        with caplog.at_level(logging.WARNING, logger="broker.replay"):
            results = _check_replay_invariants(trade_log, _dates())

        assert results["no_negative_shares"] is False
        assert any("INVARIANT FAIL" in r.message for r in caplog.records)

    def test_zero_price_fill_fails_invariant(self, caplog):
        trade_log = [
            {"action": "BUY", "ticker": "AAPL", "shares": 1.0, "price": 0.0,
             "decision_date": "2024-01-02", "fill_date": "2024-01-03"},
        ]
        with caplog.at_level(logging.WARNING, logger="broker.replay"):
            results = _check_replay_invariants(trade_log, _dates())

        assert results["no_zero_price_fills"] is False
        assert any("INVARIANT FAIL" in r.message for r in caplog.records)

    def test_unknown_action_fails_invariant(self, caplog):
        trade_log = [
            {"action": "MYSTERY_ACTION", "ticker": "AAPL", "shares": 1.0, "price": 150.0,
             "decision_date": "2024-01-02", "fill_date": "2024-01-03"},
        ]
        with caplog.at_level(logging.WARNING, logger="broker.replay"):
            results = _check_replay_invariants(trade_log, _dates())

        assert results["all_actions_known"] is False
        assert any("INVARIANT FAIL" in r.message for r in caplog.records)

    def test_empty_trade_log_passes_all_invariants(self):
        results = _check_replay_invariants([], _dates())
        assert all(results.values())

    def test_invariant_results_are_all_booleans(self):
        trade_log = [
            {"action": "BUY", "ticker": "AAPL", "shares": 1.0, "price": 150.0,
             "decision_date": "2024-01-02", "fill_date": "2024-01-03"},
        ]
        results = _check_replay_invariants(trade_log, _dates())
        for key, val in results.items():
            assert isinstance(val, bool), f"Result for '{key}' is not bool: {val}"

    def test_multiple_failures_all_reported(self, caplog):
        """Multiple invariant failures should all be reported, not just the first."""
        trade_log = [
            {"action": "BUY", "ticker": "AAPL", "shares": -1.0, "price": 0.0,
             "decision_date": "2024-01-03", "fill_date": "2024-01-02"},  # 3 failures
        ]
        with caplog.at_level(logging.WARNING, logger="broker.replay"):
            results = _check_replay_invariants(trade_log, _dates())

        failures = [k for k, v in results.items() if not v]
        assert len(failures) >= 3  # fill_before_signal, negative_shares, zero_price
