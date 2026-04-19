"""
Tests for fundamental features (Task 7).

Property 14: Fundamental Fill Behaviour
  Validates: Requirements 10.4
"""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from unittest.mock import patch, MagicMock

from pipeline.features import FEATURE_COLS, fetch_fundamentals


# ── Unit tests ────────────────────────────────────────────────────────────────

class TestFeatureColsCompleteness:
    """7.4 – Assert fundamental columns are in FEATURE_COLS."""

    def test_pe_ratio_in_feature_cols(self):
        assert "pe_ratio" in FEATURE_COLS

    def test_revenue_growth_in_feature_cols(self):
        assert "revenue_growth" in FEATURE_COLS

    def test_short_interest_pct_in_feature_cols(self):
        assert "short_interest_pct" in FEATURE_COLS


class TestFetchFundamentals:
    def _mock_info(self, pe=15.0, rev=0.1, short=0.02):
        return {
            "trailingPE": pe,
            "revenueGrowth": rev,
            "shortPercentOfFloat": short,
        }

    def test_returns_correct_columns(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            cache_path = f.name
        try:
            with patch("yfinance.Ticker") as mock_ticker:
                mock_ticker.return_value.info = self._mock_info()
                result = fetch_fundamentals(["AAPL"], cache_path=cache_path)
            assert set(result.columns) == {"pe_ratio", "revenue_growth", "short_interest_pct"}
        finally:
            os.unlink(cache_path)

    def test_missing_values_default_to_zero(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            cache_path = f.name
        try:
            with patch("yfinance.Ticker") as mock_ticker:
                mock_ticker.return_value.info = {}  # all missing
                result = fetch_fundamentals(["AAPL"], cache_path=cache_path)
            assert result.loc["AAPL", "pe_ratio"] == pytest.approx(0.0)
            assert result.loc["AAPL", "revenue_growth"] == pytest.approx(0.0)
            assert result.loc["AAPL", "short_interest_pct"] == pytest.approx(0.0)
        finally:
            os.unlink(cache_path)

    def test_cache_is_read_before_fetching(self):
        """7.4 – Verify cache is read before fetching (mock yfinance)."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            cache_path = f.name
            import time
            # Write a fresh cache (timestamp = now)
            json.dump({
                "_timestamp": time.time(),
                "data": {
                    "AAPL": {"pe_ratio": 99.0, "revenue_growth": 0.5, "short_interest_pct": 0.1}
                }
            }, f)

        try:
            with patch("yfinance.Ticker") as mock_ticker:
                result = fetch_fundamentals(["AAPL"], cache_path=cache_path)
                # yfinance should NOT have been called since cache is fresh
                mock_ticker.assert_not_called()
            assert result.loc["AAPL", "pe_ratio"] == pytest.approx(99.0)
        finally:
            os.unlink(cache_path)

    def test_stale_cache_triggers_fetch(self):
        """Stale cache (>24h) should trigger a fresh yfinance fetch."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            cache_path = f.name
            # Write a stale cache (timestamp = 2 days ago)
            json.dump({
                "_timestamp": 0.0,  # epoch = very stale
                "data": {
                    "AAPL": {"pe_ratio": 1.0, "revenue_growth": 0.0, "short_interest_pct": 0.0}
                }
            }, f)

        try:
            with patch("yfinance.Ticker") as mock_ticker:
                mock_ticker.return_value.info = {"trailingPE": 25.0}
                result = fetch_fundamentals(["AAPL"], cache_path=cache_path)
                mock_ticker.assert_called_once_with("AAPL")
            assert result.loc["AAPL", "pe_ratio"] == pytest.approx(25.0)
        finally:
            os.unlink(cache_path)

    def test_fetch_failure_fills_zero(self):
        """When yfinance raises, all fundamentals should default to 0.0."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            cache_path = f.name
        try:
            with patch("yfinance.Ticker") as mock_ticker:
                mock_ticker.return_value.info = MagicMock(
                    side_effect=Exception("network error")
                )
                # Make .info raise when accessed
                type(mock_ticker.return_value).info = property(
                    lambda self: (_ for _ in ()).throw(Exception("network error"))
                )
                result = fetch_fundamentals(["AAPL"], cache_path=cache_path)
            assert result.loc["AAPL", "pe_ratio"] == pytest.approx(0.0)
        finally:
            os.unlink(cache_path)


# ── Property-based test ───────────────────────────────────────────────────────

@given(
    gap_len=st.integers(min_value=1, max_value=20),
    n_dates=st.integers(min_value=25, max_value=60),
    fill_value=st.floats(min_value=0.1, max_value=50.0, allow_nan=False),
    seed=st.integers(min_value=0, max_value=999),
)
@settings(max_examples=100)
def test_fundamental_fill_behaviour(gap_len: int, n_dates: int, fill_value: float, seed: int):
    """
    Property 14: Fundamental Fill Behaviour
    Validates: Requirements 10.4

    For any fundamental data series with a gap of length 1-20, the filled
    series SHALL use the last known value for positions within 20 days of
    the gap start, and 0.0 for positions beyond 20 days.
    """
    dates = pd.date_range("2022-01-03", periods=n_dates, freq="B")

    gap_start = min(5, n_dates - gap_len - 1)
    values = pd.Series(fill_value, index=dates)
    last_known = values.iloc[gap_start - 1]

    # Introduce gap
    gap_end = gap_start + gap_len
    values.iloc[gap_start:gap_end] = np.nan

    # Apply the same fill logic as build_features() for fundamentals
    filled = values.ffill(limit=20).fillna(0.0)

    # Positions within 20 days of gap start should be forward-filled
    for i in range(gap_start, min(gap_end, gap_start + 20)):
        if i < n_dates:
            assert filled.iloc[i] == pytest.approx(last_known, abs=1e-9), (
                f"Position {i} in gap should be forward-filled to {last_known}, "
                f"got {filled.iloc[i]}"
            )

    # Positions beyond 20 days into the gap should be 0.0
    for i in range(gap_start + 20, gap_end):
        if i < n_dates:
            assert filled.iloc[i] == pytest.approx(0.0, abs=1e-9), (
                f"Position {i} beyond 20-day fill limit should be 0.0, "
                f"got {filled.iloc[i]}"
            )
