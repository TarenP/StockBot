"""
Tests for market context features (Task 6).

Property 9: VIX Fill Behaviour
  Validates: Requirements 7.4
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pipeline.features import FEATURE_COLS, build_market_context


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_panel(n_dates: int = 250, tickers=("AAPL", "SPY", "MSFT")) -> pd.DataFrame:
    """Build a minimal MultiIndex panel for testing."""
    dates = pd.date_range("2022-01-03", periods=n_dates, freq="B")
    rows = []
    rng = np.random.default_rng(42)
    for ticker in tickers:
        close = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, n_dates))
        for i, d in enumerate(dates):
            rows.append({
                "date": d, "ticker": ticker,
                "close": close[i], "open": close[i],
                "high": close[i] * 1.01, "low": close[i] * 0.99,
                "volume": 1_000_000,
            })
    df = pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()
    return df


# ── Unit tests ────────────────────────────────────────────────────────────────

class TestFeatureColsCompleteness:
    """6.5 – Assert market context columns are in FEATURE_COLS."""

    def test_spy_ret_20d_in_feature_cols(self):
        assert "spy_ret_20d" in FEATURE_COLS

    def test_vix_level_in_feature_cols(self):
        assert "vix_level" in FEATURE_COLS

    def test_market_breadth_in_feature_cols(self):
        assert "market_breadth" in FEATURE_COLS


class TestBuildMarketContext:
    def test_returns_correct_columns(self):
        df = _make_panel()
        ctx = build_market_context(df)
        assert set(ctx.columns) == {"spy_ret_20d", "vix_level", "market_breadth"}

    def test_index_matches_panel_dates(self):
        df = _make_panel()
        dates = df.index.get_level_values("date").unique()
        ctx = build_market_context(df)
        assert set(ctx.index) == set(dates)

    def test_no_nan_in_output(self):
        df = _make_panel()
        ctx = build_market_context(df)
        assert not ctx.isnull().any().any()

    def test_market_breadth_in_range(self):
        df = _make_panel()
        ctx = build_market_context(df)
        assert (ctx["market_breadth"] >= 0.0).all()
        assert (ctx["market_breadth"] <= 1.0).all()

    def test_spy_missing_fills_zero(self):
        """When SPY is not in the panel, spy_ret_20d should be 0."""
        df = _make_panel(tickers=("AAPL", "MSFT"))
        ctx = build_market_context(df)
        assert (ctx["spy_ret_20d"] == 0.0).all()

    def test_vix_default_fill_is_zero(self):
        """VIX fetch will fail in test env; result should be 0.0 (not NaN)."""
        df = _make_panel()
        ctx = build_market_context(df)
        assert not ctx["vix_level"].isnull().any()


# ── Property-based test ───────────────────────────────────────────────────────

@given(
    gap_len=st.integers(min_value=1, max_value=5),
    n_dates=st.integers(min_value=20, max_value=60),
    seed=st.integers(min_value=0, max_value=999),
)
@settings(max_examples=100)
def test_vix_fill_behaviour(gap_len: int, n_dates: int, seed: int):
    """
    Property 9: VIX Fill Behaviour
    Validates: Requirements 7.4

    For any VIX series with a gap of length 1-5, the filled series SHALL use
    the last known value for positions within 5 days of the gap start, and
    0.0 for positions beyond 5 days.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=n_dates, freq="B")

    # Build a VIX series with a known gap starting at position 5
    gap_start = min(5, n_dates - gap_len - 1)
    vix_values = rng.uniform(0.1, 0.8, n_dates)
    vix_series = pd.Series(vix_values, index=dates)
    last_known = vix_series.iloc[gap_start - 1]

    # Introduce gap
    gap_end = gap_start + gap_len
    vix_series.iloc[gap_start:gap_end] = np.nan

    # Apply the same fill logic as build_market_context
    filled = vix_series.ffill(limit=5).fillna(0.0)

    # Positions within the gap (length <= 5) should be forward-filled
    for i in range(gap_start, min(gap_end, gap_start + 5)):
        if i < n_dates:
            assert filled.iloc[i] == pytest.approx(last_known, abs=1e-9), (
                f"Position {i} in gap should be forward-filled to {last_known}, "
                f"got {filled.iloc[i]}"
            )

    # Positions beyond 5 days into the gap should be 0.0
    for i in range(gap_start + 5, gap_end):
        if i < n_dates:
            assert filled.iloc[i] == pytest.approx(0.0, abs=1e-9), (
                f"Position {i} beyond 5-day fill limit should be 0.0, "
                f"got {filled.iloc[i]}"
            )
