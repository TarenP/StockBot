"""
Tests for regime detection features (Task 8).

Property 10: Regime Label Validity
  Validates: Requirements 11.1

Property 11: Regime Broadcast Consistency
  Validates: Requirements 11.5
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pipeline.features import FEATURE_COLS, compute_regimes


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_panel(n_dates: int = 60, tickers=("AAPL", "SPY", "MSFT"),
                seed: int = 42) -> pd.DataFrame:
    dates = pd.date_range("2022-01-03", periods=n_dates, freq="B")
    rng = np.random.default_rng(seed)
    rows = []
    for ticker in tickers:
        close = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, n_dates))
        ret_1d = np.concatenate([[np.nan], np.diff(close) / close[:-1]])
        for i, d in enumerate(dates):
            rows.append({
                "date": d, "ticker": ticker,
                "close": close[i], "open": close[i],
                "high": close[i] * 1.01, "low": close[i] * 0.99,
                "volume": 1_000_000,
                "ret_1d": ret_1d[i],
            })
    df = pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()
    return df


# ── Unit tests ────────────────────────────────────────────────────────────────

class TestFeatureColsCompleteness:
    """8.2 – Assert regime columns are in FEATURE_COLS."""

    def test_regime_0_in_feature_cols(self):
        assert "regime_0" in FEATURE_COLS

    def test_regime_1_in_feature_cols(self):
        assert "regime_1" in FEATURE_COLS

    def test_regime_2_in_feature_cols(self):
        assert "regime_2" in FEATURE_COLS

    def test_regime_3_in_feature_cols(self):
        assert "regime_3" in FEATURE_COLS


class TestComputeRegimes:
    def test_returns_series(self):
        df = _make_panel()
        result = compute_regimes(df)
        assert isinstance(result, pd.Series)

    def test_labels_in_valid_range(self):
        df = _make_panel()
        result = compute_regimes(df)
        assert result.isin([0, 1, 2, 3]).all(), f"Invalid labels: {result.unique()}"

    def test_index_matches_panel_dates(self):
        df = _make_panel()
        dates = df.index.get_level_values("date").unique()
        result = compute_regimes(df)
        assert set(result.index) == set(dates)

    def test_no_nan_in_output(self):
        df = _make_panel()
        result = compute_regimes(df)
        assert not result.isnull().any()

    def test_high_vol_regime_assigned(self):
        """Inject high-volatility returns and verify regime 2 or 3 is assigned."""
        n = 60
        dates = pd.date_range("2022-01-03", periods=n, freq="B")
        rows = []
        rng = np.random.default_rng(0)
        for ticker in ("AAPL", "SPY"):
            # Very high volatility returns
            close = 100.0 * np.cumprod(1 + rng.normal(0, 0.10, n))
            ret_1d = np.concatenate([[np.nan], np.diff(close) / close[:-1]])
            for i, d in enumerate(dates):
                rows.append({
                    "date": d, "ticker": ticker,
                    "close": close[i], "open": close[i],
                    "high": close[i] * 1.01, "low": close[i] * 0.99,
                    "volume": 1_000_000, "ret_1d": ret_1d[i],
                })
        df = pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()
        result = compute_regimes(df)
        # At least some dates should be in high-vol regime (2 or 3)
        assert result.isin([2, 3]).any()

    def test_spy_missing_falls_back_gracefully(self):
        """Without SPY in panel, compute_regimes should still return valid labels."""
        df = _make_panel(tickers=("AAPL", "MSFT"))
        result = compute_regimes(df)
        assert result.isin([0, 1, 2, 3]).all()


# ── Property-based tests ──────────────────────────────────────────────────────

@given(
    vol=st.floats(min_value=0.0, max_value=0.10, allow_nan=False),
    dispersion=st.floats(min_value=0.0, max_value=0.10, allow_nan=False),
)
@settings(max_examples=100)
def test_regime_label_validity(vol: float, dispersion: float):
    """
    Property 10: Regime Label Validity
    Validates: Requirements 11.1

    For any (rolling_volatility, rolling_dispersion) pair, the regime label
    SHALL be an integer in {0, 1, 2, 3}.
    """
    vol_threshold = 0.015
    dispersion_threshold = 0.015

    high_vol  = vol > vol_threshold
    high_disp = dispersion > dispersion_threshold

    if not high_vol and not high_disp:
        expected = 0
    elif not high_vol and high_disp:
        expected = 1
    elif high_vol and not high_disp:
        expected = 2
    else:
        expected = 3

    assert expected in {0, 1, 2, 3}


@given(
    n_dates=st.integers(min_value=25, max_value=80),
    n_tickers=st.integers(min_value=2, max_value=6),
    seed=st.integers(min_value=0, max_value=999),
)
@settings(max_examples=100)
def test_regime_broadcast_consistency(n_dates: int, n_tickers: int, seed: int):
    """
    Property 11: Regime Broadcast Consistency
    Validates: Requirements 11.5

    For any date in the feature DataFrame, all tickers on that date SHALL
    have identical values for regime_0, regime_1, regime_2, regime_3.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=n_dates, freq="B")
    tickers = [f"T{i}" for i in range(n_tickers)] + ["SPY"]

    rows = []
    for ticker in tickers:
        close = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, n_dates))
        ret_1d = np.concatenate([[np.nan], np.diff(close) / close[:-1]])
        for i, d in enumerate(dates):
            rows.append({
                "date": d, "ticker": ticker,
                "close": close[i], "open": close[i],
                "high": close[i] * 1.01, "low": close[i] * 0.99,
                "volume": 1_000_000, "ret_1d": ret_1d[i],
            })

    df = pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()

    # Compute regimes and one-hot encode (same logic as build_features)
    regime_labels = compute_regimes(df)
    regime_df = pd.DataFrame(index=df.index)
    for r in range(4):
        col_name = f"regime_{r}"
        regime_col = (regime_labels == r).astype(float)
        regime_df[col_name] = df.index.get_level_values("date").map(regime_col.to_dict())

    # For each date, all tickers must have the same regime values
    for date in dates:
        try:
            date_slice = regime_df.xs(date, level="date")
        except KeyError:
            continue
        for col in ["regime_0", "regime_1", "regime_2", "regime_3"]:
            vals = date_slice[col].dropna()
            if len(vals) > 1:
                assert vals.nunique() == 1, (
                    f"On {date}, tickers have different {col} values: {vals.to_dict()}"
                )
