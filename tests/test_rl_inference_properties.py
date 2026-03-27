"""
Property-based tests for pipeline/rl_inference.py

Uses hypothesis to verify universal correctness properties of get_rl_targets().
A fake/mock PortfolioTransformer is used so tests do not require real checkpoints.

Properties tested:
  1.1 Determinism          — identical inputs → identical outputs
  1.2 mode="weights" sum   — weights always sum to 1.0 ± 1e-5
  1.3 mode="rank" bounds   — all rl_score values in [0, 1]
  1.4 z-score idempotency  — normalising twice == normalising once
"""

import os
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from pipeline.features import FEATURE_COLS
from pipeline.rl_inference import (
    ModelNotAvailableError,
    _zscore_df,
    get_rl_targets,
)
from pipeline.model import PortfolioTransformer


# ── Helpers ───────────────────────────────────────────────────────────────────

N_FEATURES = len(FEATURE_COLS)


def _make_df_recent(
    asset_list: list[str],
    n_dates: int,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a synthetic MultiIndex [date, ticker] DataFrame with FEATURE_COLS."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_dates, freq="B")
    rows = []
    for date in dates:
        for ticker in asset_list:
            row = rng.standard_normal(N_FEATURES)
            rows.append((date, ticker, *row))
    cols = ["date", "ticker"] + FEATURE_COLS
    df = pd.DataFrame(rows, columns=cols)
    df = df.set_index(["date", "ticker"])
    return df


def _make_checkpoint(
    asset_list: list[str],
    lookback: int = 20,
    tmp_dir: str | None = None,
) -> str:
    """
    Create a minimal valid checkpoint file for a PortfolioTransformer.
    Returns the path to the saved .pt file.
    """
    n_assets = len(asset_list)
    model_cfg = dict(
        n_assets=n_assets,
        n_features=N_FEATURES,
        lookback=lookback,
        d_model=16,
        nhead_temporal=2,
        nhead_cross=2,
        num_temporal_layers=1,
        num_cross_layers=1,
        dropout=0.0,
    )
    model = PortfolioTransformer(**model_cfg)
    model.eval()

    ckpt = {
        "model_cfg": model_cfg,
        "model_state": model.state_dict(),
        "top_n": n_assets,
        "val_sharpe": 0.5,
    }

    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp()
    path = os.path.join(tmp_dir, "test_checkpoint.pt")
    torch.save(ckpt, path)
    return path


# ── Strategies ────────────────────────────────────────────────────────────────

# Generate a small list of unique ticker strings
ticker_strategy = st.lists(
    st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ", min_size=1, max_size=4),
    min_size=2,
    max_size=8,
    unique=True,
)

# Number of dates to generate (must be >= lookback=20 for full history)
n_dates_strategy = st.integers(min_value=20, max_value=30)

# Seed for reproducibility within a single test run
seed_strategy = st.integers(min_value=0, max_value=9999)


# ── 1.1 Determinism ──────────────────────────────────────────────────────────

@settings(
    max_examples=20,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
@given(
    asset_list=ticker_strategy,
    n_dates=n_dates_strategy,
    seed=seed_strategy,
)
def test_determinism_rank_mode(asset_list, n_dates, seed):
    """
    **Property: Identical inputs produce identical outputs (deterministic inference)**
    **Validates: Requirements 1.4**

    Calling get_rl_targets twice with the same df_recent, asset_list, and
    checkpoint must return identical pd.Series values.
    """
    # Clear cache to ensure fresh load each time
    import pipeline.rl_inference as rl_mod
    rl_mod._MODEL_CACHE.clear()

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = _make_checkpoint(asset_list, lookback=20, tmp_dir=tmp_dir)
        df = _make_df_recent(asset_list, n_dates, seed=seed)

        result1 = get_rl_targets(
            df, asset_list, ckpt_path, mode="rank",
            device=torch.device("cpu"), lookback=20,
        )
        result2 = get_rl_targets(
            df, asset_list, ckpt_path, mode="rank",
            device=torch.device("cpu"), lookback=20,
        )

    pd.testing.assert_series_equal(result1, result2)


@settings(
    max_examples=20,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
@given(
    asset_list=ticker_strategy,
    n_dates=n_dates_strategy,
    seed=seed_strategy,
)
def test_determinism_weights_mode(asset_list, n_dates, seed):
    """
    **Property: Identical inputs produce identical outputs (deterministic inference)**
    **Validates: Requirements 1.4**

    Same as above but for mode="weights".
    """
    import pipeline.rl_inference as rl_mod
    rl_mod._MODEL_CACHE.clear()

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = _make_checkpoint(asset_list, lookback=20, tmp_dir=tmp_dir)
        df = _make_df_recent(asset_list, n_dates, seed=seed)

        result1 = get_rl_targets(
            df, asset_list, ckpt_path, mode="weights",
            device=torch.device("cpu"), lookback=20,
        )
        result2 = get_rl_targets(
            df, asset_list, ckpt_path, mode="weights",
            device=torch.device("cpu"), lookback=20,
        )

    pd.testing.assert_series_equal(result1, result2)


# ── 1.2 mode="weights" sum invariant ─────────────────────────────────────────

@settings(
    max_examples=25,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
@given(
    asset_list=ticker_strategy,
    n_dates=n_dates_strategy,
    seed=seed_strategy,
)
def test_weights_sum_to_one(asset_list, n_dates, seed):
    """
    **Property: RL weight series always sums to 1.0 ± 1e-5 for any valid input**
    **Validates: Requirements 1.3**

    For mode="weights", the returned Series (including CASH) must sum to 1.0
    within a tolerance of 1e-5.
    """
    import pipeline.rl_inference as rl_mod
    rl_mod._MODEL_CACHE.clear()

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = _make_checkpoint(asset_list, lookback=20, tmp_dir=tmp_dir)
        df = _make_df_recent(asset_list, n_dates, seed=seed)

        result = get_rl_targets(
            df, asset_list, ckpt_path, mode="weights",
            device=torch.device("cpu"), lookback=20,
        )

    assert abs(result.sum() - 1.0) <= 1e-5, (
        f"Weights sum to {result.sum():.8f}, expected 1.0 ± 1e-5. "
        f"asset_list={asset_list}"
    )


# ── 1.3 mode="rank" score bounds ─────────────────────────────────────────────

@settings(
    max_examples=25,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
@given(
    asset_list=ticker_strategy,
    n_dates=n_dates_strategy,
    seed=seed_strategy,
)
def test_rank_scores_in_unit_interval(asset_list, n_dates, seed):
    """
    **Property: All rl_score values are in [0, 1] for any valid asset_list**
    **Validates: Requirements 1.2**

    For mode="rank", every value in the returned Series must be in [0.0, 1.0].
    """
    import pipeline.rl_inference as rl_mod
    rl_mod._MODEL_CACHE.clear()

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = _make_checkpoint(asset_list, lookback=20, tmp_dir=tmp_dir)
        df = _make_df_recent(asset_list, n_dates, seed=seed)

        result = get_rl_targets(
            df, asset_list, ckpt_path, mode="rank",
            device=torch.device("cpu"), lookback=20,
        )

    assert (result >= 0.0).all(), (
        f"Some rl_scores are negative: {result[result < 0.0]}"
    )
    assert (result <= 1.0).all(), (
        f"Some rl_scores exceed 1.0: {result[result > 1.0]}"
    )
    assert list(result.index) == asset_list, (
        f"Index mismatch: {list(result.index)} != {asset_list}"
    )


# ── 1.4 z-score idempotency ───────────────────────────────────────────────────

@settings(
    max_examples=30,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
@given(
    n_tickers=st.integers(min_value=2, max_value=10),
    n_dates=st.integers(min_value=3, max_value=15),
    seed=seed_strategy,
)
def test_zscore_idempotency(n_tickers, n_dates, seed):
    """
    **Property: Applying the normalisation pipeline twice produces the same output as once**
    **Validates: Requirements 11.5**

    _zscore_df followed by clip(-5, 5) applied twice must equal applying it once.
    This holds because after the first application, the data is already
    cross-sectionally standardised; a second z-score of already-standardised
    data (with clipping) produces the same result.
    """
    rng = np.random.default_rng(seed)
    tickers = [f"T{i}" for i in range(n_tickers)]
    dates = pd.date_range("2023-01-01", periods=n_dates, freq="B")

    rows = []
    for date in dates:
        for ticker in tickers:
            rows.append((date, ticker, *rng.standard_normal(N_FEATURES)))
    cols = ["date", "ticker"] + FEATURE_COLS
    df = pd.DataFrame(rows, columns=cols).set_index(["date", "ticker"])

    # Apply once
    once = _zscore_df(df).clip(-5.0, 5.0).fillna(0.0)
    # Apply twice
    twice = _zscore_df(once).clip(-5.0, 5.0).fillna(0.0)

    pd.testing.assert_frame_equal(
        once, twice,
        check_exact=False,
        atol=1e-4,
        rtol=1e-4,
    )


# ── Additional: ModelNotAvailableError on missing checkpoint ──────────────────

def test_missing_checkpoint_raises():
    """Raises ModelNotAvailableError when checkpoint path does not exist."""
    import pipeline.rl_inference as rl_mod
    rl_mod._MODEL_CACHE.clear()

    df = _make_df_recent(["AAPL", "MSFT"], n_dates=25)
    with pytest.raises(ModelNotAvailableError, match="not found"):
        get_rl_targets(df, ["AAPL", "MSFT"], "/nonexistent/path.pt", mode="rank")


def test_insufficient_history_assigns_zero():
    """Tickers with fewer than lookback dates get rl_score=0.0."""
    import pipeline.rl_inference as rl_mod
    rl_mod._MODEL_CACHE.clear()

    asset_list = ["AAPL", "MSFT"]
    # Only 5 dates — less than lookback=20
    df = _make_df_recent(asset_list, n_dates=5, seed=42)

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = _make_checkpoint(asset_list, lookback=20, tmp_dir=tmp_dir)
        result = get_rl_targets(
            df, asset_list, ckpt_path, mode="rank",
            device=torch.device("cpu"), lookback=20,
        )

    assert (result == 0.0).all(), (
        f"Expected all zeros for insufficient history, got: {result}"
    )
