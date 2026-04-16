import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from pipeline.autotuner import _split_replay_holdout
from pipeline.environment import PortfolioEnv
from pipeline.model import PortfolioTransformer


def _single_asset_frame(include_close: bool) -> pd.DataFrame:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    index = pd.MultiIndex.from_product([dates, ["AAA"]], names=["date", "ticker"])
    data = {"ret_1d": [5.0, 5.0]}
    if include_close:
        data["close"] = [100.0, 110.0]
    return pd.DataFrame(data, index=index)


def test_portfolio_env_prefers_raw_close_returns_when_available():
    env = PortfolioEnv(
        _single_asset_frame(include_close=True),
        ["AAA"],
        lookback=1,
        transaction_cost=0.0,
    )

    env.reset()
    _, _, _, _, info = env.step(np.array([1.0, 0.0], dtype=np.float32))

    assert info["port_ret"] == pytest.approx(0.10, rel=1e-6, abs=1e-6)


def test_portfolio_env_falls_back_to_feature_return_without_raw_close():
    env = PortfolioEnv(
        _single_asset_frame(include_close=False),
        ["AAA"],
        lookback=1,
        transaction_cost=0.0,
    )

    env.reset()
    _, _, _, _, info = env.step(np.array([1.0, 0.0], dtype=np.float32))

    assert info["port_ret"] == pytest.approx(5.0, rel=1e-6, abs=1e-6)


def test_portfolio_env_does_not_softmax_valid_weight_vectors():
    env = PortfolioEnv(
        _single_asset_frame(include_close=True),
        ["AAA"],
        lookback=1,
        transaction_cost=0.0,
    )

    env.reset()
    _, _, _, _, info = env.step(np.array([0.0, 1.0], dtype=np.float32))

    assert info["port_ret"] == pytest.approx(0.0, rel=1e-6, abs=1e-6)


def test_get_weights_matches_dirichlet_mean():
    torch.manual_seed(0)
    model = PortfolioTransformer(
        n_assets=3,
        n_features=2,
        lookback=4,
        d_model=16,
        nhead_temporal=2,
        nhead_cross=2,
        num_temporal_layers=1,
        num_cross_layers=1,
        dropout=0.0,
    )
    obs = torch.randn(2, 4, 3, 2)

    logits, _ = model(obs)
    expected = F.softplus(logits) + 1e-6
    expected = expected / expected.sum(dim=-1, keepdim=True)
    actual = model.get_weights(obs)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_split_replay_holdout_uses_later_dates_for_validation():
    dates = pd.date_range("2023-01-02", periods=400, freq="B")
    index = pd.MultiIndex.from_product([dates, ["AAA"]], names=["date", "ticker"])
    df = pd.DataFrame({"ret_1d": np.zeros(len(index), dtype=float)}, index=index)

    search_df, holdout_df, has_holdout = _split_replay_holdout(df, replay_years=2)

    assert has_holdout is True
    search_dates = sorted(search_df.index.get_level_values("date").unique())
    holdout_dates = sorted(holdout_df.index.get_level_values("date").unique())
    assert search_dates[-1] < holdout_dates[0]
    assert len(holdout_dates) >= 63
