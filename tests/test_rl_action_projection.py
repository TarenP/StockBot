from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import Agent
from pipeline import train as train_module
from pipeline.action_projection import (
    normalize_projection_settings,
    project_actions,
)
from pipeline.backtest import run_backtest
from pipeline.data import prepare_memory_light_training_frame
from pipeline.train import train_fold


def test_softplus_projection_preserves_sum_to_one():
    logits = torch.tensor([[0.1, -0.2, 0.3, 0.0]], dtype=torch.float32)

    weights = project_actions(logits, projection="softplus")

    assert torch.isfinite(weights).all()
    assert weights.sum(dim=-1).item() == pytest.approx(1.0)


def test_softmax_projection_temperature_sharpens_weights():
    logits = torch.tensor([[0.0, 1.0, 2.0, 0.0]], dtype=torch.float32)

    smooth = project_actions(logits, projection="softmax", temperature=1.0)
    sharp = project_actions(logits, projection="softmax", temperature=0.25)

    assert sharp.max().item() > smooth.max().item()
    assert sharp.sum().item() == pytest.approx(1.0)


def test_top_k_softmax_limits_nonzero_positions():
    logits = torch.tensor([[0.0, 4.0, 3.0, 2.0, 1.0, 0.5]], dtype=torch.float32)

    weights = project_actions(logits, projection="top_k_softmax", top_k=2)
    asset_weights = weights[0, :-1]

    assert int((asset_weights > 1e-8).sum().item()) == 2


def test_top_k_softmax_preserves_sum_to_one():
    logits = torch.randn(3, 8)

    weights = project_actions(logits, projection="top_k_softmax", top_k=3, temperature=0.5)

    assert torch.isfinite(weights).all()
    assert torch.allclose(weights.sum(dim=-1), torch.ones(3), atol=1e-6)


def test_rank_top_k_limits_nonzero_positions_if_implemented():
    logits = torch.tensor([[0.0, 4.0, 3.0, 2.0, 1.0, 0.5]], dtype=torch.float32)

    weights = project_actions(logits, projection="rank_top_k", top_k=3)
    asset_weights = weights[0, :-1]

    assert int((asset_weights > 1e-8).sum().item()) == 3
    assert weights.sum().item() == pytest.approx(1.0)


def test_projection_handles_nan_or_inf_actions_safely():
    logits = torch.tensor([[float("nan"), float("inf"), -float("inf"), 0.0]], dtype=torch.float32)

    weights = project_actions(logits, projection="top_k_softmax", top_k=2)

    assert torch.isfinite(weights).all()
    assert weights.sum().item() == pytest.approx(1.0)


def test_projection_rejects_invalid_temperature():
    with pytest.raises(ValueError):
        normalize_projection_settings(projection="softmax", temperature=0.0)


def test_projection_rejects_invalid_top_k():
    with pytest.raises(ValueError):
        normalize_projection_settings(projection="top_k_softmax", top_k=0)


def _feature_frame(dates, tickers=("AAA", "BBB")):
    index = pd.MultiIndex.from_product([pd.to_datetime(dates), tickers], names=["date", "ticker"])
    return pd.DataFrame(
        {
            "ret_1d": np.linspace(0.001, 0.01, len(index), dtype=np.float32),
            "close": np.linspace(100.0, 120.0, len(index), dtype=np.float32),
        },
        index=index,
    )


def test_projection_metadata_written_to_checkpoint(monkeypatch):
    save_dir = Path("tests/_tmp/projection_metadata")
    save_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2024-01-01", periods=25, freq="B")
    df = prepare_memory_light_training_frame(_feature_frame(dates), ["ret_1d"])
    monkeypatch.setattr(
        train_module,
        "evaluate_diagnostics",
        lambda *_args, **_kwargs: {
            "sharpe": 0.5,
            "total_return": 0.02,
            "benchmark_return": 0.01,
            "max_drawdown": -0.03,
            "turnover": 0.4,
            "mean_return": 0.001,
            "std_return": 0.01,
            "n_returns": 3,
        },
    )

    ckpt_path, _ = train_fold(
        df_train=df,
        df_val=df,
        asset_list=["AAA", "BBB"],
        fold_idx=0,
        cfg={
            **train_module.PPO_CFG,
            "total_steps": 1,
            "rollout_steps": 1,
            "ppo_epochs": 0,
            "save_every": 1,
        },
        save_dir=str(save_dir),
        device=torch.device("cpu"),
        training_mode="memory_light",
        force_restart=True,
        projection_settings={
            "rl_action_projection": "top_k_softmax",
            "rl_action_temperature": 0.5,
            "rl_action_top_k": 1,
        },
    )

    meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert meta["rl_action_projection"] == "top_k_softmax"
    assert meta["rl_action_temperature"] == pytest.approx(0.5)
    assert meta["rl_action_top_k"] == 1


class _CaptureProjectionModel:
    def __init__(self):
        self.calls = []

    def eval(self):
        return self

    def get_weights(self, obs, **kwargs):
        self.calls.append(kwargs)
        n_assets = obs.shape[2]
        weights = torch.zeros((1, n_assets + 1), dtype=torch.float32, device=obs.device)
        weights[:, 0] = 1.0
        return weights


def test_predict_and_backtest_use_same_projection_config(monkeypatch):
    args = SimpleNamespace(
        rl_action_projection="top_k_softmax",
        rl_action_temperature=0.5,
        rl_action_top_k=50,
    )
    settings = Agent._resolve_rl_action_projection(args, checkpoint_meta={})
    assert settings == {
        "rl_action_projection": "top_k_softmax",
        "rl_action_temperature": 0.5,
        "rl_action_top_k": 50,
    }

    dates = pd.date_range("2024-01-01", periods=28, freq="B")
    df = _feature_frame(dates)
    model = _CaptureProjectionModel()
    monkeypatch.setattr("pipeline.backtest.fetch_spy_returns", lambda *_, **__: None)
    monkeypatch.setattr("pipeline.backtest.plot_benchmark", lambda *_, **__: None)

    run_backtest(
        model=model,
        df_test=df,
        asset_list=["AAA", "BBB"],
        device=torch.device("cpu"),
        spy_rets=np.zeros(8, dtype=np.float32),
        ckpt_n_features=1,
        projection_settings=settings,
    )

    assert model.calls
    assert model.calls[0]["projection"] == "top_k_softmax"
    assert model.calls[0]["temperature"] == pytest.approx(0.5)
    assert model.calls[0]["top_k"] == 50
