from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import Agent
from pipeline import train as train_module
from pipeline.data import (
    filter_date_range,
    prepare_memory_light_training_frame,
    walk_forward_date_ranges,
)
from pipeline.train import train_fold


def _feature_frame(dates, tickers=("AAA", "BBB"), include_unused=True):
    index = pd.MultiIndex.from_product([pd.to_datetime(dates), tickers], names=["date", "ticker"])
    data = {
        "ret_1d": np.linspace(0.001, 0.01, len(index)),
        "close": np.linspace(100.0, 120.0, len(index)),
    }
    if include_unused:
        data["unused_blob"] = np.arange(len(index), dtype=np.float64)
    return pd.DataFrame(data, index=index)


def test_memory_light_flag_parses(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["Agent.py", "--mode", "train", "--memory_light_train"],
    )

    args = Agent.parse_args()

    assert args.memory_light_train is True


def test_memory_light_fold_loader_filters_date_range():
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    df = _feature_frame(dates)

    filtered = filter_date_range(df, pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-05"))

    filtered_dates = pd.DatetimeIndex(filtered.index.get_level_values("date").unique())
    assert filtered_dates.min() == pd.Timestamp("2020-01-03")
    assert filtered_dates.max() == pd.Timestamp("2020-01-05")


def test_memory_light_casts_features_to_float32():
    df = _feature_frame(pd.date_range("2024-01-01", periods=3, freq="D"))

    prepared = prepare_memory_light_training_frame(df, ["ret_1d"])

    assert prepared["ret_1d"].dtype == np.float32


def test_memory_light_drops_unused_columns():
    df = _feature_frame(pd.date_range("2024-01-01", periods=3, freq="D"))

    prepared = prepare_memory_light_training_frame(df, ["ret_1d"])

    assert list(prepared.columns) == ["ret_1d", "close"]


def test_memory_light_writes_checkpoint_metadata(monkeypatch):
    save_dir = Path("tests/_tmp/memory_light_metadata")
    save_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2024-01-01", periods=25, freq="B")
    df = prepare_memory_light_training_frame(_feature_frame(dates), ["ret_1d"])
    monkeypatch.setattr(train_module, "evaluate", lambda *_args, **_kwargs: (0.5, 0.02))

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
    )

    meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    resume = torch.load(save_dir / "resume_fold0.pt", map_location="cpu", weights_only=False)
    assert meta["fold"] == 0
    assert meta["training_mode"] == "memory_light"
    assert meta["total_steps"] == 1
    assert meta["total_steps_requested"] == 1
    assert meta["feature_cols"] == ["ret_1d"]
    assert meta["n_features"] == 1
    assert meta["asset_list"] == ["AAA", "BBB"]
    assert meta["created_at"]
    assert resume["training_mode"] == "memory_light"


def test_memory_light_fold_cleanup_called(monkeypatch):
    save_dir = Path("tests/_tmp/memory_light_cleanup")
    save_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2010-01-01", "2021-12-31", freq="B")
    df = _feature_frame(dates, tickers=("AAA",), include_unused=False)
    calls = []

    monkeypatch.setattr(Agent, "_load_data_and_universe", lambda *_, **__: (df, ["AAA"]))
    monkeypatch.setattr(Agent, "_release_memory_light_fold", lambda fold_idx: calls.append(fold_idx))
    monkeypatch.setattr(train_module, "log_memory", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_module, "fold_is_complete", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        train_module,
        "train_fold",
        lambda **_kwargs: (str(save_dir / "best_fold0.pt"), 0.1),
    )

    args = SimpleNamespace(
        save_dir=str(save_dir),
        force_retrain=True,
        seed=42,
    )
    settings = {
        "total_steps": 1,
        "folds": 1,
        "skip_screener_train": True,
    }

    Agent._run_memory_light_train(args, settings, top_n=1)

    assert -1 in calls
    assert 0 in calls


def test_memory_light_date_ranges_do_not_materialize_fold_frames():
    dates = pd.date_range("2010-01-01", "2021-12-31", freq="B")

    ranges = walk_forward_date_ranges(dates, train_years=8, val_years=1, test_years=1)

    assert ranges
    assert set(ranges[0]) >= {"train_start", "train_end", "val_start", "val_end", "load_start", "load_end"}
