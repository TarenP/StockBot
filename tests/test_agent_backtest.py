from types import SimpleNamespace

import pandas as pd

import Agent as agent_module


def test_run_backtest_mode_uses_rl_policy_path(monkeypatch):
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    index = pd.MultiIndex.from_product([dates, ["AAA"]], names=["date", "ticker"])
    df = pd.DataFrame({"ret_1d": 0.0, "close": 10.0}, index=index)
    captured = {}

    monkeypatch.setattr(agent_module, "_best_checkpoint", lambda save_dir: "models/best_fold0.pt")
    monkeypatch.setattr(agent_module, "_load_data_and_universe", lambda top_n, include_raw_cols=False: (df, ["AAA"]))

    import torch
    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: {"top_n": 500, "fold": 0, "asset_list": ["AAA"]},
    )

    import pipeline.data as data_module
    monkeypatch.setattr(data_module, "walk_forward_split", lambda *args, **kwargs: [{"test": df}])

    import pipeline.backtest as backtest_module
    monkeypatch.setattr(backtest_module, "load_model", lambda ckpt_path, device: ("model", 1))
    monkeypatch.setattr(
        backtest_module,
        "run_backtest",
        lambda **kwargs: captured.update(kwargs) or {"policy": {}},
    )

    import broker.replay as replay_module
    monkeypatch.setattr(
        replay_module,
        "run_replay",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("broker replay should not run")),
    )

    args = SimpleNamespace(checkpoint=None, save_dir="models", top_n=None)
    agent_module.run_backtest_mode(args)

    assert captured["model"] == "model"
    assert captured["asset_list"] == ["AAA"]
    assert captured["save_plot"] == "plots/backtest.png"
    assert captured["ckpt_n_features"] == 1
