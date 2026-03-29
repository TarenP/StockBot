import numpy as np
import pandas as pd
import torch
from pathlib import Path

from pipeline.features import FEATURE_COLS
from pipeline import screener as screener_module


def test_build_samples_uses_raw_close_forward_returns():
    n_tickers = 12
    n_dates = screener_module.LOOKBACK + screener_module.FORWARD_DAYS + 1

    feat_arr = np.zeros((n_dates, n_tickers, len(FEATURE_COLS)), dtype=np.float32)
    close_arr = np.full((n_dates, n_tickers), 100.0, dtype=np.float32)
    present_mask = np.ones((n_dates, n_tickers), dtype=bool)

    anchor_idx = screener_module.LOOKBACK - 1
    future_idx = screener_module.LOOKBACK + screener_module.FORWARD_DAYS - 1
    close_arr[anchor_idx, :] = 100.0
    close_arr[future_idx, :] = np.linspace(101.0, 130.0, n_tickers)

    X, y, forward_returns, groups = screener_module._build_samples(
        feat_arr=feat_arr,
        close_arr=close_arr,
        present_mask=present_mask,
        date_indices=[screener_module.LOOKBACK],
        forward_days=screener_module.FORWARD_DAYS,
        top_pct=screener_module.TOP_PCT,
        rng=np.random.default_rng(42),
        max_t_per_date=None,
    )

    assert len(X) == n_tickers
    assert len(groups) == n_tickers
    assert int(y.sum()) >= 1
    assert y[np.argmax(forward_returns)] == 1.0


def test_evaluate_ranked_groups_scores_shortlist_quality():
    probs = np.array([0.1, 0.9, 0.2, 0.8], dtype=np.float32)
    labels = np.array([0, 1, 0, 1], dtype=np.float32)
    returns = np.array([0.01, 0.08, -0.02, 0.06], dtype=np.float32)
    groups = np.array([0, 0, 1, 1], dtype=np.int32)

    metrics = screener_module._evaluate_ranked_groups(
        probs=probs,
        labels=labels,
        forward_returns=returns,
        groups=groups,
        shortlist_size=1,
    )

    assert metrics["precision_at_k"] == 1.0
    assert metrics["recall_at_k"] == 1.0
    assert metrics["mean_return_at_k"] > metrics["baseline_return"]
    assert metrics["lift_at_k"] > 1.0


def test_run_screener_applies_filters_and_history_gate(monkeypatch):
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return x[:, -1, 0:1]

    ckpt_path = Path("models") / "test_screener_unit.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path.write_bytes(b"ok")
    monkeypatch.setattr(screener_module, "SCREENER_CKPT", str(ckpt_path))
    monkeypatch.setattr(screener_module, "load_screener", lambda device: DummyModel())

    dates = pd.date_range("2024-01-01", periods=screener_module.LOOKBACK, freq="D")
    tickers = ["KEEP", "LOWP", "LOWV"]
    rows = []
    for date in dates:
        for ticker in tickers:
            row = {col: 0.0 for col in FEATURE_COLS}
            row["ret_5d"] = {"KEEP": 3.0, "LOWP": 2.0, "LOWV": 1.0}[ticker]
            row["close"] = {"KEEP": 12.0, "LOWP": 1.0, "LOWV": 10.0}[ticker]
            row["volume"] = {"KEEP": 50_000, "LOWP": 60_000, "LOWV": 1_000}[ticker]
            rows.append((date, ticker, row))

    short_history_dates = dates[-int(screener_module.LOOKBACK * 0.5):]
    for date in short_history_dates:
        row = {col: 0.0 for col in FEATURE_COLS}
        row["ret_5d"] = 4.0
        row["close"] = 15.0
        row["volume"] = 80_000
        rows.append((date, "SHORT", row))

    index = pd.MultiIndex.from_tuples(
        [(date, ticker) for date, ticker, _ in rows],
        names=["date", "ticker"],
    )
    df = pd.DataFrame([row for _, _, row in rows], index=index)

    results = screener_module.run_screener(
        df=df,
        device=torch.device("cpu"),
        top_n=10,
        min_price=5.0,
        min_volume=10_000,
    )

    assert results["ticker"].tolist() == ["KEEP"]
