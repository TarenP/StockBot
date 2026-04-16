from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

import Agent
from pipeline import updater


def _price_rows(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for i, ticker in enumerate(tickers):
        rows.append(
            {
                "date": pd.Timestamp("2024-01-02") + pd.Timedelta(days=i),
                "open": 100.0 + i,
                "high": 101.0 + i,
                "low": 99.0 + i,
                "close": 100.5 + i,
                "volume": 1_000_000 + i,
                "ticker": ticker,
            }
        )
    return pd.DataFrame(rows)


def _workspace_parquet_path() -> Path:
    return Path.cwd() / f".test_updater_bootstrap_{uuid4().hex}.parquet"


def test_update_parquet_bootstraps_initial_universe(monkeypatch):
    parquet_path = _workspace_parquet_path()
    calls: dict[str, object] = {}

    try:
        monkeypatch.setattr(updater, "PARQUET_PATH", parquet_path)
        monkeypatch.setattr(updater, "_load_trained_universe", lambda save_dir: None)

        def fake_bootstrap(size: int) -> list[str]:
            calls["bootstrap_size"] = size
            return ["AAA", "BBB"]

        def fake_fetch(tickers: list[str], start: str, end: str) -> pd.DataFrame:
            calls["tickers"] = list(tickers)
            return _price_rows(list(tickers))

        monkeypatch.setattr(updater, "_bootstrap_universe", fake_bootstrap)
        monkeypatch.setattr(updater, "_fetch_yfinance", fake_fetch)

        n_new = updater.update_parquet(
            save_dir="models",
            bootstrap_universe_size=123,
        )

        assert n_new == 2
        assert calls["bootstrap_size"] == 123
        assert calls["tickers"] == ["AAA", "BBB"]
        assert parquet_path.exists()

        saved = pd.read_parquet(parquet_path)
        assert sorted(saved["ticker"].tolist()) == ["AAA", "BBB"]
    finally:
        parquet_path.unlink(missing_ok=True)


def test_update_parquet_raises_if_bootstrap_finds_no_tickers(monkeypatch):
    parquet_path = _workspace_parquet_path()
    try:
        monkeypatch.setattr(updater, "PARQUET_PATH", parquet_path)
        monkeypatch.setattr(updater, "_load_trained_universe", lambda save_dir: None)
        monkeypatch.setattr(updater, "_bootstrap_universe", lambda size: [])

        with pytest.raises(ValueError, match="Could not bootstrap an initial ticker universe"):
            updater.update_parquet(save_dir="models", bootstrap_universe_size=50)
    finally:
        parquet_path.unlink(missing_ok=True)


def test_ensure_price_data_bootstraps_missing_parquet(monkeypatch):
    parquet_path = _workspace_parquet_path()
    calls: dict[str, object] = {}

    try:
        monkeypatch.setattr(updater, "PARQUET_PATH", parquet_path)

        def fake_update_parquet(**kwargs) -> int:
            calls.update(kwargs)
            _price_rows(["AAA"]).set_index("date").to_parquet(parquet_path)
            return 1

        monkeypatch.setattr(updater, "update_parquet", fake_update_parquet)

        Agent._ensure_price_data(top_n=500, save_dir="models")

        assert calls["save_dir"] == "models"
        assert calls["force_full_refresh"] is True
        assert calls["bootstrap_universe_size"] == 1500
        assert parquet_path.exists()
    finally:
        parquet_path.unlink(missing_ok=True)
