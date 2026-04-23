from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

import Agent
from pipeline import sentiment
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


def _workspace_csv_path(prefix: str) -> Path:
    return Path.cwd() / f".{prefix}_{uuid4().hex}.csv"


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


def test_get_live_universe_uses_local_sources_before_bootstrap(monkeypatch):
    parquet_path = _workspace_parquet_path()
    watchlist_path = _workspace_csv_path("test_watchlist")

    try:
        pd.DataFrame({"ticker": ["BBB", "AAA"]}).to_parquet(parquet_path, index=False)
        pd.DataFrame({"ticker": ["CCC"]}).to_csv(watchlist_path, index=False)

        monkeypatch.setattr(updater, "PARQUET_PATH", parquet_path)
        monkeypatch.setattr(updater, "WATCHLIST_PATH", watchlist_path)
        monkeypatch.setattr(updater, "_load_trained_universe", lambda save_dir: ["AAA", "DDD"])
        monkeypatch.setattr(
            updater,
            "_bootstrap_universe",
            lambda target_size: pytest.fail("bootstrap should not run when local sources are sufficient"),
        )

        resolved = updater.get_live_universe(save_dir="models")

        assert resolved == ["AAA", "DDD", "BBB", "CCC"]
    finally:
        parquet_path.unlink(missing_ok=True)
        watchlist_path.unlink(missing_ok=True)


def test_update_sentiment_expands_to_live_universe(monkeypatch):
    sentiment_path = _workspace_csv_path("test_sentiment")
    calls: dict[str, object] = {}

    try:
        monkeypatch.setattr(sentiment, "SENTIMENT_PATH", sentiment_path)
        monkeypatch.setattr(
            updater,
            "get_live_universe",
            lambda preferred, save_dir, target_size: ["AAA", "BBB", "CCC"],
        )

        def fake_fetch_and_score(tickers: list[str], lookback_days: int = 7) -> pd.DataFrame:
            calls["tickers"] = list(tickers)
            calls["lookback_days"] = lookback_days
            return pd.DataFrame(
                [
                    {
                        "title": "Headline",
                        "date": "2024-01-01",
                        "stock": "AAA",
                        "neg_score": 0.1,
                        "neutral_score": 0.2,
                        "pos_score": 0.7,
                        "sentiment": "positive",
                    }
                ]
            )

        monkeypatch.setattr(sentiment, "fetch_and_score", fake_fetch_and_score)

        n_new = sentiment.update_sentiment(["AAA"], lookback_days=3, save_dir="custom-models")

        assert n_new == 1
        assert calls["tickers"] == ["AAA", "BBB", "CCC"]
        assert calls["lookback_days"] == 3

        saved = pd.read_csv(sentiment_path)
        assert saved["stock"].tolist() == ["AAA"]
    finally:
        sentiment_path.unlink(missing_ok=True)
