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


def test_update_parquet_uses_configured_universe_when_universe_missing(monkeypatch):
    parquet_path = _workspace_parquet_path()
    calls: dict[str, object] = {}

    try:
        monkeypatch.setattr(updater, "PARQUET_PATH", parquet_path)
        monkeypatch.setattr(
            updater,
            "resolve_configured_universe",
            lambda **kwargs: ["AAA", "BBB"],
        )
        monkeypatch.setattr(updater, "prune_stale_tickers", lambda df: (df, []))

        def fake_fetch(tickers: list[str], start: str, end: str) -> pd.DataFrame:
            calls["tickers"] = list(tickers)
            calls["start"] = start
            calls["end"] = end
            return _price_rows(list(tickers))

        monkeypatch.setattr(updater, "_fetch_yfinance", fake_fetch)

        n_new = updater.update_parquet(
            save_dir="models",
            config={"universe_mode": "sp500"},
        )

        assert n_new == 3
        assert calls["tickers"] == ["AAA", "BBB", "SPY"]
        assert parquet_path.exists()

        saved = pd.read_parquet(parquet_path)
        assert sorted(saved["ticker"].tolist()) == ["AAA", "BBB", "SPY"]
    finally:
        parquet_path.unlink(missing_ok=True)


def test_update_parquet_raises_if_configured_universe_is_empty(monkeypatch):
    parquet_path = _workspace_parquet_path()
    try:
        monkeypatch.setattr(updater, "PARQUET_PATH", parquet_path)
        monkeypatch.setattr(updater, "resolve_configured_universe", lambda **kwargs: [])

        with pytest.raises(ValueError, match="resolved to zero tickers"):
            updater.update_parquet(save_dir="models", config={"universe_mode": "sp500"})
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


def test_get_live_universe_uses_resolved_configured_membership(monkeypatch):
    parquet_path = _workspace_parquet_path()
    watchlist_path = _workspace_csv_path("test_watchlist")

    try:
        pd.DataFrame({"ticker": ["BBB", "AAA"]}).to_parquet(parquet_path, index=False)
        pd.DataFrame({"ticker": ["CCC"]}).to_csv(watchlist_path, index=False)

        monkeypatch.setattr(updater, "PARQUET_PATH", parquet_path)
        monkeypatch.setattr(updater, "WATCHLIST_PATH", watchlist_path)
        monkeypatch.setattr(
            updater,
            "resolve_configured_universe",
            lambda **kwargs: ["AAA", "DDD", "BBB"],
        )
        monkeypatch.setattr(
            updater,
            "constrain_to_configured_universe",
            lambda tickers, **kwargs: [ticker for ticker in (tickers or []) if ticker in {"AAA", "DDD", "BBB"}],
        )

        resolved = updater.get_live_universe(
            preferred=["DDD", "ZZZ"],
            save_dir="models",
            config={"universe_mode": "sp500"},
        )

        assert resolved == ["DDD", "AAA", "BBB"]
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
            lambda **kwargs: ["AAA", "BBB", "CCC"],
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


def test_sentiment_requests_session_ignores_env_proxy(monkeypatch):
    monkeypatch.setattr(sentiment, "_requests_session", None)

    session = sentiment._get_requests_session()

    assert session.trust_env is False


def test_score_headlines_uses_fallback_when_finbert_unavailable(monkeypatch):
    monkeypatch.setattr(
        sentiment,
        "_get_finbert",
        lambda: (_ for _ in ()).throw(RuntimeError("offline")),
    )

    scores = sentiment._score_headlines(["Company wins big contract"])

    assert len(scores) == 1
    assert scores[0]["pos_score"] > scores[0]["neg_score"]
    assert scores[0]["sentiment"] == "positive"


def test_update_sentiment_appends_to_legacy_csv_schema(monkeypatch):
    sentiment_path = _workspace_csv_path("test_legacy_sentiment")

    try:
        pd.DataFrame(
            [
                {
                    "date": "2024-01-01",
                    "stock": "AAA",
                    "neg_score": 0.2,
                    "neutral_score": 0.3,
                    "pos_score": 0.5,
                }
            ]
        ).to_csv(sentiment_path, index=False)
        monkeypatch.setattr(sentiment, "SENTIMENT_PATH", sentiment_path)

        def fake_fetch_and_score(tickers: list[str], lookback_days: int = 7) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {
                        "title": "Fresh headline",
                        "date": "2024-01-02",
                        "stock": "AAA",
                        "neg_score": 0.1,
                        "neutral_score": 0.2,
                        "pos_score": 0.7,
                        "sentiment": "positive",
                    }
                ]
            )

        monkeypatch.setattr(sentiment, "fetch_and_score", fake_fetch_and_score)

        n_new = sentiment.update_sentiment(
            ["AAA"],
            lookback_days=3,
            expand_live_universe=False,
        )

        saved = pd.read_csv(sentiment_path)
        assert n_new == 1
        assert len(saved) == 2
        assert saved.columns.tolist() == [
            "date",
            "stock",
            "neg_score",
            "neutral_score",
            "pos_score",
        ]
        assert saved.iloc[-1]["date"] == "2024-01-02"
        assert saved.iloc[-1]["stock"] == "AAA"
    finally:
        sentiment_path.unlink(missing_ok=True)


def test_update_parquet_keeps_benchmark_symbol_in_price_universe(monkeypatch):
    parquet_path = _workspace_parquet_path()
    calls: dict[str, object] = {}

    try:
        monkeypatch.setattr(updater, "PARQUET_PATH", parquet_path)
        monkeypatch.setattr(
            updater,
            "get_live_universe",
            lambda **kwargs: ["AAA", "BBB"],
        )
        monkeypatch.setattr(updater, "prune_stale_tickers", lambda df, **kwargs: (df, []))

        def fake_fetch(tickers: list[str], start: str, end: str) -> pd.DataFrame:
            calls["tickers"] = list(tickers)
            return _price_rows(list(tickers))

        monkeypatch.setattr(updater, "_fetch_yfinance", fake_fetch)

        updater.update_parquet(
            save_dir="models",
            config={"universe_mode": "tradable_us", "benchmark_symbols": "SPY"},
        )

        assert "SPY" in calls["tickers"]
        assert {"AAA", "BBB"} <= set(calls["tickers"])
    finally:
        parquet_path.unlink(missing_ok=True)


def test_get_live_universe_can_freeze_to_snapshot(monkeypatch):
    snapshot_path = _workspace_csv_path("test_universe_snapshot").with_suffix(".json")

    try:
        monkeypatch.setattr(updater, "_load_trained_universe", lambda save_dir="models": ["AAA", "BBB"])
        monkeypatch.setattr(updater, "_load_parquet_universe", lambda max_stale_days=30: ["CCC"])
        monkeypatch.setattr(updater, "_bootstrap_universe", lambda target_size=1500: ["DDD"])

        resolved = updater.get_live_universe(
            save_dir="models",
            config={
                "universe_mode": "tradable_us",
                "freeze_universe_snapshot": True,
                "universe_snapshot_path": str(snapshot_path),
                "min_broad_universe_size": 2,
            },
        )

        assert resolved[:3] == ["AAA", "BBB", "CCC"]
        assert snapshot_path.exists()

        monkeypatch.setattr(updater, "_load_trained_universe", lambda save_dir="models": ["ZZZ"])
        frozen = updater.get_live_universe(
            save_dir="models",
            config={
                "universe_mode": "tradable_us",
                "freeze_universe_snapshot": True,
                "universe_snapshot_path": str(snapshot_path),
                "min_broad_universe_size": 2,
            },
        )

        assert frozen == resolved
    finally:
        snapshot_path.unlink(missing_ok=True)


def test_get_live_universe_excludes_watchlist_by_default(monkeypatch):
    monkeypatch.setattr(updater, "_load_trained_universe", lambda save_dir="models": ["AAA"])
    monkeypatch.setattr(updater, "_load_parquet_universe", lambda max_stale_days=30: ["BBB"])
    monkeypatch.setattr(updater, "_load_watchlist_universe", lambda: ["WATCH"])
    monkeypatch.setattr(updater, "_bootstrap_universe", lambda target_size=1500: [])

    resolved = updater.get_live_universe(
        save_dir="models",
        config={
            "universe_mode": "tradable_us",
            "min_broad_universe_size": 2,
        },
    )

    assert "WATCH" not in resolved


def test_get_live_universe_can_include_watchlist_when_enabled(monkeypatch):
    monkeypatch.setattr(updater, "_load_trained_universe", lambda save_dir="models": ["AAA"])
    monkeypatch.setattr(updater, "_load_parquet_universe", lambda max_stale_days=30: ["BBB"])
    monkeypatch.setattr(updater, "_load_watchlist_universe", lambda: ["WATCH"])
    monkeypatch.setattr(updater, "_bootstrap_universe", lambda target_size=1500: [])

    resolved = updater.get_live_universe(
        save_dir="models",
        config={
            "universe_mode": "tradable_us",
            "include_watchlist_in_universe": True,
            "min_broad_universe_size": 3,
        },
    )

    assert "WATCH" in resolved
