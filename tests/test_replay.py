import numpy as np
import pandas as pd

import broker.replay as replay_module


def _feature_frame():
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    index = pd.MultiIndex.from_product(
        [dates, ["AAA", "BBB"]],
        names=["date", "ticker"],
    )
    df = pd.DataFrame(
        {
            "ret_1d": [0.01, 0.02, -0.01, 0.03],
            "ret_5d": [0.05, 0.01, 0.04, 0.00],
            "vol_ratio": [1.5, 1.2, 1.4, 1.1],
            "sent_net": [0.3, 0.1, 0.2, -0.1],
            "macd_hist": [0.2, 0.1, 0.3, -0.2],
        },
        index=index,
    )
    return df


def _price_lookup():
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    index = pd.MultiIndex.from_product(
        [dates, ["AAA", "BBB"]],
        names=["date", "ticker"],
    )
    return pd.DataFrame(
        {
            "close": [10.0, 20.0, 11.0, 21.0],
            "volume": [1_000_000.0, 800_000.0, 1_100_000.0, 850_000.0],
        },
        index=index,
    )


def test_run_replay_uses_broker_brain_historical_prices(monkeypatch):
    df_features = _feature_frame()
    price_lookup = _price_lookup()
    captured = {"prices": [], "risk_engines": 0}

    import broker.brain as brain_module

    def fake_run_cycle(self, df_slice, screener_top_n=100, risk_engine=None):
        captured["risk_engines"] += int(risk_engine is not None)
        captured["prices"].append(self._get_current_prices(["AAA"]).get("AAA"))
        return []

    monkeypatch.setattr(brain_module.BrokerBrain, "run_cycle", fake_run_cycle)

    returns, trade_log = replay_module.run_replay(
        df_features,
        price_lookup,
        strategy="heuristics_only",
        rebalance_freq=1,
        label="test",
    )

    assert len(returns) >= 2
    assert trade_log == []
    assert captured["risk_engines"] == 2
    assert captured["prices"] == [10.0, 11.0]


def test_run_replay_preserves_rl_entry_metadata(monkeypatch):
    df_features = _feature_frame()
    price_lookup = _price_lookup()
    captured = {"rl_entry_scores": []}

    import broker.brain as brain_module

    def fake_run_cycle(self, df_slice, screener_top_n=100, risk_engine=None):
        decision = brain_module.Decision(
            action="BUY",
            ticker="AAA",
            shares=1.0,
            price=10.0,
            score=0.9,
            reason="test buy",
        )
        decision._rl_score_at_entry = 0.87
        return [decision]

    original_execute = replay_module._execute_replay_decisions

    def capture_execute(portfolio, decisions, execution_spread, trade_log, date):
        original_execute(portfolio, decisions, execution_spread, trade_log, date)
        if "AAA" in portfolio.positions:
            captured["rl_entry_scores"].append(
                portfolio.positions["AAA"].get("rl_score_at_entry")
            )

    monkeypatch.setattr(brain_module.BrokerBrain, "run_cycle", fake_run_cycle)
    monkeypatch.setattr(replay_module, "_execute_replay_decisions", capture_execute)

    returns, trade_log = replay_module.run_replay(
        df_features,
        price_lookup,
        strategy="screener_rl",
        checkpoint_path="dummy.pt",
        rebalance_freq=1,
        label="test_rl",
    )

    assert len(returns) >= 2
    assert any(t["action"] == "BUY" for t in trade_log)
    assert captured["rl_entry_scores"][0] == 0.87


def test_historical_feature_score_handles_normalized_features():
    strong_report = {
        "ret_5d": 1.6,
        "ret_20d": 1.2,
        "macd_hist": 1.1,
        "vol_ratio": 1.0,
        "vol_zscore": 0.8,
        "price_pos_52w": 1.1,
        "sent_net": 1.4,
        "sent_surprise": 1.0,
        "sent_accel": 0.7,
        "sent_trend": 0.6,
        "rsi": 0.1,
        "bb_pct": 0.2,
        "atr": 0.3,
    }
    weak_report = {
        "ret_5d": -1.3,
        "ret_20d": -1.0,
        "macd_hist": -1.1,
        "vol_ratio": -0.8,
        "vol_zscore": -0.7,
        "price_pos_52w": -0.9,
        "sent_net": -1.2,
        "sent_surprise": -0.8,
        "sent_accel": -0.6,
        "sent_trend": -0.5,
        "rsi": 2.8,
        "bb_pct": 2.5,
        "atr": 2.7,
    }

    strong_score = replay_module._historical_feature_score(strong_report)
    weak_score = replay_module._historical_feature_score(weak_report)

    assert 0.0 <= weak_score <= 1.0
    assert 0.0 <= strong_score <= 1.0
    assert strong_score > 0.60
    assert weak_score < 0.50
    assert strong_score > weak_score


def test_make_historical_research_preserves_neutral_sentiment():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    index = pd.MultiIndex.from_product([dates, ["AAA"]], names=["date", "ticker"])
    df_features = pd.DataFrame(
        {
            "sent_net": [0.0] * 5,
            "sent_pos_raw": [0.45] * 5,
        },
        index=index,
    )
    price_lookup = pd.DataFrame(
        {
            "close": [10.0] * 5,
            "volume": [1_000_000.0] * 5,
        },
        index=index,
    )

    research = replay_module._make_historical_research(
        df_features,
        price_lookup,
        dates[-1],
    )
    report = research("AAA")

    assert report is not None
    assert report["sentiment"]["sentiment"] == "neutral"
    assert report["sentiment"]["pos_score"] == 0.45
    assert report["sentiment"]["neg_score"] == 0.45


def test_make_historical_research_requires_same_day_features():
    feature_dates = pd.date_range("2024-01-01", periods=5, freq="D")
    as_of_date = pd.Timestamp("2024-01-06")
    feature_index = pd.MultiIndex.from_product(
        [feature_dates, ["AAA"]],
        names=["date", "ticker"],
    )
    price_index = pd.MultiIndex.from_product(
        [feature_dates.append(pd.DatetimeIndex([as_of_date])), ["AAA"]],
        names=["date", "ticker"],
    )
    df_features = pd.DataFrame(
        {
            "sent_net": [0.0] * len(feature_index),
            "sent_pos_raw": [0.45] * len(feature_index),
        },
        index=feature_index,
    )
    price_lookup = pd.DataFrame(
        {
            "close": [10.0] * len(price_index),
            "volume": [1_000_000.0] * len(price_index),
        },
        index=price_index,
    )

    research = replay_module._make_historical_research(
        df_features,
        price_lookup,
        as_of_date,
    )

    assert research("AAA") is None


def test_make_historical_research_requires_same_day_quote():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    feature_index = pd.MultiIndex.from_product([dates, ["AAA"]], names=["date", "ticker"])
    quote_index = pd.MultiIndex.from_product([dates[:-1], ["AAA"]], names=["date", "ticker"])
    df_features = pd.DataFrame(
        {
            "sent_net": [0.0] * len(feature_index),
            "sent_pos_raw": [0.45] * len(feature_index),
        },
        index=feature_index,
    )
    price_lookup = pd.DataFrame(
        {
            "close": [10.0] * len(quote_index),
            "volume": [1_000_000.0] * len(quote_index),
        },
        index=quote_index,
    )

    research = replay_module._make_historical_research(
        df_features,
        price_lookup,
        dates[-1],
    )

    assert research("AAA") is None


def test_make_historical_research_rejects_subcent_quotes():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    index = pd.MultiIndex.from_product([dates, ["AAA"]], names=["date", "ticker"])
    df_features = pd.DataFrame(
        {
            "sent_net": [0.0] * len(index),
            "sent_pos_raw": [0.45] * len(index),
        },
        index=index,
    )
    price_lookup = pd.DataFrame(
        {
            "close": [10.0, 10.0, 10.0, 10.0, 0.0001],
            "volume": [1_000_000.0] * len(index),
        },
        index=index,
    )

    research = replay_module._make_historical_research(
        df_features,
        price_lookup,
        dates[-1],
    )

    assert research("AAA") is None


def test_adjust_replay_close_series_back_adjusts_split_like_jumps():
    close = pd.Series(
        [1.0, 4.0, 8.0, 8.0],
        index=pd.date_range("2024-01-01", periods=4, freq="D"),
        name="close",
    )

    adjusted = replay_module._adjust_replay_close_series(close, split_ratio_threshold=2.0)

    assert adjusted.tolist() == [8.0, 8.0, 8.0, 8.0]


def test_make_historical_research_skips_recent_corporate_actions():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    index = pd.MultiIndex.from_product([dates, ["AAA"]], names=["date", "ticker"])
    df_features = pd.DataFrame(
        {
            "sent_net": [0.0] * len(index),
            "sent_pos_raw": [0.45] * len(index),
        },
        index=index,
    )
    price_lookup = pd.DataFrame(
        {
            "close": [8.0, 8.0, 8.0, 8.0, 4.0],
            "close_raw": [1.0, 1.0, 1.0, 4.0, 4.0],
            "volume": [1_000_000.0] * len(index),
        },
        index=index,
    )

    research = replay_module._make_historical_research(
        df_features,
        price_lookup,
        dates[-1],
    )

    assert research("AAA") is None


def test_run_full_replay_uses_live_config_parameters(monkeypatch):
    df_features = _feature_frame()
    captured = {}

    monkeypatch.setattr(replay_module, "_build_price_lookup", lambda: _price_lookup())

    def fake_run_replay(df_features, price_lookup, **kwargs):
        captured.update(kwargs)
        return np.array([0.01, 0.0], dtype=np.float32), []

    monkeypatch.setattr(replay_module, "run_replay", fake_run_replay)

    import pipeline.benchmark as benchmark_module

    monkeypatch.setattr(
        benchmark_module,
        "fetch_spy_returns",
        lambda start, end: pd.Series([0.0, 0.0]),
    )
    monkeypatch.setattr(benchmark_module, "print_benchmark_report", lambda *args, **kwargs: None)
    monkeypatch.setattr(benchmark_module, "plot_benchmark", lambda *args, **kwargs: None)

    replay_module.run_full_replay(
        df_features=df_features,
        replay_years=1,
        live_config={
            "rl_enabled": True,
            "min_score": 0.57,
            "stop_loss": 0.09,
            "take_profit": 0.41,
            "partial_profit": 0.24,
            "penny_pct": 0.03,
            "max_sector": 0.33,
            "max_correlation": 0.72,
            "avoid_earnings": 4,
            "rl_phase": 2,
            "rl_exit_threshold": 0.27,
            "rl_conviction_drop": 0.18,
            "rl_min_score": 0.05,
        },
        checkpoint_path="models/best_fold0.pt",
    )

    assert captured["strategy"] == "screener_rl"
    assert captured["checkpoint_path"] == "models/best_fold0.pt"
    assert captured["partial_profit_pct"] == 0.24
    assert captured["max_pair_correlation"] == 0.72
    assert captured["avoid_earnings_days"] == 4
    assert captured["rl_phase"] == 2
    assert captured["rl_exit_threshold"] == 0.27
    assert captured["rl_conviction_drop"] == 0.18
    assert captured["rl_min_score"] == 0.05


def test_run_full_replay_prefers_screener_when_rl_disabled(monkeypatch):
    df_features = _feature_frame()
    captured = {}

    monkeypatch.setattr(replay_module, "_build_price_lookup", lambda: _price_lookup())

    def fake_run_replay(df_features, price_lookup, **kwargs):
        captured.update(kwargs)
        return np.array([0.01, 0.0], dtype=np.float32), []

    monkeypatch.setattr(replay_module, "run_replay", fake_run_replay)

    import pipeline.benchmark as benchmark_module

    monkeypatch.setattr(
        benchmark_module,
        "fetch_spy_returns",
        lambda start, end: pd.Series([0.0, 0.0]),
    )
    monkeypatch.setattr(benchmark_module, "print_benchmark_report", lambda *args, **kwargs: None)
    monkeypatch.setattr(benchmark_module, "plot_benchmark", lambda *args, **kwargs: None)
    monkeypatch.setattr("os.path.exists", lambda path: str(path).endswith("screener.pt"))

    replay_module.run_full_replay(
        df_features=df_features,
        replay_years=1,
        live_config={"rl_enabled": False},
        checkpoint_path=None,
    )

    assert captured["strategy"] == "screener_heuristics"
