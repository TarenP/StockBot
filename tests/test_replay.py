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
