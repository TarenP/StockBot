import numpy as np
import pandas as pd

import broker.replay as replay_module
from broker.paper_diagnostics import summarize_low_price_signal_suppression


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


def test_run_replay_returns_one_value_per_replay_date(monkeypatch):
    df_features = _feature_frame()
    price_lookup = _price_lookup()

    import broker.brain as brain_module

    monkeypatch.setattr(
        brain_module.BrokerBrain,
        "run_cycle",
        lambda self, df_slice, screener_top_n=100, risk_engine=None: [],
    )

    returns, trade_log = replay_module.run_replay(
        df_features,
        price_lookup,
        strategy="heuristics_only",
        rebalance_freq=1,
        label="length_test",
    )

    expected_len = df_features.index.get_level_values("date").nunique()
    assert len(returns) == expected_len
    assert trade_log == []


def test_run_replay_executes_trades_on_next_session(monkeypatch):
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    index = pd.MultiIndex.from_product([dates, ["AAA"]], names=["date", "ticker"])
    df_features = pd.DataFrame(
        {
            "ret_1d": [0.01, 0.02, 0.03],
            "ret_5d": [0.05, 0.06, 0.07],
            "vol_ratio": [1.2, 1.1, 1.0],
            "sent_net": [0.2, 0.1, 0.0],
            "macd_hist": [0.3, 0.2, 0.1],
        },
        index=index,
    )
    price_lookup = pd.DataFrame(
        {
            "close": [10.0, 11.0, 12.0],
            "volume": [1_000_000.0, 1_000_000.0, 1_000_000.0],
        },
        index=index,
    )

    import broker.brain as brain_module

    def fake_run_cycle(self, df_slice, screener_top_n=100, risk_engine=None):
        last_date = df_slice.index.get_level_values("date").max()
        if pd.Timestamp(last_date) != pd.Timestamp("2024-01-02"):
            return []
        return [
            brain_module.Decision(
                action="BUY",
                ticker="AAA",
                shares=1.0,
                price=10.0,
                score=0.9,
                reason="next-session test",
            )
        ]

    monkeypatch.setattr(brain_module.BrokerBrain, "run_cycle", fake_run_cycle)

    returns, trade_log = replay_module.run_replay(
        df_features,
        price_lookup,
        strategy="heuristics_only",
        rebalance_freq=1,
        label="next_session_test",
    )

    assert len(returns) == 3
    assert len(trade_log) == 1
    assert pd.Timestamp(trade_log[0]["decision_date"]) == pd.Timestamp("2024-01-02")
    assert pd.Timestamp(trade_log[0]["fill_date"]) == pd.Timestamp("2024-01-03")
    assert trade_log[0]["price"] == 11.0


def test_execute_replay_decisions_applies_execution_cost_to_sell_proceeds():
    import broker.brain as brain_module

    portfolio = replay_module.ReplayPortfolio(initial_cash=1_000.0)
    assert portfolio.buy("AAA", shares=10.0, price=10.0, reason="seed") is True

    trade_log = []
    decision = brain_module.Decision(
        action="SELL",
        ticker="AAA",
        shares=10.0,
        price=10.0,
        score=0.0,
        reason="spread test",
    )

    replay_module._execute_replay_decisions(
        portfolio=portfolio,
        decisions=[decision],
        execution_spread=0.10,
        trade_log=trade_log,
        date=pd.Timestamp("2024-01-03"),
    )

    assert portfolio.cash == 990.0
    assert trade_log[0]["shares"] == 10.0
    assert trade_log[0]["price"] == 9.0


def test_execute_replay_partial_sale_marks_partial_taken_after_fill():
    import broker.brain as brain_module

    portfolio = replay_module.ReplayPortfolio(initial_cash=1_000.0)
    assert portfolio.buy("AAA", shares=10.0, price=10.0, reason="seed") is True

    trade_log = []
    decision = brain_module.Decision(
        action="SELL_PARTIAL",
        ticker="AAA",
        shares=5.0,
        price=10.0,
        score=0.8,
        reason="Partial take-profit (25.0%), selling 50%",
    )

    replay_module._execute_replay_decisions(
        portfolio=portfolio,
        decisions=[decision],
        execution_spread=0.0,
        trade_log=trade_log,
        date=pd.Timestamp("2024-01-03"),
    )

    assert portfolio.positions["AAA"]["shares"] == 5.0
    assert portfolio.positions["AAA"]["partial_taken"] is True
    assert trade_log[0]["action"] == "SELL_PARTIAL"


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

    def capture_execute(
        portfolio,
        decisions,
        execution_spread,
        trade_log,
        date,
        price_lookup=None,
        decision_date=None,
    ):
        original_execute(
            portfolio,
            decisions,
            execution_spread,
            trade_log,
            date,
            price_lookup=price_lookup,
            decision_date=decision_date,
        )
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


def test_run_replay_accrues_cash_yield_on_idle_cash(monkeypatch):
    dates = pd.to_datetime(["2024-01-02", "2025-01-02"])
    index = pd.MultiIndex.from_product(
        [dates, ["AAA"]],
        names=["date", "ticker"],
    )
    df_features = pd.DataFrame(
        {
            "ret_1d": [0.0, 0.0],
            "ret_5d": [0.0, 0.0],
            "vol_ratio": [1.0, 1.0],
            "sent_net": [0.0, 0.0],
            "macd_hist": [0.0, 0.0],
        },
        index=index,
    )
    price_lookup = pd.DataFrame(
        {
            "close": [10.0, 10.0],
            "volume": [1_000_000.0, 1_000_000.0],
        },
        index=index,
    )

    import broker.brain as brain_module

    monkeypatch.setattr(
        brain_module.BrokerBrain,
        "run_cycle",
        lambda self, df_slice, screener_top_n=100, risk_engine=None: [],
    )

    returns, trade_log = replay_module.run_replay(
        df_features,
        price_lookup,
        strategy="heuristics_only",
        rebalance_freq=1,
        label="cash_yield_test",
    )

    assert trade_log == []
    assert len(returns) >= 2
    assert float(np.max(returns)) > 0.02


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
            "trailing_stop": 0.11,
            "trailing_activation": 0.19,
            "signal_exit_score": 0.21,
            "signal_exit_grace": 3,
            "partial_profit": 0.24,
            "max_position_pct": 0.18,
            "cash_floor": 0.02,
            "max_gross_exposure": 0.98,
            "target_volatility": 0.21,
            "vol_lookback": 30,
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
    assert captured["trailing_stop_pct"] == 0.11
    assert captured["trailing_activation_pct"] == 0.19
    assert captured["signal_exit_score"] == 0.21
    assert captured["signal_exit_grace_cycles"] == 3
    assert captured["max_position_pct"] == 0.18
    assert captured["cash_floor"] == 0.02
    assert captured["max_gross_exposure"] == 0.98
    assert captured["target_volatility"] == 0.21
    assert captured["vol_lookback"] == 30
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


def test_run_full_replay_aligns_spy_by_replay_dates(monkeypatch):
    dates = pd.to_datetime(["2024-01-02", "2024-01-04"])
    index = pd.MultiIndex.from_product([dates, ["AAA"]], names=["date", "ticker"])
    df_features = pd.DataFrame(
        {
            "ret_1d": [0.0, 0.0],
            "ret_5d": [0.0, 0.0],
            "vol_ratio": [1.0, 1.0],
            "sent_net": [0.0, 0.0],
            "macd_hist": [0.0, 0.0],
        },
        index=index,
    )
    captured = {}
    price_lookup = pd.DataFrame(
        {
            "close": [10.0, 10.5],
            "volume": [1_000_000.0, 1_000_000.0],
        },
        index=index,
    )

    monkeypatch.setattr(replay_module, "_build_price_lookup", lambda: price_lookup)
    monkeypatch.setattr(
        replay_module,
        "run_replay",
        lambda *args, **kwargs: (np.array([0.01, 0.02], dtype=float), []),
    )
    monkeypatch.setattr(
        replay_module,
        "_equal_weight_returns",
        lambda *args, **kwargs: np.array([0.0, 0.03], dtype=float),
    )

    import pipeline.benchmark as benchmark_module

    monkeypatch.setattr(
        benchmark_module,
        "fetch_spy_returns",
        lambda start, end: pd.Series(
            [0.11, 0.22, 0.33],
            index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        ),
    )
    monkeypatch.setattr(
        benchmark_module,
        "print_benchmark_report",
        lambda portfolio_rets, spy_rets, ew_rets=None, label=None: captured.update(
            {
                "portfolio": np.asarray(portfolio_rets, dtype=float),
                "spy": None if spy_rets is None else np.asarray(spy_rets, dtype=float),
                "ew": None if ew_rets is None else np.asarray(ew_rets, dtype=float),
                "label": label,
            }
        ),
    )
    monkeypatch.setattr(benchmark_module, "plot_benchmark", lambda *args, **kwargs: None)

    replay_module.run_full_replay(
        df_features=df_features,
        replay_years=1,
        live_config={},
        checkpoint_path=None,
    )

    assert captured["portfolio"].tolist() == [0.01, 0.02]
    assert captured["spy"].tolist() == [0.11, 0.33]
    assert captured["ew"].tolist() == [0.0, 0.03]
    assert captured["label"] == "Broker Replay"


def test_run_sensitivity_uses_rl_specific_grid(monkeypatch):
    labels = []

    monkeypatch.setattr(
        replay_module,
        "run_replay",
        lambda *args, label=None, **kwargs: (
            labels.append(label) or np.array([0.01, 0.0], dtype=float),
            [],
        ),
    )

    result = replay_module.run_sensitivity(
        df_features=_feature_frame(),
        price_lookup=_price_lookup(),
        live_config={"rl_enabled": True},
        strategy="screener_rl",
        checkpoint_path="models/best_fold0.pt",
    )

    assert "rl_min=5%" in labels
    assert "rl_min=10%" in labels
    assert "rl_min=20%" in labels
    assert "rl_phase=2" in labels
    assert "weak_sleeve=25%" in labels
    assert "weak_sleeve=block" in labels
    assert "weak_sleeve=cooldown2" in labels
    assert "low_price=pre_penalty" in labels
    assert "min_score=0.55" not in labels
    assert "min_score=0.65" not in labels
    assert "weak_sleeve_reentry_count" in result.columns
    assert "tokenized_high_rank_low_price_count" in result.columns
    assert "avg_top_theme_concentration" in result.columns


def test_replay_control_metrics_count_weak_reentry_and_low_price_tokenization():
    trade_log = [
        {
            "action": "BUY",
            "ticker": "LOW",
            "price": 4.0,
            "reason": (
                "rl_rank_pct=0.9500 | target_weight_pre_caps=0.2000 | "
                "final_weight=0.0300 | downweight_reason=low_price_or_penny_cap | "
                "Theme=theme_a | WeakSleeve=2/2 avg=-5.0% scale=0.50 |"
            ),
        }
    ]
    score_audit = pd.DataFrame(
        [
            {
                "cycle_date": "2024-01-02",
                "candidate_status": "buy_selected",
                "ticker": "LOW",
                "theme_bucket": "theme_a",
                "low_price_bucket": "sub_5",
                "final_weight": 0.03,
                "weak_sleeve_cap_impact": 0.03,
            },
            {
                "cycle_date": "2024-01-02",
                "candidate_status": "buy_selected",
                "ticker": "HIGH",
                "theme_bucket": "theme_b",
                "low_price_bucket": "over_10",
                "final_weight": 0.10,
                "weak_sleeve_cap_impact": 0.00,
            },
        ]
    )

    metrics = replay_module._summarize_replay_control_metrics(
        trade_log,
        score_audit,
        summarize_low_price_signal_suppression,
    )

    assert metrics["weak_sleeve_reentry_count"] == 1
    assert metrics["weak_sleeve_selected_count"] == 1
    assert metrics["tokenized_high_rank_low_price_count"] == 1
    assert np.isclose(metrics["max_top_theme_concentration"], 0.10)
    assert np.isclose(metrics["max_low_price_exposure"], 0.03)


def test_build_policy_review_report_groups_outcome_mechanism_confidence():
    sensitivity = pd.DataFrame(
        [
            {
                "params": "current_config (base)",
                "total_return": 0.08,
                "sharpe": 1.0,
                "max_drawdown": -0.10,
                "win_rate": 0.55,
                "weak_sleeve_reentry_count": 3,
                "weak_sleeve_reentry_theme_count": 1,
                "weak_sleeve_selected_count": 4,
                "tokenized_high_rank_low_price_count": 2,
                "high_rank_low_price_count": 3,
                "low_price_tokenized_rate": 0.67,
                "avg_top_theme_concentration": 0.30,
                "max_top_theme_concentration": 0.40,
                "avg_low_price_exposure": 0.08,
                "max_low_price_exposure": 0.10,
            },
            {
                "params": "weak_sleeve=block",
                "total_return": 0.07,
                "sharpe": 0.95,
                "max_drawdown": -0.08,
                "win_rate": 0.54,
                "weak_sleeve_reentry_count": 0,
                "weak_sleeve_reentry_theme_count": 0,
                "weak_sleeve_selected_count": 0,
                "tokenized_high_rank_low_price_count": 2,
                "high_rank_low_price_count": 3,
                "low_price_tokenized_rate": 0.67,
                "avg_top_theme_concentration": 0.24,
                "max_top_theme_concentration": 0.30,
                "avg_low_price_exposure": 0.08,
                "max_low_price_exposure": 0.10,
            },
            {
                "params": "low_price=pre_penalty",
                "total_return": 0.09,
                "sharpe": 1.1,
                "max_drawdown": -0.09,
                "win_rate": 0.56,
                "weak_sleeve_reentry_count": 3,
                "weak_sleeve_reentry_theme_count": 1,
                "weak_sleeve_selected_count": 4,
                "tokenized_high_rank_low_price_count": 0,
                "high_rank_low_price_count": 3,
                "low_price_tokenized_rate": 0.0,
                "avg_top_theme_concentration": 0.30,
                "max_top_theme_concentration": 0.40,
                "avg_low_price_exposure": 0.03,
                "max_low_price_exposure": 0.04,
            },
        ]
    )

    review, summary = replay_module.build_policy_review_report(sensitivity)

    assert set(review["family"]) == {"weak_sleeve", "low_price"}
    assert "current_config (base)" in set(review["variant"])
    assert "outcome_rank_score" in review.columns
    assert "mechanism_rank_score" in review.columns
    assert summary["families"]["weak_sleeve"]["rows"] == 2
    assert summary["families"]["low_price"]["small_sample_rows"] == 2


def test_rolling_window_validation_summarizes_subperiods():
    aligned = pd.DataFrame(
        {
            "portfolio": [0.02, -0.01, 0.03, 0.01, -0.02, 0.04],
            "benchmark": [0.01, 0.00, 0.01, 0.00, -0.01, 0.02],
            "equal_weight": [0.015, -0.005, 0.02, 0.005, -0.015, 0.03],
        },
        index=pd.date_range("2024-01-01", periods=6, freq="D"),
    )

    rolling_df, summary_df = replay_module._rolling_window_validation(
        aligned,
        windows=(3,),
        step=2,
    )

    assert len(rolling_df) == 3
    assert summary_df.loc[0, "window_days"] == 3
    assert summary_df.loc[0, "n_windows"] == 3
    assert "beat_rate_vs_spy" in summary_df.columns
    assert "beat_rate_vs_equal_weight" in summary_df.columns


def test_build_trade_attribution_reconstructs_fifo_closed_trades():
    trade_log = [
        {
            "fill_date": "2024-01-02",
            "action": "BUY",
            "ticker": "AAA",
            "shares": 10.0,
            "price": 10.0,
            "score": 0.9,
            "reason": "entry one",
        },
        {
            "fill_date": "2024-01-03",
            "action": "BUY",
            "ticker": "AAA",
            "shares": 5.0,
            "price": 12.0,
            "score": 0.8,
            "reason": "entry two",
        },
        {
            "fill_date": "2024-01-04",
            "action": "SELL_PARTIAL",
            "ticker": "AAA",
            "shares": 12.0,
            "price": 15.0,
            "score": 0.7,
            "reason": "Trailing stop (test)",
        },
        {
            "fill_date": "2024-01-05",
            "action": "SELL",
            "ticker": "AAA",
            "shares": 3.0,
            "price": 8.0,
            "score": 0.6,
            "reason": "Stop-loss (test)",
        },
    ]

    attribution = replay_module._build_trade_attribution(
        trade_log,
        sector_map={"AAA": "Technology"},
    )

    assert len(attribution) == 3
    assert attribution["sector"].unique().tolist() == ["Technology"]
    assert np.isclose(float(attribution["gross_pnl"].sum()), 44.0)
    assert attribution["exit_reason_bucket"].tolist() == [
        "Trailing stop",
        "Trailing stop",
        "Stop-loss",
    ]
