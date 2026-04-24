"""
Tier 1 audit tests: replay determinism.

Proves that running the same replay twice with the same inputs produces
identical results — same universe hash, same trade log, same returns.
"""

import numpy as np
import pandas as pd
import pytest

import broker.replay as replay_module
from pipeline.run_manifest import hash_ticker_list


def _feature_frame():
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    tickers = ["AAPL", "MSFT", "NVDA"]
    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    rng = np.random.default_rng(42)
    n = len(index)
    return pd.DataFrame(
        {
            "ret_1d":    rng.normal(0, 0.01, n),
            "ret_5d":    rng.normal(0, 0.02, n),
            "vol_ratio": rng.uniform(0.8, 1.5, n),
            "sent_net":  rng.uniform(-0.3, 0.3, n),
            "macd_hist": rng.normal(0, 0.1, n),
        },
        index=index,
    )


def _price_lookup():
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    tickers = ["AAPL", "MSFT", "NVDA"]
    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    return pd.DataFrame(
        {
            "close":  [150.0, 300.0, 500.0, 152.0, 302.0, 505.0, 151.0, 299.0, 498.0],
            "volume": [1e6] * 9,
        },
        index=index,
    )


class TestReplayDeterminism:

    def test_same_inputs_produce_identical_returns(self, monkeypatch):
        """Running replay twice with identical inputs must produce identical returns."""
        df = _feature_frame()
        pl = _price_lookup()

        import broker.brain as brain_module

        call_count = {"n": 0}

        def deterministic_run_cycle(self, df_slice, screener_top_n=100, risk_engine=None):
            call_count["n"] += 1
            # Always return the same decision based on call count
            if call_count["n"] % 2 == 1:
                return [
                    brain_module.Decision(
                        action="BUY", ticker="AAPL",
                        shares=1.0, price=150.0, score=0.9, reason="det test",
                    )
                ]
            return []

        monkeypatch.setattr(brain_module.BrokerBrain, "run_cycle", deterministic_run_cycle)

        call_count["n"] = 0
        rets1, log1 = replay_module.run_replay(
            df, pl, strategy="heuristics_only", rebalance_freq=1, label="run1"
        )

        call_count["n"] = 0
        rets2, log2 = replay_module.run_replay(
            df, pl, strategy="heuristics_only", rebalance_freq=1, label="run2"
        )

        np.testing.assert_array_equal(rets1, rets2)
        assert len(log1) == len(log2)
        for t1, t2 in zip(log1, log2):
            assert t1["action"] == t2["action"]
            assert t1["ticker"] == t2["ticker"]
            assert t1["price"] == t2["price"]

    def test_universe_hash_is_stable_across_runs(self, monkeypatch):
        """Universe hash computed from the same df must be identical across calls."""
        df = _feature_frame()
        tickers = df.index.get_level_values("ticker").unique().tolist()

        hash1 = hash_ticker_list(tickers)
        hash2 = hash_ticker_list(tickers)
        assert hash1 == hash2

    def test_universe_hash_changes_when_tickers_change(self):
        """Adding or removing a ticker must change the universe hash."""
        tickers_a = ["AAPL", "MSFT", "NVDA"]
        tickers_b = ["AAPL", "MSFT", "TSLA"]  # NVDA replaced by TSLA

        hash_a = hash_ticker_list(tickers_a)
        hash_b = hash_ticker_list(tickers_b)
        assert hash_a != hash_b

    def test_universe_hash_is_order_independent(self):
        """Hash must be the same regardless of ticker list order."""
        tickers_ordered = ["AAPL", "MSFT", "NVDA"]
        tickers_shuffled = ["NVDA", "AAPL", "MSFT"]

        assert hash_ticker_list(tickers_ordered) == hash_ticker_list(tickers_shuffled)

    def test_replay_returns_length_matches_date_count(self, monkeypatch):
        """Returns array length must equal number of dates in df_features."""
        df = _feature_frame()
        pl = _price_lookup()

        import broker.brain as brain_module
        monkeypatch.setattr(
            brain_module.BrokerBrain, "run_cycle",
            lambda self, df_slice, screener_top_n=100, risk_engine=None: [],
        )

        rets, _ = replay_module.run_replay(
            df, pl, strategy="heuristics_only", rebalance_freq=1, label="len_test"
        )

        n_dates = df.index.get_level_values("date").nunique()
        assert len(rets) == n_dates

    def test_config_hash_is_stable(self):
        """Same config dict must always produce the same hash."""
        from pipeline.run_manifest import hash_config
        cfg = {"min_score": 0.6, "stop_loss": 0.08, "rl_enabled": True}
        assert hash_config(cfg) == hash_config(cfg)
        assert hash_config(cfg) == hash_config(dict(cfg))

    def test_config_hash_changes_on_param_change(self):
        """Changing any config value must change the hash."""
        from pipeline.run_manifest import hash_config
        cfg_a = {"min_score": 0.6, "stop_loss": 0.08}
        cfg_b = {"min_score": 0.7, "stop_loss": 0.08}
        assert hash_config(cfg_a) != hash_config(cfg_b)

    def test_replay_trade_log_fill_dates_are_next_session(self, monkeypatch):
        """
        Decisions made on date D must fill on date D+1 (next session).
        This is the no-lookahead guarantee.
        """
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        index = pd.MultiIndex.from_product([dates, ["AAPL"]], names=["date", "ticker"])
        df = pd.DataFrame(
            {"ret_1d": [0.01, 0.02, 0.03], "ret_5d": [0.0]*3,
             "vol_ratio": [1.0]*3, "sent_net": [0.0]*3, "macd_hist": [0.0]*3},
            index=index,
        )
        pl = pd.DataFrame(
            {"close": [10.0, 11.0, 12.0], "volume": [1e6]*3},
            index=index,
        )

        import broker.brain as brain_module

        def buy_on_first_date(self, df_slice, screener_top_n=100, risk_engine=None):
            last = df_slice.index.get_level_values("date").max()
            if pd.Timestamp(last) == pd.Timestamp("2024-01-02"):
                return [brain_module.Decision(
                    action="BUY", ticker="AAPL",
                    shares=1.0, price=10.0, score=0.9, reason="det",
                )]
            return []

        monkeypatch.setattr(brain_module.BrokerBrain, "run_cycle", buy_on_first_date)

        _, log = replay_module.run_replay(
            df, pl, strategy="heuristics_only", rebalance_freq=1, label="fill_date_test"
        )

        assert len(log) == 1
        assert pd.Timestamp(log[0]["decision_date"]) == pd.Timestamp("2024-01-02")
        assert pd.Timestamp(log[0]["fill_date"]) == pd.Timestamp("2024-01-03")
        assert log[0]["price"] == 11.0  # next-session price, not decision price
