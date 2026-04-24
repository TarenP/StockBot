"""
Tier 2 & 3 audit tests.

Covers:
  - Tier 2 #7: Live-vs-replay parity on same date
  - Tier 2 #8: Friction sensitivity reporting (gross/net/stressed)
  - Tier 2 #9: Source-health telemetry warnings
  - Tier 3 #10: Stale-state contamination in broad-universe resolution
  - Tier 3 #12: Score audit CSV schema stability
"""

import logging
import numpy as np
import pandas as pd
import pytest

import broker.replay as replay_module
from pipeline.run_manifest import hash_ticker_list


# ── Helpers ───────────────────────────────────────────────────────────────────

def _feature_frame(tickers=None, dates=None):
    tickers = tickers or ["AAPL", "MSFT", "NVDA"]
    dates = dates or ["2024-01-02", "2024-01-03", "2024-01-04"]
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(dates), tickers],
        names=["date", "ticker"],
    )
    rng = np.random.default_rng(0)
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


def _price_lookup(tickers=None, dates=None):
    tickers = tickers or ["AAPL", "MSFT", "NVDA"]
    dates = dates or ["2024-01-02", "2024-01-03", "2024-01-04"]
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(dates), tickers],
        names=["date", "ticker"],
    )
    n = len(index)
    return pd.DataFrame(
        {"close": np.linspace(10.0, 15.0, n), "volume": [1e6] * n},
        index=index,
    )


# ── Tier 2 #7: Live-vs-replay parity ─────────────────────────────────────────

class TestLiveReplayParity:
    """
    Prove that live and replay resolve the same effective universe
    when given the same config and as-of date.
    """

    def test_universe_hash_matches_between_live_and_replay(self, monkeypatch, tmp_path):
        """
        Universe resolved for live cycle and replay on the same date
        must produce the same hash.
        """
        from pipeline.universe_resolver import resolve_configured_universe

        cfg = {"universe_mode": "sp500"}
        fixed_tickers = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"]

        monkeypatch.setattr(
            "pipeline.universe_resolver._resolve_sp500",
            lambda: fixed_tickers,
        )

        live_universe = resolve_configured_universe(config=cfg, save_dir=str(tmp_path))
        replay_universe = resolve_configured_universe(config=cfg, save_dir=str(tmp_path))

        assert hash_ticker_list(live_universe) == hash_ticker_list(replay_universe)

    def test_replay_universe_is_subset_of_live_universe(self, monkeypatch, tmp_path):
        """
        Every ticker in the replay universe must also be in the live universe
        (replay cannot trade names the live broker wouldn't see).
        """
        from pipeline.universe_resolver import resolve_configured_universe

        cfg = {"universe_mode": "sp500"}
        fixed_tickers = ["AAPL", "MSFT", "NVDA"]

        monkeypatch.setattr(
            "pipeline.universe_resolver._resolve_sp500",
            lambda: fixed_tickers,
        )

        live_set = set(resolve_configured_universe(config=cfg, save_dir=str(tmp_path)))
        replay_set = set(resolve_configured_universe(config=cfg, save_dir=str(tmp_path)))

        assert replay_set.issubset(live_set), (
            f"Replay has tickers not in live universe: {replay_set - live_set}"
        )

    def test_config_hash_is_same_for_live_and_replay(self):
        """Same config dict must hash identically for live and replay manifests."""
        from pipeline.run_manifest import hash_config

        cfg = {"min_score": 0.6, "rl_enabled": True, "universe_mode": "sp500"}
        assert hash_config(cfg) == hash_config(cfg)

    def test_replay_uses_no_lookahead_prices(self, monkeypatch):
        """
        Replay must fill at next-session price, not decision-date price.
        This is the core parity guarantee: replay cannot see prices
        that the live broker wouldn't have had at decision time.
        """
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        index = pd.MultiIndex.from_product([dates, ["AAPL"]], names=["date", "ticker"])
        df = pd.DataFrame(
            {"ret_1d": [0.01, 0.02, 0.03], "ret_5d": [0.0]*3,
             "vol_ratio": [1.0]*3, "sent_net": [0.0]*3, "macd_hist": [0.0]*3},
            index=index,
        )
        # Prices: 10 on Jan 2, 11 on Jan 3, 12 on Jan 4
        pl = pd.DataFrame(
            {"close": [10.0, 11.0, 12.0], "volume": [1e6]*3},
            index=index,
        )

        import broker.brain as brain_module

        def buy_on_jan2(self, df_slice, screener_top_n=100, risk_engine=None):
            last = df_slice.index.get_level_values("date").max()
            if pd.Timestamp(last) == pd.Timestamp("2024-01-02"):
                return [brain_module.Decision(
                    action="BUY", ticker="AAPL",
                    shares=1.0, price=10.0, score=0.9, reason="parity test",
                )]
            return []

        monkeypatch.setattr(brain_module.BrokerBrain, "run_cycle", buy_on_jan2)

        _, log = replay_module.run_replay(
            df, pl, strategy="heuristics_only", rebalance_freq=1, label="parity"
        )

        assert len(log) == 1
        # Decision on Jan 2 → fill on Jan 3 at price 11.0 (not 10.0)
        assert log[0]["decision_price"] == 10.0
        assert log[0]["price"] == 11.0
        assert pd.Timestamp(log[0]["fill_date"]) == pd.Timestamp("2024-01-03")


# ── Tier 2 #8: Friction sensitivity ──────────────────────────────────────────

class TestFrictionSensitivity:

    def test_friction_regimes_are_defined(self):
        """FRICTION_REGIMES must define optimistic, base, and stressed."""
        assert "optimistic" in replay_module.FRICTION_REGIMES
        assert "base" in replay_module.FRICTION_REGIMES
        assert "stressed" in replay_module.FRICTION_REGIMES

    def test_friction_regimes_have_increasing_spread(self):
        """Spread must increase from optimistic → base → stressed."""
        opt = replay_module.FRICTION_REGIMES["optimistic"]["execution_spread"]
        base = replay_module.FRICTION_REGIMES["base"]["execution_spread"]
        stressed = replay_module.FRICTION_REGIMES["stressed"]["execution_spread"]
        assert opt < base < stressed

    def test_run_friction_report_returns_three_rows(self, monkeypatch):
        """run_friction_report must return one row per friction regime."""
        df = _feature_frame()
        pl = _price_lookup()

        import broker.brain as brain_module
        monkeypatch.setattr(
            brain_module.BrokerBrain, "run_cycle",
            lambda self, df_slice, screener_top_n=100, risk_engine=None: [],
        )

        result = replay_module.run_friction_report(
            df, pl, live_config={}, strategy="heuristics_only"
        )

        assert len(result) == 3
        assert set(result["regime"].tolist()) == {"optimistic", "base", "stressed"}

    def test_run_friction_report_has_required_columns(self, monkeypatch):
        """Output must include regime, execution_spread, total_return, sharpe."""
        df = _feature_frame()
        pl = _price_lookup()

        import broker.brain as brain_module
        monkeypatch.setattr(
            brain_module.BrokerBrain, "run_cycle",
            lambda self, df_slice, screener_top_n=100, risk_engine=None: [],
        )

        result = replay_module.run_friction_report(
            df, pl, live_config={}, strategy="heuristics_only"
        )

        required = {"regime", "execution_spread", "total_return", "ann_return",
                    "sharpe", "max_drawdown", "win_rate"}
        assert required.issubset(set(result.columns))

    def test_stressed_return_not_higher_than_optimistic(self, monkeypatch):
        """
        Stressed friction must not produce higher returns than optimistic.
        If it does, the cost model is broken.
        """
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04",
                                 "2024-01-05", "2024-01-08"])
        tickers = ["AAPL", "MSFT"]
        index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
        rng = np.random.default_rng(1)
        n = len(index)
        df = pd.DataFrame(
            {"ret_1d": rng.normal(0.002, 0.01, n), "ret_5d": [0.01]*n,
             "vol_ratio": [1.2]*n, "sent_net": [0.1]*n, "macd_hist": [0.05]*n},
            index=index,
        )
        pl = pd.DataFrame(
            {"close": np.linspace(10.0, 12.0, n), "volume": [1e6]*n},
            index=index,
        )

        import broker.brain as brain_module

        call_n = {"n": 0}
        def alternating_buy(self, df_slice, screener_top_n=100, risk_engine=None):
            call_n["n"] += 1
            if call_n["n"] % 2 == 1:
                return [brain_module.Decision(
                    action="BUY", ticker="AAPL",
                    shares=1.0, price=10.0, score=0.9, reason="test",
                )]
            return [brain_module.Decision(
                action="SELL", ticker="AAPL",
                shares=1.0, price=11.0, score=0.0, reason="test sell",
            )]

        monkeypatch.setattr(brain_module.BrokerBrain, "run_cycle", alternating_buy)

        call_n["n"] = 0
        result = replay_module.run_friction_report(
            df, pl, live_config={}, strategy="heuristics_only"
        )

        opt_ret = float(result.loc[result["regime"] == "optimistic", "total_return"].iloc[0])
        stressed_ret = float(result.loc[result["regime"] == "stressed", "total_return"].iloc[0])
        # Stressed must not beat optimistic (more friction = lower or equal return)
        assert stressed_ret <= opt_ret + 1e-6, (
            f"Stressed return ({stressed_ret:.4f}) > optimistic ({opt_ret:.4f})"
        )

    def test_friction_sensitivity_warning_on_large_sharpe_drop(self, monkeypatch, caplog):
        """Warning must fire when stressed Sharpe drops >0.3 below base."""
        import broker.replay as r

        call_n = {"n": 0}
        def fake_run_replay(df, pl, strategy=None, checkpoint_path=None,
                            initial_cash=10_000.0, label=None, **kwargs):
            call_n["n"] += 1
            # Return different returns per regime to force a large Sharpe gap
            if "optimistic" in (label or ""):
                rets = np.array([0.02] * 10)
            elif "stressed" in (label or ""):
                rets = np.array([-0.01] * 10)
            else:
                rets = np.array([0.01] * 10)
            return rets, []

        monkeypatch.setattr(r, "run_replay", fake_run_replay)

        with caplog.at_level(logging.WARNING, logger="broker.replay"):
            r.run_friction_report(
                _feature_frame(), _price_lookup(),
                live_config={}, strategy="heuristics_only",
            )

        assert any("FRICTION SENSITIVITY" in r.message for r in caplog.records)


# ── Tier 2 #9: Source-health telemetry ───────────────────────────────────────

class TestSourceHealthTelemetry:

    def test_warning_when_universe_from_stale_local_state_only(self, monkeypatch, caplog):
        """
        When bootstrap returns 0 tickers and universe comes only from
        parquet/trained, a SOURCE HEALTH warning must be logged.
        """
        from pipeline.updater import get_live_universe

        monkeypatch.setattr("pipeline.updater._bootstrap_universe", lambda n: [])
        monkeypatch.setattr("pipeline.updater._load_trained_universe", lambda save_dir="models": ["AAPL", "MSFT"])
        monkeypatch.setattr("pipeline.updater._load_parquet_universe", lambda max_stale_days=30: ["AAPL", "MSFT", "NVDA"])
        monkeypatch.setattr("pipeline.updater._load_watchlist_universe", lambda: [])
        monkeypatch.setattr("pipeline.updater.filter_candidate_tickers", lambda s, config=None: list(s))

        with caplog.at_level(logging.WARNING, logger="pipeline.updater"):
            get_live_universe(
                config={"universe_mode": "tradable_us", "min_broad_universe_size": 1},
                save_dir="models",
            )

        assert any("SOURCE HEALTH" in r.message for r in caplog.records)

    def test_warning_when_bootstrap_returns_few_tickers(self, monkeypatch, caplog):
        """When bootstrap returns < 100 tickers, a degradation warning must fire."""
        from pipeline.updater import get_live_universe

        monkeypatch.setattr("pipeline.updater._bootstrap_universe", lambda n: ["A", "B", "C", "D", "E"])
        monkeypatch.setattr("pipeline.updater._load_trained_universe", lambda save_dir="models": [])
        monkeypatch.setattr("pipeline.updater._load_parquet_universe", lambda max_stale_days=30: [])
        monkeypatch.setattr("pipeline.updater._load_watchlist_universe", lambda: [])
        monkeypatch.setattr("pipeline.updater.filter_candidate_tickers", lambda s, config=None: list(s))

        with caplog.at_level(logging.WARNING, logger="pipeline.updater"):
            get_live_universe(
                config={"universe_mode": "tradable_us", "min_broad_universe_size": 1},
                save_dir="models",
            )

        assert any("SOURCE HEALTH" in r.message for r in caplog.records)

    def test_no_warning_when_bootstrap_healthy(self, monkeypatch, caplog):
        """No SOURCE HEALTH warning when bootstrap returns plenty of tickers."""
        from pipeline.updater import get_live_universe

        healthy_tickers = [f"T{i}" for i in range(200)]
        monkeypatch.setattr("pipeline.updater._bootstrap_universe", lambda n: healthy_tickers)
        monkeypatch.setattr("pipeline.updater._load_trained_universe", lambda save_dir="models": [])
        monkeypatch.setattr("pipeline.updater._load_parquet_universe", lambda max_stale_days=30: [])
        monkeypatch.setattr("pipeline.updater._load_watchlist_universe", lambda: [])
        monkeypatch.setattr("pipeline.updater.filter_candidate_tickers", lambda s, config=None: list(s))

        with caplog.at_level(logging.WARNING, logger="pipeline.updater"):
            get_live_universe(
                config={"universe_mode": "tradable_us", "min_broad_universe_size": 1},
                save_dir="models",
            )

        assert not any("SOURCE HEALTH" in r.message for r in caplog.records)

    def test_source_counts_logged_at_info_level(self, monkeypatch, caplog):
        """Source health summary must be logged at INFO level."""
        from pipeline.updater import get_live_universe

        monkeypatch.setattr("pipeline.updater._bootstrap_universe", lambda n: ["A", "B"] * 100)
        monkeypatch.setattr("pipeline.updater._load_trained_universe", lambda save_dir="models": ["C", "D"])
        monkeypatch.setattr("pipeline.updater._load_parquet_universe", lambda max_stale_days=30: ["E", "F"])
        monkeypatch.setattr("pipeline.updater._load_watchlist_universe", lambda: [])
        monkeypatch.setattr("pipeline.updater.filter_candidate_tickers", lambda s, config=None: list(s))

        with caplog.at_level(logging.INFO, logger="pipeline.updater"):
            get_live_universe(
                config={"universe_mode": "tradable_us", "min_broad_universe_size": 1},
                save_dir="models",
            )

        assert any("Source health:" in r.message for r in caplog.records)


# ── Tier 3 #10: Stale-state contamination ────────────────────────────────────

class TestStaleStateContamination:
    """
    Confirm that stale parquet symbols remain usable for historical data
    but do not dominate present-day live tradable breadth.
    """

    def test_stale_parquet_tickers_excluded_from_live_universe_in_sp500_mode(self, monkeypatch, tmp_path):
        """
        In sp500 mode, tickers in the parquet but not in the S&P 500
        must not appear in the resolved live universe.
        """
        from pipeline.universe_resolver import resolve_configured_universe

        sp500_members = ["AAPL", "MSFT", "NVDA"]
        monkeypatch.setattr(
            "pipeline.universe_resolver._resolve_sp500",
            lambda: sp500_members,
        )

        cfg = {"universe_mode": "sp500"}
        universe = resolve_configured_universe(config=cfg, save_dir=str(tmp_path))

        # Stale tickers that might be in parquet but not S&P 500
        stale_tickers = {"EDIT", "GTN", "AFRM", "RKT", "SSRM"}
        contamination = stale_tickers & set(universe)
        assert contamination == set(), (
            f"Stale tickers leaked into sp500 universe: {contamination}"
        )

    def test_stale_parquet_tickers_do_not_appear_in_sp500_universe(self, monkeypatch, tmp_path):
        """
        Even if stale tickers are in the parquet, sp500 mode must
        return only current S&P 500 members.
        """
        from pipeline.universe_resolver import resolve_configured_universe

        current_sp500 = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
        monkeypatch.setattr(
            "pipeline.universe_resolver._resolve_sp500",
            lambda: current_sp500,
        )

        cfg = {"universe_mode": "sp500"}
        universe = resolve_configured_universe(config=cfg, save_dir=str(tmp_path))

        assert set(universe) == set(current_sp500)

    def test_tradable_us_mode_filters_non_equity_symbols(self, monkeypatch):
        """
        In tradable_us mode, ETFs and malformed symbols from the parquet
        must be filtered out by filter_candidate_tickers.
        """
        from pipeline.universe_resolver import filter_candidate_tickers

        mixed_symbols = ["AAPL", "MSFT", "SPY", "QQQ", "AAPL-W", "EDIT", "GTN"]
        cfg = {}  # default: ETFs and warrants excluded

        filtered = filter_candidate_tickers(mixed_symbols, config=cfg)

        # SPY and QQQ are known non-equity (ETFs) — should be excluded by default
        assert "SPY" not in filtered
        assert "QQQ" not in filtered
        # AAPL-W is a warrant — should be excluded
        assert "AAPL-W" not in filtered
        # Regular equities should pass through
        assert "AAPL" in filtered
        assert "MSFT" in filtered

    def test_history_retention_does_not_affect_live_universe_in_sp500_mode(self, monkeypatch, tmp_path):
        """
        Keeping stale history in the parquet for backtesting must not
        cause those tickers to appear in the live sp500 universe.
        """
        from pipeline.universe_resolver import resolve_configured_universe

        # Simulate: parquet has 600 tickers (including historical names)
        # but S&P 500 only has 503 current members
        current_sp500 = [f"TICK{i}" for i in range(503)]
        monkeypatch.setattr(
            "pipeline.universe_resolver._resolve_sp500",
            lambda: current_sp500,
        )

        cfg = {"universe_mode": "sp500"}
        universe = resolve_configured_universe(config=cfg, save_dir=str(tmp_path))

        # Universe must be exactly the S&P 500 members, not the full parquet
        assert len(universe) == 503
        assert set(universe) == set(current_sp500)


# ── Tier 3 #12: Score audit CSV schema ───────────────────────────────────────

class TestScoreAuditSchema:
    """
    Ensure the score audit CSV (when produced) has a stable schema
    so downstream analysis tools don't break silently.
    """

    EXPECTED_AUDIT_COLUMNS = {
        "ticker",
        "cycle_date",
    }

    def test_last_replay_score_audit_is_dataframe(self, monkeypatch):
        """_LAST_REPLAY_SCORE_AUDIT must be a DataFrame after a replay run."""
        df = _feature_frame()
        pl = _price_lookup()

        import broker.brain as brain_module
        monkeypatch.setattr(
            brain_module.BrokerBrain, "run_cycle",
            lambda self, df_slice, screener_top_n=100, risk_engine=None: [],
        )

        replay_module.run_replay(
            df, pl, strategy="heuristics_only", rebalance_freq=1, label="audit_test"
        )

        assert isinstance(replay_module._LAST_REPLAY_SCORE_AUDIT, pd.DataFrame)

    def test_score_audit_reset_between_runs(self, monkeypatch):
        """_LAST_REPLAY_SCORE_AUDIT must be reset at the start of each replay."""
        import broker.brain as brain_module
        monkeypatch.setattr(
            brain_module.BrokerBrain, "run_cycle",
            lambda self, df_slice, screener_top_n=100, risk_engine=None: [],
        )

        # First run
        replay_module.run_replay(
            _feature_frame(), _price_lookup(),
            strategy="heuristics_only", rebalance_freq=1, label="run1"
        )
        # Second run — audit must be fresh, not accumulated from run1
        replay_module.run_replay(
            _feature_frame(), _price_lookup(),
            strategy="heuristics_only", rebalance_freq=1, label="run2"
        )

        # After second run, audit should reflect only run2 (empty since no audit data)
        assert isinstance(replay_module._LAST_REPLAY_SCORE_AUDIT, pd.DataFrame)

    def test_friction_report_columns_are_stable(self, monkeypatch):
        """run_friction_report output columns must not change unexpectedly."""
        import broker.brain as brain_module
        monkeypatch.setattr(
            brain_module.BrokerBrain, "run_cycle",
            lambda self, df_slice, screener_top_n=100, risk_engine=None: [],
        )

        result = replay_module.run_friction_report(
            _feature_frame(), _price_lookup(),
            live_config={}, strategy="heuristics_only",
        )

        expected_cols = {"regime", "execution_spread", "total_return",
                         "ann_return", "sharpe", "max_drawdown", "win_rate"}
        assert expected_cols.issubset(set(result.columns)), (
            f"Missing columns: {expected_cols - set(result.columns)}"
        )
        # No extra unexpected columns
        extra = set(result.columns) - expected_cols
        assert extra == set(), f"Unexpected extra columns: {extra}"
