import numpy as np
import pandas as pd
from unittest.mock import patch

from broker.brain import BrokerBrain


class _DummyOptions:
    positions = {}

    def check_expirations(self, *_args, **_kwargs):
        return []

    def summary_lines(self):
        return []


class _DummyPortfolio:
    def __init__(self):
        self.cash = 19_000.0
        self.positions = {
            "AAA": {
                "shares": 10.0,
                "avg_cost": 90.0,
                "last_price": 100.0,
                "partial_taken": False,
                "days_held": 0,
            }
        }
        self.options = _DummyOptions()

    def update_prices(self, prices: dict[str, float]) -> None:
        for ticker, price in prices.items():
            if ticker in self.positions:
                self.positions[ticker]["last_price"] = price

    @property
    def position_values(self) -> dict[str, float]:
        return {
            ticker: pos["shares"] * pos["last_price"]
            for ticker, pos in self.positions.items()
        }

    @property
    def equity(self) -> float:
        return self.cash + sum(self.position_values.values())


class _EmptyPortfolio:
    def __init__(self):
        self.cash = 10_000.0
        self.positions = {}
        self.options = _DummyOptions()

    def update_prices(self, prices: dict[str, float]) -> None:
        return None

    @property
    def position_values(self) -> dict[str, float]:
        return {}

    @property
    def equity(self) -> float:
        return self.cash


def _make_df_features() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    base_returns = np.sin(np.linspace(0.0, 8.0, len(dates))) * 0.02
    index = pd.MultiIndex.from_product([dates, ["AAA", "BBB"]], names=["date", "ticker"])
    rows = []
    for ret in base_returns:
        rows.append({"ret_1d": ret, "regime_0": 1.0, "regime_1": 0.0, "regime_2": 0.0, "regime_3": 0.0})
        rows.append({"ret_1d": ret, "regime_0": 1.0, "regime_1": 0.0, "regime_2": 0.0, "regime_3": 0.0})
    return pd.DataFrame(rows, index=index)


def _make_df_features_with_regime(regime: int) -> pd.DataFrame:
    df = _make_df_features().copy()
    for idx in range(4):
        df[f"regime_{idx}"] = 1.0 if idx == regime else 0.0
    return df


def _research_stub(ticker: str) -> dict:
    return {
        "ticker": ticker,
        "price": 50.0 if ticker == "BBB" else 100.0,
        "composite_score": 0.85 if ticker == "BBB" else 0.80,
        "sentiment": {"sentiment": "neutral", "sent_net": 0.0},
        "headlines": ["headline"],
        "atr": 0.02,
    }


def _run_cycle(max_pair_correlation: float) -> list:
    portfolio = _DummyPortfolio()
    brain = BrokerBrain(
        portfolio=portfolio,
        max_positions=2,
        min_score=0.60,
        max_pair_correlation=max_pair_correlation,
    )
    brain._base_min_score = 0.60
    brain._sector_map = {"AAA": "Technology", "BBB": "Technology"}

    with (
        patch.object(brain, "_screen_candidates", return_value=["BBB"]),
        patch.object(brain, "_maybe_refresh_sector_map"),
        patch.object(brain, "_get_current_prices", return_value={"AAA": 100.0}),
        patch.object(brain, "_near_earnings", return_value=False),
        patch.object(brain, "_evaluate_options", return_value=[]),
        patch("broker.brain.research", side_effect=_research_stub),
        patch("broker.brain.validate_portfolio_prices", return_value={"AAA": 100.0}),
        patch("broker.brain.score_sectors", return_value={}),
        patch("broker.brain.get_portfolio_sector_weights", return_value={"Technology": 0.05}),
        patch("broker.brain.compute_target_allocations", return_value={"Technology": 1.0}),
    ):
        return brain.run_cycle(_make_df_features(), screener_top_n=10)


def test_run_cycle_blocks_highly_correlated_new_position():
    blocked = _run_cycle(max_pair_correlation=0.80)
    allowed = _run_cycle(max_pair_correlation=1.0)

    assert not any(d.action == "BUY" and d.ticker == "BBB" for d in blocked)
    assert any(d.action == "BUY" and d.ticker == "BBB" for d in allowed)


def test_screen_candidates_prefers_known_sectors():
    portfolio = _DummyPortfolio()
    brain = BrokerBrain(portfolio=portfolio, max_positions=2, min_score=0.60)
    brain._sector_map = {"AAA": "Technology", "ETF1": "Unknown", "BBB": "Healthcare"}

    filtered = brain._filter_screened_tickers(["ETF1", "AAA", "BBB"], top_n=3)

    assert filtered == ["AAA", "BBB"]


def test_screen_candidates_uses_cpu_screener_when_device_missing(monkeypatch):
    portfolio = _DummyPortfolio()
    brain = BrokerBrain(portfolio=portfolio, max_positions=2, min_score=0.60, device=None)
    brain._sector_map = {"AAA": "Technology"}

    captured = {}

    def fake_run_screener(df_features, device, top_n=100, **_kwargs):
        captured["device"] = device
        captured["top_n"] = top_n
        return pd.DataFrame({"ticker": ["AAA"]})

    monkeypatch.setattr("os.path.exists", lambda path: True)
    monkeypatch.setattr("pipeline.screener.run_screener", fake_run_screener)

    filtered = brain._screen_candidates(_make_df_features(), top_n=7)

    assert filtered == ["AAA"]
    assert captured["top_n"] == 7
    assert captured["device"].type == "cpu"


def test_run_cycle_uses_trailing_stop_for_winner_pullback():
    portfolio = _DummyPortfolio()
    portfolio.positions["AAA"].update(
        {
            "avg_cost": 100.0,
            "last_price": 118.0,
            "peak_price": 135.0,
            "days_held": 10,
            "partial_taken": True,
        }
    )
    brain = BrokerBrain(portfolio=portfolio, max_positions=2, min_score=0.60)
    brain._sector_map = {"AAA": "Technology"}

    with (
        patch.object(brain, "_screen_candidates", return_value=[]),
        patch.object(brain, "_maybe_refresh_sector_map"),
        patch.object(brain, "_get_current_prices", return_value={"AAA": 118.0}),
        patch.object(brain, "_evaluate_options", return_value=[]),
        patch("broker.brain.validate_portfolio_prices", return_value={"AAA": 118.0}),
        patch("broker.brain.score_sectors", return_value={}),
        patch("broker.brain.get_portfolio_sector_weights", return_value={"Technology": 0.05}),
        patch("broker.brain.compute_target_allocations", return_value={"Technology": 1.0}),
    ):
        decisions = brain.run_cycle(_make_df_features_with_regime(0), screener_top_n=10)

    assert any(d.action == "SELL" and "Trailing stop" in d.reason for d in decisions)


def test_run_cycle_does_not_dump_winner_on_single_weak_signal_in_calm_regime():
    portfolio = _DummyPortfolio()
    portfolio.positions["AAA"].update(
        {
            "avg_cost": 100.0,
            "last_price": 112.0,
            "peak_price": 115.0,
            "days_held": 10,
            "weak_signal_streak": 0,
        }
    )
    brain = BrokerBrain(portfolio=portfolio, max_positions=2, min_score=0.60)
    brain._sector_map = {"AAA": "Technology"}

    with (
        patch.object(brain, "_screen_candidates", return_value=[]),
        patch.object(brain, "_maybe_refresh_sector_map"),
        patch.object(brain, "_get_current_prices", return_value={"AAA": 112.0}),
        patch.object(brain, "_evaluate_options", return_value=[]),
        patch("broker.brain.research", return_value={"composite_score": 0.05}),
        patch("broker.brain.validate_portfolio_prices", return_value={"AAA": 112.0}),
        patch("broker.brain.score_sectors", return_value={}),
        patch("broker.brain.get_portfolio_sector_weights", return_value={"Technology": 0.05}),
        patch("broker.brain.compute_target_allocations", return_value={"Technology": 1.0}),
    ):
        decisions = brain.run_cycle(_make_df_features_with_regime(0), screener_top_n=10)

    assert not any(d.action == "SELL" and "Signal deteriorated" in d.reason for d in decisions)
    assert portfolio.positions["AAA"]["weak_signal_streak"] == 1


def test_run_cycle_exits_on_weak_signal_in_risk_off_regime():
    portfolio = _DummyPortfolio()
    portfolio.positions["AAA"].update(
        {
            "avg_cost": 100.0,
            "last_price": 95.0,
            "peak_price": 101.0,
            "days_held": 10,
            "weak_signal_streak": 0,
        }
    )
    brain = BrokerBrain(portfolio=portfolio, max_positions=2, min_score=0.60)
    brain._sector_map = {"AAA": "Technology"}

    with (
        patch.object(brain, "_screen_candidates", return_value=[]),
        patch.object(brain, "_maybe_refresh_sector_map"),
        patch.object(brain, "_get_current_prices", return_value={"AAA": 95.0}),
        patch.object(brain, "_evaluate_options", return_value=[]),
        patch("broker.brain.research", return_value={"composite_score": 0.05}),
        patch("broker.brain.validate_portfolio_prices", return_value={"AAA": 95.0}),
        patch("broker.brain.score_sectors", return_value={}),
        patch("broker.brain.get_portfolio_sector_weights", return_value={"Technology": 0.05}),
        patch("broker.brain.compute_target_allocations", return_value={"Technology": 1.0}),
    ):
        decisions = brain.run_cycle(_make_df_features_with_regime(3), screener_top_n=10)

    assert any(d.action == "SELL" and "Signal deteriorated" in d.reason for d in decisions)


def test_rl_entry_floor_stays_at_configured_threshold_across_regimes():
    portfolio = _EmptyPortfolio()
    brain = BrokerBrain(
        portfolio=portfolio,
        rl_enabled=True,
        rl_checkpoint_path="models/best_fold9.pt",
        rl_min_score=0.22,
    )

    assert brain._effective_rl_entry_floor(0) == 0.22
    assert brain._effective_rl_entry_floor(1) == 0.22
    assert brain._effective_rl_entry_floor(3) == 0.22


def test_run_cycle_allows_sector_overflow_for_strong_risk_on_candidate():
    portfolio = _EmptyPortfolio()
    brain = BrokerBrain(
        portfolio=portfolio,
        max_positions=1,
        min_score=0.60,
        max_sector_pct=0.40,
        max_position_pct=0.18,
        rl_enabled=True,
        rl_checkpoint_path="models/best_fold9.pt",
    )
    brain._sector_map = {"BBB": "Technology"}

    with (
        patch.object(brain, "_screen_candidates", return_value=["BBB"]),
        patch.object(brain, "_maybe_refresh_sector_map"),
        patch.object(brain, "_get_current_prices", return_value={}),
        patch.object(brain, "_near_earnings", return_value=False),
        patch.object(brain, "_evaluate_options", return_value=[]),
        patch("broker.brain.research", return_value=_research_stub("BBB")),
        patch("broker.brain.get_rl_targets", return_value=pd.Series({"BBB": 1.0}, name="rl_score")),
        patch("broker.brain.validate_portfolio_prices", return_value={}),
        patch("broker.brain.score_sectors", return_value={}),
        patch("broker.brain.get_portfolio_sector_weights", return_value={}),
        patch("broker.brain.compute_target_allocations", return_value={"Technology": 0.10}),
        patch.object(brain, "_assert_model_available"),
    ):
        decisions = brain.run_cycle(_make_df_features_with_regime(0), screener_top_n=10)

    buy = next(d for d in decisions if d.action == "BUY" and d.ticker == "BBB")
    alloc_value = buy.shares * buy.price
    assert alloc_value > portfolio.equity * 0.10


def test_run_cycle_keeps_risk_off_candidate_within_soft_sector_budget():
    portfolio = _EmptyPortfolio()
    brain = BrokerBrain(
        portfolio=portfolio,
        max_positions=1,
        min_score=0.60,
        max_sector_pct=0.40,
        max_position_pct=0.18,
        rl_enabled=True,
        rl_checkpoint_path="models/best_fold9.pt",
    )
    brain._sector_map = {"BBB": "Technology"}

    with (
        patch.object(brain, "_screen_candidates", return_value=["BBB"]),
        patch.object(brain, "_maybe_refresh_sector_map"),
        patch.object(brain, "_get_current_prices", return_value={}),
        patch.object(brain, "_near_earnings", return_value=False),
        patch.object(brain, "_evaluate_options", return_value=[]),
        patch("broker.brain.research", return_value=_research_stub("BBB")),
        patch("broker.brain.get_rl_targets", return_value=pd.Series({"BBB": 1.0}, name="rl_score")),
        patch("broker.brain.validate_portfolio_prices", return_value={}),
        patch("broker.brain.score_sectors", return_value={}),
        patch("broker.brain.get_portfolio_sector_weights", return_value={}),
        patch("broker.brain.compute_target_allocations", return_value={"Technology": 0.10}),
        patch.object(brain, "_assert_model_available"),
    ):
        decisions = brain.run_cycle(_make_df_features_with_regime(3), screener_top_n=10)

    buy = next(d for d in decisions if d.action == "BUY" and d.ticker == "BBB")
    alloc_value = buy.shares * buy.price
    assert alloc_value <= portfolio.equity * 0.10 + 1e-6
