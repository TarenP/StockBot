import pandas as pd

import broker.brain as brain_module
from broker.brain import BrokerBrain
from pipeline.features import FEATURE_COLS


class _DummyOptions:
    positions = {}


class _DummyPortfolio:
    def __init__(self):
        self.cash = 10_000.0
        self.positions = {}
        self.options = _DummyOptions()

    @property
    def equity(self) -> float:
        return self.cash

    @property
    def position_values(self) -> dict:
        return {}


def _feature_frame() -> pd.DataFrame:
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2026-04-13"]), ["AAA"]],
        names=["date", "ticker"],
    )
    data = {col: [0.0] for col in FEATURE_COLS}
    data.update(
        {
            "ret_5d": [2.0],
            "ret_20d": [1.6],
            "macd_hist": [1.2],
            "vol_ratio": [1.5],
            "vol_zscore": [1.1],
            "price_pos_52w": [1.4],
            "sent_net": [1.3],
            "sent_surprise": [1.0],
            "sent_accel": [0.9],
            "sent_trend": [0.8],
            "rsi": [0.0],
            "bb_pct": [0.0],
            "atr": [0.4],
            "sent_pos_raw": [0.9],
            "close": [100.0],
            "volume": [1_000_000.0],
        }
    )
    return pd.DataFrame(data, index=index)


def test_run_cycle_falls_back_to_local_feature_research(monkeypatch):
    portfolio = _DummyPortfolio()
    brain = BrokerBrain(
        portfolio=portfolio,
        max_positions=5,
        min_score=0.50,
        max_sector_pct=0.30,
    )

    monkeypatch.setattr(
        brain,
        "_maybe_refresh_sector_map",
        lambda df: setattr(brain, "_sector_map", {"AAA": "Technology"}),
    )
    monkeypatch.setattr(brain, "_screen_candidates", lambda df, top_n=100: ["AAA"])
    monkeypatch.setattr(brain, "_evaluate_options", lambda researched, df: [])
    monkeypatch.setattr(brain_module, "score_sectors", lambda df, sector_map: {"Technology": 1.0})
    monkeypatch.setattr(brain_module, "get_portfolio_sector_weights", lambda positions, sector_map: {})
    monkeypatch.setattr(
        brain_module,
        "compute_target_allocations",
        lambda sector_scores, current_sector_weights, max_single_sector: {"Technology": 0.30},
    )
    monkeypatch.setattr(brain_module, "research", lambda ticker: None)
    monkeypatch.setattr(brain_module, "_get_next_earnings_date", lambda ticker: None)

    decisions = brain.run_cycle(_feature_frame(), screener_top_n=10, risk_engine=None)

    buy_decisions = [d for d in decisions if d.action == "BUY"]
    assert len(buy_decisions) == 1
    assert buy_decisions[0].ticker == "AAA"
    assert buy_decisions[0].price == 100.0
    assert "Sector=Technology" in buy_decisions[0].reason
