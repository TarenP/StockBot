import numpy as np
import pandas as pd

from pipeline.macro_dashboard import build_macro_shock_summary
from pipeline.patterns import build_pattern_features, detect_patterns_for_ticker


def _ticker_frame(n=45, breakout=False):
    dates = pd.date_range("2026-03-01", periods=n, freq="B")
    close = np.linspace(90, 100, n)
    if breakout:
        close[-1] = 106
    volume = np.full(n, 1_000_000.0)
    volume[-1] = 2_000_000.0
    frame = pd.DataFrame(
        {
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": volume,
            "atr": np.full(n, 0.02),
            "bb_pct": np.full(n, 0.55),
            "sent_surprise": np.zeros(n),
            "sent_accel": np.zeros(n),
        },
        index=dates,
    )
    return frame


def test_pattern_registry_detects_relative_strength_breakout():
    record = detect_patterns_for_ticker(_ticker_frame(breakout=True))

    names = {item["name"] for item in record["active_patterns"]}
    assert "relative_strength_breakout" in names
    assert record["pattern_score"] > 0
    assert record["pattern_confidence"] > 0


def test_build_pattern_features_handles_multiindex_panel():
    frame = _ticker_frame(breakout=True).copy()
    frame["ticker"] = "AAA"
    frame["date"] = frame.index
    panel = frame.set_index(["date", "ticker"]).sort_index()

    features = build_pattern_features(panel)

    names = {item["name"] for item in features["AAA"]["active_patterns"]}
    assert "relative_strength_breakout" in names


def test_macro_shock_summary_flags_risk_off_context():
    dates = pd.date_range("2026-03-01", periods=2, freq="B")
    rows = []
    for date in dates:
        for ticker, ret in {"AAA": -0.05, "BBB": -0.04, "CCC": 0.01}.items():
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "ret_1d": ret,
                    "ret_20d": ret,
                    "spy_ret_20d": -0.08,
                    "vix_level": 0.35,
                    "market_breadth": 0.25,
                }
            )
    panel = pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()

    summary = build_macro_shock_summary(panel)

    assert summary["available"] is True
    assert summary["risk_state"] == "risk_off"
    assert "volatility_spike" in summary["active_shocks"]
