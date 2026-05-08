"""Technical pattern diagnostics for broker candidates.

These features are intentionally descriptive. They make candidate explanations
and audits richer, but do not alter trades unless a later validation gate
explicitly promotes them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


PATTERN_NAMES = [
    "relative_strength_breakout",
    "failed_breakout",
    "volatility_contraction",
    "accumulation",
    "distribution",
    "gap_and_hold",
    "gap_and_fade",
    "post_earnings_drift",
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def _clamp(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def detect_patterns_for_ticker(frame: pd.DataFrame) -> dict[str, Any]:
    if frame is None or frame.empty or "close" not in frame.columns:
        return _empty_pattern_record()
    g = frame.sort_index().copy()
    close = pd.to_numeric(g["close"], errors="coerce")
    high = pd.to_numeric(g.get("high", close), errors="coerce")
    low = pd.to_numeric(g.get("low", close), errors="coerce")
    open_ = pd.to_numeric(g.get("open", close), errors="coerce")
    volume = pd.to_numeric(g.get("volume", 0.0), errors="coerce").fillna(0.0)
    if close.dropna().shape[0] < 25:
        return _empty_pattern_record()

    latest_close = _safe_float(close.iloc[-1])
    prev_close = _safe_float(close.iloc[-2], latest_close)
    ret_1d = latest_close / prev_close - 1.0 if prev_close > 0 else 0.0
    ret_5d = latest_close / _safe_float(close.iloc[-6], latest_close) - 1.0 if len(close) > 6 else 0.0
    ret_20d = latest_close / _safe_float(close.iloc[-21], latest_close) - 1.0 if len(close) > 21 else 0.0
    high_20_prev = _safe_float(high.shift(1).rolling(20, min_periods=10).max().iloc[-1], latest_close)
    low_20_prev = _safe_float(low.shift(1).rolling(20, min_periods=10).min().iloc[-1], latest_close)
    vol_ma20 = _safe_float(volume.rolling(20, min_periods=5).mean().iloc[-1], 0.0)
    vol_ratio = _safe_float(volume.iloc[-1] / vol_ma20, 1.0) if vol_ma20 > 0 else 1.0
    atr = pd.to_numeric(g.get("atr", pd.Series(index=g.index, data=np.nan)), errors="coerce")
    atr_now = _safe_float(atr.iloc[-1], 0.0)
    atr_ma20 = _safe_float(atr.rolling(20, min_periods=5).mean().iloc[-1], atr_now)
    bb_pct = _safe_float(g["bb_pct"].iloc[-1], 0.5) if "bb_pct" in g.columns else 0.5
    sent_surprise = _safe_float(g["sent_surprise"].iloc[-1], 0.0) if "sent_surprise" in g.columns else 0.0
    sent_accel = _safe_float(g["sent_accel"].iloc[-1], 0.0) if "sent_accel" in g.columns else 0.0

    gap = _safe_float(open_.iloc[-1] / prev_close - 1.0, 0.0) if prev_close > 0 else 0.0
    intraday_hold = _safe_float((latest_close - open_.iloc[-1]) / open_.iloc[-1], 0.0) if open_.iloc[-1] else 0.0
    breakout = latest_close > high_20_prev and ret_5d > 0
    failed_breakout = high.iloc[-1] > high_20_prev and latest_close < high_20_prev and ret_1d < 0
    contraction = atr_now > 0 and atr_ma20 > 0 and atr_now < atr_ma20 * 0.75 and 0.35 <= bb_pct <= 0.65
    accumulation = ret_20d > 0.03 and vol_ratio > 1.15 and ret_1d >= -0.01
    distribution = ret_20d < -0.03 and vol_ratio > 1.15 and ret_1d <= 0.01
    gap_and_hold = gap > 0.02 and intraday_hold >= 0 and latest_close > high_20_prev
    gap_and_fade = gap > 0.02 and intraday_hold < -0.015
    post_earnings_drift = abs(sent_surprise) > 0.15 and np.sign(sent_surprise) == np.sign(ret_5d or sent_surprise)

    raw_scores = {
        "relative_strength_breakout": _clamp((ret_20d * 2.0) + (vol_ratio - 1.0) * 0.20) if breakout else 0.0,
        "failed_breakout": -_clamp(abs(ret_1d) * 6.0 + (vol_ratio - 1.0) * 0.15, 0.0, 1.0) if failed_breakout else 0.0,
        "volatility_contraction": 0.25 if contraction else 0.0,
        "accumulation": _clamp(0.20 + ret_20d + (vol_ratio - 1.0) * 0.10) if accumulation else 0.0,
        "distribution": -_clamp(0.20 + abs(ret_20d) + (vol_ratio - 1.0) * 0.10, 0.0, 1.0) if distribution else 0.0,
        "gap_and_hold": _clamp(0.35 + gap + intraday_hold) if gap_and_hold else 0.0,
        "gap_and_fade": -_clamp(0.35 + gap + abs(intraday_hold), 0.0, 1.0) if gap_and_fade else 0.0,
        "post_earnings_drift": _clamp((sent_surprise + sent_accel) * 1.5) if post_earnings_drift else 0.0,
    }
    active = [
        {"name": name, "score": round(score, 4)}
        for name, score in raw_scores.items()
        if abs(float(score)) > 1e-9
    ]
    active = sorted(active, key=lambda item: abs(item["score"]), reverse=True)
    total_score = _clamp(sum(raw_scores.values()))
    confidence = min(1.0, 0.20 + 0.15 * len(active) + min(abs(total_score), 0.5))
    primary = active[0]["name"] if active else "none"
    return {
        "pattern_score": float(total_score),
        "pattern_confidence": float(confidence if active else 0.0),
        "primary_pattern": primary,
        "active_patterns": active[:5],
        "pattern_components": {name: float(score) for name, score in raw_scores.items()},
    }


def _empty_pattern_record() -> dict[str, Any]:
    return {
        "pattern_score": 0.0,
        "pattern_confidence": 0.0,
        "primary_pattern": "none",
        "active_patterns": [],
        "pattern_components": {name: 0.0 for name in PATTERN_NAMES},
    }


def build_pattern_features(df_features: pd.DataFrame, tickers: list[str] | None = None) -> dict[str, dict[str, Any]]:
    if df_features is None or df_features.empty or not isinstance(df_features.index, pd.MultiIndex):
        return {}
    wanted = {str(t).upper() for t in tickers or [] if str(t).strip()}
    out: dict[str, dict[str, Any]] = {}
    for ticker in df_features.index.get_level_values("ticker").unique():
        symbol = str(ticker).upper()
        if wanted and symbol not in wanted:
            continue
        try:
            out[symbol] = detect_patterns_for_ticker(df_features.xs(ticker, level="ticker"))
        except Exception:
            out[symbol] = _empty_pattern_record()
    return out

