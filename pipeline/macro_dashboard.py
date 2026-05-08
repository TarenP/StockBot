"""Human-readable macro shock diagnostics from the live feature panel."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def build_macro_shock_summary(df_features: pd.DataFrame) -> dict[str, Any]:
    if df_features is None or df_features.empty or not isinstance(df_features.index, pd.MultiIndex):
        return {
            "generated_at": datetime.now().isoformat(),
            "available": False,
            "risk_state": "unknown",
        }
    latest_date = df_features.index.get_level_values("date").max()
    snap = df_features.xs(latest_date, level="date")
    spy_ret_20d = _safe_float(snap.get("spy_ret_20d", pd.Series([0.0])).mean())
    vix_level = _safe_float(snap.get("vix_level", pd.Series([0.0])).mean())
    market_breadth = _safe_float(snap.get("market_breadth", pd.Series([0.5])).mean(), 0.5)
    ret_1d = pd.to_numeric(snap.get("ret_1d", pd.Series(dtype=float)), errors="coerce")
    ret_20d = pd.to_numeric(snap.get("ret_20d", pd.Series(dtype=float)), errors="coerce")
    dispersion = _safe_float(ret_1d.std(), 0.0)
    advancers = float((ret_1d > 0).mean()) if len(ret_1d.dropna()) else 0.0

    stress = 0.0
    stress += max(0.0, -spy_ret_20d) * 2.0
    stress += max(0.0, vix_level - 0.20) * 2.5
    stress += max(0.0, 0.45 - market_breadth)
    stress += max(0.0, dispersion - 0.025) * 3.0
    stress = min(1.0, stress)

    if stress >= 0.65:
        risk_state = "risk_off"
    elif stress >= 0.35:
        risk_state = "cautious"
    elif spy_ret_20d > 0.03 and market_breadth > 0.55:
        risk_state = "risk_on"
    else:
        risk_state = "neutral"

    top_strength = []
    top_weakness = []
    if "ticker" in snap.index.names and not ret_20d.empty:
        ranked = ret_20d.dropna().sort_values()
        top_weakness = [
            {"ticker": str(ticker), "ret_20d": round(float(value), 4)}
            for ticker, value in ranked.head(5).items()
        ]
        top_strength = [
            {"ticker": str(ticker), "ret_20d": round(float(value), 4)}
            for ticker, value in ranked.tail(5).sort_values(ascending=False).items()
        ]

    shocks = []
    if vix_level >= 0.30:
        shocks.append("volatility_spike")
    if spy_ret_20d <= -0.05:
        shocks.append("index_drawdown")
    if market_breadth <= 0.35:
        shocks.append("breadth_breakdown")
    if dispersion >= 0.04:
        shocks.append("cross_section_dispersion")

    return {
        "generated_at": datetime.now().isoformat(),
        "available": True,
        "latest_date": pd.Timestamp(latest_date).date().isoformat(),
        "risk_state": risk_state,
        "stress_score": round(float(stress), 4),
        "spy_ret_20d": round(float(spy_ret_20d), 4),
        "vix_level": round(float(vix_level), 4),
        "market_breadth": round(float(market_breadth), 4),
        "advancer_ratio": round(float(advancers), 4),
        "return_dispersion": round(float(dispersion), 4),
        "active_shocks": shocks,
        "top_strength": top_strength,
        "top_weakness": top_weakness,
    }


def write_macro_shock_summary(
    df_features: pd.DataFrame,
    output_path: str | Path = "broker/state/macro_shock_summary.json",
) -> dict[str, Any]:
    summary = build_macro_shock_summary(df_features)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary

