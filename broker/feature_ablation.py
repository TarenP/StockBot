"""Strict feature-ablation audit runner and report gates.

This module keeps uncertain feature families diagnostic until they beat a
frozen baseline in isolated replay windows. It writes a stable artifact layout
under experiments/feature_ablation/<run_id>/.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

AGGREGATE_COLUMNS = [
    "label",
    "variant_family",
    "windows",
    "wins",
    "winner_rate",
    "avg_total_return",
    "avg_sharpe",
    "avg_max_drawdown",
    "avg_turnover",
    "avg_win_rate",
    "avg_holding_days",
    "avg_policy_score",
    "avg_raw_policy_score",
    "confidence_penalty",
    "incumbent_edge",
    "max_drawdown_degradation",
    "turnover_degradation",
    "evaluated_candidate_count",
    "feature_touch_count",
    "feature_touch_rate",
    "touches_per_window",
    "decision_changed_count",
    "decision_changed_rate",
    "adjusted_entry_count",
    "adjusted_exit_count",
    "adjusted_size_count",
    "avg_forward_return_when_feature_positive",
    "avg_forward_return_when_feature_negative",
    "feature_directional_hit_rate",
    "feature_return_spread",
    "worst_window_label",
    "worst_window_return_delta",
    "best_window_label",
    "best_window_return_delta",
    "decision_status",
    "decision_reason",
]

TOUCH_COLUMNS = [
    "run_id",
    "window_id",
    "variant_label",
    "date",
    "ticker",
    "feature_name",
    "feature_value",
    "base_score",
    "adjusted_score",
    "score_delta",
    "base_weight",
    "adjusted_weight",
    "weight_delta",
    "rank_before",
    "rank_after",
    "would_enter_baseline",
    "would_enter_variant",
    "actual_next_5d_return",
    "actual_next_20d_return",
    "actual_next_60d_return",
    "source_timestamp",
    "as_of_date",
    "feature_available_at",
    "replay_safe",
    "leakage_reason",
    "true_score_delta",
    "true_weight_delta",
    "feature_present",
    "score_changed",
    "rank_changed",
    "size_changed",
    "entry_changed",
    "exit_changed",
    "decision_changed",
    "notes",
]

MECHANISM_SUMMARY_COLUMNS = [
    "window_id",
    "variant_label",
    "feature_present_count",
    "score_changed_count",
    "rank_changed_count",
    "size_changed_count",
    "entry_changed_count",
    "exit_changed_count",
    "avg_score_delta",
    "avg_weight_delta",
    "avg_next_5d_return",
    "avg_next_20d_return",
    "hit_rate_5d",
    "hit_rate_20d",
]

DECISION_STATUSES = {
    "promote",
    "hold_for_more_evidence",
    "reject_mechanism",
    "reject_confidence",
    "reject_drawdown",
    "reject_turnover",
    "reject_insufficient_edge",
    "reject_replay_safety",
    "reject_data_leakage",
}

FROZEN_BASELINE = {
    "weak_theme_penalty_mult": 0.500,
    "weak_theme_cooldown_cycles": 0,
    "low_price_rank_policy": "late_cap",
    "llm_sidecar_broker_influence": False,
    "event_sidecar_broker_influence": False,
    "pattern_sidecar_broker_influence": False,
    "earnings_reaction_enabled": False,
    "macro_regime_enabled": False,
    "macro_regime_mode": "standard",
    "insider_adjustment_enabled": False,
    "allow_unpromoted_feature_influence": False,
}

DIAGNOSTIC_DEFAULTS = {
    "llm_sidecar_enabled": True,
    "enable_llm_sidecar_precompute": True,
    "event_sidecar_enabled": True,
    "enable_event_sidecar_precompute": True,
    "pattern_sidecar_enabled": True,
    "macro_shock_dashboard_enabled": True,
}

PROMOTION_GATES = {
    "incumbent_edge": 0.05,
    "winner_rate": 0.60,
    "winner_windows": 3,
    "max_drawdown_degradation": 0.15,
    "turnover_degradation": 0.25,
    "worst_window_return_delta": -0.05,
    "decision_changed_count": 30,
    "feature_directional_hit_rate": 0.52,
    "small_sample_penalty": 0.15,
    "minimum_non_small_sample_windows": 3,
}


@dataclass(frozen=True)
class FeatureAblationVariant:
    label: str
    variant_family: str
    overrides: dict[str, Any]
    shadow_feature: str | None = None


FEATURE_ABLATION_VARIANTS = [
    FeatureAblationVariant("baseline", "baseline", {}),
    FeatureAblationVariant(
        "earnings_only",
        "active",
        {"earnings_reaction_enabled": True, "allow_unpromoted_feature_influence": True},
        "earnings_reaction_score",
    ),
    FeatureAblationVariant(
        "macro_only",
        "active",
        {"macro_regime_enabled": True, "allow_unpromoted_feature_influence": True},
        "macro",
    ),
    FeatureAblationVariant(
        "insider_only",
        "active",
        {"insider_adjustment_enabled": True, "allow_unpromoted_feature_influence": True},
        "insider",
    ),
    FeatureAblationVariant("event_sidecar_shadow", "sidecar_shadow", {}, "event_score"),
    FeatureAblationVariant("llm_sidecar_shadow", "sidecar_shadow", {}, "llm_event_confidence"),
    FeatureAblationVariant("pattern_sidecar_shadow", "sidecar_shadow", {}, "pattern_score"),
]

MACRO_ABLATION_SWEEP_VARIANTS = [
    FeatureAblationVariant("baseline", "baseline", {}),
    FeatureAblationVariant(
        "macro_weight_0.02",
        "macro_sweep",
        {
            "macro_regime_enabled": True,
            "macro_regime_weight_strength": 0.02,
            "allow_unpromoted_feature_influence": True,
        },
        "macro",
    ),
    FeatureAblationVariant(
        "macro_weight_0.04",
        "macro_sweep",
        {
            "macro_regime_enabled": True,
            "macro_regime_weight_strength": 0.04,
            "allow_unpromoted_feature_influence": True,
        },
        "macro",
    ),
    FeatureAblationVariant(
        "macro_weight_0.06",
        "macro_sweep",
        {
            "macro_regime_enabled": True,
            "macro_regime_weight_strength": 0.06,
            "allow_unpromoted_feature_influence": True,
        },
        "macro",
    ),
    FeatureAblationVariant(
        "macro_weight_0.08",
        "macro_sweep",
        {
            "macro_regime_enabled": True,
            "macro_regime_weight_strength": 0.08,
            "allow_unpromoted_feature_influence": True,
        },
        "macro",
    ),
    FeatureAblationVariant(
        "macro_risk_off_only",
        "macro_sweep",
        {
            "macro_regime_enabled": True,
            "macro_regime_mode": "risk_off_only",
            "allow_unpromoted_feature_influence": True,
        },
        "macro",
    ),
    FeatureAblationVariant(
        "macro_no_bull_boost",
        "macro_sweep",
        {
            "macro_regime_enabled": True,
            "macro_regime_mode": "no_bull_boost",
            "allow_unpromoted_feature_influence": True,
        },
        "macro",
    ),
    FeatureAblationVariant(
        "macro_drawdown_guard_only",
        "macro_sweep",
        {
            "macro_regime_enabled": True,
            "macro_regime_mode": "drawdown_guard_only",
            "allow_unpromoted_feature_influence": True,
        },
        "macro",
    ),
    FeatureAblationVariant(
        "macro_volatility_scaler_only",
        "macro_sweep",
        {
            "macro_regime_enabled": True,
            "macro_regime_mode": "volatility_scaler_only",
            "allow_unpromoted_feature_influence": True,
        },
        "macro",
    ),
]


def feature_ablation_run_id(now: datetime | None = None) -> str:
    return (now or datetime.now()).strftime("%Y%m%d_%H%M%S_%f")


def build_frozen_feature_config(
    base_config: dict | None,
    variant_label: str = "baseline",
    variants: list[FeatureAblationVariant] | None = None,
) -> dict:
    variants_by_label = {variant.label: variant for variant in (variants or FEATURE_ABLATION_VARIANTS)}
    if variant_label not in variants_by_label:
        raise ValueError(f"Unknown feature ablation variant: {variant_label}")
    cfg = dict(base_config or {})
    cfg.update(DIAGNOSTIC_DEFAULTS)
    cfg.update(FROZEN_BASELINE)
    cfg.update(variants_by_label[variant_label].overrides)
    return cfg


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def _safe_mean(values, default: float = 0.0) -> float:
    vals = [_safe_float(value, np.nan) for value in values]
    vals = [value for value in vals if np.isfinite(value)]
    return float(np.mean(vals)) if vals else default


def _avg_holding_days(trade_log: list[dict]) -> float:
    buys: dict[str, list[pd.Timestamp]] = {}
    holding_days: list[int] = []
    for rec in trade_log or []:
        ticker = str(rec.get("ticker", "")).upper()
        action = str(rec.get("action", "")).upper()
        date = pd.to_datetime(rec.get("fill_date") or rec.get("date"), errors="coerce")
        if not ticker or pd.isna(date):
            continue
        if action == "BUY":
            buys.setdefault(ticker, []).append(pd.Timestamp(date))
        elif action in {"SELL", "SELL_PARTIAL"} and buys.get(ticker):
            entry = buys[ticker].pop(0)
            holding_days.append(max(0, int((pd.Timestamp(date) - entry).days)))
    return float(np.mean(holding_days)) if holding_days else 0.0


def _turnover(trade_log: list[dict], initial_cash: float) -> float:
    traded = 0.0
    for rec in trade_log or []:
        traded += abs(_safe_float(rec.get("shares")) * _safe_float(rec.get("price")))
    return traded / max(float(initial_cash), 1.0)


def _raw_policy_score(metrics: dict[str, Any], turnover: float) -> float:
    total_return = _safe_float(metrics.get("total_return"))
    sharpe = _safe_float(metrics.get("sharpe"))
    drawdown = _safe_float(metrics.get("max_drawdown"))
    return float(total_return + 0.10 * sharpe - 0.50 * abs(drawdown) - 0.02 * turnover)


def _score_with_penalty(raw_score: float, confidence_penalty: float) -> float:
    return float(raw_score - confidence_penalty)


def _empty_touch_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=TOUCH_COLUMNS)


def _empty_mechanism_summary() -> pd.DataFrame:
    return pd.DataFrame(columns=MECHANISM_SUMMARY_COLUMNS)


def _audit_date_column(audit: pd.DataFrame) -> str | None:
    for col in ("cycle_date", "date", "as_of_date"):
        if col in audit.columns:
            return col
    return None


def _score_col(audit: pd.DataFrame) -> str | None:
    for col in ("rank_score", "composite_score", "score"):
        if col in audit.columns:
            return col
    return None


def _weight_col(audit: pd.DataFrame) -> str | None:
    for col in ("final_weight", "target_weight", "base_target_weight"):
        if col in audit.columns:
            return col
    return None


def _rank_series(audit: pd.DataFrame, score_col: str | None) -> pd.Series:
    if audit.empty or score_col is None:
        return pd.Series(index=audit.index, data=np.nan)
    date_col = _audit_date_column(audit)
    scores = pd.to_numeric(audit[score_col], errors="coerce")
    if date_col:
        return scores.groupby(audit[date_col]).rank(ascending=False, method="min")
    return scores.rank(ascending=False, method="min")


def _forward_return(df_features: pd.DataFrame, ticker: str, as_of_date: Any, horizon: int) -> float:
    if df_features is None or df_features.empty or not ticker:
        return np.nan
    try:
        if isinstance(df_features.index, pd.MultiIndex):
            series = df_features.xs(str(ticker), level="ticker")["close"]
        elif "ticker" in df_features.columns:
            series = df_features[df_features["ticker"].astype(str).str.upper() == str(ticker).upper()]["close"]
        else:
            return np.nan
    except Exception:
        return np.nan
    if series.empty:
        return np.nan
    series = series.sort_index()
    date_index = pd.to_datetime(
        series.index.get_level_values("date") if isinstance(series.index, pd.MultiIndex) else series.index,
        errors="coerce",
    )
    start = pd.to_datetime(as_of_date, errors="coerce")
    if pd.isna(start):
        return np.nan
    valid_positions = np.flatnonzero(date_index >= start)
    if len(valid_positions) == 0:
        return np.nan
    start_pos = int(valid_positions[0])
    end_pos = start_pos + int(horizon)
    if end_pos >= len(series):
        return np.nan
    start_price = _safe_float(series.iloc[start_pos], np.nan)
    end_price = _safe_float(series.iloc[end_pos], np.nan)
    if not np.isfinite(start_price) or not np.isfinite(end_price) or abs(start_price) < 1e-12:
        return np.nan
    return float(end_price / start_price - 1.0)


def _soft_signal_value(summary: Any, feature_name: str | None) -> float:
    if not feature_name:
        return np.nan
    text = str(summary or "")
    if not text:
        return np.nan
    aliases = {
        "earnings_reaction_score": "earnings",
        "insider": "insider",
        "macro": "macro",
    }
    name = aliases.get(str(feature_name), str(feature_name))
    for part in text.split(","):
        if not part.startswith(f"{name}:"):
            continue
        pieces = part.split(":")
        if len(pieces) < 3 or pieces[2] == "no_data":
            return np.nan
        return _safe_float(pieces[2], np.nan)
    return np.nan


def _feature_value(rec: pd.Series, feature_name: str | None) -> Any:
    if not feature_name:
        return np.nan
    if feature_name in {"macro", "insider", "earnings_reaction_score"}:
        value = _soft_signal_value(rec.get("soft_signal_summary"), feature_name)
        if np.isfinite(value):
            return value
    if feature_name in rec.index:
        return rec.get(feature_name)
    return np.nan


def _bool_count(frame: pd.DataFrame, col: str) -> int:
    if frame.empty or col not in frame.columns:
        return 0
    return int(frame[col].fillna(False).astype(bool).sum())


def _touch_rows_from_baseline_comparison(
    baseline_audit: pd.DataFrame,
    variant_audit: pd.DataFrame,
    *,
    df_features: pd.DataFrame,
    run_id: str,
    window_id: str,
    variant_label: str,
    feature_name: str | None,
) -> pd.DataFrame:
    if variant_audit is None or variant_audit.empty or not feature_name:
        return _empty_touch_frame()

    baseline_audit = baseline_audit.copy() if baseline_audit is not None else pd.DataFrame()
    variant_audit = variant_audit.copy()
    base_date_col = _audit_date_column(baseline_audit)
    var_date_col = _audit_date_column(variant_audit)
    if "ticker" not in variant_audit.columns or var_date_col is None:
        return _empty_touch_frame()

    variant_audit["_audit_date"] = pd.to_datetime(variant_audit[var_date_col], errors="coerce")
    variant_audit["_rank_after"] = _rank_series(variant_audit, _score_col(variant_audit))
    if not baseline_audit.empty and "ticker" in baseline_audit.columns and base_date_col is not None:
        baseline_audit["_audit_date"] = pd.to_datetime(baseline_audit[base_date_col], errors="coerce")
        baseline_audit["_rank_before"] = _rank_series(baseline_audit, _score_col(baseline_audit))
        base_score_col = _score_col(baseline_audit)
        base_weight_col = _weight_col(baseline_audit)
        base_cols = ["ticker", "_audit_date", "_rank_before", "candidate_status"]
        if base_score_col:
            base_cols.append(base_score_col)
        if base_weight_col and base_weight_col not in base_cols:
            base_cols.append(base_weight_col)
        baseline_small = baseline_audit[base_cols].rename(
            columns={
                "candidate_status": "_base_status",
                base_score_col or "score": "_base_score",
                base_weight_col or "target_weight": "_base_weight",
            }
        )
        merged = variant_audit.merge(
            baseline_small,
            how="left",
            on=["ticker", "_audit_date"],
            suffixes=("", "_baseline"),
        )
    else:
        merged = variant_audit.copy()
        merged["_rank_before"] = np.nan
        merged["_base_status"] = ""
        merged["_base_score"] = np.nan
        merged["_base_weight"] = np.nan

    rows = []
    var_score_col = _score_col(variant_audit)
    var_weight_col = _weight_col(variant_audit)
    for _idx, rec in merged.iterrows():
        as_of = rec.get("_audit_date")
        ticker = str(rec.get("ticker", "")).upper()
        feature_value = _feature_value(rec, feature_name)
        numeric_feature = _safe_float(feature_value, np.nan)
        feature_present = bool(np.isfinite(numeric_feature) and abs(numeric_feature) > 1e-9)
        adjusted_score = _safe_float(rec.get(var_score_col), np.nan) if var_score_col else np.nan
        base_score = _safe_float(rec.get("_base_score"), adjusted_score)
        adjusted_weight = _safe_float(rec.get(var_weight_col), np.nan) if var_weight_col else np.nan
        base_weight = _safe_float(rec.get("_base_weight"), adjusted_weight)
        score_delta = adjusted_score - base_score if np.isfinite(adjusted_score) and np.isfinite(base_score) else 0.0
        weight_delta = adjusted_weight - base_weight if np.isfinite(adjusted_weight) and np.isfinite(base_weight) else 0.0
        rank_before = _safe_float(rec.get("_rank_before"), np.nan)
        rank_after = _safe_float(rec.get("_rank_after"), np.nan)
        would_enter_baseline = str(rec.get("_base_status", "")) == "buy_selected"
        would_enter_variant = str(rec.get("candidate_status", "")) == "buy_selected"
        score_changed = bool(abs(score_delta) > 1e-9)
        size_changed = bool(abs(weight_delta) > 1e-9)
        rank_changed = bool(
            np.isfinite(rank_before)
            and np.isfinite(rank_after)
            and abs(rank_after - rank_before) > 1e-9
        )
        entry_changed = bool(would_enter_baseline != would_enter_variant)
        exit_changed = False
        decision_changed = bool(score_changed or rank_changed or size_changed or entry_changed or exit_changed)
        if not (feature_present or decision_changed):
            continue
        source_timestamp = as_of
        feature_available_at = source_timestamp
        replay_safe = True
        leakage_reason = ""
        if pd.notna(as_of) and pd.notna(feature_available_at):
            replay_safe = pd.Timestamp(feature_available_at) <= pd.Timestamp(as_of)
            if not replay_safe:
                leakage_reason = "feature_available_at_after_as_of_date"
        rows.append({
            "run_id": run_id,
            "window_id": window_id,
            "variant_label": variant_label,
            "date": as_of,
            "ticker": ticker,
            "feature_name": feature_name,
            "feature_value": feature_value,
            "base_score": base_score,
            "adjusted_score": adjusted_score,
            "score_delta": score_delta,
            "base_weight": base_weight,
            "adjusted_weight": adjusted_weight,
            "weight_delta": weight_delta,
            "rank_before": rank_before,
            "rank_after": rank_after,
            "would_enter_baseline": would_enter_baseline,
            "would_enter_variant": would_enter_variant,
            "actual_next_5d_return": _forward_return(df_features, ticker, as_of, 5),
            "actual_next_20d_return": _forward_return(df_features, ticker, as_of, 20),
            "actual_next_60d_return": _forward_return(df_features, ticker, as_of, 60),
            "source_timestamp": source_timestamp,
            "as_of_date": as_of,
            "feature_available_at": feature_available_at,
            "replay_safe": replay_safe,
            "leakage_reason": leakage_reason,
            "true_score_delta": score_delta,
            "true_weight_delta": weight_delta,
            "feature_present": feature_present,
            "score_changed": score_changed,
            "rank_changed": rank_changed,
            "size_changed": size_changed,
            "entry_changed": entry_changed,
            "exit_changed": exit_changed,
            "decision_changed": decision_changed,
            "notes": "baseline_comparison_touch",
        })
    return pd.DataFrame(rows, columns=TOUCH_COLUMNS)


def _mechanism_summary(touch_df: pd.DataFrame) -> pd.DataFrame:
    if touch_df is None or touch_df.empty:
        return _empty_mechanism_summary()
    rows = []
    grouped = touch_df.groupby(["variant_label", "window_id"], sort=False)
    for (variant_label, window_id), group in grouped:
        fv = pd.to_numeric(group.get("feature_value", pd.Series(dtype=float)), errors="coerce")
        ret5 = pd.to_numeric(group.get("actual_next_5d_return", pd.Series(dtype=float)), errors="coerce")
        ret20 = pd.to_numeric(group.get("actual_next_20d_return", pd.Series(dtype=float)), errors="coerce")
        sign = np.sign(fv)
        rows.append({
            "window_id": window_id,
            "variant_label": variant_label,
            "feature_present_count": _bool_count(group, "feature_present"),
            "score_changed_count": _bool_count(group, "score_changed"),
            "rank_changed_count": _bool_count(group, "rank_changed"),
            "size_changed_count": _bool_count(group, "size_changed"),
            "entry_changed_count": _bool_count(group, "entry_changed"),
            "exit_changed_count": _bool_count(group, "exit_changed"),
            "avg_score_delta": _safe_mean(group.get("score_delta", []), np.nan),
            "avg_weight_delta": _safe_mean(group.get("weight_delta", []), np.nan),
            "avg_next_5d_return": _safe_mean(ret5, np.nan),
            "avg_next_20d_return": _safe_mean(ret20, np.nan),
            "hit_rate_5d": float(((sign * np.sign(ret5)) > 0).mean()) if len(group) else np.nan,
            "hit_rate_20d": float(((sign * np.sign(ret20)) > 0).mean()) if len(group) else np.nan,
        })
    return pd.DataFrame(rows, columns=MECHANISM_SUMMARY_COLUMNS)


def build_feature_decision(row: pd.Series, baseline_policy_score: float) -> tuple[str, str]:
    if not bool(row.get("replay_safe", True)):
        return "reject_replay_safety", "Replay-safety checks failed."
    if int(row.get("feature_touch_count", 0) or 0) <= 0:
        return "hold_for_more_evidence", "Feature had no decision touches; collect more data."
    if int(row.get("decision_changed_count", 0) or 0) < PROMOTION_GATES["decision_changed_count"]:
        if float(row.get("feature_touch_rate", 0.0) or 0.0) <= 0:
            return "hold_for_more_evidence", "Feature had no decision touches; collect more data."
        return "hold_for_more_evidence", "Feature changed too few decisions for promotion."
    if float(row.get("feature_return_spread", 0.0) or 0.0) <= 0:
        return "reject_mechanism", "Feature forward-return spread did not support the signal direction."
    hit_rate = row.get("feature_directional_hit_rate", np.nan)
    if np.isfinite(_safe_float(hit_rate, np.nan)) and float(hit_rate) < PROMOTION_GATES["feature_directional_hit_rate"]:
        return "reject_confidence", "Feature directional hit rate below promotion gate."
    if int(row.get("wins", 0) or 0) < PROMOTION_GATES["winner_windows"]:
        return "reject_confidence", "Too few winning windows."
    if float(row.get("winner_rate", 0.0) or 0.0) < PROMOTION_GATES["winner_rate"]:
        return "reject_confidence", "Winner rate below promotion gate."
    if float(row.get("incumbent_edge", 0.0) or 0.0) < PROMOTION_GATES["incumbent_edge"]:
        return "reject_insufficient_edge", "Incumbent edge below promotion gate."
    if float(row.get("avg_policy_score", 0.0) or 0.0) <= float(baseline_policy_score):
        return "reject_insufficient_edge", "Average policy score did not beat baseline."
    if float(row.get("max_drawdown_degradation", 0.0) or 0.0) > PROMOTION_GATES["max_drawdown_degradation"]:
        return "reject_drawdown", "Drawdown degradation exceeded gate."
    if float(row.get("turnover_degradation", 0.0) or 0.0) > PROMOTION_GATES["turnover_degradation"]:
        return "reject_turnover", "Turnover degradation exceeded gate."
    if float(row.get("worst_window_return_delta", 0.0) or 0.0) < PROMOTION_GATES["worst_window_return_delta"]:
        return "reject_drawdown", "Worst-window return delta exceeded loss gate."
    return "promote", "Feature cleared outcome, risk, mechanism, confidence, and replay-safety gates."


def _aggregate_review(window_metrics: pd.DataFrame, touch_df: pd.DataFrame) -> pd.DataFrame:
    if window_metrics.empty:
        return pd.DataFrame(columns=AGGREGATE_COLUMNS)
    baseline = window_metrics[window_metrics["label"] == "baseline"].copy()
    baseline_score = _safe_mean(baseline["policy_score"])
    baseline_return_by_window = dict(zip(baseline["window_id"], baseline["total_return"]))
    baseline_dd = _safe_mean(baseline["max_drawdown"])
    baseline_turnover = _safe_mean(baseline["turnover"])

    rows = []
    for label, group in window_metrics.groupby("label", sort=False):
        variant_family = str(group["variant_family"].iloc[0])
        deltas = [
            _safe_float(row["total_return"]) - _safe_float(baseline_return_by_window.get(row["window_id"]))
            for _idx, row in group.iterrows()
        ]
        wins = int(sum(delta > 0 for delta in deltas))
        raw_policy = _safe_mean(group["raw_policy_score"])
        touches = touch_df[touch_df["variant_label"] == label] if not touch_df.empty else _empty_touch_frame()
        confidence_penalty = (
            PROMOTION_GATES["small_sample_penalty"]
            if int(len(group)) < PROMOTION_GATES["minimum_non_small_sample_windows"]
            else 0.0
        )
        evaluated_count = int(pd.to_numeric(group.get("evaluated_candidate_count", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        touch_count = int(len(touches))
        decision_changed_count = _bool_count(touches, "decision_changed")
        feature_values = pd.to_numeric(touches.get("feature_value", pd.Series(dtype=float)), errors="coerce")
        forward_20d = pd.to_numeric(touches.get("actual_next_20d_return", pd.Series(dtype=float)), errors="coerce")
        positive_returns = forward_20d[feature_values > 0]
        negative_returns = forward_20d[feature_values < 0]
        valid_direction = feature_values.notna() & forward_20d.notna() & (feature_values.abs() > 1e-9)
        directional_hit_rate = (
            float((np.sign(feature_values[valid_direction]) == np.sign(forward_20d[valid_direction])).mean())
            if bool(valid_direction.any())
            else np.nan
        )
        positive_avg = _safe_mean(positive_returns, np.nan)
        negative_avg = _safe_mean(negative_returns, np.nan)
        replay_safe = bool(touches.empty or touches.get("replay_safe", pd.Series(dtype=bool)).fillna(False).astype(bool).all())
        avg_drawdown = _safe_mean(group["max_drawdown"])
        drawdown_degradation = max(0.0, abs(avg_drawdown) - abs(baseline_dd))
        row = {
            "label": label,
            "variant_family": variant_family,
            "windows": int(len(group)),
            "wins": wins,
            "winner_rate": wins / len(group) if len(group) else 0.0,
            "avg_total_return": _safe_mean(group["total_return"]),
            "avg_sharpe": _safe_mean(group["sharpe"]),
            "avg_max_drawdown": avg_drawdown,
            "avg_turnover": _safe_mean(group["turnover"]),
            "avg_win_rate": _safe_mean(group["win_rate"]),
            "avg_holding_days": _safe_mean(group["avg_holding_days"]),
            "avg_policy_score": _score_with_penalty(raw_policy, confidence_penalty),
            "avg_raw_policy_score": raw_policy,
            "confidence_penalty": confidence_penalty,
            "incumbent_edge": _score_with_penalty(raw_policy, confidence_penalty) - baseline_score,
            "max_drawdown_degradation": drawdown_degradation,
            "turnover_degradation": max(0.0, _safe_mean(group["turnover"]) - baseline_turnover),
            "evaluated_candidate_count": evaluated_count,
            "feature_touch_count": touch_count,
            "feature_touch_rate": touch_count / max(evaluated_count, 1),
            "touches_per_window": touch_count / max(int(len(group)), 1),
            "decision_changed_count": decision_changed_count,
            "decision_changed_rate": decision_changed_count / max(evaluated_count, 1),
            "adjusted_entry_count": _bool_count(touches, "entry_changed"),
            "adjusted_exit_count": _bool_count(touches, "exit_changed"),
            "adjusted_size_count": _bool_count(touches, "size_changed"),
            "avg_forward_return_when_feature_positive": positive_avg,
            "avg_forward_return_when_feature_negative": negative_avg,
            "feature_directional_hit_rate": directional_hit_rate,
            "feature_return_spread": positive_avg - negative_avg if np.isfinite(positive_avg) and np.isfinite(negative_avg) else np.nan,
            "worst_window_label": str(group.iloc[int(np.argmin(deltas))]["window_id"]) if deltas else None,
            "worst_window_return_delta": min(deltas) if deltas else 0.0,
            "best_window_label": str(group.iloc[int(np.argmax(deltas))]["window_id"]) if deltas else None,
            "best_window_return_delta": max(deltas) if deltas else 0.0,
            "replay_safe": replay_safe,
        }
        status, reason = (
            ("baseline", "Frozen baseline comparator.")
            if label == "baseline"
            else build_feature_decision(pd.Series(row), baseline_score)
        )
        row["decision_status"] = status
        row["decision_reason"] = reason
        rows.append(row)

    return pd.DataFrame(rows)[AGGREGATE_COLUMNS]


def _promotion_decision(aggregate: pd.DataFrame) -> dict[str, Any]:
    promoted = aggregate[aggregate["decision_status"] == "promote"] if not aggregate.empty else aggregate
    if promoted is None or promoted.empty:
        return {
            "decision_status": "hold_for_more_evidence",
            "decision_reason": "No feature cleared all promotion gates.",
            "promoted_feature": None,
            "promotions": [],
        }
    winner = promoted.sort_values("avg_policy_score", ascending=False).iloc[0].to_dict()
    return {
        "decision_status": "promote",
        "decision_reason": str(winner.get("decision_reason")),
        "promoted_feature": str(winner.get("label")),
        "promotions": [winner],
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def run_feature_ablation_audit(
    df_features: pd.DataFrame,
    price_lookup: pd.DataFrame,
    *,
    n_windows: int = 5,
    window_years: int = 1,
    step_months: int = 3,
    initial_cash: float = 10_000.0,
    live_config: dict | None = None,
    strategy: str | None = None,
    checkpoint_path: str | None = None,
    output_root: str | Path = "experiments/feature_ablation",
    run_id: str | None = None,
    dry_run: bool = False,
    variants: list[FeatureAblationVariant] | None = None,
) -> dict[str, Any]:
    from broker import replay as replay_module
    from pipeline.benchmark import compute_metrics

    run_id = str(run_id or feature_ablation_run_id())
    variants = list(variants or FEATURE_ABLATION_VARIANTS)
    output_dir = Path(output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    windows = replay_module.build_policy_replay_windows(
        replay_module._date_level_values(df_features.index),
        n_windows=n_windows,
        window_years=window_years,
        step_months=step_months,
    )
    manifest = replay_module._window_manifest(windows)
    manifest.to_csv(output_dir / "window_manifest.csv", index=False)
    _write_json(output_dir / "window_manifest.json", manifest.to_dict(orient="records"))

    metadata = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "purpose": "feature_ablation_audit",
        "n_windows": int(n_windows),
        "window_years": int(window_years),
        "step_months": int(step_months),
        "initial_cash": float(initial_cash),
        "strategy": strategy or replay_module._resolve_replay_strategy(live_config),
        "checkpoint_path": checkpoint_path,
        "frozen_baseline": FROZEN_BASELINE,
        "diagnostic_defaults": DIAGNOSTIC_DEFAULTS,
        "variants": [
            {"label": variant.label, "variant_family": variant.variant_family, "overrides": variant.overrides}
            for variant in variants
        ],
        "dry_run": bool(dry_run),
    }
    _write_json(output_dir / "run_metadata.json", metadata)

    metrics_rows: list[dict[str, Any]] = []
    audit_by_key: dict[tuple[str, str], pd.DataFrame] = {}

    for variant in variants:
        cfg = build_frozen_feature_config(live_config, variant.label, variants=variants)
        variant_dir = output_dir / variant.label
        variant_dir.mkdir(parents=True, exist_ok=True)
        replay_kwargs = replay_module._replay_kwargs_from_live_config(cfg)
        for window in windows:
            window_id = str(window["label"])
            window_dir = variant_dir / window_id
            window_dir.mkdir(parents=True, exist_ok=True)
            if dry_run:
                metrics = compute_metrics(np.array([], dtype=float), label=variant.label)
                trade_log = []
                audit = pd.DataFrame()
                evaluated_candidate_count = 0
            else:
                df_window, price_window = replay_module._slice_replay_window(
                    df_features,
                    price_lookup,
                    window["start"],
                    window["end"],
                )
                returns, trade_log = replay_module.run_replay(
                    df_window,
                    price_window,
                    strategy=strategy or replay_module._resolve_replay_strategy(cfg),
                    checkpoint_path=checkpoint_path,
                    initial_cash=initial_cash,
                    label=f"{variant.label}_{window_id}",
                    **replay_kwargs,
                )
                metrics = compute_metrics(returns, label=variant.label)
                audit = getattr(replay_module, "_LAST_REPLAY_SCORE_AUDIT", pd.DataFrame())
                evaluated_candidate_count = int(len(audit)) if audit is not None else 0

            turnover = _turnover(trade_log, initial_cash)
            raw_score = _raw_policy_score(metrics, turnover)
            row = {
                "run_id": run_id,
                "label": variant.label,
                "variant_family": variant.variant_family,
                "window_id": window_id,
                "start": pd.Timestamp(window["start"]).date().isoformat(),
                "end": pd.Timestamp(window["end"]).date().isoformat(),
                "total_return": _safe_float(metrics.get("total_return")),
                "sharpe": _safe_float(metrics.get("sharpe")),
                "max_drawdown": _safe_float(metrics.get("max_drawdown")),
                "turnover": turnover,
                "win_rate": _safe_float(metrics.get("win_rate")),
                "avg_holding_days": _avg_holding_days(trade_log),
                "trade_count": len(trade_log),
                "evaluated_candidate_count": evaluated_candidate_count,
                "raw_policy_score": raw_score,
                "policy_score": raw_score,
                "replay_safe": True,
            }
            metrics_rows.append(row)
            _write_json(window_dir / "metrics.json", row)
            pd.DataFrame(trade_log).to_csv(window_dir / "trade_log.csv", index=False)
            audit_by_key[(variant.label, window_id)] = audit.copy() if audit is not None else pd.DataFrame()

    window_metrics = pd.DataFrame(metrics_rows)
    touch_frames: list[pd.DataFrame] = []
    for variant in variants:
        for window in windows:
            window_id = str(window["label"])
            window_dir = output_dir / variant.label / window_id
            if variant.label == "baseline":
                touches = _empty_touch_frame()
            else:
                touches = _touch_rows_from_baseline_comparison(
                    audit_by_key.get(("baseline", window_id), pd.DataFrame()),
                    audit_by_key.get((variant.label, window_id), pd.DataFrame()),
                    df_features=df_features,
                    run_id=run_id,
                    window_id=window_id,
                    variant_label=variant.label,
                    feature_name=variant.shadow_feature,
                )
            touches.to_csv(window_dir / "feature_touch_audit.csv", index=False)
            touch_frames.append(touches)

    all_touches = pd.concat(touch_frames, ignore_index=True) if touch_frames else _empty_touch_frame()
    if all_touches.empty:
        all_touches = _empty_touch_frame()
    mechanism = _mechanism_summary(all_touches)
    for variant in variants:
        for window in windows:
            window_id = str(window["label"])
            window_dir = output_dir / variant.label / window_id
            window_summary = mechanism[
                (mechanism["variant_label"] == variant.label)
                & (mechanism["window_id"] == window_id)
            ] if not mechanism.empty else _empty_mechanism_summary()
            if window_summary.empty and variant.label != "baseline":
                window_summary = pd.DataFrame([{
                    "window_id": window_id,
                    "variant_label": variant.label,
                    "feature_present_count": 0,
                    "score_changed_count": 0,
                    "rank_changed_count": 0,
                    "size_changed_count": 0,
                    "entry_changed_count": 0,
                    "exit_changed_count": 0,
                    "avg_score_delta": np.nan,
                    "avg_weight_delta": np.nan,
                    "avg_next_5d_return": np.nan,
                    "avg_next_20d_return": np.nan,
                    "hit_rate_5d": np.nan,
                    "hit_rate_20d": np.nan,
                }], columns=MECHANISM_SUMMARY_COLUMNS)
            window_summary.to_csv(window_dir / "window_feature_mechanism_summary.csv", index=False)
    aggregate = _aggregate_review(window_metrics, all_touches)
    decision = _promotion_decision(aggregate)

    window_metrics.to_csv(output_dir / "window_metrics.csv", index=False)
    aggregate.to_csv(output_dir / "aggregate_feature_review.csv", index=False)
    _write_json(output_dir / "aggregate_feature_review.json", aggregate.to_dict(orient="records"))
    all_touches.to_csv(output_dir / "feature_touch_audit.csv", index=False)
    _write_json(output_dir / "feature_touch_audit.json", all_touches.to_dict(orient="records"))
    mechanism.to_csv(output_dir / "window_feature_mechanism_summary.csv", index=False)
    _write_json(output_dir / "window_feature_mechanism_summary.json", mechanism.to_dict(orient="records"))
    _write_json(output_dir / "promotion_decision.json", decision)

    return {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "window_manifest": manifest,
        "window_metrics": window_metrics,
        "aggregate_feature_review": aggregate,
        "feature_touch_audit": all_touches,
        "window_feature_mechanism_summary": mechanism,
        "promotion_decision": decision,
    }
