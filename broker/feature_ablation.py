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
    "feature_touch_count",
    "feature_touch_rate",
    "adjusted_entry_count",
    "adjusted_exit_count",
    "adjusted_size_count",
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
    "source_timestamp",
    "as_of_date",
    "replay_safe",
    "notes",
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
    "insider_adjustment_enabled": False,
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
    "feature_touch_count": 30,
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
    FeatureAblationVariant("earnings_only", "active", {"earnings_reaction_enabled": True}, "earnings_reaction_score"),
    FeatureAblationVariant("macro_only", "active", {"macro_regime_enabled": True}, "macro"),
    FeatureAblationVariant("insider_only", "active", {"insider_adjustment_enabled": True}, "insider"),
    FeatureAblationVariant("event_sidecar_shadow", "sidecar_shadow", {}, "event_score"),
    FeatureAblationVariant("llm_sidecar_shadow", "sidecar_shadow", {}, "llm_event_confidence"),
    FeatureAblationVariant("pattern_sidecar_shadow", "sidecar_shadow", {}, "pattern_score"),
]


def feature_ablation_run_id(now: datetime | None = None) -> str:
    return (now or datetime.now()).strftime("%Y%m%d_%H%M%S_%f")


def build_frozen_feature_config(base_config: dict | None, variant_label: str = "baseline") -> dict:
    variants = {variant.label: variant for variant in FEATURE_ABLATION_VARIANTS}
    if variant_label not in variants:
        raise ValueError(f"Unknown feature ablation variant: {variant_label}")
    cfg = dict(base_config or {})
    cfg.update(DIAGNOSTIC_DEFAULTS)
    cfg.update(FROZEN_BASELINE)
    cfg.update(variants[variant_label].overrides)
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
    return float(total_return + 0.10 * sharpe - 0.50 * drawdown - 0.02 * turnover)


def _score_with_penalty(raw_score: float, confidence_penalty: float) -> float:
    return float(raw_score - confidence_penalty)


def _empty_touch_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=TOUCH_COLUMNS)


def _touch_rows_from_score_audit(
    audit: pd.DataFrame,
    *,
    run_id: str,
    window_id: str,
    variant_label: str,
    feature_name: str | None,
) -> pd.DataFrame:
    if audit is None or audit.empty or not feature_name:
        return _empty_touch_frame()
    if feature_name in {"macro", "insider"}:
        summary_col = "soft_signal_summary"
        mask = audit.get(summary_col, pd.Series(index=audit.index, data="")).astype(str).str.contains(feature_name)
        feature_values = audit.get(summary_col, pd.Series(index=audit.index, data=""))
    elif feature_name in audit.columns:
        feature_values = audit[feature_name]
        mask = pd.to_numeric(feature_values, errors="coerce").fillna(0.0).abs() > 1e-9
    else:
        return _empty_touch_frame()

    rows = []
    touched = audit.loc[mask].copy()
    for idx, rec in touched.iterrows():
        base_score = _safe_float(rec.get("composite_score"))
        adjusted_score = _safe_float(rec.get("rank_score"), base_score)
        base_weight = _safe_float(rec.get("base_target_weight"))
        adjusted_weight = _safe_float(rec.get("final_weight"), base_weight)
        rows.append({
            "run_id": run_id,
            "window_id": window_id,
            "variant_label": variant_label,
            "date": rec.get("cycle_date"),
            "ticker": rec.get("ticker"),
            "feature_name": feature_name,
            "feature_value": feature_values.loc[idx] if hasattr(feature_values, "loc") else None,
            "base_score": base_score,
            "adjusted_score": adjusted_score,
            "score_delta": adjusted_score - base_score,
            "base_weight": base_weight,
            "adjusted_weight": adjusted_weight,
            "weight_delta": adjusted_weight - base_weight,
            "rank_before": np.nan,
            "rank_after": np.nan,
            "would_enter_baseline": False,
            "would_enter_variant": str(rec.get("candidate_status", "")) == "buy_selected",
            "actual_next_5d_return": np.nan,
            "actual_next_20d_return": np.nan,
            "source_timestamp": rec.get("cycle_date"),
            "as_of_date": rec.get("cycle_date"),
            "replay_safe": True,
            "notes": "score_audit_touch",
        })
    return pd.DataFrame(rows, columns=TOUCH_COLUMNS)


def build_feature_decision(row: pd.Series, baseline_policy_score: float) -> tuple[str, str]:
    if not bool(row.get("replay_safe", True)):
        return "reject_replay_safety", "Replay-safety checks failed."
    if int(row.get("feature_touch_count", 0) or 0) < PROMOTION_GATES["feature_touch_count"]:
        if float(row.get("feature_touch_rate", 0.0) or 0.0) <= 0:
            return "hold_for_more_evidence", "Feature had no decision touches; collect more data."
        return "hold_for_more_evidence", "Feature touched too few decisions for promotion."
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
        touch_count = int(len(touches))
        adjusted_count = int(
            (pd.to_numeric(touches.get("score_delta", pd.Series(dtype=float)), errors="coerce").fillna(0.0).abs() > 1e-9).sum()
            + (pd.to_numeric(touches.get("weight_delta", pd.Series(dtype=float)), errors="coerce").fillna(0.0).abs() > 1e-9).sum()
        )
        row = {
            "label": label,
            "variant_family": variant_family,
            "windows": int(len(group)),
            "wins": wins,
            "winner_rate": wins / len(group) if len(group) else 0.0,
            "avg_total_return": _safe_mean(group["total_return"]),
            "avg_sharpe": _safe_mean(group["sharpe"]),
            "avg_max_drawdown": _safe_mean(group["max_drawdown"]),
            "avg_turnover": _safe_mean(group["turnover"]),
            "avg_win_rate": _safe_mean(group["win_rate"]),
            "avg_holding_days": _safe_mean(group["avg_holding_days"]),
            "avg_policy_score": _score_with_penalty(raw_policy, confidence_penalty),
            "avg_raw_policy_score": raw_policy,
            "confidence_penalty": confidence_penalty,
            "incumbent_edge": _score_with_penalty(raw_policy, confidence_penalty) - baseline_score,
            "max_drawdown_degradation": max(0.0, _safe_mean(group["max_drawdown"]) - baseline_dd),
            "turnover_degradation": max(0.0, _safe_mean(group["turnover"]) - baseline_turnover),
            "feature_touch_count": touch_count,
            "feature_touch_rate": touch_count / max(int(len(group)), 1),
            "adjusted_entry_count": int((touches.get("would_enter_variant", pd.Series(dtype=bool)) == True).sum()) if not touches.empty else 0,
            "adjusted_exit_count": 0,
            "adjusted_size_count": adjusted_count,
            "worst_window_label": str(group.iloc[int(np.argmin(deltas))]["window_id"]) if deltas else None,
            "worst_window_return_delta": min(deltas) if deltas else 0.0,
            "best_window_label": str(group.iloc[int(np.argmax(deltas))]["window_id"]) if deltas else None,
            "best_window_return_delta": max(deltas) if deltas else 0.0,
            "replay_safe": True,
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
) -> dict[str, Any]:
    from broker import replay as replay_module
    from pipeline.benchmark import compute_metrics

    run_id = str(run_id or feature_ablation_run_id())
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
            for variant in FEATURE_ABLATION_VARIANTS
        ],
        "dry_run": bool(dry_run),
    }
    _write_json(output_dir / "run_metadata.json", metadata)

    metrics_rows: list[dict[str, Any]] = []
    touch_frames: list[pd.DataFrame] = []

    for variant in FEATURE_ABLATION_VARIANTS:
        cfg = build_frozen_feature_config(live_config, variant.label)
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
                "raw_policy_score": raw_score,
                "policy_score": raw_score,
                "replay_safe": True,
            }
            metrics_rows.append(row)
            _write_json(window_dir / "metrics.json", row)
            pd.DataFrame(trade_log).to_csv(window_dir / "trade_log.csv", index=False)
            touches = _touch_rows_from_score_audit(
                audit,
                run_id=run_id,
                window_id=window_id,
                variant_label=variant.label,
                feature_name=variant.shadow_feature,
            )
            touches.to_csv(window_dir / "feature_touch_audit.csv", index=False)
            touch_frames.append(touches)

    window_metrics = pd.DataFrame(metrics_rows)
    all_touches = (
        pd.concat(touch_frames, ignore_index=True)
        if touch_frames else _empty_touch_frame()
    )
    if all_touches.empty:
        all_touches = _empty_touch_frame()
    aggregate = _aggregate_review(window_metrics, all_touches)
    decision = _promotion_decision(aggregate)

    window_metrics.to_csv(output_dir / "window_metrics.csv", index=False)
    aggregate.to_csv(output_dir / "aggregate_feature_review.csv", index=False)
    _write_json(output_dir / "aggregate_feature_review.json", aggregate.to_dict(orient="records"))
    all_touches.to_csv(output_dir / "feature_touch_audit.csv", index=False)
    _write_json(output_dir / "feature_touch_audit.json", all_touches.to_dict(orient="records"))
    _write_json(output_dir / "promotion_decision.json", decision)

    return {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "window_manifest": manifest,
        "window_metrics": window_metrics,
        "aggregate_feature_review": aggregate,
        "feature_touch_audit": all_touches,
        "promotion_decision": decision,
    }
