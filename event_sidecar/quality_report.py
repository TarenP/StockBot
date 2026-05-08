"""Quality and validation reporting for event-sidecar diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from event_sidecar.cache import EventSidecarCache


def _load_price_panel(price_path: str | Path) -> pd.DataFrame:
    path = Path(price_path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _coerce_panel_index(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    if isinstance(frame.index, pd.MultiIndex) and {"date", "ticker"}.issubset(frame.index.names):
        return frame.sort_index()
    if {"date", "ticker"}.issubset(frame.columns):
        return frame.set_index(["date", "ticker"]).sort_index()
    return pd.DataFrame()


def _future_return(panel: pd.DataFrame, ticker: str, as_of_date: str | None, horizon: int) -> float | None:
    if panel.empty or not as_of_date:
        return None
    try:
        series = panel.xs(str(ticker).upper(), level="ticker")["close"].sort_index()
    except Exception:
        return None
    dates = pd.to_datetime(series.index, errors="coerce")
    series.index = dates
    series = series[series.index.notna()].sort_index()
    if series.empty:
        return None
    as_of = pd.Timestamp(as_of_date)
    future = series[series.index >= as_of]
    if len(future) <= horizon:
        return None
    start = float(future.iloc[0])
    end = float(future.iloc[horizon])
    if not np.isfinite(start) or start <= 0 or not np.isfinite(end):
        return None
    return end / start - 1.0


def build_event_sidecar_quality_report(
    *,
    cache_dir: str | Path = "broker/state/event_sidecar",
    price_path: str | Path = "MasterDS/stooq_panel.parquet",
    min_samples_for_influence: int = 30,
    min_directional_accuracy: float = 0.52,
    max_abs_mean_error: float = 0.03,
) -> dict[str, Any]:
    cache = EventSidecarCache(cache_dir)
    panel = _coerce_panel_index(_load_price_panel(price_path))
    rows = []
    if cache.features_dir.exists():
        for path in sorted(cache.features_dir.glob("*.json")):
            record = cache.get_feature_record(path.stem)
            if record is None or record.mention_count <= 0:
                continue
            row = {
                "ticker": record.ticker,
                "as_of_date": record.as_of_date,
                "event_score": record.event_score,
                "event_risk_score": record.event_risk_score,
                "event_opportunity_score": record.event_opportunity_score,
                "mention_count": record.mention_count,
                "source_count": record.source_count,
                "confidence": record.confidence,
            }
            for horizon in (1, 5, 20):
                row[f"ret_fwd_{horizon}d"] = _future_return(
                    panel,
                    record.ticker,
                    record.as_of_date,
                    horizon,
                )
            rows.append(row)

    frame = pd.DataFrame(rows)
    horizon_stats: dict[str, Any] = {}
    for horizon in (1, 5, 20):
        col = f"ret_fwd_{horizon}d"
        if frame.empty or col not in frame:
            horizon_stats[col] = {"samples": 0}
            continue
        sample = frame.dropna(subset=["event_score", col]).copy()
        if sample.empty:
            horizon_stats[col] = {"samples": 0}
            continue
        direction_match = (
            np.sign(sample["event_score"].astype(float))
            == np.sign(sample[col].astype(float))
        )
        horizon_stats[col] = {
            "samples": int(len(sample)),
            "mean_forward_return": round(float(sample[col].mean()), 6),
            "mean_event_score": round(float(sample["event_score"].mean()), 6),
            "directional_accuracy": round(float(direction_match.mean()), 6),
            "abs_mean_error": round(float((sample[col] - sample["event_score"]).abs().mean()), 6),
        }

    primary = horizon_stats.get("ret_fwd_5d", {})
    samples = int(primary.get("samples", 0) or 0)
    directional_accuracy = float(primary.get("directional_accuracy", 0.0) or 0.0)
    abs_mean_error = float(primary.get("abs_mean_error", 1.0) or 1.0)
    influence_allowed = (
        samples >= int(min_samples_for_influence)
        and directional_accuracy >= float(min_directional_accuracy)
        and abs_mean_error <= float(max_abs_mean_error)
    )
    failed = []
    if samples < int(min_samples_for_influence):
        failed.append("samples")
    if directional_accuracy < float(min_directional_accuracy):
        failed.append("directional_accuracy")
    if abs_mean_error > float(max_abs_mean_error):
        failed.append("abs_mean_error")

    source_counts: dict[str, int] = {}
    event_type_counts: dict[str, int] = {}
    for event in cache.iter_events() or []:
        source_counts[event.source] = source_counts.get(event.source, 0) + 1
        event_type_counts[event.event_type] = event_type_counts.get(event.event_type, 0) + 1

    return {
        "raw_events": sum(source_counts.values()),
        "feature_records": int(len(frame)),
        "validated_records": int(frame["ret_fwd_5d"].notna().sum()) if not frame.empty else 0,
        "source_counts": dict(sorted(source_counts.items())),
        "event_type_counts": dict(sorted(event_type_counts.items())),
        "horizon_stats": horizon_stats,
        "go_no_go": {
            "influence_allowed": bool(influence_allowed),
            "decision": "influence_ready" if influence_allowed else "diagnostics_only",
            "failed_criteria": failed,
        },
    }


def write_event_sidecar_quality_report(
    output_path: str | Path = "broker/state/event_sidecar_quality_report.json",
    **kwargs,
) -> dict[str, Any]:
    report = build_event_sidecar_quality_report(**kwargs)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report

