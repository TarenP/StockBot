"""Adapters from cached event records to broker-readable diagnostics."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from event_sidecar.cache import EventSidecarCache
from event_sidecar.impact import build_ticker_event_features


def precompute_event_features(
    tickers: list[str],
    *,
    cache_dir: str | Path = "broker/state/event_sidecar",
    sector_map: dict[str, str] | None = None,
    lookback_days: int = 14,
    as_of_date: str | None = None,
) -> dict[str, Any]:
    cache = EventSidecarCache(cache_dir)
    events = list(cache.iter_recent_events(lookback_days=lookback_days))
    records = build_ticker_event_features(
        events,
        tickers,
        sector_map=sector_map,
        as_of_date=as_of_date or datetime.now().date().isoformat(),
    )
    for record in records:
        cache.put_feature_record(record)
    return {
        "events_considered": len(events),
        "feature_records": len(records),
        "tickers": len({str(t).upper() for t in tickers or []}),
    }


def load_cached_event_features(
    tickers: list[str],
    *,
    cache_dir: str | Path = "broker/state/event_sidecar",
    min_confidence: float = 0.0,
    as_of_date: str | None = None,
) -> dict[str, dict]:
    cache = EventSidecarCache(cache_dir)
    return cache.load_features(tickers, min_confidence=min_confidence, as_of_date=as_of_date)


def summarize_event_features(
    tickers: list[str] | None = None,
    *,
    cache_dir: str | Path = "broker/state/event_sidecar",
    output_path: str | Path = "broker/state/event_sidecar_summary.json",
) -> dict[str, Any]:
    cache = EventSidecarCache(cache_dir)
    symbols = sorted({str(t).upper() for t in tickers or [] if str(t).strip()})
    if not symbols and cache.features_dir.exists():
        symbols = sorted(path.stem.upper() for path in cache.features_dir.glob("*.json"))

    features = cache.load_features(symbols)
    eventful = {
        ticker: payload
        for ticker, payload in features.items()
        if int(payload.get("mention_count", 0) or 0) > 0
    }
    risk = [
        (ticker, float(payload.get("event_risk_score", 0.0) or 0.0))
        for ticker, payload in eventful.items()
    ]
    opportunity = [
        (ticker, float(payload.get("event_opportunity_score", 0.0) or 0.0))
        for ticker, payload in eventful.items()
    ]
    event_type_counts: dict[str, int] = {}
    for payload in eventful.values():
        for event_type in payload.get("top_event_types") or []:
            event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1

    summary = {
        "enabled": True,
        "broker_influence": False,
        "tickers_checked": len(symbols),
        "tickers_with_events": len(eventful),
        "top_risk_tickers": [
            {"ticker": ticker, "risk": round(score, 4)}
            for ticker, score in sorted(risk, key=lambda item: item[1], reverse=True)[:10]
            if score > 0
        ],
        "top_opportunity_tickers": [
            {"ticker": ticker, "opportunity": round(score, 4)}
            for ticker, score in sorted(opportunity, key=lambda item: item[1], reverse=True)[:10]
            if score > 0
        ],
        "event_type_counts": dict(sorted(event_type_counts.items())),
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary

