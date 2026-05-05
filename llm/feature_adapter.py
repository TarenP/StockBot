"""Replay-safe adapter from cached sidecar outputs to broker features."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from llm.cache import LLMCache
from llm.memo_writer import write_event_memo
from llm.schemas import SidecarFeatureRecord, TranscriptEventParse


def build_feature_record(
    parsed: TranscriptEventParse,
    *,
    min_confidence: float = 0.65,
) -> SidecarFeatureRecord:
    features = parsed.compact_features(min_confidence=min_confidence)
    return SidecarFeatureRecord(
        ticker=str(parsed.ticker).upper(),
        as_of_date=parsed.as_of_date,
        features=features,
        memo=write_event_memo(features),
        source_id=parsed.source_id,
    )


def persist_feature_record(
    parsed: TranscriptEventParse,
    *,
    cache: LLMCache | None = None,
    min_confidence: float = 0.65,
) -> SidecarFeatureRecord:
    cache = cache or LLMCache()
    record = build_feature_record(parsed, min_confidence=min_confidence)
    cache.put_feature_record(record)
    return record


def load_cached_sidecar_features(
    tickers: Iterable[str],
    *,
    cache_dir: str | Path = "broker/state/llm_cache",
    min_confidence: float = 0.65,
    as_of_date: str | None = None,
) -> dict[str, dict]:
    cache = LLMCache(cache_dir)
    features: dict[str, dict] = {}
    for ticker in tickers or []:
        symbol = str(ticker).upper()
        record = cache.get_feature_record(symbol)
        if record is None:
            continue
        record_date = record.as_of_date
        if as_of_date and record_date and str(record_date) > str(as_of_date):
            continue
        payload = dict(record.features or {})
        confidence = float(payload.get("llm_event_confidence", 0.0) or 0.0)
        if confidence < float(min_confidence):
            payload["llm_event_trusted"] = False
        features[symbol] = payload
    return features

