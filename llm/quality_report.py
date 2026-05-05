"""Coverage and manual-review reporting for the local AI sidecar."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from llm.cache import LLMCache
from llm.schemas import TranscriptEventParse


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _iter_json_files(path: Path):
    if not path.exists():
        return
    for item in sorted(path.glob("*.json")):
        try:
            yield item, json.loads(item.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            yield item, None


def build_sidecar_quality_report(
    *,
    cache_dir: str | Path = "broker/state/llm_cache",
    document_store_dir: str | Path = "broker/state/document_store",
    tickers: Iterable[str] | None = None,
    min_confidence: float = 0.65,
    manual_review_limit: int = 50,
) -> dict:
    cache = LLMCache(cache_dir)
    ticker_set = {str(t).upper() for t in tickers or [] if str(t).strip()}

    raw_docs = []
    for path, payload in _iter_json_files(Path(document_store_dir)) or []:
        if not isinstance(payload, dict):
            raw_docs.append({"path": str(path), "valid_json": False})
            continue
        ticker = str(payload.get("ticker", "")).upper()
        if ticker_set and ticker not in ticker_set:
            continue
        raw_docs.append({
            "doc_id": payload.get("doc_id") or path.stem,
            "ticker": ticker,
            "source_type": payload.get("source_type", "unknown"),
            "as_of_date": payload.get("as_of_date"),
            "valid_json": True,
        })

    parses = []
    invalid_parses = 0
    for path, payload in _iter_json_files(cache.parses_dir) or []:
        if not isinstance(payload, dict):
            invalid_parses += 1
            continue
        try:
            parsed = TranscriptEventParse.model_validate(payload)
        except Exception:
            invalid_parses += 1
            continue
        if ticker_set and str(parsed.ticker).upper() not in ticker_set:
            continue
        parses.append(parsed)

    features = {}
    for ticker in sorted(ticker_set) if ticker_set else []:
        record = cache.get_feature_record(ticker)
        if record is not None:
            features[ticker] = record.features or {}
    if not ticker_set:
        for path, payload in _iter_json_files(cache.features_dir) or []:
            if isinstance(payload, dict):
                ticker = str(payload.get("ticker") or path.stem).upper()
                features[ticker] = payload.get("features") or {}

    trusted = [
        p for p in parses
        if float(p.confidence or 0.0) >= float(min_confidence)
    ]
    low_confidence = [
        p for p in parses
        if float(p.confidence or 0.0) < float(min_confidence)
    ]
    hallucination_review_queue = []
    for parsed in sorted(parses, key=lambda p: float(p.confidence or 0.0), reverse=True):
        risks = [str(risk) for risk in (parsed.top_risks or [])]
        needs_review = (
            float(parsed.confidence or 0.0) >= 0.85
            and (
                parsed.thesis_impact in {"strengthens", "weakens"}
                or len(risks) >= 3
            )
        )
        if needs_review:
            hallucination_review_queue.append({
                "ticker": str(parsed.ticker).upper(),
                "source_id": parsed.source_id,
                "as_of_date": parsed.as_of_date,
                "confidence": float(parsed.confidence or 0.0),
                "thesis_impact": parsed.thesis_impact,
                "top_risks": risks[:5],
                "evidence": list(parsed.evidence or [])[:3],
                "review_status": "needs_manual_review",
            })

    parsed_doc_ids = {p.source_id for p in parses if p.source_id}
    raw_doc_ids = {row.get("doc_id") for row in raw_docs if row.get("doc_id")}
    unparsed_doc_ids = sorted(raw_doc_ids - parsed_doc_ids)
    coverage_denominator = len(raw_doc_ids)
    coverage = (len(parsed_doc_ids & raw_doc_ids) / coverage_denominator) if coverage_denominator else None

    return {
        "generated_at": _utc_now(),
        "min_confidence": float(min_confidence),
        "raw_documents": len(raw_docs),
        "valid_raw_documents": sum(1 for row in raw_docs if row.get("valid_json")),
        "parsed_documents": len(parses),
        "invalid_parse_files": invalid_parses,
        "trusted_parses": len(trusted),
        "low_confidence_parses": len(low_confidence),
        "feature_records": len(features),
        "document_parse_coverage": coverage,
        "unparsed_document_ids": unparsed_doc_ids[:100],
        "confidence": {
            "avg": (
                sum(float(p.confidence or 0.0) for p in parses) / len(parses)
                if parses else None
            ),
            "min": min((float(p.confidence or 0.0) for p in parses), default=None),
            "max": max((float(p.confidence or 0.0) for p in parses), default=None),
        },
        "manual_review_queue": hallucination_review_queue[:manual_review_limit],
    }


def write_sidecar_quality_report(
    output_path: str | Path = "broker/state/llm_sidecar_quality_report.json",
    **kwargs,
) -> Path:
    report = build_sidecar_quality_report(**kwargs)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path
