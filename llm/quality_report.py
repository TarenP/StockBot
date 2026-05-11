"""Coverage and manual-review reporting for the local AI sidecar."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
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


def _safe_event_type(value: object) -> str:
    text = str(value or "unknown").strip().lower()
    return text or "unknown"


def _safe_ticker(value: object) -> str:
    return str(value or "").strip().upper()


def _parse_date(value: object) -> date | None:
    if not value:
        return None
    text = str(value).strip()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            return None


def _confidence_bucket(confidence: float) -> str:
    if confidence < 0.50:
        return "lt_0_50"
    if confidence < 0.65:
        return "0_50_to_0_65"
    if confidence < 0.85:
        return "0_65_to_0_85"
    return "gte_0_85"


def _new_event_type_bucket() -> dict:
    return {
        "raw_documents": 0,
        "parsed_documents": 0,
        "trusted_parses": 0,
        "low_confidence_parses": 0,
        "invalid_parse_files": 0,
        "manual_review_queue_count": 0,
        "raw_doc_ids": set(),
        "parsed_doc_ids": set(),
        "confidence_values": [],
        "confidence_buckets": {
            "lt_0_50": 0,
            "0_50_to_0_65": 0,
            "0_65_to_0_85": 0,
            "gte_0_85": 0,
        },
        "parse_age_days": [],
    }


def _finalize_event_type_bucket(bucket: dict) -> dict:
    raw_doc_ids = bucket.pop("raw_doc_ids", set())
    parsed_doc_ids = bucket.pop("parsed_doc_ids", set())
    confidence_values = bucket.pop("confidence_values", [])
    parse_age_days = bucket.pop("parse_age_days", [])
    denominator = len(raw_doc_ids)
    coverage = (len(parsed_doc_ids & raw_doc_ids) / denominator) if denominator else None
    invalid_denominator = bucket["parsed_documents"] + bucket["invalid_parse_files"]
    invalid_rate = (
        bucket["invalid_parse_files"] / invalid_denominator
        if invalid_denominator else 0.0
    )
    return {
        **bucket,
        "document_parse_coverage": coverage,
        "invalid_parse_rate": invalid_rate,
        "confidence": {
            "avg": (
                sum(confidence_values) / len(confidence_values)
                if confidence_values else None
            ),
            "min": min(confidence_values, default=None),
            "max": max(confidence_values, default=None),
        },
        "parse_age_days": {
            "avg": (
                sum(parse_age_days) / len(parse_age_days)
                if parse_age_days else None
            ),
            "max": max(parse_age_days, default=None),
        },
    }


def build_sidecar_quality_report(
    *,
    cache_dir: str | Path = "broker/state/llm_cache",
    document_store_dir: str | Path = "broker/state/document_store",
    tickers: Iterable[str] | None = None,
    min_confidence: float = 0.65,
    manual_review_limit: int = 50,
    min_coverage: float = 0.50,
    max_invalid_parse_rate: float = 0.10,
    max_manual_review_queue: int = 50,
    min_trusted_parses: int = 20,
    max_staleness_days: int = 14,
) -> dict:
    cache = LLMCache(cache_dir)
    ticker_set = {str(t).upper() for t in tickers or [] if str(t).strip()}
    today = datetime.now(timezone.utc).date()

    raw_docs = []
    by_event_type: dict[str, dict] = {}
    for path, payload in _iter_json_files(Path(document_store_dir)) or []:
        if not isinstance(payload, dict):
            raw_docs.append({"path": str(path), "valid_json": False})
            by_event_type.setdefault("unknown", _new_event_type_bucket())["raw_documents"] += 1
            continue
        ticker = _safe_ticker(payload.get("ticker"))
        if ticker_set and ticker not in ticker_set:
            continue
        source_type = _safe_event_type(payload.get("source_type"))
        doc_id = str(payload.get("doc_id") or path.stem)
        raw_docs.append({
            "doc_id": doc_id,
            "ticker": ticker,
            "source_type": source_type,
            "as_of_date": payload.get("as_of_date"),
            "valid_json": True,
        })
        bucket = by_event_type.setdefault(source_type, _new_event_type_bucket())
        bucket["raw_documents"] += 1
        bucket["raw_doc_ids"].add(doc_id)

    parses = []
    invalid_parses = 0
    invalid_parse_by_event_type: dict[str, int] = {}
    for path, payload in _iter_json_files(cache.parses_dir) or []:
        if not isinstance(payload, dict):
            invalid_parses += 1
            invalid_parse_by_event_type["unknown"] = invalid_parse_by_event_type.get("unknown", 0) + 1
            continue
        try:
            parsed = TranscriptEventParse.model_validate(payload)
        except Exception:
            invalid_parses += 1
            source_type = _safe_event_type(payload.get("source_type"))
            invalid_parse_by_event_type[source_type] = invalid_parse_by_event_type.get(source_type, 0) + 1
            continue
        if ticker_set and _safe_ticker(parsed.ticker) not in ticker_set:
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
        if bool(getattr(p, "valid", True)) and float(p.confidence or 0.0) >= float(min_confidence)
    ]
    invalid_parse_records = [
        p for p in parses
        if not bool(getattr(p, "valid", True))
    ]
    low_confidence = [
        p for p in parses
        if float(p.confidence or 0.0) < float(min_confidence)
    ]
    hallucination_review_queue = []
    manual_review_counts_by_event_type: dict[str, int] = {}
    for parsed in sorted(parses, key=lambda p: float(p.confidence or 0.0), reverse=True):
        risks = [str(risk) for risk in (parsed.top_risks or [])]
        source_type = _safe_event_type(parsed.source_type)
        needs_review = bool(getattr(parsed, "manual_review_required", False)) or (
            bool(getattr(parsed, "valid", True))
            and float(parsed.confidence or 0.0) >= 0.85
            and (
                parsed.thesis_impact in {"strengthens", "weakens"}
                or len(risks) >= 3
            )
        )
        if needs_review:
            manual_review_counts_by_event_type[source_type] = manual_review_counts_by_event_type.get(source_type, 0) + 1
            hallucination_review_queue.append({
                "ticker": _safe_ticker(parsed.ticker),
                "source_type": source_type,
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
    cache_hit_rate = coverage

    parse_ages = []
    trusted_parse_count_by_ticker: dict[str, int] = {}
    parse_count_by_ticker: dict[str, int] = {}
    confidence_buckets = {
        "lt_0_50": 0,
        "0_50_to_0_65": 0,
        "0_65_to_0_85": 0,
        "gte_0_85": 0,
    }
    for parsed in parses:
        ticker = _safe_ticker(parsed.ticker)
        source_type = _safe_event_type(parsed.source_type)
        doc_id = parsed.source_id
        confidence = float(parsed.confidence or 0.0)
        is_trusted = bool(getattr(parsed, "valid", True)) and confidence >= float(min_confidence)
        parse_count_by_ticker[ticker] = parse_count_by_ticker.get(ticker, 0) + 1
        if is_trusted:
            trusted_parse_count_by_ticker[ticker] = trusted_parse_count_by_ticker.get(ticker, 0) + 1
        bucket_name = _confidence_bucket(confidence)
        confidence_buckets[bucket_name] += 1

        event_bucket = by_event_type.setdefault(source_type, _new_event_type_bucket())
        event_bucket["parsed_documents"] += 1
        if is_trusted:
            event_bucket["trusted_parses"] += 1
        else:
            event_bucket["low_confidence_parses"] += 1
        if doc_id:
            event_bucket["parsed_doc_ids"].add(doc_id)
        event_bucket["confidence_values"].append(confidence)
        event_bucket["confidence_buckets"][bucket_name] += 1

        parsed_date = _parse_date(parsed.as_of_date)
        if parsed_date:
            age = max((today - parsed_date).days, 0)
            parse_ages.append(age)
            event_bucket["parse_age_days"].append(age)

    for source_type, count in invalid_parse_by_event_type.items():
        by_event_type.setdefault(source_type, _new_event_type_bucket())["invalid_parse_files"] += count
    for source_type, count in manual_review_counts_by_event_type.items():
        by_event_type.setdefault(source_type, _new_event_type_bucket())["manual_review_queue_count"] += count

    invalid_parse_total = invalid_parses + len(invalid_parse_records)
    invalid_parse_denominator = len(parses) + invalid_parses
    invalid_parse_rate = (
        invalid_parse_total / invalid_parse_denominator
        if invalid_parse_denominator else 0.0
    )
    stale_parse_count = sum(1 for age in parse_ages if age > int(max_staleness_days))
    stale_parse_rate = (stale_parse_count / len(parse_ages)) if parse_ages else 0.0
    manual_review_count = len(hallucination_review_queue)
    criteria = {
        "coverage": {
            "passed": coverage is not None and coverage >= float(min_coverage),
            "actual": coverage,
            "threshold": float(min_coverage),
        },
        "invalid_parse_rate": {
            "passed": invalid_parse_rate <= float(max_invalid_parse_rate),
            "actual": invalid_parse_rate,
            "threshold": float(max_invalid_parse_rate),
        },
        "manual_review_queue": {
            "passed": manual_review_count <= int(max_manual_review_queue),
            "actual": manual_review_count,
            "threshold": int(max_manual_review_queue),
        },
        "trusted_parses": {
            "passed": len(trusted) >= int(min_trusted_parses),
            "actual": len(trusted),
            "threshold": int(min_trusted_parses),
        },
        "staleness": {
            "passed": stale_parse_rate <= 0.20,
            "actual": stale_parse_rate,
            "threshold": 0.20,
            "max_staleness_days": int(max_staleness_days),
        },
    }
    failed_criteria = [name for name, row in criteria.items() if not row["passed"]]
    if failed_criteria:
        decision = "collect_more_data"
    else:
        decision = "manual_review_ready"
    confidence_values = [float(p.confidence or 0.0) for p in parses]
    confidence_avg = (sum(confidence_values) / len(confidence_values)) if confidence_values else None
    influence_reason = (
        "Broker influence disabled by policy; sidecar output is diagnostic-only."
        if not failed_criteria
        else "Broker influence disabled by policy; failed criteria: " + ", ".join(failed_criteria)
    )

    return {
        "generated_at": _utc_now(),
        "min_confidence": float(min_confidence),
        "raw_docs": len(raw_docs),
        "raw_documents": len(raw_docs),
        "valid_raw_documents": sum(1 for row in raw_docs if row.get("valid_json")),
        "parsed_docs": len(parses),
        "parsed_documents": len(parses),
        "invalid_parses": invalid_parse_total,
        "invalid_parse_files": invalid_parses,
        "invalid_parse_records": len(invalid_parse_records),
        "trusted_parses": len(trusted),
        "low_confidence_parses": len(low_confidence),
        "feature_records": len(features),
        "coverage": coverage,
        "document_parse_coverage": coverage,
        "cache_hit_rate": cache_hit_rate,
        "invalid_parse_rate": invalid_parse_rate,
        "confidence_min": min(confidence_values, default=None),
        "confidence_max": max(confidence_values, default=None),
        "confidence_avg": confidence_avg,
        "stale_parse_count": stale_parse_count,
        "influence_allowed": False,
        "reason": influence_reason,
        "unparsed_document_ids": unparsed_doc_ids[:100],
        "confidence": {
            "avg": confidence_avg,
            "min": min(confidence_values, default=None),
            "max": max(confidence_values, default=None),
        },
        "confidence_buckets": confidence_buckets,
        "by_event_type": {
            source_type: _finalize_event_type_bucket(bucket)
            for source_type, bucket in sorted(by_event_type.items())
        },
        "parse_count_by_ticker": dict(sorted(parse_count_by_ticker.items())),
        "trusted_parse_count_by_ticker": dict(sorted(trusted_parse_count_by_ticker.items())),
        "staleness": {
            "avg_parse_age_days": (
                sum(parse_ages) / len(parse_ages)
                if parse_ages else None
            ),
            "max_parse_age_days": max(parse_ages, default=None),
            "stale_parse_count": stale_parse_count,
            "stale_parse_rate": stale_parse_rate,
            "max_staleness_days": int(max_staleness_days),
        },
        "manual_review_queue": hallucination_review_queue[:manual_review_limit],
        "go_no_go": {
            "decision": decision,
            "influence_allowed": False,
            "tiny_penalty_experiment_ready_after_manual_review": decision == "manual_review_ready",
            "failed_criteria": failed_criteria,
            "criteria": criteria,
            "notes": [
                "Broker influence remains disabled by policy; this report only gates manual review and future paper experiments.",
                "Hard vetoes, exits, sizing authority, and policy promotion remain disconnected from the sidecar.",
            ],
        },
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
