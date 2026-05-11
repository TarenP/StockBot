"""Transcript/event parser using cached local Ollama JSON extraction."""

from __future__ import annotations

import logging

from llm.cache import LLMCache, stable_doc_id
from llm.client import OllamaClient
from llm.prompts import TRANSCRIPT_EVENT_SYSTEM, TRANSCRIPT_EVENT_USER
from llm.schemas import TranscriptEventParse

logger = logging.getLogger(__name__)


def _fallback_parse(
    ticker: str,
    *,
    source_type: str,
    as_of_date: str | None,
    doc_id: str,
    error: str,
    model: str = "degraded_no_ollama",
) -> TranscriptEventParse:
    return TranscriptEventParse(
        ticker=str(ticker).upper(),
        doc_id=doc_id,
        source_type=source_type or "unknown",
        source_id=doc_id,
        source_date=as_of_date,
        as_of_date=as_of_date,
        event_type=source_type or "unknown",
        sentiment_label="unknown",
        guidance_direction="unknown",
        management_tone="unknown",
        demand_outlook="unknown",
        margin_outlook="unknown",
        thesis_impact="unknown",
        top_risks=[],
        confidence=0.0,
        risk_score=0.0,
        opportunity_score=0.0,
        summary="",
        contradictions=[],
        manual_review_required=False,
        valid=False,
        error=str(error)[:500],
        evidence=[],
        model=model,
    )


def parse_transcript_event(
    ticker: str,
    text: str,
    *,
    source_type: str = "transcript",
    as_of_date: str | None = None,
    source_id: str | None = None,
    client: OllamaClient | None = None,
    cache: LLMCache | None = None,
    max_chars: int = 24000,
    force: bool = False,
) -> TranscriptEventParse:
    cache = cache or LLMCache()
    doc_id = source_id or stable_doc_id(ticker, source_type, as_of_date, text)
    if not force:
        cached = cache.get_parse(doc_id)
        if cached is not None:
            return cached

    if client is None:
        client = OllamaClient()
    clipped = (text or "")[:max_chars]
    prompt = TRANSCRIPT_EVENT_USER.format(
        ticker=str(ticker).upper(),
        source_type=source_type,
        as_of_date=as_of_date or "unknown",
        text=clipped,
    )
    try:
        payload = client.generate_json(TRANSCRIPT_EVENT_SYSTEM, prompt)
        payload.setdefault("ticker", str(ticker).upper())
        payload.setdefault("doc_id", doc_id)
        payload.setdefault("source_type", source_type)
        payload.setdefault("source_id", doc_id)
        payload.setdefault("source_date", as_of_date)
        payload.setdefault("as_of_date", as_of_date)
        payload.setdefault("event_type", source_type or "unknown")
        payload.setdefault("sentiment_label", "unknown")
        payload.setdefault("risk_score", 0.0)
        payload.setdefault("opportunity_score", 0.0)
        payload.setdefault("summary", "")
        payload.setdefault("contradictions", [])
        payload.setdefault("manual_review_required", False)
        payload.setdefault("valid", True)
        payload.setdefault("error", None)
        payload.setdefault("model", client.model)
        parsed = TranscriptEventParse.model_validate(payload)
    except Exception as exc:
        logger.warning("LLM parse fallback for %s: %s", doc_id, exc)
        parsed = _fallback_parse(
            ticker,
            source_type=source_type,
            as_of_date=as_of_date,
            doc_id=doc_id,
            error=exc,
        )
    cache.put_parse(doc_id, parsed)
    return parsed


def parse_cached_or_degrade(
    ticker: str,
    text: str,
    *,
    source_type: str = "transcript",
    as_of_date: str | None = None,
    source_id: str | None = None,
    cache: LLMCache | None = None,
) -> TranscriptEventParse:
    cache = cache or LLMCache()
    doc_id = source_id or stable_doc_id(ticker, source_type, as_of_date, text)
    cached = cache.get_parse(doc_id)
    if cached is not None:
        return cached
    return TranscriptEventParse(
        ticker=str(ticker).upper(),
        doc_id=doc_id,
        source_type=source_type,
        source_id=doc_id,
        source_date=as_of_date,
        as_of_date=as_of_date,
        event_type=source_type,
        sentiment_label="unknown",
        guidance_direction="unknown",
        management_tone="unknown",
        demand_outlook="unknown",
        margin_outlook="unknown",
        thesis_impact="unknown",
        top_risks=[],
        confidence=0.0,
        risk_score=0.0,
        opportunity_score=0.0,
        summary="",
        contradictions=[],
        manual_review_required=False,
        valid=False,
        error="no cached parse available",
        evidence=[],
        model="degraded_no_cache",
    )
