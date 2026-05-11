import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from broker.brain import BrokerBrain
from broker.orchestrator import _run_llm_sidecar_task
from data_sources.document_fetcher import fetch_documents_for_universe
from llm.cache import LLMCache
from llm.feature_adapter import build_feature_record, load_cached_sidecar_features, persist_feature_record
from llm.schemas import SidecarFeatureRecord, TranscriptEventParse
from llm.transcript_parser import parse_transcript_event


class _FailingClient:
    model = "test-unavailable"

    def generate_json(self, _system_prompt, _user_prompt):
        raise ConnectionError("ollama unavailable")


class _Portfolio:
    positions = {}


def _test_dir() -> Path:
    path = Path("tests/_tmp") / f"llm_sidecar_rebuild_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_parse_transcript_event_fallback_without_ollama():
    base = _test_dir()
    cache = LLMCache(base / "cache")

    parsed = parse_transcript_event(
        "ABC",
        "Management discussed demand and margin risk.",
        source_type="transcript",
        as_of_date="2026-05-01",
        source_id="doc-fallback",
        client=_FailingClient(),
        cache=cache,
    )

    cached = cache.get_parse("doc-fallback")
    assert parsed.valid is False
    assert parsed.confidence == 0.0
    assert parsed.manual_review_required is False
    assert "ollama unavailable" in parsed.error
    assert cached.valid is False


def test_persist_feature_record_writes_compact_json():
    base = _test_dir()
    cache = LLMCache(base / "cache")
    parsed = TranscriptEventParse(
        ticker="abc",
        doc_id="doc-2",
        source_type="8-k",
        source_id="doc-2",
        source_date="2026-05-02",
        as_of_date="2026-05-02",
        event_type="guidance",
        sentiment_label="positive",
        confidence=0.81,
        risk_score=0.2,
        opportunity_score=0.7,
        summary="Guidance improved.",
        valid=True,
    )

    record = persist_feature_record(parsed, cache=cache, min_confidence=0.65)
    payload = json.loads(cache.feature_path("ABC").read_text(encoding="utf-8"))

    assert record.broker_influence is False
    assert payload["broker_influence"] is False
    assert payload["features"]["ticker"] == "ABC"
    assert payload["features"]["source_date"] == "2026-05-02"
    assert payload["features"]["event_type"] == "guidance"
    assert payload["features"]["trusted"] is True
    assert payload["features"]["broker_influence"] is False


def test_llm_feature_records_always_default_broker_influence_false():
    parsed = TranscriptEventParse(
        ticker="ABC",
        doc_id="doc-feature",
        source_type="8-k",
        source_id="doc-feature",
        as_of_date="2026-05-03",
        confidence=0.95,
        risk_score=0.9,
        opportunity_score=0.8,
        valid=True,
    )

    record = build_feature_record(parsed)

    assert record.broker_influence is False
    assert record.features["broker_influence"] is False
    assert record.features["trusted"] is True


def test_load_cached_sidecar_features_forces_broker_influence_false():
    base = _test_dir()
    cache = LLMCache(base / "cache")
    cache.put_feature_record(
        SidecarFeatureRecord(
            ticker="ABC",
            as_of_date="2026-05-03",
            features={
                "ticker": "ABC",
                "llm_event_confidence": 0.99,
                "llm_event_trusted": True,
                "trusted": True,
                "broker_influence": True,
            },
            broker_influence=True,
        )
    )

    loaded = load_cached_sidecar_features(["ABC"], cache_dir=base / "cache")

    assert loaded["ABC"]["broker_influence"] is False
    assert loaded["ABC"]["llm_event_trusted"] is True


def test_low_confidence_parse_is_untrusted_and_zeroes_risk_opportunity():
    parsed = TranscriptEventParse(
        ticker="ABC",
        doc_id="doc-low",
        source_type="8-k",
        source_id="doc-low",
        as_of_date="2026-05-03",
        confidence=0.40,
        risk_score=0.9,
        opportunity_score=0.8,
        valid=True,
    )

    features = parsed.compact_features(min_confidence=0.65)

    assert features["trusted"] is False
    assert features["llm_event_trusted"] is False
    assert features["risk_score"] == 0.0
    assert features["opportunity_score"] == 0.0
    assert features["broker_influence"] is False


def test_invalid_parse_is_untrusted_even_with_high_confidence():
    parsed = TranscriptEventParse(
        ticker="ABC",
        doc_id="doc-invalid",
        source_type="8-k",
        source_id="doc-invalid",
        as_of_date="2026-05-03",
        confidence=0.99,
        risk_score=0.9,
        opportunity_score=0.8,
        valid=False,
        error="invalid json fallback",
    )

    features = parsed.compact_features(min_confidence=0.65)

    assert features["trusted"] is False
    assert features["llm_event_trusted"] is False
    assert features["risk_score"] == 0.0
    assert features["opportunity_score"] == 0.0
    assert features["broker_influence"] is False


def test_document_fetch_failure_is_non_blocking(monkeypatch):
    base = _test_dir()

    def _raise_fetch(*_args, **_kwargs):
        raise RuntimeError("source unavailable")

    monkeypatch.setattr("data_sources.document_fetcher.fetch_sec_filings", _raise_fetch)
    monkeypatch.setattr("data_sources.document_fetcher.fetch_earnings_news", _raise_fetch)

    summary = fetch_documents_for_universe(
        ["ABC", "XYZ"],
        store_dir=str(base / "docs"),
        fetch_sec=True,
        fetch_news=True,
    )

    assert summary["sec_stored"] == 0
    assert summary["news_stored"] == 0
    assert summary["tickers_processed"] == 2


def test_broker_sidecar_precompute_empty_store_does_not_crash(monkeypatch):
    base = _test_dir()

    def _fake_fetch_documents_for_universe(**_kwargs):
        return {"sec_stored": 0, "news_stored": 0, "errors": []}

    monkeypatch.setattr(
        "data_sources.document_fetcher.fetch_documents_for_universe",
        _fake_fetch_documents_for_universe,
    )
    monkeypatch.setattr("pipeline.checkpoints.load_checkpoint_asset_list", lambda **_kwargs: [])
    monkeypatch.setattr("pipeline.universe_resolver.resolve_configured_universe", lambda **_kwargs: [])
    monkeypatch.setattr(
        "llm.quality_report.write_sidecar_quality_report",
        lambda *_args, **_kwargs: base / "quality.json",
    )

    result = _run_llm_sidecar_task(
        {
            "llm_document_store_dir": str(base / "docs"),
            "llm_cache_dir": str(base / "cache"),
            "llm_max_docs_per_run": 5,
            "llm_fetch_sec_filings": False,
            "llm_fetch_earnings_news": False,
            "llm_sidecar_min_confidence": 0.65,
        },
        state={},
        now=datetime(2026, 5, 11, tzinfo=timezone.utc),
    )

    assert result["processed"] == 0
    assert result["failed"] == 0


def test_sidecar_diagnostics_do_not_change_broker_decisions():
    brain = BrokerBrain(
        portfolio=_Portfolio(),
        llm_sidecar_features={
            "ABC": {
                "llm_event_confidence": 0.99,
                "llm_event_trusted": True,
                "guidance_direction": "negative",
                "management_tone": "negative",
                "demand_outlook": "negative",
                "margin_outlook": "negative",
                "thesis_impact": "weakens",
                "broker_influence": False,
            }
        },
        llm_sidecar_broker_influence=False,
    )
    report = {"ticker": "ABC", "composite_score": 0.9}

    brain._attach_llm_sidecar_features("ABC", report)

    assert report["composite_score"] == 0.9
    assert report["llm_broker_influence"] is False
    assert report["llm_diagnostic_only"] is True
    assert "earnings_reaction_score" not in report
