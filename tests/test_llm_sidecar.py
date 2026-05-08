import json
from pathlib import Path
from uuid import uuid4

from broker.brain import BrokerBrain
from llm.cache import LLMCache, stable_doc_id
from llm.feature_adapter import build_feature_record, load_cached_sidecar_features
from llm.quality_report import build_sidecar_quality_report
from llm.schemas import TranscriptEventParse
from llm.transcript_parser import parse_cached_or_degrade


class _Portfolio:
    positions = {}


def _test_cache_dir() -> Path:
    path = Path("tests/_tmp") / f"llm_sidecar_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_transcript_event_schema_and_feature_compaction():
    parsed = TranscriptEventParse(
        ticker="abc",
        source_type="transcript",
        source_id="doc1",
        as_of_date="2026-05-01",
        guidance_direction="positive",
        management_tone="mixed",
        demand_outlook="positive",
        margin_outlook="neutral",
        thesis_impact="strengthens",
        top_risks=["inventory"],
        confidence=0.82,
        model="test",
    )

    features = parsed.compact_features(min_confidence=0.65)

    assert features["ticker"] == "ABC"
    assert features["llm_event_trusted"] is True
    assert features["thesis_impact"] == "strengthens"


def test_llm_cache_round_trips_parse_and_features():
    cache_dir = _test_cache_dir()
    cache = LLMCache(cache_dir)
    parsed = TranscriptEventParse(
        ticker="AAA",
        source_type="8-k",
        source_id="doc2",
        as_of_date="2026-05-01",
        thesis_impact="weakens",
        confidence=0.9,
    )
    cache.put_parse("doc2", parsed)
    record = build_feature_record(parsed, min_confidence=0.65)
    cache.put_feature_record(record)

    assert cache.get_parse("doc2").ticker == "AAA"
    loaded = load_cached_sidecar_features(["AAA"], cache_dir=cache_dir)
    assert loaded["AAA"]["llm_event_trusted"] is True
    assert loaded["AAA"]["thesis_impact"] == "weakens"


def test_parser_degrades_without_cache_and_without_ollama_call():
    cache_dir = _test_cache_dir()
    text = "Management discussed demand but no cached parse exists."
    doc_id = stable_doc_id("AAA", "transcript", "2026-05-01", text)
    parsed = parse_cached_or_degrade(
        "AAA",
        text,
        source_type="transcript",
        as_of_date="2026-05-01",
        cache=LLMCache(cache_dir),
    )

    assert parsed.source_id == doc_id
    assert parsed.confidence == 0.0
    assert parsed.model == "degraded_no_cache"


def test_brain_attaches_sidecar_features_without_broker_influence():
    features = {
        "AAA": {
            "llm_event_confidence": 0.95,
            "llm_event_trusted": True,
            "guidance_direction": "negative",
            "management_tone": "negative",
            "demand_outlook": "negative",
            "margin_outlook": "negative",
            "thesis_impact": "weakens",
            "top_risks": ["demand"],
        }
    }
    brain = BrokerBrain(
        portfolio=_Portfolio(),
        llm_sidecar_features=features,
        llm_sidecar_broker_influence=False,
    )
    report = {"ticker": "AAA", "composite_score": 0.9}

    brain._attach_llm_sidecar_features("AAA", report)

    assert report["llm_event_trusted"] is True
    assert report["llm_thesis_impact"] == "weakens"
    assert "earnings_reaction_score" not in report


def test_llm_sidecar_influence_requires_explicit_experiment_gate():
    features = {
        "AAA": {
            "llm_event_confidence": 0.95,
            "llm_event_trusted": True,
            "guidance_direction": "negative",
            "management_tone": "negative",
            "demand_outlook": "negative",
            "margin_outlook": "negative",
            "thesis_impact": "weakens",
        }
    }
    brain = BrokerBrain(
        portfolio=_Portfolio(),
        llm_sidecar_features=features,
        llm_sidecar_broker_influence=True,
        allow_unpromoted_feature_influence=False,
    )
    report = {"ticker": "AAA", "composite_score": 0.9}

    brain._attach_llm_sidecar_features("AAA", report)

    assert report["llm_diagnostic_only"] is True
    assert report["llm_broker_influence"] is False
    assert "earnings_reaction_score" not in report


def test_brain_attaches_event_sidecar_features_without_broker_influence():
    brain = BrokerBrain(
        portfolio=_Portfolio(),
        event_sidecar_features={
            "AAA": {
                "event_score": -0.2,
                "event_risk_score": 0.3,
                "event_opportunity_score": 0.0,
                "crowd_sentiment_score": -0.1,
                "mention_count": 2,
                "source_count": 1,
                "top_event_types": ["conflict"],
                "confidence": 0.7,
            }
        },
        event_sidecar_broker_influence=False,
    )
    report = {"ticker": "AAA", "composite_score": 0.9}

    brain._attach_event_sidecar_features("AAA", report)

    assert report["event_risk_score"] == 0.3
    assert report["event_top_types"] == ["conflict"]
    assert "earnings_reaction_score" not in report


def test_replay_safety_blocks_future_dated_cached_features():
    cache_dir = _test_cache_dir()
    parsed = TranscriptEventParse(
        ticker="AAA",
        source_type="transcript",
        source_id="future",
        as_of_date="2026-06-01",
        thesis_impact="strengthens",
        confidence=0.99,
    )
    cache = LLMCache(cache_dir)
    cache.put_feature_record(build_feature_record(parsed))

    loaded = load_cached_sidecar_features(["AAA"], cache_dir=cache_dir, as_of_date="2026-05-01")

    assert loaded == {}


def test_quality_report_tracks_coverage_and_manual_review_queue():
    cache_dir = _test_cache_dir()
    doc_store = cache_dir / "docs"
    doc_store.mkdir(parents=True, exist_ok=True)
    raw_doc = {
        "doc_id": "doc-review",
        "ticker": "AAA",
        "source_type": "transcript",
        "as_of_date": "2026-05-01",
        "text": "Demand improved but management cited margin risk.",
    }
    (doc_store / "doc-review.json").write_text(json.dumps(raw_doc), encoding="utf-8")

    parsed = TranscriptEventParse(
        ticker="AAA",
        source_type="transcript",
        source_id="doc-review",
        as_of_date="2026-05-01",
        guidance_direction="positive",
        management_tone="positive",
        demand_outlook="positive",
        margin_outlook="negative",
        thesis_impact="strengthens",
        top_risks=["margin", "inventory", "competition"],
        confidence=0.91,
    )
    cache = LLMCache(cache_dir)
    cache.put_parse("doc-review", parsed)
    cache.put_feature_record(build_feature_record(parsed))

    report = build_sidecar_quality_report(
        cache_dir=cache_dir,
        document_store_dir=doc_store,
        tickers=["AAA"],
        min_trusted_parses=1,
    )

    assert report["raw_documents"] == 1
    assert report["parsed_documents"] == 1
    assert report["trusted_parses"] == 1
    assert report["document_parse_coverage"] == 1.0
    assert report["cache_hit_rate"] == 1.0
    assert report["by_event_type"]["transcript"]["raw_documents"] == 1
    assert report["by_event_type"]["transcript"]["trusted_parses"] == 1
    assert report["by_event_type"]["transcript"]["confidence_buckets"]["gte_0_85"] == 1
    assert report["trusted_parse_count_by_ticker"]["AAA"] == 1
    assert report["go_no_go"]["influence_allowed"] is False
    assert report["go_no_go"]["decision"] == "manual_review_ready"
    assert len(report["manual_review_queue"]) == 1


def test_quality_report_blocks_influence_when_coverage_is_missing():
    cache_dir = _test_cache_dir()
    doc_store = cache_dir / "docs"
    doc_store.mkdir(parents=True, exist_ok=True)

    report = build_sidecar_quality_report(
        cache_dir=cache_dir,
        document_store_dir=doc_store,
    )

    assert report["raw_documents"] == 0
    assert report["go_no_go"]["influence_allowed"] is False
    assert report["go_no_go"]["decision"] == "collect_more_data"
    assert "coverage" in report["go_no_go"]["failed_criteria"]
