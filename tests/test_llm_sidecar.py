from pathlib import Path
from uuid import uuid4

from broker.brain import BrokerBrain
from llm.cache import LLMCache, stable_doc_id
from llm.feature_adapter import build_feature_record, load_cached_sidecar_features
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
