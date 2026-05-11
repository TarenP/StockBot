from pathlib import Path
from uuid import uuid4

from llm.cache import LLMCache
from llm.feature_adapter import build_feature_record
from llm.schemas import TranscriptEventParse


def _test_dir() -> Path:
    path = Path("tests/_tmp") / f"llm_cache_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_llm_cache_write_read_parse():
    cache = LLMCache(_test_dir())
    parsed = TranscriptEventParse(
        ticker="ABC",
        doc_id="doc-1",
        source_type="8-k",
        source_id="doc-1",
        source_date="2026-05-01",
        as_of_date="2026-05-01",
        event_type="8-k",
        sentiment_label="neutral",
        confidence=0.72,
        valid=True,
    )

    cache.put_parse("doc-1", parsed)
    cache.put_feature_record(build_feature_record(parsed))

    loaded_parse = cache.get_parse("doc-1")
    loaded_feature = cache.get_feature_record("ABC")

    assert loaded_parse.ticker == "ABC"
    assert loaded_parse.doc_id == "doc-1"
    assert loaded_feature.ticker == "ABC"
    assert loaded_feature.broker_influence is False
    assert loaded_feature.features["broker_influence"] is False
