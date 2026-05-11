from pathlib import Path
from uuid import uuid4

from llm.cache import LLMCache
from llm.quality_report import build_sidecar_quality_report
from llm.schemas import TranscriptEventParse


def _test_dir() -> Path:
    path = Path("tests/_tmp") / f"llm_quality_report_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_quality_report_empty_cache_is_valid():
    base = _test_dir()
    report = build_sidecar_quality_report(
        cache_dir=base / "cache",
        document_store_dir=base / "docs",
    )

    assert report["raw_docs"] == 0
    assert report["parsed_docs"] == 0
    assert report["invalid_parses"] == 0
    assert report["coverage"] is None
    assert report["manual_review_queue"] == []


def test_quality_report_sets_influence_allowed_false_by_default():
    base = _test_dir()
    docs = base / "docs"
    docs.mkdir()
    (docs / "doc-1.json").write_text(
        '{"doc_id":"doc-1","ticker":"ABC","source_type":"8-k","as_of_date":"2026-05-01","text":"ok"}',
        encoding="utf-8",
    )
    cache = LLMCache(base / "cache")
    cache.put_parse(
        "doc-1",
        TranscriptEventParse(
            ticker="ABC",
            doc_id="doc-1",
            source_type="8-k",
            source_id="doc-1",
            source_date="2026-05-01",
            as_of_date="2026-05-01",
            event_type="8-k",
            sentiment_label="neutral",
            confidence=0.9,
            valid=True,
        ),
    )

    report = build_sidecar_quality_report(
        cache_dir=base / "cache",
        document_store_dir=docs,
        min_trusted_parses=1,
    )

    assert report["raw_docs"] == 1
    assert report["parsed_docs"] == 1
    assert report["trusted_parses"] == 1
    assert report["coverage"] == 1.0
    assert report["confidence_min"] == 0.9
    assert report["confidence_max"] == 0.9
    assert report["confidence_avg"] == 0.9
    assert report["influence_allowed"] is False
    assert report["go_no_go"]["influence_allowed"] is False
    assert "disabled" in report["reason"].lower()


def test_quality_report_never_allows_influence_by_default():
    base = _test_dir()
    report = build_sidecar_quality_report(
        cache_dir=base / "cache",
        document_store_dir=base / "docs",
        min_coverage=0.0,
        min_trusted_parses=0,
    )

    assert report["influence_allowed"] is False
    assert report["go_no_go"]["influence_allowed"] is False
    assert "disabled" in report["reason"].lower()
