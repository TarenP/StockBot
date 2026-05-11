from pathlib import Path
from uuid import uuid4

from data_sources.document_store import DocumentStore


def _test_dir() -> Path:
    path = Path("tests/_tmp") / f"document_store_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_document_store_write_read_list():
    store = DocumentStore(_test_dir())

    doc_id = store.put(
        "abc",
        "Management discussed demand and margin risk.",
        source_type="transcript",
        as_of_date="2026-05-01",
        source_id="abc-transcript-1",
        source_date="2026-05-01",
        metadata={"url": "local"},
    )

    doc = store.get(doc_id)
    docs = store.list_documents()

    assert doc_id == "abc-transcript-1"
    assert doc["ticker"] == "ABC"
    assert doc["source_type"] == "transcript"
    assert doc["source_date"] == "2026-05-01"
    assert doc["metadata"]["url"] == "local"
    assert [row["doc_id"] for row in docs] == [doc_id]


def test_document_store_dedupes_doc_id():
    store = DocumentStore(_test_dir())

    first = store.put("ABC", "old text", source_type="8-k", source_id="same-doc")
    second = store.put("ABC", "new text", source_type="8-k", source_id="same-doc")

    assert first == second == "same-doc"
    assert len(store.list_documents()) == 1
    assert store.get("same-doc")["text"] == "new text"
