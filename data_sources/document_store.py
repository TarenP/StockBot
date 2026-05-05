"""Local raw-document store for sidecar precompute jobs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm.cache import stable_doc_id


class DocumentStore:
    def __init__(self, root: str | Path = "broker/state/document_store"):
        self.root = Path(root)

    def put(
        self,
        ticker: str,
        text: str,
        *,
        source_type: str,
        as_of_date: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        doc_id = stable_doc_id(ticker, source_type, as_of_date, text)
        path = self.root / f"{doc_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "doc_id": doc_id,
                    "ticker": str(ticker).upper(),
                    "source_type": source_type,
                    "as_of_date": as_of_date,
                    "text": text,
                    "metadata": metadata or {},
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return doc_id

    def get(self, doc_id: str) -> dict[str, Any] | None:
        try:
            return json.loads((self.root / f"{doc_id}.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def iter_documents(self):
        if not self.root.exists():
            return
        for path in sorted(self.root.glob("*.json")):
            payload = self.get(path.stem)
            if payload is not None:
                yield payload
