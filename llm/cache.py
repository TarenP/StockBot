"""File-backed cache for sidecar parses and compact features."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from llm.schemas import SidecarFeatureRecord, TranscriptEventParse

DEFAULT_CACHE_DIR = Path("broker/state/llm_cache")


def stable_doc_id(ticker: str, source_type: str, as_of_date: str | None, text: str) -> str:
    seed = "|".join([str(ticker).upper(), str(source_type), str(as_of_date or ""), text])
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]


class LLMCache:
    def __init__(self, root: str | Path = DEFAULT_CACHE_DIR):
        self.root = Path(root)
        self.parses_dir = self.root / "parses"
        self.features_dir = self.root / "features"
        self.memos_dir = self.root / "memos"
        self.parses_dir.mkdir(parents=True, exist_ok=True)
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.memos_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, folder: Path, key: str) -> Path:
        clean = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in key)
        return folder / f"{clean}.json"

    def read_json(self, path: Path) -> dict[str, Any] | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def write_json(self, path: Path, payload: dict[str, Any]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        return path

    def get_parse(self, doc_id: str) -> TranscriptEventParse | None:
        payload = self.read_json(self._path(self.parses_dir, doc_id))
        if payload is None:
            return None
        try:
            return TranscriptEventParse.model_validate(payload)
        except Exception:
            return None

    def put_parse(self, doc_id: str, parsed: TranscriptEventParse) -> Path:
        return self.write_json(self._path(self.parses_dir, doc_id), parsed.model_dump(mode="json"))

    def get_feature_record(self, ticker: str) -> SidecarFeatureRecord | None:
        payload = self.read_json(self._path(self.features_dir, str(ticker).upper()))
        if payload is None:
            return None
        try:
            return SidecarFeatureRecord.model_validate(payload)
        except Exception:
            return None

    def put_feature_record(self, record: SidecarFeatureRecord) -> Path:
        return self.write_json(
            self._path(self.features_dir, str(record.ticker).upper()),
            record.model_dump(mode="json"),
        )

    def feature_path(self, ticker: str) -> Path:
        return self._path(self.features_dir, str(ticker).upper())
