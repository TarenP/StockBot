"""File-backed cache for market-event sidecar records."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from event_sidecar.schemas import MarketEventRecord, TickerEventFeatureRecord


def stable_event_id(source: str, title: str, as_of_date: str | None = None, url: str | None = None) -> str:
    payload = "|".join([
        str(source or "").strip().lower(),
        str(as_of_date or "").strip(),
        str(url or "").strip().lower(),
        str(title or "").strip().lower(),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


class EventSidecarCache:
    def __init__(self, root: str | Path = "broker/state/event_sidecar"):
        self.root = Path(root)
        self.events_dir = self.root / "events"
        self.features_dir = self.root / "features"

    def put_event(self, record: MarketEventRecord) -> Path:
        path = self.events_dir / f"{record.event_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(record.model_dump(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return path

    def get_event(self, event_id: str) -> MarketEventRecord | None:
        try:
            payload = json.loads((self.events_dir / f"{event_id}.json").read_text(encoding="utf-8"))
            return MarketEventRecord.model_validate(payload)
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            return None

    def iter_events(self, *, since_date: str | None = None):
        if not self.events_dir.exists():
            return
        for path in sorted(self.events_dir.glob("*.json")):
            event = self.get_event(path.stem)
            if event is None:
                continue
            if since_date and (event.as_of_date or "") < since_date:
                continue
            yield event

    def iter_recent_events(self, lookback_days: int = 14, now: datetime | None = None):
        cutoff = (now or datetime.now()).date() - timedelta(days=max(1, int(lookback_days)))
        yield from self.iter_events(since_date=cutoff.isoformat()) or []

    def put_feature_record(self, record: TickerEventFeatureRecord) -> Path:
        path = self.features_dir / f"{record.ticker}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(record.model_dump(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return path

    def get_feature_record(self, ticker: str) -> TickerEventFeatureRecord | None:
        symbol = str(ticker).strip().upper()
        try:
            payload = json.loads((self.features_dir / f"{symbol}.json").read_text(encoding="utf-8"))
            return TickerEventFeatureRecord.model_validate(payload)
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            return None

    def load_features(
        self,
        tickers: list[str],
        *,
        min_confidence: float = 0.0,
        as_of_date: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        features: dict[str, dict[str, Any]] = {}
        for ticker in tickers or []:
            record = self.get_feature_record(ticker)
            if record is None:
                continue
            if as_of_date and record.as_of_date and record.as_of_date > as_of_date:
                continue
            if record.confidence < float(min_confidence):
                continue
            features[record.ticker] = record.features
        return features

