"""Typed schemas for market-event sidecar records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _clean_strings(values) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    return [str(value).strip() for value in values if str(value).strip()]


@dataclass
class MarketEventRecord:
    event_id: str
    source: str = "unknown"
    event_type: str = "market_news"
    title: str = ""
    summary: str = ""
    url: str | None = None
    as_of_date: str | None = None
    geography: str | None = None
    tickers: list[str] = field(default_factory=list)
    sectors: list[str] = field(default_factory=list)
    sentiment_score: float = 0.0
    severity: float = 0.0
    confidence: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.event_id = str(self.event_id)
        self.source = str(self.source or "unknown")
        self.event_type = str(self.event_type or "market_news")
        self.title = str(self.title or "")
        self.summary = str(self.summary or "")
        self.tickers = sorted({ticker.upper() for ticker in _clean_strings(self.tickers)})
        self.sectors = sorted(set(_clean_strings(self.sectors)))
        self.sentiment_score = _clamp(self.sentiment_score, -1.0, 1.0)
        self.severity = _clamp(self.severity, 0.0, 1.0)
        self.confidence = _clamp(self.confidence, 0.0, 1.0)
        self.metadata = dict(self.metadata or {})

    def model_dump(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "source": self.source,
            "event_type": self.event_type,
            "title": self.title,
            "summary": self.summary,
            "url": self.url,
            "as_of_date": self.as_of_date,
            "geography": self.geography,
            "tickers": list(self.tickers),
            "sectors": list(self.sectors),
            "sentiment_score": self.sentiment_score,
            "severity": self.severity,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def model_validate(cls, payload: dict[str, Any]) -> "MarketEventRecord":
        return cls(**dict(payload or {}))


@dataclass
class TickerEventFeatureRecord:
    ticker: str
    as_of_date: str | None = None
    event_score: float = 0.0
    event_risk_score: float = 0.0
    event_opportunity_score: float = 0.0
    crowd_sentiment_score: float = 0.0
    crowd_mention_velocity: float = 0.0
    mention_count: int = 0
    source_count: int = 0
    top_event_types: list[str] = field(default_factory=list)
    top_events: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    broker_influence: bool = False

    def __post_init__(self) -> None:
        self.ticker = str(self.ticker).strip().upper()
        self.event_score = _clamp(self.event_score, -1.0, 1.0)
        self.event_risk_score = _clamp(self.event_risk_score, 0.0, 1.0)
        self.event_opportunity_score = _clamp(self.event_opportunity_score, 0.0, 1.0)
        self.crowd_sentiment_score = _clamp(self.crowd_sentiment_score, -1.0, 1.0)
        self.crowd_mention_velocity = _clamp(self.crowd_mention_velocity, -1.0, 1.0)
        self.confidence = _clamp(self.confidence, 0.0, 1.0)
        self.mention_count = max(0, int(self.mention_count or 0))
        self.source_count = max(0, int(self.source_count or 0))
        self.top_event_types = [str(item) for item in self.top_event_types[:5]]
        self.top_events = [dict(item) for item in self.top_events[:5]]
        self.broker_influence = bool(self.broker_influence)

    @property
    def features(self) -> dict[str, Any]:
        return {
            "event_score": self.event_score,
            "event_risk_score": self.event_risk_score,
            "event_opportunity_score": self.event_opportunity_score,
            "crowd_sentiment_score": self.crowd_sentiment_score,
            "crowd_mention_velocity": self.crowd_mention_velocity,
            "mention_count": self.mention_count,
            "source_count": self.source_count,
            "top_event_types": list(self.top_event_types),
            "top_events": list(self.top_events),
            "confidence": self.confidence,
            "broker_influence": self.broker_influence,
            "as_of_date": self.as_of_date,
        }

    def model_dump(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "as_of_date": self.as_of_date,
            **self.features,
        }

    @classmethod
    def model_validate(cls, payload: dict[str, Any]) -> "TickerEventFeatureRecord":
        return cls(**dict(payload or {}))
