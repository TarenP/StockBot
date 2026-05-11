"""Typed schemas for local-LLM sidecar outputs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

try:
    from pydantic import BaseModel as _PydanticBaseModel
    from pydantic import Field, ValidationError
    PYDANTIC_AVAILABLE = True

    class BaseModel(_PydanticBaseModel):
        @classmethod
        def model_validate(cls, payload):
            validator = getattr(super(), "model_validate", None)
            if callable(validator):
                return validator(payload)
            return cls.parse_obj(payload)

        def model_dump(self, mode: str | None = None):
            dumper = getattr(super(), "model_dump", None)
            if callable(dumper):
                return dumper(mode=mode)
            return self.dict()
except ModuleNotFoundError:  # pragma: no cover - production deps should install pydantic.
    PYDANTIC_AVAILABLE = False

    class ValidationError(ValueError):
        pass

    class _FieldInfo:
        def __init__(self, default=None, default_factory=None, **_kwargs):
            self.default = default
            self.default_factory = default_factory

    def Field(default=None, default_factory=None, **kwargs):
        return _FieldInfo(default, default_factory, **kwargs)

    class BaseModel:
        def __init__(self, **kwargs):
            annotations = getattr(self, "__annotations__", {})
            for key in annotations:
                default = getattr(type(self), key, None)
                if isinstance(default, _FieldInfo):
                    if default.default_factory is not None:
                        value = default.default_factory()
                    else:
                        value = default.default
                else:
                    value = default
                setattr(self, key, kwargs.get(key, value))

        @classmethod
        def model_validate(cls, payload):
            if not isinstance(payload, dict):
                raise ValidationError("payload must be a dict")
            return cls(**payload)

        def model_dump(self, mode: str | None = None):
            _ = mode
            return dict(self.__dict__)


Direction = Literal["positive", "neutral", "negative", "mixed", "unknown"]
Tone = Literal["positive", "neutral", "negative", "mixed", "unknown"]
ThesisImpact = Literal["strengthens", "neutral", "weakens", "mixed", "unknown"]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class TranscriptEventParse(BaseModel):
    ticker: str = Field(..., description="Uppercase ticker symbol")
    doc_id: str = Field("", description="Stable raw document id")
    source_type: str = Field("unknown", description="transcript, 8-k, press_release, or event_cluster")
    source_id: str = Field("", description="Stable document or cache source id")
    source_date: str | None = Field(None, description="Original source date in ISO format")
    as_of_date: str | None = Field(None, description="Document date in ISO format")
    parsed_at: str = Field(default_factory=_utc_now_iso)
    event_type: str = "unknown"
    sentiment_label: Tone = "unknown"
    guidance_direction: Direction = "unknown"
    management_tone: Tone = "unknown"
    demand_outlook: Direction = "unknown"
    margin_outlook: Direction = "unknown"
    thesis_impact: ThesisImpact = "unknown"
    top_risks: list[str] = Field(default_factory=list)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    risk_score: float = Field(0.0, ge=0.0, le=1.0)
    opportunity_score: float = Field(0.0, ge=0.0, le=1.0)
    summary: str = ""
    contradictions: list[str] = Field(default_factory=list)
    manual_review_required: bool = False
    valid: bool = True
    error: str | None = None
    evidence: list[str] = Field(default_factory=list)
    model: str = "unknown"
    generated_at: str = Field(default_factory=_utc_now_iso)
    schema_version: int = 1

    def compact_features(self, min_confidence: float = 0.65) -> dict[str, Any]:
        trusted = bool(self.valid) and float(self.confidence or 0.0) >= float(min_confidence)
        doc_id = self.doc_id or self.source_id
        feature_timestamp = self.parsed_at or self.generated_at
        return {
            "ticker": str(self.ticker).upper(),
            "as_of_date": self.as_of_date,
            "source_date": self.source_date or self.as_of_date,
            "feature_timestamp": feature_timestamp,
            "event_type": self.event_type or self.source_type or "unknown",
            "confidence": float(self.confidence or 0.0),
            "risk_score": float(self.risk_score or 0.0) if trusted else 0.0,
            "opportunity_score": float(self.opportunity_score or 0.0) if trusted else 0.0,
            "manual_review_required": bool(self.manual_review_required),
            "trusted": bool(trusted),
            "broker_influence": False,
            "llm_event_confidence": float(self.confidence or 0.0),
            "llm_event_trusted": bool(trusted),
            "guidance_direction": self.guidance_direction if trusted else "unknown",
            "management_tone": self.management_tone if trusted else "unknown",
            "demand_outlook": self.demand_outlook if trusted else "unknown",
            "margin_outlook": self.margin_outlook if trusted else "unknown",
            "thesis_impact": self.thesis_impact if trusted else "unknown",
            "top_risks": list(self.top_risks or [])[:5] if trusted else [],
            "source_id": self.source_id,
            "doc_id": doc_id,
            "source_type": self.source_type,
            "schema_version": self.schema_version,
        }


class SidecarFeatureRecord(BaseModel):
    ticker: str
    as_of_date: str | None = None
    source_date: str | None = None
    feature_timestamp: str = Field(default_factory=_utc_now_iso)
    features: dict[str, Any] = Field(default_factory=dict)
    memo: str | None = None
    source_id: str = ""
    broker_influence: bool = False
    generated_at: str = Field(default_factory=_utc_now_iso)
    schema_version: int = 1
