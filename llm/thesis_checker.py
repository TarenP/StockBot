"""Convert sidecar parses into thesis-health diagnostics."""

from __future__ import annotations

NEGATIVE_VALUES = {"negative", "weakens"}
POSITIVE_VALUES = {"positive", "strengthens"}


def thesis_health(parsed_features: dict, min_confidence: float = 0.65) -> dict:
    confidence = float(parsed_features.get("llm_event_confidence", 0.0) or 0.0)
    if confidence < min_confidence:
        return {"status": "unknown", "score": 0.0, "reason": "low_confidence"}

    fields = [
        parsed_features.get("guidance_direction"),
        parsed_features.get("management_tone"),
        parsed_features.get("demand_outlook"),
        parsed_features.get("margin_outlook"),
        parsed_features.get("thesis_impact"),
    ]
    score = 0
    for value in fields:
        if value in POSITIVE_VALUES:
            score += 1
        elif value in NEGATIVE_VALUES:
            score -= 1
    if score > 0:
        status = "supportive"
    elif score < 0:
        status = "challenged"
    else:
        status = "neutral"
    return {"status": status, "score": float(score), "reason": "trusted_parse"}

