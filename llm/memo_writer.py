"""Human-readable broker memo writer from structured sidecar features."""

from __future__ import annotations


def write_event_memo(features: dict) -> str:
    ticker = str(features.get("ticker", "")).upper()
    confidence = float(features.get("llm_event_confidence", 0.0) or 0.0)
    parts = [
        f"{ticker} event parse",
        f"confidence={confidence:.2f}",
        f"guidance={features.get('guidance_direction', 'unknown')}",
        f"tone={features.get('management_tone', 'unknown')}",
        f"demand={features.get('demand_outlook', 'unknown')}",
        f"margin={features.get('margin_outlook', 'unknown')}",
        f"thesis={features.get('thesis_impact', 'unknown')}",
    ]
    risks = features.get("top_risks") or []
    if risks:
        parts.append("risks=" + "; ".join(str(risk) for risk in risks[:5]))
    return " | ".join(parts)

