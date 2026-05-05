"""Helpers for event-type classification from text and SEC metadata."""

from __future__ import annotations

KEYWORDS = {
    "earnings": ("earnings", "quarter", "revenue", "eps", "guidance"),
    "8-k": ("form 8-k", "item 2.02", "item 7.01", "current report"),
    "press_release": ("press release", "announced", "company reports"),
    "risk_event": ("investigation", "restatement", "downgrade", "lawsuit", "recall"),
}


def classify_event(text: str, form_type: str | None = None) -> dict:
    haystack = (text or "").lower()
    labels = []
    if form_type and str(form_type).upper() == "8-K":
        labels.append("8-k")
    for label, terms in KEYWORDS.items():
        if any(term in haystack for term in terms):
            labels.append(label)
    if not labels:
        labels.append("unknown")
    return {"primary_event": labels[0], "labels": sorted(set(labels))}

