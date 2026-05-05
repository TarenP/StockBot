"""Prompt templates for local Ollama extraction."""

TRANSCRIPT_EVENT_SYSTEM = """You extract compact portfolio diagnostics from financial text.
Return only valid JSON matching the requested schema. Do not recommend trades,
weights, stops, or risk limits. If evidence is insufficient, use unknown/neutral
and lower confidence."""

TRANSCRIPT_EVENT_USER = """Ticker: {ticker}
Source type: {source_type}
Document date: {as_of_date}

Extract:
- guidance_direction: positive, neutral, negative, mixed, or unknown
- management_tone: positive, neutral, negative, mixed, or unknown
- demand_outlook: positive, neutral, negative, mixed, or unknown
- margin_outlook: positive, neutral, negative, mixed, or unknown
- thesis_impact: strengthens, neutral, weakens, mixed, or unknown
- top_risks: up to five short risk phrases
- confidence: 0.0 to 1.0
- evidence: up to three short support snippets

Text:
{text}
"""

