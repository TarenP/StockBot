"""
Portfolio exposure helpers.

These diagnostics describe economic exposure that sector labels miss. They are
intentionally lightweight and deterministic so live runs, replay audits, and
status reports all use the same buckets.
"""

from __future__ import annotations


THEME_BUCKETS: dict[str, str] = {
    "AFRM": "consumer_credit_finance",
    "HOOD": "consumer_credit_finance",
    "RKT": "consumer_credit_finance",
    "UWMC": "consumer_credit_finance",
    "SOFI": "consumer_credit_finance",
    "PYPL": "consumer_credit_finance",
    "COIN": "consumer_credit_finance",
    "HL": "precious_metals_miners",
    "SSRM": "precious_metals_miners",
    "AGI": "precious_metals_miners",
    "NEM": "precious_metals_miners",
    "GOLD": "precious_metals_miners",
    "PAAS": "precious_metals_miners",
    "SNAP": "speculative_growth_turnaround",
}


def theme_bucket(ticker: str, sector: str | None = None) -> str:
    """Return a stable economic theme bucket for concentration diagnostics."""
    symbol = str(ticker or "").upper()
    if symbol in THEME_BUCKETS:
        return THEME_BUCKETS[symbol]
    normalized_sector = str(sector or "Unknown").strip().lower().replace(" ", "_")
    return f"sector_{normalized_sector or 'unknown'}"


def low_price_bucket(price: float, penny_threshold: float = 5.0) -> str:
    """Classify a holding by price-risk bucket."""
    try:
        px = float(price)
    except Exception:
        return "unknown"
    if px < float(penny_threshold):
        return "sub_5"
    if px < 10.0:
        return "5_to_10"
    return "over_10"


def position_value(position: dict) -> float:
    return float(position.get("shares", 0.0)) * float(position.get("last_price", 0.0))


def exposure_weights(values: dict[str, float]) -> dict[str, float]:
    total = float(sum(max(0.0, v) for v in values.values()))
    if total <= 0.0:
        return {}
    return {bucket: max(0.0, value) / total for bucket, value in values.items()}


def effective_bet_count(weights: dict[str, float]) -> float:
    """Herfindahl effective number of independent buckets."""
    denom = sum(float(w) ** 2 for w in weights.values() if float(w) > 0.0)
    if denom <= 0.0:
        return 0.0
    return 1.0 / denom


def portfolio_theme_values(
    positions: dict,
    sector_map: dict[str, str] | None = None,
) -> dict[str, float]:
    sector_map = sector_map or {}
    values: dict[str, float] = {}
    for ticker, pos in positions.items():
        sector = sector_map.get(str(ticker).upper(), "Unknown")
        bucket = theme_bucket(str(ticker), sector)
        values[bucket] = values.get(bucket, 0.0) + position_value(pos)
    return values


def portfolio_low_price_values(
    positions: dict,
    penny_threshold: float = 5.0,
) -> dict[str, float]:
    values = {"sub_5": 0.0, "5_to_10": 0.0, "over_10": 0.0}
    for pos in positions.values():
        bucket = low_price_bucket(float(pos.get("last_price", 0.0)), penny_threshold)
        values[bucket] = values.get(bucket, 0.0) + position_value(pos)
    return values
