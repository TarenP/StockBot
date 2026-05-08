"""Map market events and crowd sentiment into ticker diagnostics."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from typing import Iterable

import pandas as pd

from event_sidecar.schemas import MarketEventRecord, TickerEventFeatureRecord


EVENT_KEYWORDS = {
    "conflict": {"war", "missile", "invasion", "attack", "ceasefire", "military", "conflict"},
    "sanctions": {"sanction", "export control", "embargo", "tariff", "trade restriction"},
    "oil_shock": {"oil", "crude", "opec", "pipeline", "refinery", "energy supply"},
    "rates": {"fed", "rate cut", "rate hike", "treasury yield", "bond yield"},
    "inflation": {"inflation", "cpi", "ppi", "prices", "wage pressure"},
    "supply_chain": {"supply chain", "shipping", "port", "freight", "shortage"},
    "cyber": {"cyberattack", "ransomware", "data breach", "hack"},
    "earnings": {"earnings", "guidance", "revenue", "eps", "profit", "outlook"},
}

POSITIVE_WORDS = {
    "beat", "beats", "growth", "upgrade", "raises", "record", "strong", "surge",
    "optimism", "ceasefire", "deal", "recovery", "approval", "profit",
}
NEGATIVE_WORDS = {
    "miss", "cuts", "downgrade", "weak", "slump", "risk", "lawsuit", "probe",
    "attack", "war", "sanction", "shortage", "inflation", "recession", "breach",
}

SECTOR_IMPACTS: dict[str, dict[str, float]] = {
    "conflict": {
        "Energy": 0.20,
        "Industrials": 0.10,
        "Technology": -0.04,
        "Consumer Discretionary": -0.15,
        "Communication Services": -0.05,
    },
    "sanctions": {
        "Energy": 0.10,
        "Materials": 0.05,
        "Industrials": -0.08,
        "Technology": -0.12,
        "Consumer Discretionary": -0.08,
    },
    "oil_shock": {
        "Energy": 0.25,
        "Materials": 0.05,
        "Industrials": -0.10,
        "Consumer Discretionary": -0.15,
        "Communication Services": -0.05,
    },
    "rates": {
        "Financials": 0.08,
        "Real Estate": -0.18,
        "Utilities": -0.12,
        "Technology": -0.08,
        "Consumer Discretionary": -0.08,
    },
    "inflation": {
        "Energy": 0.10,
        "Materials": 0.08,
        "Consumer Staples": 0.03,
        "Consumer Discretionary": -0.12,
        "Real Estate": -0.08,
    },
    "supply_chain": {
        "Industrials": -0.12,
        "Consumer Discretionary": -0.10,
        "Technology": -0.08,
        "Materials": 0.04,
    },
    "cyber": {
        "Technology": -0.04,
        "Financials": -0.06,
        "Communication Services": -0.04,
    },
    "earnings": {},
}


def infer_event_type(text: str) -> str:
    haystack = str(text or "").lower()
    hits = {
        event_type: sum(1 for keyword in keywords if keyword in haystack)
        for event_type, keywords in EVENT_KEYWORDS.items()
    }
    best, count = max(hits.items(), key=lambda item: item[1])
    return best if count > 0 else "market_news"


def lexicon_sentiment_score(text: str) -> float:
    words = str(text or "").lower()
    positive = sum(1 for word in POSITIVE_WORDS if word in words)
    negative = sum(1 for word in NEGATIVE_WORDS if word in words)
    total = positive + negative
    if total == 0:
        return 0.0
    return max(-1.0, min(1.0, (positive - negative) / total))


def infer_sectors(event_type: str) -> list[str]:
    return sorted(SECTOR_IMPACTS.get(event_type, {}).keys())


def _event_weight(event: MarketEventRecord) -> float:
    severity = max(float(event.severity or 0.0), 0.25)
    confidence = max(float(event.confidence or 0.0), 0.25)
    return min(1.0, severity * confidence)


def _ticker_direct_match(event: MarketEventRecord, ticker: str) -> bool:
    if ticker in event.tickers:
        return True
    if len(ticker) < 2:
        return False
    needle = f" {ticker.lower()} "
    text = f" {event.title} {event.summary} ".lower()
    return needle in text


def build_ticker_event_features(
    events: Iterable[MarketEventRecord],
    tickers: list[str],
    *,
    sector_map: dict[str, str] | None = None,
    as_of_date: str | None = None,
) -> list[TickerEventFeatureRecord]:
    symbols = sorted({str(t).strip().upper() for t in tickers or [] if str(t).strip()})
    if not symbols:
        return []
    sector_map = {str(k).upper(): str(v) for k, v in (sector_map or {}).items()}
    events = list(events or [])
    as_of_date = as_of_date or datetime.now().date().isoformat()
    records: list[TickerEventFeatureRecord] = []

    for ticker in symbols:
        sector = sector_map.get(ticker, "Unknown")
        event_scores: list[float] = []
        risk_scores: list[float] = []
        opportunity_scores: list[float] = []
        crowd_scores: list[float] = []
        crowd_dates: list[pd.Timestamp] = []
        source_names: set[str] = set()
        top_events: list[dict] = []
        event_types: Counter[str] = Counter()

        for event in events:
            direct = _ticker_direct_match(event, ticker)
            sector_impact = SECTOR_IMPACTS.get(event.event_type, {}).get(sector, 0.0)
            if not direct and abs(sector_impact) < 1e-9:
                continue

            weight = _event_weight(event)
            direct_impact = float(event.sentiment_score or 0.0) * (0.30 if direct else 0.0)
            impact = (sector_impact + direct_impact) * weight
            event_scores.append(impact)
            if impact < 0:
                risk_scores.append(abs(impact))
            elif impact > 0:
                opportunity_scores.append(impact)
            if event.source in {"reddit", "stocktwits", "social", "sentiment_csv"}:
                crowd_scores.append(float(event.sentiment_score or 0.0) * weight)
                if event.as_of_date:
                    date = pd.to_datetime(event.as_of_date, errors="coerce")
                    if pd.notna(date):
                        crowd_dates.append(pd.Timestamp(date).normalize())
            source_names.add(event.source)
            event_types[event.event_type] += 1
            top_events.append(
                {
                    "event_id": event.event_id,
                    "source": event.source,
                    "event_type": event.event_type,
                    "title": event.title[:160],
                    "score": round(impact, 4),
                    "severity": round(float(event.severity or 0.0), 4),
                }
            )

        if not event_scores:
            records.append(TickerEventFeatureRecord(ticker=ticker, as_of_date=as_of_date))
            continue

        ranked_events = sorted(top_events, key=lambda item: abs(float(item["score"])), reverse=True)[:5]
        crowd_velocity = 0.0
        if crowd_dates:
            latest_date = max(crowd_dates)
            recent = sum(1 for date in crowd_dates if (latest_date - date).days <= 3)
            prior = sum(1 for date in crowd_dates if 3 < (latest_date - date).days <= 14)
            prior_expected = max(1.0, prior * (3.0 / 11.0))
            crowd_velocity = max(-1.0, min(1.0, (recent - prior_expected) / prior_expected))
        records.append(
            TickerEventFeatureRecord(
                ticker=ticker,
                as_of_date=as_of_date,
                event_score=float(max(-1.0, min(1.0, sum(event_scores)))),
                event_risk_score=float(max(risk_scores) if risk_scores else 0.0),
                event_opportunity_score=float(max(opportunity_scores) if opportunity_scores else 0.0),
                crowd_sentiment_score=float(sum(crowd_scores) / len(crowd_scores)) if crowd_scores else 0.0,
                crowd_mention_velocity=float(crowd_velocity),
                mention_count=len(event_scores),
                source_count=len(source_names),
                top_event_types=[name for name, _count in event_types.most_common(5)],
                top_events=ranked_events,
                confidence=min(1.0, 0.25 + 0.15 * len(source_names) + 0.05 * len(event_scores)),
                broker_influence=False,
            )
        )
    return records


def build_events_from_sentiment_frame(frame: pd.DataFrame, tickers: list[str], limit: int = 500) -> list[MarketEventRecord]:
    if frame is None or frame.empty:
        return []
    required = {"title", "date", "stock"}
    if not required.issubset(frame.columns):
        return []
    allowed = {str(t).upper() for t in tickers or []}
    rows = frame.copy()
    rows["stock"] = rows["stock"].astype(str).str.upper()
    if allowed:
        rows = rows[rows["stock"].isin(allowed)]
    rows = rows.tail(max(1, int(limit)))
    events: list[MarketEventRecord] = []
    from event_sidecar.cache import stable_event_id

    for row in rows.to_dict("records"):
        title = str(row.get("title") or "")
        if not title:
            continue
        ticker = str(row.get("stock") or "").upper()
        pos = float(row.get("pos_score", 0.0) or 0.0)
        neg = float(row.get("neg_score", 0.0) or 0.0)
        score = max(-1.0, min(1.0, pos - neg))
        event_type = infer_event_type(title)
        as_of_date = str(row.get("date") or "")[:10] or None
        events.append(
            MarketEventRecord(
                event_id=stable_event_id("sentiment_csv", title, as_of_date, ticker),
                source="sentiment_csv",
                event_type=event_type,
                title=title,
                as_of_date=as_of_date,
                tickers=[ticker],
                sectors=infer_sectors(event_type),
                sentiment_score=score,
                severity=min(1.0, abs(score) + 0.20),
                confidence=0.65,
            )
        )
    return events
