"""Public-source event fetchers for the market-event sidecar."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from event_sidecar.cache import EventSidecarCache, stable_event_id
from event_sidecar.impact import (
    build_events_from_sentiment_frame,
    infer_event_type,
    infer_sectors,
    lexicon_sentiment_score,
)
from event_sidecar.schemas import MarketEventRecord

logger = logging.getLogger(__name__)

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
DEFAULT_GLOBAL_QUERIES = [
    "war OR missile OR invasion OR ceasefire",
    "sanctions OR export controls OR tariff",
    "oil OR crude OR OPEC OR pipeline",
    "Federal Reserve OR inflation OR treasury yields",
    "supply chain OR shipping OR port disruption",
    "cyberattack OR ransomware OR data breach",
]


def _gdelt_timespan(lookback_days: int) -> str:
    days = max(1, min(30, int(lookback_days)))
    return f"{days}d"


def fetch_gdelt_events(
    *,
    lookback_days: int = 3,
    queries: list[str] | None = None,
    max_records_per_query: int = 15,
    timeout: int = 20,
) -> list[MarketEventRecord]:
    events: list[MarketEventRecord] = []
    for query in queries or DEFAULT_GLOBAL_QUERIES:
        params = {
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "sort": "HybridRel",
            "timespan": _gdelt_timespan(lookback_days),
            "maxrecords": max(1, min(50, int(max_records_per_query))),
        }
        try:
            response = requests.get(GDELT_DOC_API, params=params, timeout=timeout)
            response.raise_for_status()
            articles = response.json().get("articles", [])
        except Exception as exc:
            logger.debug("GDELT fetch failed for query %r: %s", query, exc)
            continue

        for article in articles:
            title = str(article.get("title") or "").strip()
            if not title:
                continue
            url = article.get("url")
            seendate = str(article.get("seendate") or "")[:8]
            as_of_date = None
            if len(seendate) == 8 and seendate.isdigit():
                as_of_date = f"{seendate[:4]}-{seendate[4:6]}-{seendate[6:8]}"
            text = " ".join([title, str(article.get("sourceCountry") or ""), query])
            event_type = infer_event_type(text)
            sentiment = lexicon_sentiment_score(text)
            severity = min(1.0, 0.35 + abs(sentiment) * 0.35)
            events.append(
                MarketEventRecord(
                    event_id=stable_event_id("gdelt", title, as_of_date, url),
                    source="gdelt",
                    event_type=event_type,
                    title=title,
                    summary=str(article.get("domain") or ""),
                    url=url,
                    as_of_date=as_of_date or datetime.now().date().isoformat(),
                    geography=article.get("sourceCountry"),
                    sectors=infer_sectors(event_type),
                    sentiment_score=sentiment,
                    severity=severity,
                    confidence=0.60,
                    metadata={
                        "query": query,
                        "domain": article.get("domain"),
                        "language": article.get("language"),
                    },
                )
            )
    return events


def load_sentiment_csv_events(
    tickers: list[str],
    *,
    sentiment_path: str | Path = "Sentiment/analyst_ratings_with_sentiment.csv",
    limit: int = 500,
) -> list[MarketEventRecord]:
    try:
        import pandas as pd

        path = Path(sentiment_path)
        if not path.exists():
            return []
        return build_events_from_sentiment_frame(pd.read_csv(path), tickers, limit=limit)
    except Exception as exc:
        logger.debug("Could not load sentiment CSV events: %s", exc)
        return []


def fetch_and_store_market_events(
    tickers: list[str],
    *,
    cache_dir: str | Path = "broker/state/event_sidecar",
    lookback_days: int = 3,
    include_gdelt: bool = True,
    include_sentiment_csv: bool = True,
    gdelt_queries: list[str] | None = None,
    max_gdelt_records_per_query: int = 15,
    sentiment_csv_limit: int = 500,
) -> dict[str, Any]:
    cache = EventSidecarCache(cache_dir)
    events: list[MarketEventRecord] = []
    if include_gdelt:
        events.extend(
            fetch_gdelt_events(
                lookback_days=lookback_days,
                queries=gdelt_queries,
                max_records_per_query=max_gdelt_records_per_query,
            )
        )
    if include_sentiment_csv:
        events.extend(
            load_sentiment_csv_events(
                tickers,
                limit=sentiment_csv_limit,
            )
        )

    stored = 0
    seen: set[str] = set()
    for event in events:
        if event.event_id in seen:
            continue
        seen.add(event.event_id)
        cache.put_event(event)
        stored += 1

    by_type: dict[str, int] = {}
    for event in events:
        by_type[event.event_type] = by_type.get(event.event_type, 0) + 1
    return {
        "stored_events": stored,
        "candidate_events": len(events),
        "by_event_type": by_type,
    }

