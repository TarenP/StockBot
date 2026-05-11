"""
Auto-fetch documents for the LLM sidecar.

Fetches recent SEC filings (8-K, 10-Q, 10-K) and earnings call transcripts
for a list of tickers and stores them in the document store for sidecar
precompute.

Sources:
  - SEC EDGAR (8-K, 10-Q, 10-K) — free, no API key
  - yfinance earnings calendar — for earnings dates
  - Finviz/Yahoo RSS — for earnings-related news (fallback)

All fetches are rate-limited and cached by doc_id so the same document
is never stored twice.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# SEC rate limit: 10 requests/second max, we use 0.15s between requests
_SEC_DELAY = 0.15
_DEFAULT_LOOKBACK_DAYS = 90
_FORMS_TO_FETCH = {"8-K"}  # 10-Q/10-K are too long and mostly financial tables; 8-K is event-driven narrative
DEFAULT_SEC_USER_AGENT = os.getenv(
    "STOCKBOT_SEC_USER_AGENT",
    "StockBot research sidecar; set STOCKBOT_SEC_USER_AGENT with contact email",
)
_SEC_USER_AGENT_WARNING_EMITTED = False


def get_sec_user_agent() -> str:
    """Return the configured SEC User-Agent and warn once if contact info is missing."""
    global _SEC_USER_AGENT_WARNING_EMITTED
    configured = os.getenv("STOCKBOT_SEC_USER_AGENT")
    if configured:
        return configured
    if not _SEC_USER_AGENT_WARNING_EMITTED:
        logger.warning(
            "STOCKBOT_SEC_USER_AGENT is not set; SEC requests will use a generic project User-Agent. "
            "Set STOCKBOT_SEC_USER_AGENT with contact information before live SEC fetches."
        )
        _SEC_USER_AGENT_WARNING_EMITTED = True
    return DEFAULT_SEC_USER_AGENT


def _load_ticker_cik_map(cache_path: str = "broker/state/company_tickers.json") -> dict[str, str]:
    """Load or refresh the SEC ticker→CIK map."""
    import json
    import requests

    path = Path(cache_path)

    # Refresh if missing or older than 7 days
    if not path.exists() or (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).days > 7:
        try:
            logger.info("Refreshing SEC ticker→CIK map...")
            r = requests.get(
                "https://www.sec.gov/files/company_tickers.json",
                headers={"User-Agent": get_sec_user_agent()},
                timeout=30,
            )
            r.raise_for_status()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(r.text, encoding="utf-8")
            logger.info("SEC ticker map saved: %d entries", len(r.json()))
        except Exception as exc:
            logger.warning("Could not refresh SEC ticker map: %s", exc)

    from data_sources.sec_edgar import load_ticker_cik_map
    return load_ticker_cik_map(path)


def _is_useful_for_llm(text: str) -> bool:
    """
    Return False for documents that are XBRL, pure financial tables,
    or otherwise not useful for narrative extraction.
    """
    if len(text) < 200:
        return False
    # XBRL files are tag soup — no narrative
    if text.strip().startswith("<?xml") or "<xbrl" in text[:500].lower():
        return False
    # Pure financial tables with no sentences
    words = text.split()
    if len(words) < 50:
        return False
    # If >60% of tokens are numbers/symbols, it's a table not narrative
    numeric = sum(1 for w in words[:200] if w.replace(".", "").replace(",", "").replace("$", "").replace("-", "").isdigit())
    if numeric / min(len(words), 200) > 0.5:
        return False
    return True


def _strip_html(text: str) -> str:
    """Remove HTML tags, entities, and normalize whitespace."""
    import re
    import html as _html

    text = _html.unescape(text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-zA-Z#0-9]+;", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fetch_sec_filings(
    tickers: list[str],
    store_dir: str = "broker/state/document_store",
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
    forms: set[str] | None = None,
    max_chars: int = 24000,
    user_agent: str | None = None,
) -> dict[str, int]:
    """
    Fetch recent SEC filings for a list of tickers and store them.

    Returns dict of {ticker: n_stored}.
    """
    from data_sources.document_store import DocumentStore
    from data_sources.sec_edgar import (
        get_submissions, recent_filings, download_filing_document
    )

    store = DocumentStore(store_dir)
    user_agent = user_agent or get_sec_user_agent()
    ticker_cik = _load_ticker_cik_map()
    forms_to_fetch = forms or _FORMS_TO_FETCH
    cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    results: dict[str, int] = {}

    for ticker in tickers:
        ticker = ticker.upper()
        cik = ticker_cik.get(ticker)
        if not cik:
            logger.debug("No CIK found for %s — skipping SEC fetch", ticker)
            continue

        try:
            submissions = get_submissions(cik, user_agent=user_agent)
            filings = recent_filings(submissions, forms=forms_to_fetch)
            stored = 0

            for filing in filings:
                filing_date = str(filing.get("filingDate") or "")
                if filing_date < cutoff:
                    continue

                accession = filing.get("accessionNumber")
                primary_doc = filing.get("primaryDocument")
                form = filing.get("form", "filing")

                if not accession or not primary_doc:
                    continue

                # Skip non-text documents
                if not any(primary_doc.lower().endswith(ext) for ext in (".htm", ".html", ".txt")):
                    continue

                try:
                    raw_text = download_filing_document(
                        cik, accession, primary_doc, user_agent=user_agent
                    )
                    clean_text = _strip_html(raw_text)[:max_chars]

                    if len(clean_text) < 200:
                        continue

                    if not _is_useful_for_llm(clean_text):
                        logger.debug("Skipping %s %s for %s — not useful for LLM (XBRL/table)", form, filing_date, ticker)
                        continue

                    doc_id = store.put(
                        ticker=ticker,
                        text=clean_text,
                        source_type=form.lower().replace("-", "_"),
                        as_of_date=filing_date,
                        metadata={
                            "form": form,
                            "accession": accession,
                            "primary_document": primary_doc,
                            "cik": cik,
                        },
                    )
                    stored += 1
                    logger.debug("Stored %s %s for %s (doc_id=%s)", form, filing_date, ticker, doc_id)
                    time.sleep(_SEC_DELAY)

                except Exception as exc:
                    logger.debug("Failed to fetch %s filing for %s: %s", form, ticker, exc)
                    time.sleep(_SEC_DELAY)

            results[ticker] = stored
            if stored:
                logger.info("SEC: stored %d filing(s) for %s", stored, ticker)

        except Exception as exc:
            logger.warning("SEC fetch failed for %s (CIK=%s): %s", ticker, cik, exc)
            results[ticker] = 0

        time.sleep(_SEC_DELAY)

    return results


def fetch_earnings_news(
    tickers: list[str],
    store_dir: str = "broker/state/document_store",
    lookback_days: int = 30,
    max_chars: int = 24000,
) -> dict[str, int]:
    """
    Fetch earnings-related news articles for tickers using yfinance.
    Stores them as 'earnings_news' documents for sidecar parsing.
    """
    import yfinance as yf
    from data_sources.document_store import DocumentStore

    store = DocumentStore(store_dir)
    results: dict[str, int] = {}
    cutoff = datetime.now() - timedelta(days=lookback_days)

    for ticker in tickers:
        ticker = ticker.upper()
        stored = 0
        try:
            t = yf.Ticker(ticker)
            news = t.news or []
            for item in news:
                pub_ts = item.get("providerPublishTime", 0)
                if pub_ts:
                    pub_date = datetime.fromtimestamp(pub_ts)
                    if pub_date < cutoff:
                        continue
                    as_of_date = pub_date.strftime("%Y-%m-%d")
                else:
                    as_of_date = datetime.now().strftime("%Y-%m-%d")

                title = item.get("title", "")
                summary = item.get("summary", "") or item.get("description", "")
                text = f"{title}\n\n{summary}".strip()

                if len(text) < 50:
                    continue

                # Only store earnings/guidance/revenue related news
                keywords = {"earnings", "revenue", "guidance", "eps", "profit",
                            "loss", "beat", "miss", "outlook", "forecast", "quarter"}
                if not any(kw in text.lower() for kw in keywords):
                    continue

                store.put(
                    ticker=ticker,
                    text=text[:max_chars],
                    source_type="earnings_news",
                    as_of_date=as_of_date,
                    metadata={"source": item.get("publisher", ""), "url": item.get("link", "")},
                )
                stored += 1

        except Exception as exc:
            logger.debug("Earnings news fetch failed for %s: %s", ticker, exc)

        results[ticker] = stored
        if stored:
            logger.info("Earnings news: stored %d article(s) for %s", stored, ticker)

    return results


def fetch_documents_for_universe(
    tickers: list[str],
    store_dir: str = "broker/state/document_store",
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
    fetch_sec: bool = True,
    fetch_news: bool = True,
    max_chars: int = 24000,
) -> dict[str, Any]:
    """
    Main entry point. Fetch all document types for a list of tickers.

    Called by the orchestrator before the LLM sidecar precompute task.
    Gracefully skips tickers where fetching fails.

    Returns summary dict with counts.
    """
    if not tickers:
        return {"sec_stored": 0, "news_stored": 0, "tickers_processed": 0}

    logger.info(
        "Fetching documents for %d tickers (lookback=%dd, sec=%s, news=%s)",
        len(tickers), lookback_days, fetch_sec, fetch_news,
    )

    sec_results: dict[str, int] = {}
    news_results: dict[str, int] = {}

    if fetch_sec:
        try:
            sec_results = fetch_sec_filings(
                tickers,
                store_dir=store_dir,
                lookback_days=lookback_days,
                max_chars=max_chars,
            )
        except Exception as exc:
            logger.warning("SEC document fetch failed: %s", exc)

    if fetch_news:
        try:
            news_results = fetch_earnings_news(
                tickers,
                store_dir=store_dir,
                lookback_days=min(lookback_days, 30),
                max_chars=max_chars,
            )
        except Exception as exc:
            logger.warning("Earnings news fetch failed: %s", exc)

    total_sec = sum(sec_results.values())
    total_news = sum(news_results.values())

    logger.info(
        "Document fetch complete: %d SEC filings, %d news articles across %d tickers",
        total_sec, total_news, len(tickers),
    )

    return {
        "sec_stored": total_sec,
        "news_stored": total_news,
        "tickers_processed": len(tickers),
        "sec_by_ticker": sec_results,
        "news_by_ticker": news_results,
    }
