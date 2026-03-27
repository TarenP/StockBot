"""
News sentiment pipeline.

Sources (in priority order, all free):
  1. NewsAPI (newsapi.org) — free tier: 100 req/day, 1 month history
     Set env var NEWSAPI_KEY to enable.
  2. Finviz RSS  — no key, no cap, scrapes per-ticker news feed
  3. Yahoo Finance RSS — no key, no cap, per-ticker feed

Scores headlines with FinBERT (ProsusAI/finbert) — the same model
family used to generate your existing sentiment CSV.

Output: appends new rows to Sentiment/analyst_ratings_with_sentiment.csv
        in the exact same schema as the existing data.
"""

import os
import time
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

SENTIMENT_PATH = Path("Sentiment/analyst_ratings_with_sentiment.csv")
NEWSAPI_KEY    = os.getenv("NEWSAPI_KEY", "")

# ── FinBERT loader (lazy — only loads when first needed) ─────────────────────

_finbert_pipeline = None

def _get_finbert():
    global _finbert_pipeline
    if _finbert_pipeline is None:
        try:
            from transformers import pipeline as hf_pipeline
            logger.info("Loading FinBERT model (first run downloads ~500MB)...")
            _finbert_pipeline = hf_pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                top_k=None,          # return all 3 label scores
                device=-1,           # CPU; set to 0 for GPU
                truncation=True,
                max_length=512,
            )
            logger.info("FinBERT loaded.")
        except ImportError:
            raise ImportError(
                "transformers and torch are required for sentiment scoring.\n"
                "Run: pip install transformers torch"
            )
    return _finbert_pipeline


def _score_headlines(headlines: list[str]) -> list[dict]:
    """
    Score a list of headline strings with FinBERT.
    Returns list of dicts with keys: neg_score, neutral_score, pos_score, sentiment.
    """
    if not headlines:
        return []

    pipe    = _get_finbert()
    results = []

    batches = list(range(0, len(headlines), 32))
    pbar = tqdm(batches, desc="  Scoring headlines", unit="batch",
                leave=False, colour="yellow")
    for i in pbar:
        batch = headlines[i:i + 32]
        pbar.set_postfix(headlines=f"{i}–{i+len(batch)}")
        try:
            outputs = pipe(batch)
            for out in outputs:
                scores = {item["label"].lower(): item["score"] for item in out}
                neg  = scores.get("negative", 0.0)
                neu  = scores.get("neutral",  0.0)
                pos  = scores.get("positive", 0.0)
                label = max(scores, key=scores.get)
                # Normalise label to match existing CSV schema
                label = "positive" if label == "positive" else "negative"
                results.append({
                    "neg_score":     neg,
                    "neutral_score": neu,
                    "pos_score":     pos,
                    "sentiment":     label,
                })
        except Exception as e:
            logger.warning(f"FinBERT scoring error: {e}")
            for _ in batch:
                results.append({"neg_score": 0.5, "neutral_score": 0.05,
                                 "pos_score": 0.45, "sentiment": "negative"})
    return results


# ── Source 1: NewsAPI ─────────────────────────────────────────────────────────

def _fetch_newsapi(ticker: str, from_date: str, to_date: str) -> list[dict]:
    """Fetch headlines from NewsAPI. Requires NEWSAPI_KEY env var."""
    if not NEWSAPI_KEY:
        return []
    url = "https://newsapi.org/v2/everything"
    params = {
        "q":        ticker,
        "from":     from_date,
        "to":       to_date,
        "language": "en",
        "sortBy":   "publishedAt",
        "pageSize": 20,
        "apiKey":   NEWSAPI_KEY,
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            articles = r.json().get("articles", [])
            return [
                {
                    "title": a["title"],
                    "date":  a["publishedAt"][:10],
                    "stock": ticker,
                }
                for a in articles if a.get("title")
            ]
    except Exception as e:
        logger.debug(f"NewsAPI error for {ticker}: {e}")
    return []


# ── Source 2: Finviz RSS ──────────────────────────────────────────────────────

def _fetch_finviz_rss(ticker: str) -> list[dict]:
    """Scrape Finviz news RSS — no key, no rate limit (be polite)."""
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; StockBot/1.0)"}
    rows = []
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        news_table = soup.find(id="news-table")
        if not news_table:
            return []
        today = datetime.today().date()
        current_date = today
        for row in news_table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 2:
                continue
            date_cell = cells[0].text.strip()
            title     = cells[1].get_text(strip=True)
            # Finviz shows "Today HH:MM" or "MMM-DD-YY HH:MM"
            if "Today" in date_cell or ":" in date_cell and len(date_cell) < 10:
                current_date = today
            else:
                try:
                    current_date = datetime.strptime(date_cell.split()[0], "%b-%d-%y").date()
                except ValueError:
                    pass
            rows.append({"title": title, "date": str(current_date), "stock": ticker})
    except Exception as e:
        logger.debug(f"Finviz scrape error for {ticker}: {e}")
    return rows


# ── Source 3: Yahoo Finance RSS ───────────────────────────────────────────────

def _fetch_yahoo_rss(ticker: str) -> list[dict]:
    """Yahoo Finance RSS feed — no key, no cap."""
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    rows = []
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.content, "xml")
        for item in soup.find_all("item")[:20]:
            title   = item.find("title")
            pub_date = item.find("pubDate")
            if not title:
                continue
            date_str = str(datetime.today().date())
            if pub_date:
                try:
                    date_str = pd.to_datetime(pub_date.text).strftime("%Y-%m-%d")
                except Exception:
                    pass
            rows.append({"title": title.text.strip(), "date": date_str, "stock": ticker})
    except Exception as e:
        logger.debug(f"Yahoo RSS error for {ticker}: {e}")
    return rows


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_and_score(
    tickers: list[str],
    lookback_days: int = 7,
) -> pd.DataFrame:
    """
    Fetch recent headlines for all tickers and score with FinBERT.
    Returns a DataFrame in the same schema as the existing sentiment CSV.
    """
    from_date = (datetime.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    to_date   = datetime.today().strftime("%Y-%m-%d")

    all_rows = []
    pbar = tqdm(tickers, desc="Fetching news", unit="ticker", colour="green")
    for ticker in pbar:
        pbar.set_postfix(ticker=ticker)
        raw = []

        # Try NewsAPI first (best quality), fall back to scrapers
        if NEWSAPI_KEY:
            raw = _fetch_newsapi(ticker, from_date, to_date)
            time.sleep(0.1)

        if not raw:
            raw = _fetch_finviz_rss(ticker)
            time.sleep(0.15)

        if not raw:
            raw = _fetch_yahoo_rss(ticker)
            time.sleep(0.15)

        if raw:
            headlines = [r["title"] for r in raw]
            scores    = _score_headlines(headlines)
            for row, score in zip(raw, scores):
                all_rows.append({
                    "title":         row["title"],
                    "date":          row["date"],
                    "stock":         row["stock"],
                    "neg_score":     score["neg_score"],
                    "neutral_score": score["neutral_score"],
                    "pos_score":     score["pos_score"],
                    "sentiment":     score["sentiment"],
                })
            pbar.set_postfix(ticker=ticker, headlines=len(raw))

    if not all_rows:
        return pd.DataFrame()

    return pd.DataFrame(all_rows)


def update_sentiment(tickers: list[str], lookback_days: int = 7) -> int:
    """
    Fetch + score new headlines and append to the sentiment CSV.
    Deduplicates on (title, date, stock).
    Returns number of new rows added.
    """
    SENTIMENT_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fetching news for {len(tickers)} tickers (last {lookback_days} days)...")
    new_df = fetch_and_score(tickers, lookback_days=lookback_days)

    if new_df.empty:
        logger.info("No new headlines found.")
        return 0

    # Load existing and deduplicate
    if SENTIMENT_PATH.exists():
        existing = pd.read_csv(SENTIMENT_PATH)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    before = len(combined)
    combined = combined.drop_duplicates(subset=["title", "date", "stock"], keep="last")
    combined = combined.reset_index(drop=True)
    n_new = len(combined) - (before - len(new_df))

    combined.to_csv(SENTIMENT_PATH, index=False)
    logger.info(f"Sentiment updated. {n_new} new rows. Total: {len(combined):,}")
    return n_new
