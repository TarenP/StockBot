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
from collections import Counter
from contextlib import contextmanager
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
SENTIMENT_CORE_COLUMNS = ["date", "stock", "neg_score", "neutral_score", "pos_score"]
_SOURCE_FAILURES: Counter[str] = Counter()
_requests_session: requests.Session | None = None
_PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)

# ── FinBERT loader (lazy — only loads when first needed) ─────────────────────

_finbert_pipeline = None


def _get_requests_session() -> requests.Session:
    """
    Use a direct requests session for public news endpoints.

    Some local/sandbox environments set proxy variables that point at a proxy
    listener which is not actually running. The price updater uses yfinance and
    may still work in that situation, while direct news requests fail with
    ProxyError. Disabling env proxy inheritance keeps sentiment refreshes from
    silently starving.
    """
    global _requests_session
    if _requests_session is None:
        _requests_session = requests.Session()
        _requests_session.trust_env = False
    return _requests_session


def _http_get(url: str, **kwargs) -> requests.Response:
    return _get_requests_session().get(url, **kwargs)


def _record_source_failure(source: str, reason: str) -> None:
    _SOURCE_FAILURES[f"{source}:{reason}"] += 1


@contextmanager
def _without_proxy_env():
    saved = {key: os.environ.pop(key) for key in _PROXY_ENV_VARS if key in os.environ}
    try:
        yield
    finally:
        os.environ.update(saved)

def _get_finbert():
    global _finbert_pipeline
    if _finbert_pipeline is None:
        try:
            import torch
            from transformers import pipeline as hf_pipeline
            logger.info("Loading FinBERT model (first run downloads ~500MB)...")
            device = 0 if torch.cuda.is_available() else -1
            with _without_proxy_env():
                _finbert_pipeline = hf_pipeline(
                    "text-classification",
                    model="ProsusAI/finbert",
                    tokenizer="ProsusAI/finbert",
                    top_k=None,
                    device=device,
                    framework="pt",
                    truncation=True,
                    max_length=512,
                )
            logger.info("FinBERT loaded on %s.", "GPU" if device == 0 else "CPU")
        except ImportError:
            raise ImportError(
                "transformers and torch are required for sentiment scoring.\n"
                "Run: pip install transformers torch"
            )
    return _finbert_pipeline


_POSITIVE_WORDS = {
    "beat", "beats", "bullish", "buy", "growth", "higher", "outperform",
    "profit", "raise", "raised", "rally", "record", "strong", "surge",
    "upgrade", "upside", "wins",
}
_NEGATIVE_WORDS = {
    "bearish", "cut", "downgrade", "falls", "fraud", "lawsuit", "loss",
    "miss", "probe", "recall", "risk", "slump", "weak", "warning",
}


def _fallback_score_headline(headline: str) -> dict:
    text = str(headline or "").lower()
    pos_hits = sum(1 for word in _POSITIVE_WORDS if word in text)
    neg_hits = sum(1 for word in _NEGATIVE_WORDS if word in text)
    raw = float(pos_hits - neg_hits)
    pos = float(np.clip(0.45 + raw * 0.12, 0.05, 0.90))
    neg = float(np.clip(0.45 - raw * 0.12, 0.05, 0.90))
    neutral = float(max(0.05, 1.0 - pos - neg))
    total = pos + neg + neutral
    pos /= total
    neg /= total
    neutral /= total
    label = "positive" if pos >= neg else "negative"
    return {
        "neg_score": neg,
        "neutral_score": neutral,
        "pos_score": pos,
        "sentiment": label,
    }


def _score_headlines(headlines: list[str]) -> list[dict]:
    """
    Score a list of headline strings with FinBERT.
    Returns list of dicts with keys: neg_score, neutral_score, pos_score, sentiment.
    """
    if not headlines:
        return []

    try:
        pipe = _get_finbert()
    except Exception as exc:
        logger.warning(
            "FinBERT unavailable (%s); using lexical sentiment fallback for %d headline(s).",
            exc,
            len(headlines),
        )
        return [_fallback_score_headline(headline) for headline in headlines]

    results = []

    batches = list(range(0, len(headlines), 64))
    pbar = tqdm(batches, desc="  Scoring headlines", unit="batch",
                leave=False, colour="yellow")
    for i in pbar:
        batch = headlines[i:i + 64]
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
        r = _http_get(url, params=params, timeout=10)
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
        _record_source_failure("newsapi", f"http_{r.status_code}")
    except Exception as e:
        _record_source_failure("newsapi", type(e).__name__)
        logger.debug(f"NewsAPI error for {ticker}: {e}")
    return []


# ── Source 2: Finviz RSS ──────────────────────────────────────────────────────

def _fetch_finviz_rss(ticker: str) -> list[dict]:
    """Scrape Finviz news RSS — no key, no rate limit (be polite)."""
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; StockBot/1.0)"}
    rows = []
    try:
        r = _http_get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            _record_source_failure("finviz", f"http_{r.status_code}")
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        news_table = soup.find(id="news-table")
        if not news_table:
            _record_source_failure("finviz", "missing_news_table")
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
        _record_source_failure("finviz", type(e).__name__)
        logger.debug(f"Finviz scrape error for {ticker}: {e}")
    return rows


# ── Source 3: Yahoo Finance RSS ───────────────────────────────────────────────

def _fetch_yahoo_rss(ticker: str) -> list[dict]:
    """Yahoo Finance RSS feed — no key, no cap."""
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    rows = []
    try:
        r = _http_get(url, timeout=10)
        if r.status_code != 200:
            _record_source_failure("yahoo_rss", f"http_{r.status_code}")
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
        _record_source_failure("yahoo_rss", type(e).__name__)
        logger.debug(f"Yahoo RSS error for {ticker}: {e}")
    return rows


# ── Public API ────────────────────────────────────────────────────────────────

def _normalize_tickers(tickers: list[str] | None) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for ticker in tickers or []:
        symbol = str(ticker).strip().upper().replace(".", "-")
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        cleaned.append(symbol)
    return cleaned


def _resolve_sentiment_tickers(
    tickers: list[str] | None,
    save_dir: str,
    live_target_size: int | None,
    expand_live_universe: bool,
) -> list[str]:
    seed_tickers = _normalize_tickers(tickers)
    if not expand_live_universe:
        return seed_tickers

    from pipeline.updater import get_live_universe

    resolved = get_live_universe(
        preferred=seed_tickers or None,
        save_dir=save_dir,
        target_size=live_target_size,
    )
    return resolved or seed_tickers


def fetch_and_score(
    tickers: list[str],
    lookback_days: int = 7,
) -> pd.DataFrame:
    """
    Fetch recent headlines for all tickers and score with FinBERT.
    Collects all headlines first, then scores in one batched pass for speed.
    Returns a DataFrame in the same schema as the existing sentiment CSV.
    """
    tickers = _normalize_tickers(tickers)
    if not tickers:
        return pd.DataFrame()

    from_date = (datetime.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    to_date   = datetime.today().strftime("%Y-%m-%d")
    start_date = pd.Timestamp(from_date).date()
    end_date = pd.Timestamp(to_date).date()
    failure_snapshot = Counter(_SOURCE_FAILURES)

    def _is_recent(row: dict) -> bool:
        try:
            row_date = pd.Timestamp(row.get("date")).date()
        except Exception:
            return False
        return start_date <= row_date <= end_date

    # ── Phase 1: fetch all headlines (fast — just HTTP) ───────────────────────
    raw_rows: list[dict] = []
    source_hit_counts: Counter[str] = Counter()
    pbar = tqdm(tickers, desc="Fetching news", unit="ticker", colour="green")
    for ticker in pbar:
        pbar.set_postfix(ticker=ticker)
        raw = []
        source = None

        if NEWSAPI_KEY:
            raw = _fetch_newsapi(ticker, from_date, to_date)
            source = "newsapi" if raw else None
            time.sleep(0.05)

        if not raw:
            raw = _fetch_finviz_rss(ticker)
            source = "finviz" if raw else None
            time.sleep(0.10)

        if not raw:
            raw = _fetch_yahoo_rss(ticker)
            source = "yahoo_rss" if raw else None
            time.sleep(0.10)

        if raw:
            raw = [row for row in raw if _is_recent(row)]
        if raw and source:
            source_hit_counts[source] += 1
        raw_rows.extend(raw)

    if not raw_rows:
        failures = _SOURCE_FAILURES - failure_snapshot
        if failures:
            top_failures = ", ".join(
                f"{key}={count}" for key, count in failures.most_common(5)
            )
            logger.warning(
                "No recent headlines found; source failures during fetch: %s",
                top_failures,
            )
        else:
            logger.info(
                "No recent headlines found from configured sources for %d tickers.",
                len(tickers),
            )
        return pd.DataFrame()
    if source_hit_counts:
        logger.info(
            "Headline source coverage: %s",
            ", ".join(f"{key}={value}" for key, value in source_hit_counts.items()),
        )

    # ── Phase 2: score all headlines in one batched FinBERT pass ─────────────
    all_headlines = [r["title"] for r in raw_rows]
    logger.info("Scoring %d headlines with FinBERT...", len(all_headlines))
    scores = _score_headlines(all_headlines)

    all_rows = []
    for row, score in zip(raw_rows, scores):
        all_rows.append({
            "title":         row["title"],
            "date":          row["date"],
            "stock":         row["stock"],
            "neg_score":     score["neg_score"],
            "neutral_score": score["neutral_score"],
            "pos_score":     score["pos_score"],
            "sentiment":     score["sentiment"],
        })

    return pd.DataFrame(all_rows)


def _read_existing_keys_for_dates(
    path: Path,
    key_cols: list[str],
    dates: set[str],
) -> set[tuple]:
    if not dates:
        return set()

    keys: set[tuple] = set()
    try:
        for chunk in pd.read_csv(path, usecols=key_cols, chunksize=1_000_000):
            chunk["date"] = chunk["date"].astype(str).str[:10]
            chunk = chunk[chunk["date"].isin(dates)]
            if chunk.empty:
                continue
            for col in key_cols:
                chunk[col] = chunk[col].astype(str)
            keys.update(tuple(row) for row in chunk[key_cols].itertuples(index=False, name=None))
    except ValueError:
        return set()
    return keys


def _storage_frame_for_existing_schema(new_df: pd.DataFrame, existing_cols: list[str]) -> pd.DataFrame:
    frame = new_df.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    frame = frame.dropna(subset=["date", "stock"])

    if "title" not in existing_cols:
        core = frame[SENTIMENT_CORE_COLUMNS].copy()
        return (
            core.groupby(["date", "stock"], as_index=False)[
                ["neg_score", "neutral_score", "pos_score"]
            ]
            .mean()
            .loc[:, SENTIMENT_CORE_COLUMNS]
        )

    for col in existing_cols:
        if col not in frame.columns:
            frame[col] = np.nan
    return frame[existing_cols]


def update_sentiment(
    tickers: list[str] | None = None,
    lookback_days: int = 7,
    save_dir: str = "models",
    live_target_size: int | None = None,
    expand_live_universe: bool = True,
) -> int:
    """
    Fetch + score new headlines and append to the sentiment CSV.
    Deduplicates on (title, date, stock).
    Returns number of new rows added.
    """
    SENTIMENT_PATH.parent.mkdir(parents=True, exist_ok=True)

    tickers = _resolve_sentiment_tickers(
        tickers=tickers,
        save_dir=save_dir,
        live_target_size=live_target_size,
        expand_live_universe=expand_live_universe,
    )
    if not tickers:
        logger.info("No tickers resolved for sentiment update.")
        return 0

    logger.info(f"Fetching news for {len(tickers)} tickers (last {lookback_days} days)...")
    new_df = fetch_and_score(tickers, lookback_days=lookback_days)

    if new_df.empty:
        logger.info("No new headlines found.")
        return 0

    if SENTIMENT_PATH.exists():
        existing_cols = pd.read_csv(SENTIMENT_PATH, nrows=0).columns.tolist()
        storage_df = _storage_frame_for_existing_schema(new_df, existing_cols)
        if storage_df.empty:
            logger.info("No new sentiment rows after schema normalization.")
            return 0

        key_cols = ["title", "date", "stock"] if "title" in existing_cols else ["date", "stock"]
        candidate_dates = set(storage_df["date"].astype(str).str[:10])
        existing_keys = _read_existing_keys_for_dates(
            SENTIMENT_PATH,
            key_cols=key_cols,
            dates=candidate_dates,
        )
        if existing_keys:
            key_tuples = storage_df[key_cols].astype(str).itertuples(index=False, name=None)
            keep_mask = [tuple(row) not in existing_keys for row in key_tuples]
            storage_df = storage_df.loc[keep_mask]

        if storage_df.empty:
            logger.info("Sentiment already up to date for fetched headline dates.")
            return 0

        storage_df.to_csv(
            SENTIMENT_PATH,
            mode="a",
            header=False,
            index=False,
            columns=existing_cols,
        )
        n_new = len(storage_df)
    else:
        new_df.to_csv(SENTIMENT_PATH, index=False)
        n_new = len(new_df)

    logger.info(f"Sentiment updated. {n_new} new rows appended.")
    return n_new
