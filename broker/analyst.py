"""
On-demand stock analyst.
Fetches price history + live news for any ticker on request,
computes features, and returns a scored research report.
Used by the broker brain to investigate specific stocks.
"""

import logging
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from pipeline.features import _compute_all, FEATURE_COLS
from pipeline.sentiment import _fetch_finviz_rss, _fetch_yahoo_rss, _score_headlines

logger = logging.getLogger(__name__)

logging.getLogger("yfinance").setLevel(logging.CRITICAL)


@contextmanager
def _quiet():
    with open(os.devnull, "w") as dn:
        old = sys.stderr
        sys.stderr = dn
        try:
            yield
        finally:
            sys.stderr = old


def fetch_ticker_data(ticker: str, days: int = 90) -> pd.DataFrame | None:
    """
    Fetch OHLCV for a single ticker going back `days` days.
    Returns a DataFrame indexed by date, or None if unavailable.
    """
    start = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    end   = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        with _quiet():
            raw = yf.download(ticker, start=start, end=end,
                              auto_adjust=True, progress=False)
        if raw.empty:
            return None
        raw = raw[["Open", "High", "Low", "Close", "Volume"]].rename(columns=str.lower)
        raw.index = pd.to_datetime(raw.index).normalize()
        raw.index.name = "date"
        raw = raw.dropna(subset=["close"])
        raw = raw[raw["close"] > 0]
        return raw if not raw.empty else None
    except Exception:
        return None


def fetch_ticker_sentiment(ticker: str) -> dict:
    """
    Fetch and score latest news headlines for a single ticker.
    Returns dict with sentiment scores and headline list.
    """
    raw = _fetch_finviz_rss(ticker) or _fetch_yahoo_rss(ticker)
    if not raw:
        return {"pos_score": 0.45, "neg_score": 0.45, "neutral_score": 0.05,
                "headlines": [], "sentiment": "neutral"}

    headlines = [r["title"] for r in raw]
    scores    = _score_headlines(headlines)

    if not scores:
        return {"pos_score": 0.45, "neg_score": 0.45, "neutral_score": 0.05,
                "headlines": headlines, "sentiment": "neutral"}

    avg_pos  = float(np.mean([s["pos_score"]     for s in scores]))
    avg_neg  = float(np.mean([s["neg_score"]     for s in scores]))
    avg_neu  = float(np.mean([s["neutral_score"] for s in scores]))
    label    = "positive" if avg_pos > avg_neg else "negative"

    return {
        "pos_score":     avg_pos,
        "neg_score":     avg_neg,
        "neutral_score": avg_neu,
        "sentiment":     label,
        "headlines":     headlines[:5],   # top 5 for logging
        "sent_net":      avg_pos - avg_neg,
    }


def research(ticker: str, days: int = 90) -> dict | None:
    """
    Full research report for a ticker: price features + sentiment.
    Returns a dict with all signals, or None if data unavailable.
    """
    price_df = fetch_ticker_data(ticker, days=days)
    if price_df is None or len(price_df) < 20:
        return None

    # Inject sentiment into price df
    sent = fetch_ticker_sentiment(ticker)
    price_df["pos_score"]     = sent["pos_score"]
    price_df["neg_score"]     = sent["neg_score"]
    price_df["neutral_score"] = sent["neutral_score"]

    # Compute features
    try:
        feat_df = _compute_all(price_df.copy())
    except Exception as e:
        logger.debug(f"Feature error for {ticker}: {e}")
        return None

    # Get latest row
    latest = feat_df.iloc[-1]
    report = {
        "ticker":       ticker,
        "price":        float(price_df["close"].iloc[-1]),
        "volume":       float(price_df["volume"].iloc[-1]),
        "sentiment":    sent,
        "headlines":    sent.get("headlines", []),
    }

    for col in FEATURE_COLS:
        if col in feat_df.columns:
            val = latest.get(col, np.nan)
            report[col] = float(val) if pd.notna(val) else 0.0

    # Composite score: weighted combination of key signals
    report["composite_score"] = _composite_score(report, sent)
    return report


def _composite_score(report: dict, sent: dict) -> float:
    """
    Simple weighted signal score in [0, 1].
    Higher = stronger buy signal.
    """
    scores = []

    # Momentum (higher = better)
    mom5  = report.get("ret_5d",  0.0)
    mom20 = report.get("ret_20d", 0.0)
    scores.append(np.clip(0.5 + mom5  * 5, 0, 1) * 0.15)
    scores.append(np.clip(0.5 + mom20 * 2, 0, 1) * 0.10)

    # RSI: prefer 40-65 range (not overbought/oversold)
    rsi = report.get("rsi", 50.0)
    rsi_score = 1.0 - abs(rsi - 52.5) / 52.5
    scores.append(np.clip(rsi_score, 0, 1) * 0.10)

    # MACD histogram positive = bullish
    macd = report.get("macd_hist", 0.0)
    scores.append(np.clip(0.5 + macd * 10, 0, 1) * 0.10)

    # Bollinger: prefer middle of band (not at extremes)
    bb = report.get("bb_pct", 0.5)
    bb_score = 1.0 - abs(bb - 0.5) * 2
    scores.append(np.clip(bb_score, 0, 1) * 0.05)

    # Volume surge = attention
    vol = report.get("vol_ratio", 1.0)
    scores.append(np.clip(vol / 3.0, 0, 1) * 0.10)

    # Sentiment (most important)
    sent_net = sent.get("sent_net", 0.0)
    scores.append(np.clip(0.5 + sent_net, 0, 1) * 0.20)

    # Sentiment surprise (sudden positive news)
    surprise = report.get("sent_surprise", 0.0)
    scores.append(np.clip(0.5 + surprise * 2, 0, 1) * 0.20)

    return float(sum(scores))
