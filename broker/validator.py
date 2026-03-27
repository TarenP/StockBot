"""
Data quality validator.

When a suspicious price move is detected (>30% in one day), it doesn't
just discard it — it cross-checks multiple sources to determine if the
move is real before deciding what to do.

Verification sources:
  1. yfinance (second fetch with different parameters)
  2. Finviz price quote
  3. News check — is there a headline explaining the move?

Decision:
  - Confirmed real  → accept the data, flag as high-volatility event
  - Likely bad data → use previous close, log warning
  - Uncertain       → use previous close conservatively, log for review
"""

import logging
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)

# Moves larger than this trigger cross-verification
SUSPICIOUS_MOVE_THRESHOLD = 0.30   # 30% single-day move


def check_move_is_real(
    ticker: str,
    reported_price: float,
    prev_price: float,
    move_pct: float,
) -> tuple[bool, str]:
    """
    Cross-verify a suspicious price move across multiple sources.

    Returns:
        (is_real: bool, explanation: str)
    """
    logger.info(
        f"  Verifying suspicious move: {ticker} "
        f"{move_pct:+.1%} (${prev_price:.3f} → ${reported_price:.3f})"
    )

    confirmations = 0
    explanations  = []

    # ── Source 1: Re-fetch from yfinance with fresh session ───────────────────
    try:
        import os, sys
        from contextlib import contextmanager

        @contextmanager
        def _quiet():
            with open(os.devnull, "w") as dn:
                old = sys.stderr; sys.stderr = dn
                try: yield
                finally: sys.stderr = old

        with _quiet():
            ticker_obj = yf.Ticker(ticker)
            hist = ticker_obj.history(period="5d", auto_adjust=True)

        if not hist.empty:
            yf_price = float(hist["Close"].iloc[-1])
            yf_move  = (yf_price - prev_price) / prev_price if prev_price > 0 else 0

            if abs(yf_move) >= abs(move_pct) * 0.7:
                confirmations += 1
                explanations.append(f"yfinance confirms: ${yf_price:.3f} ({yf_move:+.1%})")
            else:
                explanations.append(
                    f"yfinance disagrees: ${yf_price:.3f} ({yf_move:+.1%})"
                )
    except Exception as e:
        explanations.append(f"yfinance check failed: {e}")

    # ── Source 2: Finviz price quote ──────────────────────────────────────────
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; StockBot/1.0)"}
        url     = f"https://finviz.com/quote.ashx?t={ticker}"
        r       = requests.get(url, headers=headers, timeout=8)
        if r.status_code == 200:
            from bs4 import BeautifulSoup
            soup  = BeautifulSoup(r.text, "html.parser")
            # Finviz shows price in a specific table cell
            price_cell = soup.find("strong", class_="quote-price")
            if not price_cell:
                # Try alternate selector
                cells = soup.find_all("td", class_="snapshot-td2")
                for cell in cells:
                    try:
                        val = float(cell.text.strip().replace(",", ""))
                        if 0.001 < val < 1_000_000:
                            finviz_price = val
                            finviz_move  = (finviz_price - prev_price) / prev_price
                            if abs(finviz_move) >= abs(move_pct) * 0.7:
                                confirmations += 1
                                explanations.append(
                                    f"Finviz confirms: ${finviz_price:.3f} ({finviz_move:+.1%})"
                                )
                            else:
                                explanations.append(
                                    f"Finviz disagrees: ${finviz_price:.3f}"
                                )
                            break
                    except ValueError:
                        pass
        time.sleep(0.3)
    except Exception as e:
        explanations.append(f"Finviz check failed: {e}")

    # ── Source 3: News check — is there a headline explaining the move? ───────
    try:
        from pipeline.sentiment import _fetch_finviz_rss, _fetch_yahoo_rss
        news = _fetch_finviz_rss(ticker) or _fetch_yahoo_rss(ticker)

        if news:
            today     = datetime.today().date()
            yesterday = today - timedelta(days=1)
            recent    = [
                n for n in news
                if str(today) in n.get("date", "") or str(yesterday) in n.get("date", "")
            ]

            # Keywords that explain large moves
            move_keywords = [
                "earnings", "acquisition", "merger", "fda", "approval", "buyout",
                "bankruptcy", "delisted", "fraud", "sec", "lawsuit", "guidance",
                "upgrade", "downgrade", "beat", "miss", "revenue", "profit",
                "dividend", "split", "offering", "dilut", "short", "squeeze",
            ]

            for article in recent:
                title_lower = article.get("title", "").lower()
                if any(kw in title_lower for kw in move_keywords):
                    confirmations += 1
                    explanations.append(
                        f"News explains move: \"{article['title'][:80]}\""
                    )
                    break
            else:
                if recent:
                    explanations.append(
                        f"Recent news found but no clear catalyst: "
                        f"\"{recent[0].get('title', '')[:60]}\""
                    )
                else:
                    explanations.append("No recent news found")
    except Exception as e:
        explanations.append(f"News check failed: {e}")

    # ── Decision ──────────────────────────────────────────────────────────────
    is_real     = confirmations >= 2
    explanation = " | ".join(explanations)

    if is_real:
        logger.info(f"  ✓ Move CONFIRMED real ({confirmations}/3 sources agree): {explanation}")
    else:
        logger.warning(
            f"  ✗ Move LIKELY BAD DATA ({confirmations}/3 sources agree): {explanation}"
        )

    return is_real, explanation


def validate_price_update(
    ticker: str,
    new_price: float,
    old_price: float,
) -> tuple[float, bool, str]:
    """
    Validate a price update. If the move is suspicious, cross-verify.

    Returns:
        (validated_price, is_clean, note)
        - validated_price: price to use (new_price if real, old_price if bad data)
        - is_clean: True if no issues
        - note: explanation string
    """
    if old_price <= 0 or new_price <= 0:
        return new_price, True, ""

    move_pct = (new_price - old_price) / old_price

    if abs(move_pct) < SUSPICIOUS_MOVE_THRESHOLD:
        return new_price, True, ""

    # Suspicious — cross-verify
    is_real, explanation = check_move_is_real(ticker, new_price, old_price, move_pct)

    if is_real:
        note = f"LARGE MOVE CONFIRMED: {move_pct:+.1%} | {explanation}"
        return new_price, True, note
    else:
        note = f"SUSPICIOUS MOVE REJECTED: {move_pct:+.1%} | {explanation}"
        return old_price, False, note


def validate_portfolio_prices(
    positions: dict,
    new_prices: dict[str, float],
) -> dict[str, float]:
    """
    Validate all price updates for held positions.
    Returns cleaned price dict — bad data replaced with previous close.
    """
    validated = {}
    for ticker, new_price in new_prices.items():
        if ticker not in positions:
            validated[ticker] = new_price
            continue

        old_price = positions[ticker].get("last_price", 0.0)
        clean_price, is_clean, note = validate_price_update(ticker, new_price, old_price)

        if note:
            logger.warning(f"  {ticker}: {note}")

        validated[ticker] = clean_price

    return validated
