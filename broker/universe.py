"""
Dynamic universe manager.
Discovers new stocks continuously — checks for tickers not yet in the
parquet, validates them via yfinance, and adds them to the watchlist.
"""

import logging
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

WATCHLIST_PATH = Path("broker/state/watchlist.csv")


@contextmanager
def _quiet():
    with open(os.devnull, "w") as dn:
        old = sys.stderr; sys.stderr = dn
        try: yield
        finally: sys.stderr = old


def load_watchlist() -> set[str]:
    if WATCHLIST_PATH.exists():
        df = pd.read_csv(WATCHLIST_PATH)
        return set(df["ticker"].str.upper().tolist())
    return set()


def save_watchlist(tickers: set[str]):
    WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ticker": sorted(tickers)}).to_csv(WATCHLIST_PATH, index=False)


def get_parquet_universe() -> set[str]:
    path = Path("MasterDS/stooq_panel.parquet")
    if not path.exists():
        return set()
    df = pd.read_parquet(path, columns=["ticker"])
    return set(df["ticker"].str.upper().unique())


def discover_new_tickers(max_new: int = 200) -> list[str]:
    """
    Discover tickers not yet in the parquet by scraping:
    1. Finviz screener (all US stocks)
    2. Yahoo Finance trending
    Returns list of new validated tickers.
    """
    known    = get_parquet_universe()
    watchlist = load_watchlist()
    already  = known | watchlist
    found    = set()

    # ── Source 1: Finviz full screener ───────────────────────────────────────
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; StockBot/1.0)"}
        # Finviz screener — all stocks, sorted by volume
        for page in range(1, 6):   # 5 pages × ~20 tickers
            url = f"https://finviz.com/screener.ashx?v=111&o=-volume&r={1 + (page-1)*20}"
            r   = requests.get(url, headers=headers, timeout=10)
            if r.status_code != 200:
                break
            from bs4 import BeautifulSoup
            soup  = BeautifulSoup(r.text, "html.parser")
            cells = soup.find_all("a", class_="screener-link-primary")
            for cell in cells:
                t = cell.text.strip().upper()
                if t and t not in already:
                    found.add(t)
            time.sleep(0.5)
    except Exception as e:
        logger.debug(f"Finviz discovery error: {e}")

    # ── Source 2: Yahoo Finance trending ─────────────────────────────────────
    try:
        url = "https://finance.yahoo.com/trending-tickers"
        r   = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        if r.status_code == 200:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", {"data-symbol": True}):
                t = a["data-symbol"].upper()
                if t and "." not in t and t not in already:
                    found.add(t)
    except Exception as e:
        logger.debug(f"Yahoo trending error: {e}")

    if not found:
        return []

    # ── Validate via yfinance (check they actually trade) ────────────────────
    candidates = list(found)[:max_new * 2]
    valid      = []

    logger.info(f"Validating {len(candidates)} new ticker candidates...")
    for ticker in tqdm(candidates, desc="Validating new tickers",
                       unit="ticker", colour="cyan", dynamic_ncols=True):
        try:
            with _quiet():
                info = yf.Ticker(ticker).fast_info
            price = getattr(info, "last_price", None)
            if price and price > 0:
                valid.append(ticker)
                if len(valid) >= max_new:
                    break
        except Exception:
            pass
        time.sleep(0.05)

    logger.info(f"Found {len(valid)} new valid tickers.")
    return valid


def refresh_universe(max_new: int = 200) -> list[str]:
    """
    Discover new tickers, add to watchlist, fetch their price history,
    and append to the master parquet.
    Returns list of newly added tickers.
    """
    new_tickers = discover_new_tickers(max_new=max_new)
    if not new_tickers:
        logger.info("No new tickers discovered.")
        return []

    # Add to watchlist
    watchlist = load_watchlist()
    watchlist.update(new_tickers)
    save_watchlist(watchlist)

    # Fetch 2 years of history for new tickers
    from pipeline.updater import _fetch_yfinance, PARQUET_PATH
    start = (datetime.today() - timedelta(days=730)).strftime("%Y-%m-%d")
    end   = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")

    logger.info(f"Fetching history for {len(new_tickers)} new tickers...")
    new_df = _fetch_yfinance(new_tickers, start, end)

    if new_df.empty:
        return []

    new_df["date"] = pd.to_datetime(new_df["date"]).dt.normalize()
    new_df = new_df.set_index("date").sort_index()

    # Append to parquet
    if PARQUET_PATH.exists():
        existing = pd.read_parquet(PARQUET_PATH)
        combined = pd.concat([existing, new_df])
        combined = combined.reset_index()
        combined["date"] = pd.to_datetime(combined["date"]).dt.normalize()
        combined = combined.drop_duplicates(subset=["date", "ticker"], keep="last")
        combined = combined.set_index("date").sort_index()
    else:
        combined = new_df

    combined.to_parquet(PARQUET_PATH, index=True)
    logger.info(f"Added {len(new_tickers)} new tickers to parquet.")
    return new_tickers
