"""
Live data updater.
Fetches latest OHLCV from yfinance for the trained universe only
and appends new rows to the master parquet.
"""

import time
import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Silence yfinance's own error output entirely
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)


@contextmanager
def _suppress_stderr():
    """Redirect stderr to devnull to swallow yfinance's direct print() errors."""
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

PARQUET_PATH = Path("MasterDS/stooq_panel.parquet")
CHUNK_SIZE   = 50   # tickers per yfinance batch request


def _load_trained_universe(save_dir: str = "models") -> list[str] | None:
    """
    Read the universe (top_n tickers) from the best available checkpoint.
    Falls back to None if no checkpoint found.
    """
    import glob, torch
    ckpts = sorted(glob.glob(f"{save_dir}/best_fold*.pt"))
    if not ckpts:
        return None
    try:
        meta = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
        # model_cfg stores n_assets which tells us the universe size,
        # but we need the actual ticker list — stored in asset_list if present
        return meta.get("asset_list", None)
    except Exception:
        return None


def _fetch_yfinance(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Download OHLCV for a list of tickers via yfinance.
    Silently skips delisted / missing tickers.
    Returns flat DataFrame with columns [date, open, high, low, close, volume, ticker].
    """
    rows = []
    chunks = [tickers[i:i + CHUNK_SIZE] for i in range(0, len(tickers), CHUNK_SIZE)]

    pbar = tqdm(chunks, desc="Fetching prices", unit="batch", colour="cyan",
                dynamic_ncols=True)
    for chunk in pbar:
        pbar.set_postfix(first=chunk[0])
        try:
            with _suppress_stderr():
                raw = yf.download(
                    chunk,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                )
            if raw.empty:
                continue

            if isinstance(raw.columns, pd.MultiIndex):
                for ticker in chunk:
                    try:
                        t_df = raw.xs(ticker, axis=1, level=1)
                        t_df = t_df[["Open", "High", "Low", "Close", "Volume"]].rename(columns=str.lower)
                        t_df = t_df.dropna(subset=["close"])
                        t_df = t_df[t_df["close"] > 0]
                        if t_df.empty:
                            continue
                        t_df["ticker"] = ticker.upper()
                        t_df.index = pd.to_datetime(t_df.index).normalize()
                        t_df.index.name = "date"
                        rows.append(t_df.reset_index())
                    except (KeyError, Exception):
                        pass   # delisted or missing — skip silently
            else:
                # Single ticker returned
                raw = raw[["Open", "High", "Low", "Close", "Volume"]].rename(columns=str.lower)
                raw = raw.dropna(subset=["close"])
                raw = raw[raw["close"] > 0]
                if not raw.empty:
                    raw["ticker"] = chunk[0].upper()
                    raw.index = pd.to_datetime(raw.index).normalize()
                    raw.index.name = "date"
                    rows.append(raw.reset_index())

        except Exception as e:
            # Log at debug level — delisted errors are noisy and expected
            logger.debug(f"Skipping chunk {chunk[:2]}: {e}")

        time.sleep(0.3)

    if not rows:
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True)
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]
    return df


def update_parquet(
    universe: list[str] | None = None,
    force_full_refresh: bool = False,
    save_dir: str = "models",
) -> int:
    """
    Append new trading days to the master parquet.

    Only updates tickers in `universe`. If universe is None, tries to load
    it from the best checkpoint. Falls back to all parquet tickers only as
    a last resort (and warns loudly).

    Args:
        universe:           explicit list of tickers to update
        force_full_refresh: re-download the last 30 days (fixes gaps)
        save_dir:           where to look for checkpoints

    Returns:
        Number of new rows appended.
    """
    PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ── Load existing parquet ────────────────────────────────────────────────
    if PARQUET_PATH.exists():
        existing  = pd.read_parquet(PARQUET_PATH)
        last_date = pd.to_datetime(existing.index).max().normalize()
    else:
        existing  = None
        last_date = None

    # ── Resolve universe — prefer checkpoint, then explicit arg, then warn ───
    if universe is None:
        universe = _load_trained_universe(save_dir)

    if universe is None and existing is not None:
        logger.warning(
            "No checkpoint found to determine universe. "
            "Falling back to all tickers in parquet — this may be slow. "
            "Run --mode train first to set a universe."
        )
        universe = existing["ticker"].unique().tolist()
    elif universe is None:
        raise ValueError("No universe and no existing parquet. Run --mode train first.")

    logger.info(f"Updating {len(universe)} tickers.")

    # ── Date range ───────────────────────────────────────────────────────────
    if last_date is not None:
        if force_full_refresh:
            fetch_start = (last_date - timedelta(days=30)).strftime("%Y-%m-%d")
        else:
            fetch_start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        fetch_start = "2010-01-01"

    fetch_end = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    today_str = datetime.today().strftime("%Y-%m-%d")

    if fetch_start >= today_str:
        logger.info("Price data is already up to date.")
        return 0

    logger.info(f"Fetching from {fetch_start} to {fetch_end}...")
    new_df = _fetch_yfinance(universe, fetch_start, fetch_end)

    if new_df.empty:
        logger.info("No new price data returned.")
        return 0

    new_df["date"] = pd.to_datetime(new_df["date"]).dt.normalize()
    new_df = new_df.set_index("date").sort_index()

    # ── Merge with existing ──────────────────────────────────────────────────
    if existing is not None:
        if force_full_refresh:
            cutoff   = pd.to_datetime(fetch_start)
            existing = existing[pd.to_datetime(existing.index) < cutoff]
        combined = pd.concat([existing, new_df])
    else:
        combined = new_df

    combined = combined.reset_index()
    combined["date"] = pd.to_datetime(combined["date"]).dt.normalize()
    combined = combined.drop_duplicates(subset=["date", "ticker"], keep="last")
    combined = combined.set_index("date").sort_index()

    combined.to_parquet(PARQUET_PATH, index=True)

    n_new = len(new_df)
    logger.info(f"Done. {n_new} new rows added. Total: {len(combined):,}")
    return n_new
