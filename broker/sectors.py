"""
Sector intelligence module.

The broker decides its own sector allocations dynamically based on:
  - Sector momentum (which sectors are trending)
  - Sector sentiment (news tone across the sector)
  - Diversification penalty (concentration risk)
  - Market regime (risk-on vs risk-off)

Rather than hard-coding "max 25% per sector", the brain scores each sector
and allocates proportionally — it can go overweight a hot sector but pays
an increasing diversification penalty as concentration grows.
"""

import logging
import numpy as np
import pandas as pd
import requests
import time
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

# ── GICS sector map (ticker → sector) ────────────────────────────────────────
# Populated on first use via yfinance, then cached to disk

SECTOR_CACHE_PATH = Path("broker/state/sector_cache.json")

# Fallback static map for common tickers (avoids API calls for well-known stocks)
_STATIC_SECTOR_MAP = {
    # Technology
    "AAPL":"Technology","MSFT":"Technology","NVDA":"Technology","GOOGL":"Technology",
    "META":"Technology","AVGO":"Technology","ORCL":"Technology","CRM":"Technology",
    "AMD":"Technology","INTC":"Technology","QCOM":"Technology","TXN":"Technology",
    "AMAT":"Technology","MU":"Technology","KLAC":"Technology","LRCX":"Technology",
    # Healthcare
    "LLY":"Healthcare","UNH":"Healthcare","JNJ":"Healthcare","ABBV":"Healthcare",
    "MRK":"Healthcare","TMO":"Healthcare","ABT":"Healthcare","DHR":"Healthcare",
    "PFE":"Healthcare","AMGN":"Healthcare","ISRG":"Healthcare","GILD":"Healthcare",
    # Financials
    "BRK-B":"Financials","JPM":"Financials","V":"Financials","MA":"Financials",
    "BAC":"Financials","WFC":"Financials","GS":"Financials","MS":"Financials",
    "BLK":"Financials","SCHW":"Financials","AXP":"Financials","C":"Financials",
    # Consumer Discretionary
    "AMZN":"Consumer Discretionary","TSLA":"Consumer Discretionary",
    "HD":"Consumer Discretionary","MCD":"Consumer Discretionary",
    "NKE":"Consumer Discretionary","SBUX":"Consumer Discretionary",
    "TJX":"Consumer Discretionary","LOW":"Consumer Discretionary",
    # Consumer Staples
    "WMT":"Consumer Staples","PG":"Consumer Staples","KO":"Consumer Staples",
    "PEP":"Consumer Staples","COST":"Consumer Staples","PM":"Consumer Staples",
    "MO":"Consumer Staples","CL":"Consumer Staples",
    # Energy
    "XOM":"Energy","CVX":"Energy","COP":"Energy","EOG":"Energy",
    "SLB":"Energy","MPC":"Energy","PSX":"Energy","VLO":"Energy",
    # Industrials
    "GE":"Industrials","CAT":"Industrials","RTX":"Industrials","HON":"Industrials",
    "UPS":"Industrials","BA":"Industrials","LMT":"Industrials","DE":"Industrials",
    # Communication Services
    "NFLX":"Communication Services","DIS":"Communication Services",
    "CMCSA":"Communication Services","T":"Communication Services",
    "VZ":"Communication Services","TMUS":"Communication Services",
    # Utilities
    "NEE":"Utilities","DUK":"Utilities","SO":"Utilities","D":"Utilities",
    # Real Estate
    "PLD":"Real Estate","AMT":"Real Estate","EQIX":"Real Estate",
    # Materials
    "LIN":"Materials","APD":"Materials","SHW":"Materials","FCX":"Materials",
}

ALL_SECTORS = [
    "Technology", "Healthcare", "Financials", "Consumer Discretionary",
    "Consumer Staples", "Energy", "Industrials", "Communication Services",
    "Utilities", "Real Estate", "Materials", "Unknown",
]


def get_sector(ticker: str) -> str:
    """Return GICS sector for a ticker. Tries cache, static map, then yfinance."""
    ticker = ticker.upper()

    # Static map first (instant)
    if ticker in _STATIC_SECTOR_MAP:
        return _STATIC_SECTOR_MAP[ticker]

    # Disk cache
    cache = _load_sector_cache()
    if ticker in cache:
        return cache[ticker]

    # yfinance lookup
    try:
        import yfinance as yf
        info   = yf.Ticker(ticker).info
        sector = info.get("sector", "Unknown") or "Unknown"
        cache[ticker] = sector
        _save_sector_cache(cache)
        return sector
    except Exception:
        return "Unknown"


def get_sectors_bulk(tickers: list[str]) -> dict[str, str]:
    """Batch sector lookup — uses cache aggressively to minimise API calls."""
    result = {}
    need_lookup = []

    cache = _load_sector_cache()
    for t in tickers:
        t = t.upper()
        if t in _STATIC_SECTOR_MAP:
            result[t] = _STATIC_SECTOR_MAP[t]
        elif t in cache:
            result[t] = cache[t]
        else:
            need_lookup.append(t)

    if need_lookup:
        import yfinance as yf
        for ticker in need_lookup:
            try:
                info   = yf.Ticker(ticker).info
                sector = info.get("sector", "Unknown") or "Unknown"
                result[ticker] = sector
                cache[ticker]  = sector
                time.sleep(0.05)
            except Exception:
                result[ticker] = "Unknown"
        _save_sector_cache(cache)

    return result


def _load_sector_cache() -> dict:
    if SECTOR_CACHE_PATH.exists():
        import json
        try:
            return json.loads(SECTOR_CACHE_PATH.read_text())
        except Exception:
            pass
    return {}


def _save_sector_cache(cache: dict):
    import json
    SECTOR_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SECTOR_CACHE_PATH.write_text(json.dumps(cache, indent=2))


# ── Sector scoring ────────────────────────────────────────────────────────────

def score_sectors(
    df_features: pd.DataFrame,
    sector_map: dict[str, str],
) -> dict[str, float]:
    """
    Score each sector 0–1 based on:
      - Average momentum of stocks in that sector
      - Average sentiment of stocks in that sector
      - Breadth (% of stocks in sector with positive momentum)

    Returns dict of sector → score.
    """
    dates     = sorted(df_features.index.get_level_values("date").unique())
    last_date = dates[-1]

    try:
        snap = df_features.loc[last_date].copy()
    except KeyError:
        return {s: 0.5 for s in ALL_SECTORS}

    snap.index = snap.index.str.upper()
    snap["sector"] = snap.index.map(lambda t: sector_map.get(t, "Unknown"))

    sector_scores = {}
    for sector in ALL_SECTORS:
        group = snap[snap["sector"] == sector]
        if len(group) < 2:
            sector_scores[sector] = 0.5   # neutral if too few stocks
            continue

        # Momentum score
        mom5  = group.get("ret_5d",   pd.Series(dtype=float)).mean()
        mom20 = group.get("ret_20d",  pd.Series(dtype=float)).mean()
        mom_score = np.clip(0.5 + float(mom5 or 0) * 3 + float(mom20 or 0), 0, 1)

        # Sentiment score
        sent = group.get("sent_net", pd.Series(dtype=float)).mean()
        sent_score = np.clip(0.5 + float(sent or 0), 0, 1)

        # Breadth: % of stocks with positive 5d return
        breadth = (group.get("ret_5d", pd.Series(dtype=float)) > 0).mean()
        breadth_score = float(breadth or 0.5)

        # Volume surge (sector-wide attention)
        vol = group.get("vol_ratio", pd.Series(dtype=float)).mean()
        vol_score = np.clip(float(vol or 1.0) / 2.0, 0, 1)

        sector_scores[sector] = (
            mom_score   * 0.35 +
            sent_score  * 0.30 +
            breadth_score * 0.25 +
            vol_score   * 0.10
        )

    return sector_scores


def compute_target_allocations(
    sector_scores: dict[str, float],
    current_sector_weights: dict[str, float],
    max_single_sector: float = 0.40,
    min_sectors: int = 3,
) -> dict[str, float]:
    """
    Convert sector scores into target portfolio allocation percentages.

    The broker decides its own allocations — higher-scoring sectors get
    more weight, but a diversification penalty kicks in as concentration
    grows. The penalty is quadratic: going from 20%→30% in one sector
    costs more than going from 10%→20%.

    Args:
        sector_scores:           raw 0–1 scores per sector
        current_sector_weights:  current % of portfolio in each sector
        max_single_sector:       hard cap per sector (default 40%)
        min_sectors:             minimum number of sectors to hold

    Returns:
        dict of sector → target allocation fraction (sums to ≤ 1.0)
    """
    # Filter to sectors with meaningful scores
    active = {s: v for s, v in sector_scores.items()
              if v > 0.4 and s != "Unknown"}

    if not active:
        # Fallback: equal weight across top 5 sectors
        top5 = sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        return {s: 0.15 for s, _ in top5}

    # Apply diversification penalty to current overweights
    # Penalty = current_weight² × 2  (quadratic — punishes concentration)
    penalised = {}
    for sector, score in active.items():
        current_w = current_sector_weights.get(sector, 0.0)
        penalty   = current_w ** 2 * 2.0
        penalised[sector] = max(0.0, score - penalty)

    # Softmax to get proportional allocations
    sectors = list(penalised.keys())
    raw     = np.array([penalised[s] for s in sectors])

    if raw.sum() < 1e-9:
        raw = np.ones(len(sectors))

    # Temperature-scaled softmax (lower temp = more concentrated)
    temp    = 0.3
    exp_raw = np.exp(raw / temp)
    weights = exp_raw / exp_raw.sum()

    # Apply hard cap
    weights = np.minimum(weights, max_single_sector)
    weights = weights / weights.sum()   # renormalise after capping

    # Ensure minimum sector count
    if len(sectors) < min_sectors:
        # Add neutral-weight sectors to reach minimum
        missing = min_sectors - len(sectors)
        extras  = [s for s in ALL_SECTORS
                   if s not in sectors and s != "Unknown"][:missing]
        for s in extras:
            sectors.append(s)
            weights = np.append(weights, 0.05)
        weights = weights / weights.sum()

    target = dict(zip(sectors, weights.tolist()))

    logger.info("Sector allocation targets:")
    for s, w in sorted(target.items(), key=lambda x: x[1], reverse=True):
        current = current_sector_weights.get(s, 0.0)
        arrow   = "↑" if w > current + 0.02 else ("↓" if w < current - 0.02 else "→")
        logger.info(f"  {s:<28} {w:.1%}  {arrow}  (current: {current:.1%})")

    return target


def get_portfolio_sector_weights(
    positions: dict,
    sector_map: dict[str, str],
) -> dict[str, float]:
    """
    Calculate current sector weights in the portfolio.
    Returns dict of sector → fraction of total position value.
    """
    total_value = sum(
        p["shares"] * p["last_price"] for p in positions.values()
    )
    if total_value <= 0:
        return {}

    sector_values: dict[str, float] = {}
    for ticker, pos in positions.items():
        sector = sector_map.get(ticker.upper(), "Unknown")
        value  = pos["shares"] * pos["last_price"]
        sector_values[sector] = sector_values.get(sector, 0.0) + value

    return {s: v / total_value for s, v in sector_values.items()}
