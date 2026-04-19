"""
Feature engineering: technical indicators + sentiment signals + normalization.

Optimised for speed:
- Single groupby pass per ticker (all indicators computed together)
- Vectorized numpy rolling slope (no Python-level rolling.apply)
- Cross-sectional normalisation via unstack/stack (avoids slow groupby.transform)
"""

import json
import logging
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ── Vectorized rolling linear slope ──────────────────────────────────────────

def _rolling_slope(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Compute rolling linear regression slope over `window` periods.
    Fully vectorized — no Python loops over rows.
    """
    n = len(arr)
    out = np.full(n, np.nan)
    if n < window:
        return out

    # Pre-compute x values and their stats (constant across all windows)
    x = np.arange(window, dtype=np.float64)
    x_mean = x.mean()
    x_var  = ((x - x_mean) ** 2).sum()

    if x_var == 0:
        return out

    # Sliding window using stride tricks
    shape   = (n - window + 1, window)
    strides = (arr.strides[0], arr.strides[0])
    windows = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

    y_mean = windows.mean(axis=1)
    # cov(x, y) = mean((x - x_mean)(y - y_mean))
    cov    = ((windows - y_mean[:, None]) * (x - x_mean)).sum(axis=1)
    slopes = cov / x_var

    out[window - 1:] = slopes
    return out


# ── All indicators in one pass per ticker ────────────────────────────────────

def _compute_all(g: pd.DataFrame) -> pd.DataFrame:
    """Compute every feature for a single ticker's time series."""
    c = g["close"].values.astype(np.float64)
    h = g["high"].values.astype(np.float64)
    l = g["low"].values.astype(np.float64)
    v = g["volume"].values.astype(np.float64)
    n = len(c)

    # ── Returns ──────────────────────────────────────────────────────────────
    def pct(arr, k):
        out = np.full(n, np.nan)
        out[k:] = (arr[k:] / arr[:-k]) - 1.0
        return out

    g["ret_1d"]  = pct(c, 1)
    g["ret_5d"]  = pct(c, 5)
    g["ret_20d"] = pct(c, 20)

    # ── RSI (14) ─────────────────────────────────────────────────────────────
    delta = np.diff(c, prepend=np.nan)
    gain  = pd.Series(np.where(delta > 0, delta, 0.0)).rolling(14).mean().values
    loss  = pd.Series(np.where(delta < 0, -delta, 0.0)).rolling(14).mean().values
    rs    = gain / (loss + 1e-9)
    g["rsi"] = 100.0 - (100.0 / (1.0 + rs))

    # ── MACD ─────────────────────────────────────────────────────────────────
    cs = pd.Series(c)
    ema12 = cs.ewm(span=12, adjust=False).mean().values
    ema26 = cs.ewm(span=26, adjust=False).mean().values
    macd  = ema12 - ema26
    macd_sig = pd.Series(macd).ewm(span=9, adjust=False).mean().values
    g["macd_hist"] = macd - macd_sig

    # ── Bollinger Bands (20) ─────────────────────────────────────────────────
    cs20_mean = cs.rolling(20).mean().values
    cs20_std  = cs.rolling(20).std().values
    bb_upper  = cs20_mean + 2 * cs20_std
    bb_lower  = cs20_mean - 2 * cs20_std
    g["bb_pct"] = (c - bb_lower) / (bb_upper - bb_lower + 1e-9)

    # ── ATR (14) ─────────────────────────────────────────────────────────────
    prev_c = np.roll(c, 1); prev_c[0] = c[0]
    tr     = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    g["atr"] = pd.Series(tr).rolling(14).mean().values / (c + 1e-9)

    # ── Volume features ───────────────────────────────────────────────────────
    vs       = pd.Series(v)
    vol_ma20 = vs.rolling(20).mean().values
    vol_std  = vs.rolling(20).std().values
    g["vol_ratio"]  = v / (vol_ma20 + 1e-9)
    g["vol_zscore"] = (v - vol_ma20) / (vol_std + 1e-9)

    # ── 52-week price position ────────────────────────────────────────────────
    high52 = cs.rolling(252).max().values
    low52  = cs.rolling(252).min().values
    g["price_pos_52w"] = (c - low52) / (high52 - low52 + 1e-9)

    # ── Sentiment features ────────────────────────────────────────────────────
    has_sentiment = "pos_score" in g.columns and g["pos_score"].notna().any()

    if has_sentiment:
        pos = g["pos_score"].ffill(limit=3).fillna(0.45).values
        neg = g["neg_score"].ffill(limit=3).fillna(0.45).values

        sent_net = pos - neg
        sn = pd.Series(sent_net)

        g["sent_net"]      = sent_net
        g["sent_ma3"]      = sn.rolling(3,  min_periods=1).mean().values
        g["sent_ma7"]      = sn.rolling(7,  min_periods=1).mean().values
        g["sent_ma14"]     = sn.rolling(14, min_periods=1).mean().values
        g["sent_surprise"] = sent_net - g["sent_ma14"].values
        g["sent_accel"]    = g["sent_ma3"].values - g["sent_ma7"].values
        # Vectorized slope — no Python rolling.apply
        g["sent_trend"]    = _rolling_slope(sent_net, window=7)
        g["sent_pos_raw"]  = pos
        neg_ma14           = sn.rolling(14, min_periods=1).mean().values
        g["sent_neg_spike"]= neg - neg_ma14
    else:
        for col in _SENTIMENT_FEATURE_COLS:
            g[col] = 0.0

    return g


_SENTIMENT_FEATURE_COLS = [
    "sent_net", "sent_ma3", "sent_ma7", "sent_ma14",
    "sent_surprise", "sent_accel", "sent_trend",
    "sent_pos_raw", "sent_neg_spike",
]

FEATURE_COLS = [
    "ret_1d", "ret_5d", "ret_20d",
    "rsi", "macd_hist", "bb_pct", "atr",
    "vol_ratio", "vol_zscore", "price_pos_52w",
    "sent_net", "sent_ma3", "sent_ma7", "sent_ma14",
    "sent_surprise", "sent_accel", "sent_trend",
    "sent_pos_raw", "sent_neg_spike",
    # Market context features (Task 6)
    "spy_ret_20d", "vix_level", "market_breadth",
    # Fundamental features (Task 7)
    "pe_ratio", "revenue_growth", "short_interest_pct",
    # Regime features (Task 8)
    "regime_0", "regime_1", "regime_2", "regime_3",
]


# ── Market context features (Task 6) ─────────────────────────────────────────

def build_market_context(
    df: pd.DataFrame,
    spy_ticker: str = "SPY",
    vix_ticker: str = "^VIX",
    breadth_window: int = 200,
    spy_ret_window: int = 20,
) -> pd.DataFrame:
    """
    Compute market-wide context features from the panel.

    Returns a date-indexed DataFrame with columns:
      spy_ret_20d  – SPY 20-day return (from panel if available, else 0)
      vix_level    – VIX / 100 (fetched from yfinance)
      market_breadth – fraction of tickers where close > 200d rolling mean
    """
    dates = df.index.get_level_values("date").unique().sort_values()

    # ── SPY 20-day return ─────────────────────────────────────────────────────
    tickers = df.index.get_level_values("ticker").unique()
    if spy_ticker in tickers:
        spy_close = df.xs(spy_ticker, level="ticker")["close"]
        spy_ret_20d = spy_close.pct_change(spy_ret_window).reindex(dates)
    else:
        spy_ret_20d = pd.Series(0.0, index=dates)

    # ── VIX level ─────────────────────────────────────────────────────────────
    try:
        start = dates.min().strftime("%Y-%m-%d")
        end   = (dates.max() + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        vix_raw = yf.download(vix_ticker, start=start, end=end,
                              progress=False, auto_adjust=True)
        if not vix_raw.empty:
            # Flatten MultiIndex columns if present
            if isinstance(vix_raw.columns, pd.MultiIndex):
                vix_raw.columns = ["_".join(str(c) for c in col).strip("_")
                                   for col in vix_raw.columns]
            close_col = next((c for c in vix_raw.columns
                              if c.lower().startswith("close")), None)
            if close_col:
                vix_close = vix_raw[close_col]
            else:
                vix_close = vix_raw.iloc[:, 0]
            vix_close = vix_close.squeeze()
            vix_close.index = pd.to_datetime(vix_close.index).normalize()
            vix_level = (vix_close / 100.0).reindex(dates)
        else:
            vix_level = pd.Series(np.nan, index=dates)
    except Exception as exc:
        logger.warning("VIX fetch failed: %s — filling with 0.0", exc)
        vix_level = pd.Series(np.nan, index=dates)

    # ── Market breadth ────────────────────────────────────────────────────────
    try:
        close_panel = df["close"].unstack(level="ticker")
        rolling_mean = close_panel.rolling(breadth_window, min_periods=1).mean()
        above = (close_panel > rolling_mean).sum(axis=1)
        total = close_panel.notna().sum(axis=1)
        market_breadth = (above / total.replace(0, np.nan)).reindex(dates).squeeze()
    except Exception as exc:
        logger.warning("Market breadth computation failed: %s — filling with 0.5", exc)
        market_breadth = pd.Series(np.nan, index=dates)

    # Ensure all inputs are 1-D Series before assembling
    spy_ret_20d    = pd.Series(spy_ret_20d.squeeze(),    index=dates)
    vix_level      = pd.Series(vix_level.squeeze(),      index=dates)
    market_breadth = pd.Series(market_breadth.squeeze(), index=dates)

    # ── Assemble and fill ─────────────────────────────────────────────────────
    ctx = pd.DataFrame({
        "spy_ret_20d":    spy_ret_20d,
        "vix_level":      vix_level,
        "market_breadth": market_breadth,
    }, index=dates)

    # Forward-fill VIX and breadth up to 5 days; then fill defaults
    ctx["vix_level"]      = ctx["vix_level"].ffill(limit=5).fillna(0.0)
    ctx["market_breadth"] = ctx["market_breadth"].ffill(limit=5).fillna(0.5)
    ctx["spy_ret_20d"]    = ctx["spy_ret_20d"].fillna(0.0)

    return ctx


# ── Fundamental features (Task 7) ────────────────────────────────────────────

FUNDAMENTALS_CACHE = "models/fundamentals_cache.json"
FUNDAMENTALS_STALENESS_HOURS = 24


def fetch_fundamentals(
    tickers: list,
    cache_path: str = FUNDAMENTALS_CACHE,
) -> pd.DataFrame:
    """
    Fetch trailingPE, revenueGrowth, shortPercentOfFloat from yfinance.
    Caches results in a JSON file with 24-hour staleness.

    Returns a ticker-indexed DataFrame with columns:
      pe_ratio, revenue_growth, short_interest_pct
    Missing values default to 0.0.
    """
    now = datetime.now(tz=timezone.utc).timestamp()
    cache: dict = {}

    # ── Load cache ────────────────────────────────────────────────────────────
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    cache_ts   = cache.get("_timestamp", 0)
    cache_data = cache.get("data", {})
    cache_stale = (now - cache_ts) > FUNDAMENTALS_STALENESS_HOURS * 3600

    results: dict = {}

    for ticker in tickers:
        if not cache_stale and ticker in cache_data:
            results[ticker] = cache_data[ticker]
            continue
        try:
            info = yf.Ticker(ticker).info
            results[ticker] = {
                "pe_ratio":          float(info.get("trailingPE")          or 0.0),
                "revenue_growth":    float(info.get("revenueGrowth")       or 0.0),
                "short_interest_pct": float(info.get("shortPercentOfFloat") or 0.0),
            }
        except Exception as exc:
            logger.debug("Fundamentals fetch failed for %s: %s", ticker, exc)
            results[ticker] = {
                "pe_ratio": 0.0, "revenue_growth": 0.0, "short_interest_pct": 0.0
            }

    # ── Persist cache ─────────────────────────────────────────────────────────
    try:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({"_timestamp": now, "data": results}, f)
    except Exception as exc:
        logger.warning("Could not write fundamentals cache: %s", exc)

    df_fund = pd.DataFrame.from_dict(results, orient="index",
                                     columns=["pe_ratio", "revenue_growth",
                                              "short_interest_pct"])
    df_fund = df_fund.fillna(0.0)
    return df_fund


# ── Regime detection features (Task 8) ───────────────────────────────────────

def compute_regimes(
    df: pd.DataFrame,
    vol_window: int = 20,
    dispersion_window: int = 20,
    vol_threshold: float = 0.015,
    dispersion_threshold: float = 0.015,
    spy_ticker: str = "SPY",
) -> pd.Series:
    """
    Compute market regime labels (0-3) from panel data.

    Regime matrix (vol × dispersion):
      0: low vol,  low dispersion  (calm bull)
      1: low vol,  high dispersion (trending/rotation)
      2: high vol, low dispersion  (choppy)
      3: high vol, high dispersion (risk-off/bear)

    Returns a date-indexed Series of integer labels.
    """
    dates = df.index.get_level_values("date").unique().sort_values()

    # ── SPY realised vol ──────────────────────────────────────────────────────
    tickers = df.index.get_level_values("ticker").unique()
    if spy_ticker in tickers:
        spy_close = df.xs(spy_ticker, level="ticker")["close"].reindex(dates)
        spy_ret   = spy_close.pct_change()
    elif "spy_ret_20d" in df.columns:
        # Fallback: use the already-computed spy_ret_20d column (first ticker)
        spy_ret = df["spy_ret_20d"].groupby(level="date").first().reindex(dates)
    else:
        spy_ret = pd.Series(0.0, index=dates)

    rolling_vol = spy_ret.rolling(vol_window, min_periods=1).std()

    # ── Cross-sectional return dispersion ─────────────────────────────────────
    try:
        ret_panel = df["ret_1d"].unstack(level="ticker").reindex(dates)
        dispersion = ret_panel.rolling(dispersion_window, min_periods=1).std().mean(axis=1)
    except Exception:
        dispersion = pd.Series(0.0, index=dates)

    # ── Assign regime labels ──────────────────────────────────────────────────
    high_vol  = (rolling_vol  > vol_threshold).fillna(False)
    high_disp = (dispersion   > dispersion_threshold).fillna(False)

    regime = pd.Series(0, index=dates, dtype=int)
    regime[~high_vol &  high_disp] = 1
    regime[ high_vol & ~high_disp] = 2
    regime[ high_vol &  high_disp] = 3

    return regime


# ── Public API ────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: MultiIndex [date, ticker], columns include OHLCV + optional sentiment.
    Returns df with FEATURE_COLS, cross-sectionally z-scored.
    """
    df = df.copy().sort_index()

    tickers = df.index.get_level_values("ticker").unique()

    # ── Single pass per ticker ────────────────────────────────────────────────
    parts = []
    for ticker in tqdm(tickers, desc="Building features", unit="ticker",
                       colour="magenta", dynamic_ncols=True):
        g = df.xs(ticker, level="ticker").copy()
        g = _compute_all(g)
        g["ticker"] = ticker
        parts.append(g)

    df = pd.concat(parts)
    df = df.reset_index().set_index(["date", "ticker"]).sort_index()

    # ── Market context features (broadcast per date) ──────────────────────────
    tqdm.write("  Computing market context features...")
    market_ctx = build_market_context(df)
    df = df.join(market_ctx, on="date", how="left")

    # ── Fundamental features (broadcast per ticker) ───────────────────────────
    tqdm.write("  Fetching fundamental features...")
    fund = fetch_fundamentals(list(tickers))
    # Merge by ticker level, then forward-fill within each ticker (up to 20 days)
    fund_cols = ["pe_ratio", "revenue_growth", "short_interest_pct"]
    for col in fund_cols:
        df[col] = df.index.get_level_values("ticker").map(fund[col].to_dict())
        df[col] = (
            df[col]
            .groupby(level="ticker")
            .transform(lambda s: s.ffill(limit=20))
            .fillna(0.0)
        )

    # ── Regime features (broadcast per date) ──────────────────────────────────
    tqdm.write("  Computing regime features...")
    regime_labels = compute_regimes(df)
    # One-hot encode into 4 binary columns
    for r in range(4):
        col_name = f"regime_{r}"
        regime_col = (regime_labels == r).astype(float)
        df[col_name] = df.index.get_level_values("date").map(regime_col.to_dict())
        df[col_name] = df[col_name].ffill().fillna(0.0)

    # ── Keep only feature columns ─────────────────────────────────────────────
    keep = [c for c in FEATURE_COLS if c in df.columns]
    df   = df[keep]

    # ── Cross-sectional z-score (memory-safe, per-date) ──────────────────────
    tqdm.write("  Normalising features...")
    # Compute per-date mean and std for each feature, then subtract/divide
    # Using transform on the MultiIndex directly — no unstack needed
    grp  = df.groupby(level="date")
    mean = grp.transform("mean")
    std  = grp.transform("std").replace(0, 1e-9).fillna(1e-9)
    df   = (df - mean) / std

    df = df.clip(-5, 5)
    df = df[df.isnull().mean(axis=1) < 0.5]
    df = df.fillna(0.0)

    return df
