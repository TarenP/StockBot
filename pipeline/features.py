"""
Feature engineering: technical indicators + sentiment signals + normalization.

Optimised for speed:
- Single groupby pass per ticker (all indicators computed together)
- Vectorized numpy rolling slope (no Python-level rolling.apply)
- Cross-sectional normalisation via unstack/stack (avoids slow groupby.transform)
"""

import numpy as np
import pandas as pd
from tqdm import tqdm


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
]


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
