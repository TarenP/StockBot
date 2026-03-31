"""
Data loading and train/val/test splitting with no leakage.
Uses a walk-forward (expanding window) split strategy.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pipeline.features import build_features
from broker.sectors import get_cached_sector_map


DATA_DIR = Path("MasterDS")
SENTIMENT_LAG_SESSIONS = 1


def _prefer_classified_tickers(tickers: list[str], top_n: int) -> list[str]:
    """
    Prefer tickers that map to a known sector in the local cache/static map.
    This trims ETF/fund-heavy "Unknown" names out of the core stock universe
    whenever we already have enough classified common-stock candidates.
    """
    if not tickers:
        return []

    sector_map = get_cached_sector_map(tickers)
    classified = [ticker for ticker in tickers if sector_map.get(ticker, "Unknown") != "Unknown"]
    unknown = [ticker for ticker in tickers if sector_map.get(ticker, "Unknown") == "Unknown"]
    ordered = classified + unknown
    return ordered[:top_n]


def _select_universe_from_raw(
    df_prices: pd.DataFrame,
    top_n: int,
    lookback_years: int = 5,
    min_coverage: float = 0.7,
    min_price: float = 2.0,
    min_avg_volume: float = 500_000,
) -> list[str]:
    """
    Pick the top_n liquid tickers directly from raw price data (before feature
    engineering) so we only build features for the tickers we actually need.
    """
    dates  = df_prices.index.get_level_values("date").unique()
    cutoff = dates.max() - pd.DateOffset(years=lookback_years)
    recent = df_prices[df_prices.index.get_level_values("date") >= cutoff]

    n_dates = recent.index.get_level_values("date").nunique()
    counts  = recent.groupby(level="ticker").size()
    liquid  = counts[counts >= min_coverage * n_dates].index

    recent  = recent[recent.index.get_level_values("ticker").isin(liquid)]
    stats   = recent.groupby(level="ticker")[["close", "volume"]].mean()
    liquid  = stats[(stats["close"] >= min_price) & (stats["volume"] >= min_avg_volume)].index

    vol_rank = stats.loc[liquid, "volume"].sort_values(ascending=False)
    selected = _prefer_classified_tickers(vol_rank.index.tolist(), top_n)
    return sorted(selected)


def _lag_sentiment_to_next_trading_session(
    df_sent: pd.DataFrame,
    df_prices: pd.DataFrame,
    lag_sessions: int = SENTIMENT_LAG_SESSIONS,
) -> pd.DataFrame:
    """
    Move sentiment to the next available trading session to avoid using same-day
    headlines in the day's features. This is a conservative point-in-time fix:
    all sentiment is applied starting on the next market session.
    """
    if lag_sessions <= 0 or df_sent.empty:
        return df_sent

    sent_cols = list(df_sent.columns)
    shifted_parts: list[pd.DataFrame] = []

    for ticker, grp in df_sent.groupby(level="ticker", sort=False):
        try:
            trading_dates = df_prices.xs(ticker, level="ticker").index.to_numpy()
        except KeyError:
            continue
        if len(trading_dates) == 0:
            continue

        grp_reset = grp.reset_index()
        sentiment_dates = grp_reset["date"].to_numpy()
        target_pos = np.searchsorted(trading_dates, sentiment_dates, side="right")
        target_pos = target_pos + (lag_sessions - 1)
        valid = target_pos < len(trading_dates)
        if not np.any(valid):
            continue

        shifted = grp_reset.loc[valid, ["ticker", *sent_cols]].copy()
        shifted.insert(0, "date", trading_dates[target_pos[valid]])
        shifted_parts.append(shifted)

    if not shifted_parts:
        return df_sent.iloc[0:0]

    shifted_df = pd.concat(shifted_parts, ignore_index=True)
    shifted_df = shifted_df.groupby(["date", "ticker"], sort=True)[sent_cols].mean()
    return shifted_df


def load_master(
    price_path: str = str(DATA_DIR / "stooq_panel.parquet"),
    sentiment_path: str = "Sentiment/analyst_ratings_with_sentiment.csv",
    min_history_days: int = 252,
    min_price: float = 2.0,
    min_avg_volume: float = 500_000,
    universe: list[str] | None = None,   # pre-filter to these tickers before features
    top_n: int = 500,                    # used to auto-select universe if not provided
    include_raw_cols: bool = False,
    sentiment_lag_sessions: int = SENTIMENT_LAG_SESSIONS,
) -> pd.DataFrame:
    """
    Load, merge, filter, and feature-engineer the master dataset.
    Returns MultiIndex [date, ticker] DataFrame with feature columns.

    Pass `universe` to skip the auto-selection step (faster on repeated calls).
    """
    tqdm.write("Loading price data...")
    if not Path(price_path).exists():
        raise FileNotFoundError(
            f"\n\nPrice data not found at '{price_path}'.\n"
            "You need to train the model first:\n\n"
            "    python Agent.py --mode train --folds 10\n\n"
            "This downloads historical data and trains the RL model (~2-6 hours).\n"
            "After that, just run: python Broker.py"
        )
    df_prices = pd.read_parquet(price_path)

    with tqdm(total=5, desc="Preparing data", unit="step", colour="cyan",
              dynamic_ncols=True) as pbar:

        # ── 1. Normalise index ───────────────────────────────────────────────
        pbar.set_description("Normalising index")
        df_prices = df_prices.reset_index()
        df_prices["date"] = (
            pd.to_datetime(df_prices["date"], utc=True, errors="coerce")
              .dt.tz_convert(None).dt.normalize()
        )
        df_prices["ticker"] = df_prices["ticker"].str.upper()
        df_prices = df_prices.set_index(["date", "ticker"]).sort_index()
        pbar.update(1)

        # ── 2. Basic liquidity filter (history + price + volume) ─────────────
        pbar.set_description("Filtering illiquid stocks")
        counts = df_prices.groupby(level="ticker").size()
        valid  = counts[counts >= min_history_days].index
        df_prices = df_prices[df_prices.index.get_level_values("ticker").isin(valid)]

        stats = df_prices.groupby(level="ticker")[["close", "volume"]].mean()
        valid = stats[(stats["close"] >= min_price) & (stats["volume"] >= min_avg_volume)].index
        df_prices = df_prices[df_prices.index.get_level_values("ticker").isin(valid)]
        pbar.update(1)

        # ── 3. Universe pre-filter (BEFORE feature engineering) ──────────────
        pbar.set_description(f"Selecting universe")
        if universe is None:
            universe = _select_universe_from_raw(
                df_prices, top_n=top_n,
                min_price=min_price, min_avg_volume=min_avg_volume,
            )
        df_prices = df_prices[df_prices.index.get_level_values("ticker").isin(universe)]
        tqdm.write(f"  Universe: {len(universe)} tickers "
                   f"({df_prices.index.get_level_values('date').nunique()} trading days)")
        pbar.update(1)

        # ── 4. Sentiment merge ───────────────────────────────────────────────
        pbar.set_description("Merging sentiment")
        try:
            df_sent = pd.read_csv(sentiment_path)
            # Handle both old tz-aware strings and new plain date strings
            df_sent["date"] = pd.to_datetime(
                df_sent["date"], utc=True, errors="coerce"
            ).dt.tz_convert(None).dt.normalize()
            df_sent = df_sent.dropna(subset=["date"])
            df_sent["ticker"] = df_sent["stock"].str.upper()
            # Only keep sentiment for tickers in our universe
            df_sent = df_sent[df_sent["ticker"].isin(universe)]
            sent_cols = ["neg_score", "neutral_score", "pos_score"]
            df_sent = df_sent.groupby(["date", "ticker"])[sent_cols].mean()
            df_sent = _lag_sentiment_to_next_trading_session(
                df_sent,
                df_prices,
                lag_sessions=sentiment_lag_sessions,
            )
            df_prices = df_prices.join(df_sent, how="left")
            n_sent = df_prices["pos_score"].notna().sum()
            tqdm.write(f"  Sentiment rows matched: {n_sent:,}")
        except FileNotFoundError:
            tqdm.write("  Sentiment file not found — skipping.")
        pbar.update(1)

        # ── 5. Feature engineering (only on universe tickers) ────────────────
        pbar.set_description("Building features")
        df_features = build_features(df_prices)
        if include_raw_cols:
            raw_cols = df_prices[["close", "volume"]].copy()
            df_features = df_features.join(raw_cols, how="left")
        pbar.update(1)

    tqdm.write(f"  Tickers : {df_features.index.get_level_values('ticker').nunique()}")
    tqdm.write(f"  Features: {df_features.shape[1]}")
    tqdm.write(f"  Rows    : {len(df_features):,}")
    tqdm.write(f"  Dates   : {df_features.index.get_level_values('date').min().date()} "
               f"to {df_features.index.get_level_values('date').max().date()}")

    return df_features


def walk_forward_split(
    df: pd.DataFrame,
    train_years: int = 10,
    val_years: int = 1,
    test_years: int = 1,
) -> list[dict]:
    dates = sorted(df.index.get_level_values("date").unique())
    dates = pd.DatetimeIndex(dates)

    folds      = []
    val_delta  = pd.DateOffset(years=val_years)
    test_delta = pd.DateOffset(years=test_years)

    fold_start = dates[0] + pd.DateOffset(years=train_years)
    while fold_start + val_delta + test_delta <= dates[-1]:
        val_start  = fold_start
        test_start = val_start  + val_delta
        test_end   = test_start + test_delta

        train_idx = dates[dates < val_start]
        val_idx   = dates[(dates >= val_start)  & (dates < test_start)]
        test_idx  = dates[(dates >= test_start) & (dates < test_end)]

        if len(train_idx) and len(val_idx) and len(test_idx):
            folds.append({
                "train": df[df.index.get_level_values("date").isin(train_idx)],
                "val":   df[df.index.get_level_values("date").isin(val_idx)],
                "test":  df[df.index.get_level_values("date").isin(test_idx)],
            })

        fold_start = test_end

    return folds


def get_asset_universe(
    df: pd.DataFrame,
    top_n: int = 500,
    lookback_years: int = 5,
    min_coverage: float = 0.7,
    min_price: float = 2.0,
    min_avg_volume: float = 500_000,
) -> list[str]:
    """Select top_n liquid tickers from an already-loaded feature DataFrame."""
    dates  = sorted(df.index.get_level_values("date").unique())
    cutoff = pd.Timestamp(dates[-1]) - pd.DateOffset(years=lookback_years)
    recent = df[df.index.get_level_values("date") >= cutoff]

    n_dates = recent.index.get_level_values("date").nunique()
    counts  = recent.groupby(level="ticker").size()
    liquid  = counts[counts >= min_coverage * n_dates].index

    # Use row count as volume proxy (features are already normalised)
    ranked = counts[counts.index.isin(liquid)].sort_values(ascending=False)
    selected = _prefer_classified_tickers(ranked.index.tolist(), top_n)
    return sorted(selected)
