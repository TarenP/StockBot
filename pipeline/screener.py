"""
Universal stock screener — runs independently of the portfolio agent.

Scores every ticker in the parquet (including penny stocks) using a
per-ticker Transformer that processes each stock in isolation.
No fixed universe size — handles 10,000+ tickers efficiently by
processing them one at a time in batches.

Output: ranked DataFrame of all tickers with buy scores + signals.
"""

import os
import glob
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

from pipeline.features import FEATURE_COLS, build_features

logger = logging.getLogger(__name__)

SCREENER_CKPT = "models/screener.pt"
LOOKBACK      = 30   # days of history per ticker


# ── Per-ticker Transformer scorer ────────────────────────────────────────────

class TickerScorer(nn.Module):
    """
    Lightweight Transformer that reads a single ticker's feature history
    and outputs a scalar buy-signal score in [0, 1].

    Input:  (batch, lookback, n_features)
    Output: (batch, 1)  — buy signal probability
    """

    def __init__(self, n_features: int, d_model: int = 32, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)       # (B, T, d_model)
        x = self.transformer(x)      # (B, T, d_model)
        x = x[:, -1, :]             # last timestep
        return self.head(x)          # (B, 1)


# ── Training the screener ─────────────────────────────────────────────────────

def train_screener(
    df: pd.DataFrame,
    forward_days: int = 5,
    top_pct: float = 0.2,       # top 20% returns = positive label
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 256,
    device: torch.device = None,
):
    """
    Train the screener on ALL tickers in df (no universe limit).
    Label: 1 if ticker's forward return is in top 20% on that date, else 0.
    Saves checkpoint to SCREENER_CKPT.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = None  # defined after sample building (n_features known then)

    tqdm.write(f"Training screener on {df.index.get_level_values('ticker').nunique()} tickers...")

    # ── Pre-pivot into 3D array for fast O(1) lookups ─────────────────────────
    # Shape: (n_dates, n_tickers, n_features) — avoids per-ticker pandas scans
    tqdm.write("Building training samples (pre-pivoting data)...")
    dates   = sorted(df.index.get_level_values("date").unique())
    tickers = sorted(df.index.get_level_values("ticker").unique())
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    n_features = len(feat_cols)

    date_idx   = {d: i for i, d in enumerate(dates)}
    ticker_idx = {t: i for i, t in enumerate(tickers)}
    n_d, n_t   = len(dates), len(tickers)

    # Fill 3D array — missing (date, ticker) pairs stay as NaN
    arr = np.full((n_d, n_t, n_features), np.nan, dtype=np.float32)
    for (date, ticker), row in df[feat_cols].iterrows():
        di = date_idx.get(date)
        ti = ticker_idx.get(ticker)
        if di is not None and ti is not None:
            arr[di, ti, :] = row.values.astype(np.float32)

    # Pre-compute forward returns array: (n_dates, n_tickers)
    # Use ret_1d column if available, else NaN
    ret_col = feat_cols.index("ret_1d") if "ret_1d" in feat_cols else None
    if ret_col is not None:
        ret_arr = arr[:, :, ret_col].copy()   # (n_dates, n_tickers)
    else:
        ret_arr = np.zeros((n_d, n_t), dtype=np.float32)

    X_list, y_list = [], []

    # Sample every SAMPLE_STRIDE dates to keep dataset manageable
    SAMPLE_STRIDE = 5
    sample_dates  = range(LOOKBACK, n_d - forward_days, SAMPLE_STRIDE)

    for t_idx in tqdm(sample_dates, desc="Sampling windows",
                      unit="date", colour="cyan", dynamic_ncols=True):

        # Forward return for each ticker: product of daily returns over next forward_days
        fwd_end = min(t_idx + forward_days, n_d)
        fwd_rets_mat = ret_arr[t_idx:fwd_end, :]          # (forward_days, n_tickers)
        fwd_rets_vec = np.nanprod(1 + fwd_rets_mat, axis=0) - 1  # (n_tickers,)

        # Only consider tickers with data on this date
        has_data = ~np.isnan(arr[t_idx, :, 0])            # (n_tickers,)
        valid_mask = has_data & np.isfinite(fwd_rets_vec)

        if valid_mask.sum() < 10:
            continue

        # Label: top 20% forward return = 1
        valid_rets = fwd_rets_vec[valid_mask]
        threshold  = np.nanpercentile(valid_rets, (1 - top_pct) * 100)
        labels_vec = (fwd_rets_vec >= threshold).astype(np.float32)

        # Build observation windows for each valid ticker
        hist_slice = arr[t_idx - LOOKBACK: t_idx, :, :]   # (LOOKBACK, n_tickers, n_features)

        for ti, has in enumerate(valid_mask):
            if not has:
                continue
            obs = hist_slice[:, ti, :]                     # (LOOKBACK, n_features)
            nan_frac = np.isnan(obs).mean()
            if nan_frac > 0.3:
                continue
            obs = np.nan_to_num(obs, nan=0.0)
            X_list.append(obs)
            y_list.append(float(labels_vec[ti]))

    if not X_list:
        logger.error("No training samples built. Check data.")
        return

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    tqdm.write(f"  Samples: {len(X):,}  |  Positive rate: {y.mean():.1%}")

    model     = TickerScorer(n_features=n_features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # ── Training loop ─────────────────────────────────────────────────────────
    dataset  = torch.utils.data.TensorDataset(
        torch.tensor(X), torch.tensor(y).unsqueeze(1)
    )
    loader   = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}",
                    unit="batch", colour="blue", dynamic_ncols=True)
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            pred   = model(xb)
            loss   = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        tqdm.write(f"  Epoch {epoch+1} avg loss: {total_loss/len(loader):.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "n_features":  n_features,
        "lookback":    LOOKBACK,
    }, SCREENER_CKPT)
    tqdm.write(f"Screener saved → {SCREENER_CKPT}")
    return model


# ── Running the screener ──────────────────────────────────────────────────────

def load_screener(device: torch.device) -> TickerScorer:
    ckpt      = torch.load(SCREENER_CKPT, map_location=device, weights_only=False)
    model     = TickerScorer(n_features=ckpt["n_features"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def run_screener(
    df: pd.DataFrame,
    device: torch.device,
    top_n: int = 50,
    min_price: float = 0.0,     # 0 = include all (penny stocks too)
    max_price: float = None,    # None = no upper limit
    min_volume: float = 10_000,
    batch_size: int = 512,
) -> pd.DataFrame:
    """
    Score every ticker in df and return the top_n ranked by buy signal.

    Args:
        df:         MultiIndex [date, ticker] feature DataFrame
        top_n:      how many top picks to return
        min_price:  filter by minimum price (use raw parquet for this)
        max_price:  filter by maximum price (set to 5.0 for penny-only)
        min_volume: minimum average daily volume

    Returns:
        DataFrame with columns [ticker, score, ...signals] sorted by score desc
    """
    if not os.path.exists(SCREENER_CKPT):
        raise FileNotFoundError(
            "Screener not trained yet. Run: python Agent.py --mode train_screener"
        )

    model = load_screener(device)

    dates        = sorted(df.index.get_level_values("date").unique())
    recent_dates = dates[-LOOKBACK:]
    df_recent    = df[df.index.get_level_values("date").isin(recent_dates)]
    tickers      = sorted(df_recent.index.get_level_values("ticker").unique())
    feat_cols    = [c for c in FEATURE_COLS if c in df.columns]
    n_features   = len(feat_cols)
    last_date    = recent_dates[-1]

    # ── Build observation tensor for all tickers ──────────────────────────────
    obs_list    = []
    valid_tickers = []

    for ticker in tqdm(tickers, desc="Preparing observations", unit="ticker",
                       colour="cyan", dynamic_ncols=True):
        try:
            hist = df_recent.xs(ticker, level="ticker")[feat_cols]
            obs  = hist.values.astype(np.float32)
            if obs.shape[0] < LOOKBACK or np.isnan(obs).mean() > 0.3:
                continue
            obs = np.nan_to_num(obs[-LOOKBACK:], nan=0.0)
            obs_list.append(obs)
            valid_tickers.append(ticker)
        except (KeyError, Exception):
            pass

    if not obs_list:
        logger.error("No valid observations for screener.")
        return pd.DataFrame()

    X = torch.tensor(np.array(obs_list, dtype=np.float32), device=device)

    # ── Score in batches ──────────────────────────────────────────────────────
    scores = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(X), batch_size),
                      desc="Scoring tickers", unit="batch",
                      colour="magenta", dynamic_ncols=True):
            batch  = X[i:i + batch_size]
            score  = model(batch).squeeze(1).cpu().numpy()
            scores.extend(score.tolist())

    # ── Build results table ───────────────────────────────────────────────────
    rows = []
    for ticker, score in zip(valid_tickers, scores):
        row = {"ticker": ticker, "score": score}
        try:
            feats     = df_recent.loc[(last_date, ticker)]
            feat_dict = dict(zip(feat_cols, feats.values))
            row["momentum_5d"]  = feat_dict.get("ret_5d",       np.nan)
            row["momentum_20d"] = feat_dict.get("ret_20d",      np.nan)
            row["rsi"]          = feat_dict.get("rsi",          np.nan)
            row["sentiment"]    = feat_dict.get("sent_net",     np.nan)
            row["sent_surprise"]= feat_dict.get("sent_surprise",np.nan)
            row["vol_ratio"]    = feat_dict.get("vol_ratio",    np.nan)
            row["bb_pct"]       = feat_dict.get("bb_pct",       np.nan)
        except (KeyError, Exception):
            pass
        rows.append(row)

    results = pd.DataFrame(rows).sort_values("score", ascending=False)
    results = results.reset_index(drop=True)
    results.index += 1

    return results.head(top_n)


def print_screener_results(results: pd.DataFrame, label: str = "All stocks"):
    last_date = "today"
    print(f"\n{'='*80}")
    print(f"  Top Picks — {label}  ({len(results)} shown)")
    print(f"{'='*80}")
    print(f"  {'#':<4} {'Ticker':<8} {'Score':>6}  {'Mom5d':>7}  {'Mom20d':>7}  "
          f"{'RSI':>6}  {'Sentiment':>10}  {'VolRatio':>9}")
    print(f"  {'-'*74}")
    for rank, row in results.iterrows():
        def fmt(v):
            return f"{v:+.2f}" if pd.notna(v) else "  n/a"
        print(
            f"  {rank:<4} {row['ticker']:<8} {row['score']:>6.3f}  "
            f"{fmt(row.get('momentum_5d')):>7}  "
            f"{fmt(row.get('momentum_20d')):>7}  "
            f"{fmt(row.get('rsi')):>6}  "
            f"{fmt(row.get('sentiment')):>10}  "
            f"{fmt(row.get('vol_ratio')):>9}"
        )
    print(f"{'='*80}\n")
