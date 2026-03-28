"""
Universal stock screener — runs independently of the portfolio agent.

Scores every ticker using a bidirectional GRU with attention pooling,
trained to identify stocks in the top decile of 20-day forward returns.

Architecture improvements over baseline:
  - Attention pooling over all GRU timesteps (vs just last hidden state)
  - Deeper model: 3-layer biGRU, hidden=128
  - Residual connection from input projection to attention output
  - Label smoothing to reduce overconfidence on noisy targets
  - Cosine annealing with linear warmup
  - 3x more samples: stride=5, max 600 tickers/date
  - 40-day lookback (captures medium-term momentum/mean-reversion)
  - 20-day forward return (higher signal-to-noise than 10-day)
  - Cache versioned by (n_rows, LOOKBACK, forward_days, top_pct, stride, max_t)
"""

import hashlib
import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

from pipeline.features import FEATURE_COLS

logger = logging.getLogger(__name__)

SCREENER_CKPT    = "models/screener.pt"
SCREENER_SAMPLES = "models/screener_samples.npz"
LOOKBACK         = 40    # days of history per ticker
FORWARD_DAYS     = 20    # forward return window for labels
TOP_PCT          = 0.10  # top 10% = positive label
SAMPLE_STRIDE    = 5     # sample every N dates
MAX_T_PER_DATE   = 600   # max tickers per sampled date


# ── Model ─────────────────────────────────────────────────────────────────────

class TickerScorer(nn.Module):
    """
    Bidirectional GRU with attention pooling.

    Input:  (batch, lookback, n_features)
    Output: (batch, 1) — raw logit

    Key improvements over last-timestep GRU:
      - Attention pooling: learns which timesteps matter most
      - Residual from projected input to attention context
      - Deeper (3 layers) with higher hidden dim (128)
    """

    def __init__(self, n_features: int, hidden: int = 128,
                 num_layers: int = 3, dropout: float = 0.25):
        super().__init__()
        self.input_norm = nn.LayerNorm(n_features)
        self.input_proj = nn.Linear(n_features, hidden)

        self.gru = nn.GRU(
            input_size    = hidden,
            hidden_size   = hidden,
            num_layers    = num_layers,
            batch_first   = True,
            dropout       = dropout if num_layers > 1 else 0.0,
            bidirectional = True,
        )

        # Attention: score each timestep, softmax, weighted sum
        self.attn = nn.Linear(hidden * 2, 1)

        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2 + hidden),   # attn context + residual
            nn.Linear(hidden * 2 + hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, n_features)
        x_norm = self.input_norm(x)
        proj   = self.input_proj(x_norm)          # (B, T, hidden) — residual

        out, _ = self.gru(proj)                   # (B, T, hidden*2)

        # Attention pooling
        scores  = self.attn(out).squeeze(-1)      # (B, T)
        weights = torch.softmax(scores, dim=1)    # (B, T)
        context = (out * weights.unsqueeze(-1)).sum(dim=1)  # (B, hidden*2)

        # Residual: mean of projected input
        residual = proj.mean(dim=1)               # (B, hidden)

        combined = torch.cat([context, residual], dim=-1)  # (B, hidden*3)
        return self.head(combined)                # (B, 1)


# ── Normalisation ─────────────────────────────────────────────────────────────

def _cross_sectional_zscore(arr: np.ndarray) -> np.ndarray:
    """
    Cross-sectional z-score in-place per date. Clips to [-5, 5].
    arr: (n_dates, n_tickers, n_features) — modified in place.
    """
    for di in range(arr.shape[0]):
        sl  = arr[di]
        mu  = np.nanmean(sl, axis=0)
        std = np.nanstd( sl, axis=0)
        std[std < 1e-9] = 1e-9
        arr[di] = np.clip((sl - mu) / std, -5.0, 5.0)
    np.nan_to_num(arr, copy=False, nan=0.0)
    return arr


def _cache_key(n_rows: int) -> str:
    """Version string that invalidates cache when any sample parameter changes."""
    params = f"{n_rows}_{LOOKBACK}_{FORWARD_DAYS}_{TOP_PCT}_{SAMPLE_STRIDE}_{MAX_T_PER_DATE}"
    return hashlib.md5(params.encode()).hexdigest()[:12]


# ── Training ──────────────────────────────────────────────────────────────────

def train_screener(
    df: pd.DataFrame,
    forward_days: int  = FORWARD_DAYS,
    top_pct: float     = TOP_PCT,
    val_frac: float    = 0.15,
    epochs: int        = 20,
    lr: float          = 3e-4,
    batch_size: int    = 512,
    device: torch.device = None,
    label_smoothing: float = 0.05,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tqdm.write(f"Training screener on {df.index.get_level_values('ticker').nunique()} tickers "
               f"| device={device}")

    feat_cols  = [c for c in FEATURE_COLS if c in df.columns]
    n_features = len(feat_cols)
    n_rows     = len(df)
    ck         = _cache_key(n_rows)

    # ── Load or build samples ─────────────────────────────────────────────────
    X_train = y_train = X_val = y_val = None

    if Path(SCREENER_SAMPLES).exists():
        try:
            cached = np.load(SCREENER_SAMPLES, allow_pickle=False)
            if str(cached.get("cache_key", b"")) == ck:
                X_train = cached["X_train"]
                y_train = cached["y_train"]
                X_val   = cached["X_val"]
                y_val   = cached["y_val"]
                tqdm.write(
                    f"  Loaded cached samples: {len(X_train):,} train  "
                    f"{len(X_val):,} val  (key={ck})"
                )
        except Exception as exc:
            tqdm.write(f"  Cache load failed ({exc}) — rebuilding")

    if X_train is None:
        tqdm.write("Building training samples (pre-pivoting data)...")
        dates     = sorted(df.index.get_level_values("date").unique())
        tickers   = sorted(df.index.get_level_values("ticker").unique())
        date_idx  = {d: i for i, d in enumerate(dates)}
        ticker_idx= {t: i for i, t in enumerate(tickers)}
        n_d, n_t  = len(dates), len(tickers)

        arr_raw = np.full((n_d, n_t, n_features), np.nan, dtype=np.float32)
        for (date, ticker), row in df[feat_cols].iterrows():
            di = date_idx.get(date)
            ti = ticker_idx.get(ticker)
            if di is not None and ti is not None:
                arr_raw[di, ti, :] = row.values.astype(np.float32)

        # Extract raw ret_1d BEFORE normalising
        ret_col = feat_cols.index("ret_1d") if "ret_1d" in feat_cols else None
        ret_arr = (arr_raw[:, :, ret_col].copy() if ret_col is not None
                   else np.zeros((n_d, n_t), dtype=np.float32))

        tqdm.write("  Normalising features (cross-sectional z-score)...")
        arr_norm = _cross_sectional_zscore(arr_raw)   # in-place, arr_raw → arr_norm
        del arr_raw

        sample_dates = list(range(LOOKBACK, n_d - forward_days, SAMPLE_STRIDE))
        split_idx    = int(len(sample_dates) * (1 - val_frac))
        rng          = np.random.default_rng(42)

        def _build(date_indices, desc):
            Xo, yo = [], []
            for t_idx in tqdm(date_indices, desc=f"  {desc}", unit="date",
                              colour="cyan", dynamic_ncols=True, leave=False):
                fwd_end      = min(t_idx + forward_days, n_d)
                fwd_rets_vec = np.nanprod(1 + ret_arr[t_idx:fwd_end, :], axis=0) - 1

                has_data   = arr_norm[t_idx, :, 0] != 0
                valid_mask = has_data & np.isfinite(fwd_rets_vec)
                if valid_mask.sum() < 10:
                    continue

                threshold  = np.nanpercentile(fwd_rets_vec[valid_mask], (1 - top_pct) * 100)
                labels_vec = (fwd_rets_vec >= threshold).astype(np.float32)

                hist_slice    = arr_norm[t_idx - LOOKBACK: t_idx, :, :]
                valid_indices = np.where(valid_mask)[0]
                if len(valid_indices) > MAX_T_PER_DATE:
                    valid_indices = rng.choice(valid_indices, MAX_T_PER_DATE, replace=False)

                for ti in valid_indices:
                    obs = hist_slice[:, ti, :]
                    if np.isnan(obs).mean() > 0.3:
                        continue
                    Xo.append(np.nan_to_num(obs, nan=0.0))
                    yo.append(float(labels_vec[ti]))
            return np.array(Xo, dtype=np.float32), np.array(yo, dtype=np.float32)

        X_train, y_train = _build(sample_dates[:split_idx], "Train")
        X_val,   y_val   = _build(sample_dates[split_idx:], "Val")

        os.makedirs("models", exist_ok=True)
        np.savez_compressed(
            SCREENER_SAMPLES,
            X_train=X_train, y_train=y_train,
            X_val=X_val,     y_val=y_val,
            cache_key=np.array(ck),
        )
        tqdm.write(f"  Samples cached → {SCREENER_SAMPLES}")

    if len(X_train) == 0:
        logger.error("No training samples built.")
        return

    tqdm.write(
        f"  Train: {len(X_train):,}  pos={y_train.mean():.1%} | "
        f"Val: {len(X_val):,}  pos={y_val.mean():.1%}"
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = TickerScorer(n_features=n_features).to(device)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tqdm.write(f"  Model: {n_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Linear warmup for first 10% of steps, then cosine decay
    total_steps  = epochs * (len(X_train) // batch_size + 1)
    warmup_steps = total_steps // 10

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Label smoothing: soft targets instead of hard 0/1
    pos_weight = torch.tensor([(1 - top_pct) / top_pct], device=device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1)
    )
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    best_auc   = 0.0
    best_state = None
    step       = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1:>2}/{epochs}",
                    unit="batch", colour="blue", dynamic_ncols=True)
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)

            # Label smoothing
            yb_smooth = yb * (1 - label_smoothing) + 0.5 * label_smoothing

            pred = model(xb)
            loss = criterion(pred, yb_smooth)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg_loss = total_loss / len(loader)

        # Validation AUC + top-decile return
        val_auc = 0.0
        if len(X_val) > 0:
            model.eval()
            all_probs = []
            with torch.no_grad():
                for i in range(0, len(X_val), 2048):
                    xv     = torch.tensor(X_val[i:i+2048], device=device)
                    logits = model(xv).squeeze(1).cpu().numpy()
                    all_probs.extend((1 / (1 + np.exp(-logits))).tolist())
            probs = np.array(all_probs)
            try:
                from sklearn.metrics import roc_auc_score
                val_auc = roc_auc_score(y_val, probs)
            except Exception:
                pass

            if val_auc > best_auc:
                best_auc   = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        tqdm.write(
            f"  Epoch {epoch+1:>2}/{epochs}  loss={avg_loss:.4f}  "
            f"val_auc={val_auc:.4f}"
            + (" ← best" if val_auc == best_auc and val_auc > 0 else "")
        )

    if best_state is not None:
        model.load_state_dict(best_state)
        tqdm.write(f"  Restored best model (val AUC={best_auc:.4f})")

    os.makedirs("models", exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "n_features":  n_features,
        "lookback":    LOOKBACK,
        "val_auc":     best_auc,
    }, SCREENER_CKPT)
    tqdm.write(f"Screener saved → {SCREENER_CKPT}  (best val AUC={best_auc:.4f})")
    return model


# ── Inference ─────────────────────────────────────────────────────────────────

def load_screener(device: torch.device) -> TickerScorer:
    ckpt  = torch.load(SCREENER_CKPT, map_location=device, weights_only=False)
    model = TickerScorer(n_features=ckpt["n_features"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def run_screener(
    df: pd.DataFrame,
    device: torch.device,
    top_n: int = 50,
    min_price: float = 0.0,
    max_price: float = None,
    min_volume: float = 10_000,
    batch_size: int = 512,
) -> pd.DataFrame:
    """
    Score every ticker in df and return the top_n ranked by buy signal.
    Applies the same cross-sectional z-score normalisation used in training.
    """
    if not os.path.exists(SCREENER_CKPT):
        raise FileNotFoundError(
            "Screener not trained yet. Run: python Agent.py --mode train_screener"
        )

    model     = load_screener(device)
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    dates     = sorted(df.index.get_level_values("date").unique())
    last_date = dates[-1]

    # Build (LOOKBACK, n_tickers, n_features) with same normalisation as training
    recent_dates = dates[-LOOKBACK:]
    df_recent    = df[df.index.get_level_values("date").isin(recent_dates)]
    tickers      = sorted(df_recent.index.get_level_values("ticker").unique())
    ticker_idx   = {t: i for i, t in enumerate(tickers)}
    n_t          = len(tickers)
    n_features   = len(feat_cols)

    arr = np.full((LOOKBACK, n_t, n_features), np.nan, dtype=np.float32)
    for li, date in enumerate(recent_dates):
        try:
            snap = df_recent.loc[date][feat_cols]
            for ticker, row in snap.iterrows():
                ti = ticker_idx.get(ticker)
                if ti is not None:
                    arr[li, ti, :] = row.values.astype(np.float32)
        except KeyError:
            pass

    # Cross-sectional z-score per date
    for li in range(LOOKBACK):
        sl  = arr[li]
        mu  = np.nanmean(sl, axis=0)
        std = np.nanstd( sl, axis=0)
        std[std < 1e-9] = 1e-9
        arr[li] = np.clip((sl - mu) / std, -5.0, 5.0)
    arr = np.nan_to_num(arr, nan=0.0)

    obs_list, valid_tickers = [], []
    for ti, ticker in enumerate(tickers):
        obs = arr[:, ti, :]
        obs_list.append(obs)
        valid_tickers.append(ticker)

    if not obs_list:
        logger.error("No valid observations for screener.")
        return pd.DataFrame()

    X = torch.tensor(np.array(obs_list, dtype=np.float32), device=device)

    scores = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            logits = model(X[i:i + batch_size]).squeeze(1)
            scores.extend(torch.sigmoid(logits).cpu().numpy().tolist())

    rows = []
    for ticker, score in zip(valid_tickers, scores):
        row = {"ticker": ticker, "score": score}
        try:
            feats     = df_recent.loc[(last_date, ticker)]
            feat_dict = dict(zip(feat_cols, feats.values))
            row["momentum_5d"]   = feat_dict.get("ret_5d",        np.nan)
            row["momentum_20d"]  = feat_dict.get("ret_20d",       np.nan)
            row["rsi"]           = feat_dict.get("rsi",           np.nan)
            row["sentiment"]     = feat_dict.get("sent_net",      np.nan)
            row["sent_surprise"] = feat_dict.get("sent_surprise", np.nan)
            row["vol_ratio"]     = feat_dict.get("vol_ratio",     np.nan)
        except (KeyError, Exception):
            pass
        rows.append(row)

    results = (
        pd.DataFrame(rows)
          .sort_values("score", ascending=False)
          .reset_index(drop=True)
    )
    results.index += 1
    return results.head(top_n)


def print_screener_results(results: pd.DataFrame, label: str = "All stocks"):
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
