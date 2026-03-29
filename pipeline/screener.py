"""
Universal stock screener - runs independently of the portfolio agent.

The screener's real job in this bot is to produce a high-quality shortlist
for the broker and RL ranker. Training is therefore optimized for shortlist
quality on unseen dates, not just generic classifier AUC.
"""

import hashlib
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from pipeline.features import FEATURE_COLS

logger = logging.getLogger(__name__)

SCREENER_CKPT = "models/screener.pt"
SCREENER_SAMPLES = "models/screener_samples.npz"
SCREENER_CACHE_VERSION = 2
LOOKBACK = 40
FORWARD_DAYS = 20
TOP_PCT = 0.10
SAMPLE_STRIDE = 5
MAX_T_PER_DATE = 600
EVAL_TOP_N = 50
MIN_HISTORY_COVERAGE = 0.70


class TickerScorer(nn.Module):
    """
    Bidirectional GRU with attention pooling.

    Input:  (batch, lookback, n_features)
    Output: (batch, 1) - raw logit
    """

    def __init__(
        self,
        n_features: int,
        hidden: int = 128,
        num_layers: int = 3,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(n_features)
        self.input_proj = nn.Linear(n_features, hidden)

        self.gru = nn.GRU(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        self.attn = nn.Linear(hidden * 2, 1)

        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2 + hidden),
            nn.Linear(hidden * 2 + hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.input_norm(x)
        proj = self.input_proj(x_norm)
        out, _ = self.gru(proj)

        scores = self.attn(out).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = (out * weights.unsqueeze(-1)).sum(dim=1)
        residual = proj.mean(dim=1)

        combined = torch.cat([context, residual], dim=-1)
        return self.head(combined)


def _cache_key(
    df: pd.DataFrame,
    feat_cols: list[str],
    forward_days: int,
    top_pct: float,
    val_frac: float,
    test_frac: float,
    eval_top_n: int,
) -> str:
    """Invalidate caches when the dataset span or screener params change."""
    dates = df.index.get_level_values("date")
    tickers = df.index.get_level_values("ticker")
    params = "_".join(
        [
            f"cache_v={SCREENER_CACHE_VERSION}",
            str(len(df)),
            str(dates.nunique()),
            str(tickers.nunique()),
            str(dates.min()),
            str(dates.max()),
            ",".join(feat_cols),
            str(forward_days),
            str(top_pct),
            str(val_frac),
            str(test_frac),
            str(LOOKBACK),
            str(SAMPLE_STRIDE),
            str(MAX_T_PER_DATE),
            str(eval_top_n),
        ]
    )
    return hashlib.md5(params.encode()).hexdigest()[:12]


def _prepare_screener_arrays(
    df: pd.DataFrame,
    feat_cols: list[str],
) -> tuple[list, list, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build aligned feature, close, volume, and row-presence arrays."""
    dates = sorted(df.index.get_level_values("date").unique())
    tickers = sorted(df.index.get_level_values("ticker").unique())
    date_idx = {d: i for i, d in enumerate(dates)}
    ticker_idx = {t: i for i, t in enumerate(tickers)}
    n_d, n_t = len(dates), len(tickers)

    feat_arr = np.full((n_d, n_t, len(feat_cols)), np.nan, dtype=np.float32)
    close_arr = np.full((n_d, n_t), np.nan, dtype=np.float32)
    volume_arr = np.full((n_d, n_t), np.nan, dtype=np.float32)
    present_mask = np.zeros((n_d, n_t), dtype=bool)

    cols = feat_cols + [c for c in ("close", "volume") if c in df.columns]
    for (date, ticker), row in tqdm(
        df[cols].iterrows(),
        total=len(df),
        desc="  Aligning arrays",
        unit="row",
        colour="cyan",
        dynamic_ncols=True,
        leave=False,
    ):
        di = date_idx.get(date)
        ti = ticker_idx.get(ticker)
        if di is None or ti is None:
            continue

        feat_arr[di, ti, :] = row[feat_cols].values.astype(np.float32)
        if "close" in row.index:
            close_arr[di, ti] = float(row["close"])
        if "volume" in row.index:
            volume_arr[di, ti] = float(row["volume"])
        present_mask[di, ti] = True

    feat_arr = np.nan_to_num(np.clip(feat_arr, -5.0, 5.0), nan=0.0)
    return dates, tickers, feat_arr, close_arr, volume_arr, present_mask


def _sample_date_splits(
    n_dates: int,
    forward_days: int,
    val_frac: float,
    test_frac: float,
) -> tuple[list[int], list[int], list[int]]:
    sample_dates = list(range(LOOKBACK, n_dates - forward_days + 1))
    if len(sample_dates) < 10:
        raise ValueError("Not enough screener history to build train/val/test splits.")

    train_end = max(1, int(len(sample_dates) * (1 - val_frac - test_frac)))
    val_end = max(train_end + 1, int(len(sample_dates) * (1 - test_frac)))
    val_end = min(val_end, len(sample_dates) - 1)

    train_dates = sample_dates[:train_end:SAMPLE_STRIDE]
    val_dates = sample_dates[train_end:val_end]
    test_dates = sample_dates[val_end:]

    if not train_dates or not val_dates or not test_dates:
        raise ValueError("Screener split is empty; adjust val/test fractions.")
    return train_dates, val_dates, test_dates


def _build_samples(
    feat_arr: np.ndarray,
    close_arr: np.ndarray,
    present_mask: np.ndarray,
    date_indices: list[int],
    forward_days: int,
    top_pct: float,
    rng: np.random.Generator,
    max_t_per_date: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Xo, yo, ro, go = [], [], [], []

    for t_idx in tqdm(
        date_indices,
        desc="  Samples",
        unit="date",
        colour="cyan",
        dynamic_ncols=True,
        leave=False,
    ):
        anchor_idx = t_idx - 1
        future_idx = t_idx + forward_days - 1

        history_coverage = present_mask[t_idx - LOOKBACK:t_idx].mean(axis=0)
        base_close = close_arr[anchor_idx]
        future_close = close_arr[future_idx]
        valid_mask = (
            (history_coverage >= MIN_HISTORY_COVERAGE)
            & np.isfinite(base_close)
            & np.isfinite(future_close)
            & (base_close > 0)
        )
        if valid_mask.sum() < 10:
            continue

        fwd_rets_vec = (future_close / base_close) - 1.0
        threshold = np.nanpercentile(fwd_rets_vec[valid_mask], (1 - top_pct) * 100)
        labels_vec = (fwd_rets_vec >= threshold).astype(np.float32)

        valid_indices = np.where(valid_mask)[0]
        if max_t_per_date is not None and len(valid_indices) > max_t_per_date:
            valid_indices = rng.choice(valid_indices, max_t_per_date, replace=False)

        hist_slice = feat_arr[t_idx - LOOKBACK:t_idx, :, :]
        for ti in valid_indices:
            Xo.append(hist_slice[:, ti, :])
            yo.append(float(labels_vec[ti]))
            ro.append(float(fwd_rets_vec[ti]))
            go.append(int(anchor_idx))

    return (
        np.array(Xo, dtype=np.float32),
        np.array(yo, dtype=np.float32),
        np.array(ro, dtype=np.float32),
        np.array(go, dtype=np.int32),
    )


def _evaluate_ranked_groups(
    probs: np.ndarray,
    labels: np.ndarray,
    forward_returns: np.ndarray,
    groups: np.ndarray,
    shortlist_size: int,
) -> dict[str, float]:
    """Evaluate as a ranked shortlist, matching the screener's role in the bot."""
    if len(probs) == 0:
        return {
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "mean_return_at_k": 0.0,
            "baseline_return": 0.0,
            "lift_at_k": 0.0,
        }

    precisions = []
    recalls = []
    top_returns = []
    baseline_returns = []
    lifts = []

    # _build_samples appends examples date-by-date, so groups arrive in
    # contiguous order already. Avoiding a full argsort here saves a large
    # temporary int64 index allocation during validation/test on big datasets.
    groups_view = groups
    probs_view = probs
    labels_view = labels
    returns_view = forward_returns

    boundaries = np.flatnonzero(np.diff(groups_view)) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [len(groups_view)]))

    for start, stop in zip(starts, stops):
        if stop - start < 2:
            continue

        grp_probs = probs_view[start:stop]
        grp_labels = labels_view[start:stop]
        grp_returns = returns_view[start:stop]
        k = min(shortlist_size, len(grp_probs))

        order = np.argsort(grp_probs)[-k:]
        top_labels = grp_labels[order]
        top_grp_returns = grp_returns[order]

        positive_count = max(float(grp_labels.sum()), 1.0)
        baseline_hit = float(grp_labels.mean())
        precision = float(top_labels.mean())
        recall = float(top_labels.sum() / positive_count)

        precisions.append(precision)
        recalls.append(recall)
        top_returns.append(float(np.mean(top_grp_returns)))
        baseline_returns.append(float(np.mean(grp_returns)))
        lifts.append(precision / max(baseline_hit, 1e-9))

    if not precisions:
        return {
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "mean_return_at_k": 0.0,
            "baseline_return": 0.0,
            "lift_at_k": 0.0,
        }

    return {
        "precision_at_k": float(np.mean(precisions)),
        "recall_at_k": float(np.mean(recalls)),
        "mean_return_at_k": float(np.mean(top_returns)),
        "baseline_return": float(np.mean(baseline_returns)),
        "lift_at_k": float(np.mean(lifts)),
    }


def _heuristic_scores_from_windows(X: np.ndarray, feat_cols: list[str]) -> np.ndarray:
    """
    Lightweight cross-sectional alpha built from the most recent normalized
    screener features. This complements the neural model on obvious momentum /
    sentiment setups and can be blended using validation.
    """
    if len(X) == 0:
        return np.array([], dtype=np.float32)

    idx = {col: i for i, col in enumerate(feat_cols)}
    latest = X[:, -1, :]
    recent5 = X[:, -5:, :].mean(axis=1)
    recent20 = X.mean(axis=1)

    def col(arr: np.ndarray, name: str) -> np.ndarray:
        pos = idx.get(name)
        if pos is None:
            return np.zeros(len(arr), dtype=np.float32)
        return arr[:, pos]

    alpha = np.zeros(len(X), dtype=np.float32)
    alpha += 0.24 * col(latest, "ret_20d")
    alpha += 0.16 * col(latest, "ret_5d")
    alpha += 0.12 * col(latest, "macd_hist")
    alpha += 0.10 * col(latest, "price_pos_52w")
    alpha += 0.08 * col(latest, "vol_ratio")
    alpha += 0.10 * col(latest, "sent_net")
    alpha += 0.08 * col(latest, "sent_surprise")
    alpha += 0.05 * col(recent5, "sent_accel")
    alpha += 0.04 * col(recent5, "sent_trend")
    alpha += 0.03 * col(recent20, "vol_ratio")

    alpha = np.clip(alpha, -8.0, 8.0)
    return (1.0 / (1.0 + np.exp(-alpha))).astype(np.float32)


def _blend_scores(
    model_probs: np.ndarray,
    heuristic_probs: np.ndarray,
    blend_weight: float,
) -> np.ndarray:
    if len(model_probs) == 0:
        return model_probs
    blend_weight = float(np.clip(blend_weight, 0.0, 1.0))
    return (
        blend_weight * model_probs
        + (1.0 - blend_weight) * heuristic_probs
    ).astype(np.float32)


def _score_epoch(metrics: dict[str, float]) -> float:
    """
    Score a checkpoint based on shortlist quality.

    Precision matters most because the broker only researches the top part of
    the screener output. Recall still matters so we do not drop too many true
    winners before RL/rules can rerank them.
    """
    return (
        0.55 * metrics["precision_at_k"]
        + 0.25 * metrics["recall_at_k"]
        + 0.20 * max(metrics["mean_return_at_k"] - metrics["baseline_return"], 0.0)
    )


def _predict_in_chunks(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    desc: str,
    chunk_size: int = 2048,
) -> np.ndarray:
    if len(X) == 0:
        return np.array([], dtype=np.float32)

    probs = []
    total_chunks = (len(X) + chunk_size - 1) // chunk_size
    with torch.no_grad():
        for start in tqdm(
            range(0, len(X), chunk_size),
            total=total_chunks,
            desc=desc,
            unit="chunk",
            colour="yellow",
            dynamic_ncols=True,
            leave=False,
        ):
            xv = torch.tensor(X[start:start + chunk_size], device=device)
            logits = model(xv).squeeze(1).cpu().numpy()
            probs.extend((1 / (1 + np.exp(-logits))).tolist())
    return np.array(probs, dtype=np.float32)


def train_screener(
    df: pd.DataFrame,
    forward_days: int = FORWARD_DAYS,
    top_pct: float = TOP_PCT,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    epochs: int = 20,
    lr: float = 3e-4,
    batch_size: int = 512,
    device: torch.device = None,
    label_smoothing: float = 0.05,
    eval_top_n: int = EVAL_TOP_N,
    force_rebuild_cache: bool = False,
):
    """
    Train the screener on real forward returns.

    Requirements for df:
      - Engineered feature columns from FEATURE_COLS
      - Raw close prices in a 'close' column
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if "close" not in df.columns:
        raise ValueError(
            "Screener training requires raw 'close' prices. "
            "Load data with include_raw_cols=True."
        )

    tqdm.write(
        f"Training screener on {df.index.get_level_values('ticker').nunique()} tickers "
        f"| device={device}"
    )

    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    n_features = len(feat_cols)
    ck = _cache_key(df, feat_cols, forward_days, top_pct, val_frac, test_frac, eval_top_n)

    X_train = y_train = r_train = g_train = None
    X_val = y_val = r_val = g_val = None
    X_test = y_test = r_test = g_test = None

    if force_rebuild_cache:
        tqdm.write("  Force retrain enabled - rebuilding screener samples cache.")
    elif Path(SCREENER_SAMPLES).exists():
        try:
            cached = np.load(SCREENER_SAMPLES, allow_pickle=False)
            if str(cached.get("cache_key", b"")) == ck:
                X_train = cached["X_train"]
                y_train = cached["y_train"]
                r_train = cached["r_train"]
                g_train = cached["g_train"]
                X_val = cached["X_val"]
                y_val = cached["y_val"]
                r_val = cached["r_val"]
                g_val = cached["g_val"]
                X_test = cached["X_test"]
                y_test = cached["y_test"]
                r_test = cached["r_test"]
                g_test = cached["g_test"]
                tqdm.write(
                    f"  Loaded cached samples: {len(X_train):,} train  "
                    f"{len(X_val):,} val  {len(X_test):,} test  (key={ck})"
                )
        except Exception as exc:
            tqdm.write(f"  Cache load failed ({exc}) - rebuilding")

    if X_train is None:
        tqdm.write("Building screener samples from raw closes + engineered features...")
        dates, _tickers, feat_arr, close_arr, _volume_arr, present_mask = _prepare_screener_arrays(
            df, feat_cols
        )
        rng = np.random.default_rng(42)
        train_dates, val_dates, test_dates = _sample_date_splits(
            n_dates=len(dates),
            forward_days=forward_days,
            val_frac=val_frac,
            test_frac=test_frac,
        )
        tqdm.write(
            f"  Split dates: train={len(train_dates)}  val={len(val_dates)}  test={len(test_dates)}"
        )

        X_train, y_train, r_train, g_train = _build_samples(
            feat_arr,
            close_arr,
            present_mask,
            train_dates,
            forward_days,
            top_pct,
            rng,
            MAX_T_PER_DATE,
        )
        X_val, y_val, r_val, g_val = _build_samples(
            feat_arr,
            close_arr,
            present_mask,
            val_dates,
            forward_days,
            top_pct,
            rng,
            None,
        )
        X_test, y_test, r_test, g_test = _build_samples(
            feat_arr,
            close_arr,
            present_mask,
            test_dates,
            forward_days,
            top_pct,
            rng,
            None,
        )

        os.makedirs("models", exist_ok=True)
        np.savez_compressed(
            SCREENER_SAMPLES,
            X_train=X_train,
            y_train=y_train,
            r_train=r_train,
            g_train=g_train,
            X_val=X_val,
            y_val=y_val,
            r_val=r_val,
            g_val=g_val,
            X_test=X_test,
            y_test=y_test,
            r_test=r_test,
            g_test=g_test,
            cache_key=np.array(ck),
        )
        tqdm.write(f"  Samples cached -> {SCREENER_SAMPLES}")

    if len(X_train) == 0:
        logger.error("No screener training samples built.")
        return

    tqdm.write(
        f"  Train: {len(X_train):,}  pos={y_train.mean():.1%} | "
        f"Val: {len(X_val):,}  pos={y_val.mean():.1%} | "
        f"Test: {len(X_test):,}  pos={y_test.mean():.1%}"
    )

    model = TickerScorer(n_features=n_features).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tqdm.write(f"  Model: {n_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = epochs * (len(X_train) // batch_size + 1)
    warmup_steps = total_steps // 10

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    pos_weight = torch.tensor([(1 - top_pct) / top_pct], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_train),
        torch.tensor(y_train).unsqueeze(1),
    )
    loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    best_score = float("-inf")
    best_state = None
    best_val_metrics: dict[str, float] = {}
    best_blend_weight = 1.0
    val_heuristic_probs = _heuristic_scores_from_windows(X_val, feat_cols)
    test_heuristic_probs = _heuristic_scores_from_windows(X_test, feat_cols)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(
            loader,
            desc=f"Epoch {epoch + 1:>2}/{epochs}",
            unit="batch",
            colour="blue",
            dynamic_ncols=True,
        )
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            yb_smooth = yb * (1 - label_smoothing) + 0.5 * label_smoothing

            pred = model(xb)
            loss = criterion(pred, yb_smooth)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        avg_loss = total_loss / max(len(loader), 1)
        val_auc = 0.0
        val_metrics = {
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "mean_return_at_k": 0.0,
            "baseline_return": 0.0,
            "lift_at_k": 0.0,
        }

        if len(X_val) > 0:
            model.eval()
            probs = _predict_in_chunks(
                model=model,
                X=X_val,
                device=device,
                desc="  Validation",
            )
            try:
                from sklearn.metrics import roc_auc_score

                val_auc = roc_auc_score(y_val, probs)
            except Exception:
                pass

            blend_candidates = [1.0, 0.85, 0.70, 0.55]
            blended_metrics = None
            blended_score = float("-inf")
            blended_weight = 1.0
            for blend_weight in blend_candidates:
                eval_probs = _blend_scores(probs, val_heuristic_probs, blend_weight)
                metrics = _evaluate_ranked_groups(
                    probs=eval_probs,
                    labels=y_val,
                    forward_returns=r_val,
                    groups=g_val,
                    shortlist_size=eval_top_n,
                )
                score = _score_epoch(metrics)
                if score > blended_score:
                    blended_score = score
                    blended_metrics = metrics
                    blended_weight = blend_weight

            val_metrics = blended_metrics or {
                "precision_at_k": 0.0,
                "recall_at_k": 0.0,
                "mean_return_at_k": 0.0,
                "baseline_return": 0.0,
                "lift_at_k": 0.0,
            }
            model_score = blended_score
            if model_score > best_score:
                best_score = model_score
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_val_metrics = dict(val_metrics)
                best_val_metrics["val_auc"] = float(val_auc)
                best_blend_weight = blended_weight

        tqdm.write(
            f"  Epoch {epoch + 1:>2}/{epochs}  loss={avg_loss:.4f}  "
            f"val_auc={val_auc:.4f}  "
            f"p@{eval_top_n}={val_metrics['precision_at_k']:.1%}  "
            f"r@{eval_top_n}={val_metrics['recall_at_k']:.1%}  "
            f"ret@{eval_top_n}={val_metrics['mean_return_at_k']:.2%}  "
            f"lift={val_metrics['lift_at_k']:.2f}  "
            f"blend={best_blend_weight:.2f}"
            + (" <- best" if best_state is not None and best_score == _score_epoch(val_metrics) else "")
        )

    if best_state is not None:
        model.load_state_dict(best_state)
        tqdm.write(
            f"  Restored best model "
            f"(p@{eval_top_n}={best_val_metrics.get('precision_at_k', 0.0):.1%}, "
            f"r@{eval_top_n}={best_val_metrics.get('recall_at_k', 0.0):.1%}, "
            f"blend={best_blend_weight:.2f})"
        )

    test_auc = 0.0
    test_metrics = {
        "precision_at_k": 0.0,
        "recall_at_k": 0.0,
        "mean_return_at_k": 0.0,
        "baseline_return": 0.0,
        "lift_at_k": 0.0,
    }
    if len(X_test) > 0:
        model.eval()
        test_probs = _predict_in_chunks(
            model=model,
            X=X_test,
            device=device,
            desc="  Testing",
        )
        try:
            from sklearn.metrics import roc_auc_score

            test_auc = roc_auc_score(y_test, test_probs)
        except Exception:
            pass
        test_probs = _blend_scores(test_probs, test_heuristic_probs, best_blend_weight)
        test_metrics = _evaluate_ranked_groups(
            probs=test_probs,
            labels=y_test,
            forward_returns=r_test,
            groups=g_test,
            shortlist_size=eval_top_n,
        )
        tqdm.write(
            f"  Test  auc={test_auc:.4f}  "
            f"p@{eval_top_n}={test_metrics['precision_at_k']:.1%}  "
            f"r@{eval_top_n}={test_metrics['recall_at_k']:.1%}  "
            f"ret@{eval_top_n}={test_metrics['mean_return_at_k']:.2%}  "
            f"lift={test_metrics['lift_at_k']:.2f}"
        )

    os.makedirs("models", exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "n_features": n_features,
            "feature_cols": feat_cols,
            "lookback": LOOKBACK,
            "forward_days": forward_days,
            "top_pct": top_pct,
            "eval_top_n": eval_top_n,
            "blend_weight": float(best_blend_weight),
            "val_metrics": best_val_metrics,
            "test_metrics": {**test_metrics, "test_auc": float(test_auc)},
        },
        SCREENER_CKPT,
    )
    tqdm.write(
        f"Screener saved -> {SCREENER_CKPT}  "
        f"(best p@{eval_top_n}={best_val_metrics.get('precision_at_k', 0.0):.1%})"
    )
    return model


def load_screener(device: torch.device) -> TickerScorer:
    ckpt = torch.load(SCREENER_CKPT, map_location=device, weights_only=False)
    model = TickerScorer(n_features=ckpt["n_features"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model._blend_weight = float(ckpt.get("blend_weight", 1.0))
    model._feature_cols = ckpt.get("feature_cols", [])
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

    The model expects engineered features from FEATURE_COLS. If raw close/volume
    columns are also present, they are used for final-day eligibility filters.
    """
    if not os.path.exists(SCREENER_CKPT):
        raise FileNotFoundError(
            "Screener not trained yet. Run: python Agent.py --mode train_screener"
        )

    model = load_screener(device)
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    dates = sorted(df.index.get_level_values("date").unique())
    if len(dates) < LOOKBACK:
        raise ValueError(
            f"Need at least {LOOKBACK} dates for screening; found {len(dates)}."
        )

    last_date = dates[-1]
    recent_dates = dates[-LOOKBACK:]
    df_recent = df[df.index.get_level_values("date").isin(recent_dates)]
    tickers = sorted(df_recent.index.get_level_values("ticker").unique())
    ticker_idx = {t: i for i, t in enumerate(tickers)}
    n_t = len(tickers)
    n_features = len(feat_cols)

    arr = np.full((LOOKBACK, n_t, n_features), np.nan, dtype=np.float32)
    present_mask = np.zeros((LOOKBACK, n_t), dtype=bool)
    for li, date in enumerate(recent_dates):
        try:
            snap = df_recent.loc[date][feat_cols]
            for ticker, row in snap.iterrows():
                ti = ticker_idx.get(ticker)
                if ti is None:
                    continue
                arr[li, ti, :] = row.values.astype(np.float32)
                present_mask[li, ti] = True
        except KeyError:
            pass
    arr = np.nan_to_num(np.clip(arr, -5.0, 5.0), nan=0.0)

    latest_close = None
    latest_volume = None
    if "close" in df_recent.columns:
        try:
            latest_close = df_recent.loc[last_date]["close"]
        except (KeyError, Exception):
            latest_close = None
    if "volume" in df_recent.columns:
        try:
            latest_volume = df_recent.loc[last_date]["volume"]
        except (KeyError, Exception):
            latest_volume = None

    obs_list, valid_tickers = [], []
    for ti, ticker in enumerate(tickers):
        if present_mask[:, ti].mean() < MIN_HISTORY_COVERAGE:
            continue

        if latest_close is not None:
            try:
                price = float(latest_close.loc[ticker])
                if price < min_price:
                    continue
                if max_price is not None and price > max_price:
                    continue
            except Exception:
                pass

        if latest_volume is not None:
            try:
                if float(latest_volume.loc[ticker]) < min_volume:
                    continue
            except Exception:
                pass

        obs_list.append(arr[:, ti, :])
        valid_tickers.append(ticker)

    if not obs_list:
        logger.error("No valid observations for screener.")
        return pd.DataFrame()

    obs_arr = np.array(obs_list, dtype=np.float32)
    X = torch.tensor(obs_arr, device=device)

    scores = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            logits = model(X[i:i + batch_size]).squeeze(1)
            scores.extend(torch.sigmoid(logits).cpu().numpy().tolist())

    heuristic_scores = _heuristic_scores_from_windows(obs_arr, feat_cols)
    blend_weight = float(getattr(model, "_blend_weight", 1.0))
    scores = _blend_scores(
        np.array(scores, dtype=np.float32),
        heuristic_scores,
        blend_weight,
    ).tolist()

    rows = []
    for ticker, score in zip(valid_tickers, scores):
        row = {"ticker": ticker, "score": score}
        try:
            feats = df_recent.loc[(last_date, ticker)]
            feat_dict = dict(zip(feat_cols, feats[feat_cols].values))
            row["momentum_5d"] = feat_dict.get("ret_5d", np.nan)
            row["momentum_20d"] = feat_dict.get("ret_20d", np.nan)
            row["rsi"] = feat_dict.get("rsi", np.nan)
            row["sentiment"] = feat_dict.get("sent_net", np.nan)
            row["sent_surprise"] = feat_dict.get("sent_surprise", np.nan)
            row["vol_ratio"] = feat_dict.get("vol_ratio", np.nan)
            if "close" in feats.index:
                row["close"] = float(feats["close"])
            if "volume" in feats.index:
                row["volume"] = float(feats["volume"])
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
    print(f"\n{'=' * 80}")
    print(f"  Top Picks - {label}  ({len(results)} shown)")
    print(f"{'=' * 80}")
    print(
        f"  {'#':<4} {'Ticker':<8} {'Score':>6}  {'Mom5d':>7}  {'Mom20d':>7}  "
        f"{'RSI':>6}  {'Sentiment':>10}  {'VolRatio':>9}"
    )
    print(f"  {'-' * 74}")
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
    print(f"{'=' * 80}\n")
