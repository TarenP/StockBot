#!/usr/bin/env python
# train_topselector_lstm.py – LSTM version of Top‑Selector
# -----------------------------------------------------------
# Set‑up (unchanged except for matplotlib import)
# -----------------------------------------------------------
import os, math
from pathlib import Path
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import IterableDataset, DataLoader
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    roc_auc_score, classification_report,
)
import matplotlib.pyplot as plt  # <‑‑ added

torch.multiprocessing.set_start_method("spawn", force=True)

PARQUET_PATH = Path("MasterDS/Master_cleaned.parquet")
RETURN_COL   = "ret_5d_future"
BATCH_SIZE   = 256
EPOCHS       = 15
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW       = 10
POS_RATIO    = 0.30              # oversample positives inside each batch
NUM_WORKERS  = max(1, min(6, os.cpu_count() - 2))

# -----------------------------------------------------------
# Utility: load metadata once – UNCHANGED
# -----------------------------------------------------------
pf = pq.ParquetFile(PARQUET_PATH)
ALL_COLS   = pf.schema_arrow.names
FEATURE_COLS = [c for c in ALL_COLS if c not in {RETURN_COL, "date", "ticker"}]

# unique dates → split 80/20
all_dates = set()
for rg in range(pf.num_row_groups):
    all_dates.update(
        pf.read_row_group(rg, columns=["date"]).to_pandas()["date"].unique()
    )
all_dates = sorted(all_dates)
split = int(len(all_dates) * 0.8)
TRAIN_DATES = set(all_dates[:split])
TEST_DATES  = set(all_dates[split:])

# ticker‑to‑id
tickers = set()
for rg in range(pf.num_row_groups):
    tickers.update(
        pf.read_row_group(rg, columns=["ticker"]).to_pandas()["ticker"].unique()
    )
TICKER2ID = {t: i for i, t in enumerate(sorted(tickers))}
TICKER_VOCAB = len(TICKER2ID)

# scaler fit on small sample
samples = []
for i in range(min(20, pf.num_row_groups)):
    df = pf.read_row_group(i).to_pandas()
    df = df[df["date"].isin(TRAIN_DATES)]
    if not df.empty:
        samples.append(df.sample(3000, random_state=42)[FEATURE_COLS])
SCALER = StandardScaler().fit(pd.concat(samples).values)

# 95‑th pct cut‑off per date
cutoff = defaultdict(list)
for batch in pf.iter_batches(batch_size=500_000):
    df = batch.to_pandas()[["date", RETURN_COL]]
    for d, g in df.groupby("date"):
        cutoff[d].extend(g[RETURN_COL].values)
CUTOFF_MAP = {d: np.quantile(v, 0.95) for d, v in cutoff.items()}

# total number of rows across all row‑groups
total_rows = sum(pf.metadata.row_group(i).num_rows
                 for i in range(pf.num_row_groups))

# train uses 80% of those rows, at BATCH_SIZE per batch
TRAIN_STEPS = int((total_rows * 0.8) // BATCH_SIZE)

print(f"{TRAIN_STEPS} training batches per epoch")

# -----------------------------------------------------------
# Dataset – UNCHANGED
# -----------------------------------------------------------
class ParquetSequenceDataset(IterableDataset):
    def __init__(self, dates, pos_ratio):
        super().__init__()
        self.dates = set(dates)
        self.pos_ratio = pos_ratio

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        
        pf_local = pq.ParquetFile(PARQUET_PATH)
        history = defaultdict(lambda: deque(maxlen=WINDOW))
        buf_X, buf_y, buf_meta = [], [], []

        for rg_idx in range(worker_id, pf_local.num_row_groups, num_workers):
            try:
                df = pf_local.read_row_group(rg_idx).to_pandas()
                df = df[df["date"].isin(self.dates)]
                if df.empty:
                    continue
                
                df.sort_values(["ticker", "date"], inplace=True)
                df["label"] = (df[RETURN_COL] >= df["date"].map(CUTOFF_MAP)).astype(int)

                for row in df.itertuples(index=False):
                    feats = np.array([getattr(row, c) for c in FEATURE_COLS],
                                     dtype=np.float32)
                    hist = history[row.ticker]
                    hist.append(feats)
                    if len(hist) < WINDOW: 
                        continue

                    buf_X.append(np.stack(hist))
                    buf_y.append(float(row.label))
                    buf_meta.append((row.date, row.ticker))

                    if len(buf_y) >= BATCH_SIZE:
                        batch = self._process_batch(buf_X[:BATCH_SIZE], buf_y[:BATCH_SIZE], buf_meta[:BATCH_SIZE])
                        buf_X, buf_y, buf_meta = buf_X[BATCH_SIZE:], buf_y[BATCH_SIZE:], buf_meta[BATCH_SIZE:]
                        yield batch
            except Exception as e:
                print(f"Error processing row group {rg_idx}: {e}")
                continue

        while buf_X:
            batch_size = min(len(buf_X), BATCH_SIZE)
            yield self._process_batch(buf_X[:batch_size], buf_y[:batch_size], buf_meta[:batch_size])
            buf_X, buf_y, buf_meta = buf_X[batch_size:], buf_y[batch_size:], buf_meta[batch_size:]

    def _process_batch(self, X, y, meta):
        X, y, meta = np.array(X), np.array(y, dtype=np.float32), np.array(meta)
        
        if self.pos_ratio:
            pos_indices = np.where(y == 1)[0]
            neg_indices = np.where(y == 0)[0]
            
            if len(pos_indices) > 0 and len(neg_indices) > 0:
                n_pos = int(BATCH_SIZE * self.pos_ratio)
                n_neg = BATCH_SIZE - n_pos
                sel_pos = np.random.choice(pos_indices, min(n_pos, len(pos_indices)), 
                                          replace=n_pos > len(pos_indices))
                sel_neg = np.random.choice(neg_indices, min(n_neg, len(neg_indices)), 
                                          replace=n_neg > len(neg_indices))
                selected = np.concatenate([sel_pos, sel_neg])
                np.random.shuffle(selected)
                X, y, meta = X[selected], y[selected], meta[selected]
        
        flat = SCALER.transform(X.reshape(-1, X.shape[-1]))
        X = np.clip(flat, -5, 5).reshape(X.shape)
        
        return (
            torch.from_numpy(X.astype(np.float32)),
            torch.from_numpy(y),
            meta
        )

# -----------------------------------------------------------
# Model – ***REPLACED WITH LSTM***
# -----------------------------------------------------------
class LSTMClassifier(nn.Module):
    """Bidirectional LSTM with mean‑pooling head."""
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 bidirectional: bool = True, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, 1)
        )

    def forward(self, x):  # x: [B, WINDOW, input_dim]
        h, _ = self.lstm(x)           # h: [B, WINDOW, out_dim]
        pooled = h.mean(dim=1)        # mean‑pool across time
        return self.head(pooled).squeeze(1)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.):
        super().__init__(); self.a, self.g = alpha, gamma
    def forward(self, logit, y):
        bce = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
        p   = torch.sigmoid(logit)
        pt  = y*p + (1-y)*(1-p)
        return (self.a * (1-pt).pow(self.g) * bce).mean()

# -----------------------------------------------------------
# Train / Eval – progress bar retained
# -----------------------------------------------------------

def train():
    train_ds = ParquetSequenceDataset(TRAIN_DATES, POS_RATIO)
    train_loader = DataLoader(
        train_ds, batch_size=None, num_workers=NUM_WORKERS,
        pin_memory=True, persistent_workers=bool(NUM_WORKERS),
        prefetch_factor=2 if NUM_WORKERS else 0,
    )

    model = LSTMClassifier(len(FEATURE_COLS)).to(DEVICE)  # <‑‑ NEW MODEL
    opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=3e-4,
        steps_per_epoch=TRAIN_STEPS,
        epochs=EPOCHS,
        pct_start=0.2,
    )
    scaler = GradScaler()
    crit = FocalLoss().to(DEVICE)

    epoch_losses = []
    for epoch in range(1, EPOCHS+1):
        model.train(); running_loss = 0.0; step_count = 0
        pbar = tqdm(total=TRAIN_STEPS, desc=f"Epoch {epoch}/{EPOCHS}",
                    mininterval=0.5, ncols=100)
        try:
            for xb, yb, _ in train_loader:
                if step_count >= TRAIN_STEPS:
                    break
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad(set_to_none=True)
                with autocast(device_type=DEVICE.type, dtype=torch.float16):
                    loss = crit(model(xb), yb)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                scheduler.step()
                running_loss += loss.item(); step_count += 1
                pbar.update(1)
                pbar.set_postfix({'loss': f"{loss.item():.4f}",
                                   'avg_loss': f"{running_loss/step_count:.4f}"})
        except Exception as e:
            print(f"\nTraining error at epoch {epoch}, step {step_count}: {e}")
        finally:
            pbar.close()
        epoch_loss = running_loss / max(step_count, 1)
        epoch_losses.append(epoch_loss)
        print(f"→ Epoch {epoch} mean loss: {epoch_loss:.4f}")
        torch.save(model.state_dict(), f"models/TopSelector_epoch{epoch}.pt")

    torch.save(model.state_dict(), "models/TopSelector_final.pt")
    return model, epoch_losses

# -----------------------------------------------------------
# Evaluation (unchanged)
# -----------------------------------------------------------

def evaluate(model):
    test_ds = ParquetSequenceDataset(TEST_DATES, pos_ratio=None)
    test_loader = DataLoader(test_ds, batch_size=None, num_workers=0)

    y_true, y_prob = [], []
    model.eval(); pbar = tqdm(desc="Evaluating", unit="batch")
    with torch.no_grad():
        for xb, yb, _ in test_loader:
            xb = xb.to(DEVICE); y_true.extend(yb.numpy())
            for i in range(0, xb.size(0), 512):
                chunk = xb[i:i+512]
                y_prob.extend(torch.sigmoid(model(chunk)).cpu().numpy())
            pbar.update(1)
    pbar.close()
    y_true, y_prob = np.array(y_true, dtype=int), np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    print("\nEvaluation Metrics")
    print("Accuracy       :", accuracy_score(y_true, y_pred))
    print("Balanced Acc   :", balanced_accuracy_score(y_true, y_pred))
    print("AUROC          :", roc_auc_score(y_true, y_prob))
    print(classification_report(y_true, y_pred, digits=3))

# -----------------------------------------------------------
# Entrypoint – plot fix
# -----------------------------------------------------------

def main():
    os.makedirs("models", exist_ok=True)
    print(f"Using {DEVICE} | workers={NUM_WORKERS}")
    model, epoch_losses = train()
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, EPOCHS+1), epoch_losses, marker='o')
    plt.title("Training: Mean Loss per Epoch")
    plt.xlabel("Epoch"); plt.ylabel("Mean Loss")
    plt.grid(True); plt.tight_layout()
    plt.savefig("training_loss.png")
    evaluate(model)

if __name__ == "__main__":
    main()
