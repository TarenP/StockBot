#!/usr/bin/env python
# test_topselector_lstm.py – Batched evaluation script aligned with training worker logic
# -----------------------------------------------------------------------------
import os
from pathlib import Path
from collections import defaultdict, deque

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    roc_auc_score, classification_report
)
from tqdm.auto import tqdm

# ---------------- user-config -------------------------------------------------
PARQUET_PATH  = Path("MasterDS/Master_cleaned.parquet")
MODEL_WEIGHTS = Path("models/TopSelector_epoch4.pt")
RETURN_COL    = "ret_5d_future"
BATCH_SIZE    = 512          # match train BATCH_SIZE
WINDOW        = 10
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS   = max(1, min(6, os.cpu_count() - 2))
# -----------------------------------------------------------------------------

# 1) ---------- metadata -------------------------------------------------------
pf = pq.ParquetFile(PARQUET_PATH)
ALL_COLS     = pf.schema_arrow.names
FEATURE_COLS = [c for c in ALL_COLS if c not in {RETURN_COL, "date", "ticker"}]

# reproduce 80/20 date split
dates = []
for rg in range(pf.num_row_groups):
    dates.extend(pf.read_row_group(rg, columns=["date"]).column(0).to_pylist())
all_dates = sorted(set(dates))
split = int(len(all_dates) * 0.8)
TEST_DATES = set(all_dates[split:])

# fit StandardScaler on small sample
samples = []
for rg in range(min(10, pf.num_row_groups)):
    df = pf.read_row_group(rg).to_pandas()
    if not df.empty:
        samples.append(df.sample(1000, random_state=0)[FEATURE_COLS])
SCALER = StandardScaler().fit(np.vstack(samples))

# per-day cutoff map
cutoff = defaultdict(list)
for batch in pf.iter_batches(batch_size=2_000_000):
    df = batch.to_pandas()[["date", RETURN_COL]]
    for d, g in df.groupby("date"):
        cutoff[d].extend(g[RETURN_COL].values)
CUTOFF_MAP = {d: np.quantile(v, 0.95) for d, v in cutoff.items()}

# 2) ---------- LSTM Model Definition -----------------------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2,
                 bidirectional=True, dropout=0.2):
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

    def forward(self, x):
        h, _ = self.lstm(x)
        pooled = h.mean(dim=1)
        return self.head(pooled).squeeze(1)

# 3) ---------- streaming test dataset mirroring train ------------------------
class ParquetSequenceDataset(IterableDataset):
    def __init__(self, dates):
        super().__init__()
        self.dates = set(dates)

    def __iter__(self):
        worker = get_worker_info()
        wid = worker.id if worker else 0
        nworkers = worker.num_workers if worker else 1

        pf_local = pq.ParquetFile(PARQUET_PATH)
        history = defaultdict(lambda: deque(maxlen=WINDOW))
        buf_X, buf_y = [], []

        # round-robin rowgroups
        for rg in range(wid, pf_local.num_row_groups, nworkers):
            df = pf_local.read_row_group(rg).to_pandas()
            df = df[df["date"].isin(self.dates)]
            if df.empty:
                continue
            df.sort_values(["ticker","date"], inplace=True)
            df["label"] = (df[RETURN_COL] >= df["date"].map(CUTOFF_MAP)).astype(int)

            for row in df.itertuples(index=False):
                feats = np.array([getattr(row,c) for c in FEATURE_COLS], np.float32)
                hist = history[row.ticker]
                hist.append(feats)
                if len(hist) < WINDOW:
                    continue
                buf_X.append(np.stack(hist))
                buf_y.append(row.label)

                if len(buf_y) >= BATCH_SIZE:
                    Xb = np.vstack(buf_X).reshape(-1, WINDOW, len(FEATURE_COLS))
                    Xb = SCALER.transform(Xb.reshape(-1, Xb.shape[-1])).reshape(Xb.shape)
                    Xb = torch.from_numpy(Xb)
                    yb = torch.tensor(buf_y, dtype=torch.float32)
                    buf_X, buf_y = [], []
                    yield Xb, yb

        # flush remainder
        while buf_y:
            Xb = np.vstack(buf_X).reshape(-1, WINDOW, len(FEATURE_COLS))
            Xb = SCALER.transform(Xb.reshape(-1, Xb.shape[-1])).reshape(Xb.shape)
            Xb = torch.from_numpy(Xb)
            yb = torch.tensor(buf_y, dtype=torch.float32)
            yield Xb, yb
            buf_X, buf_y = [], []

# 4) ---------- evaluation loop -----------------------------------------------
def evaluate():
    print(f"Loading model on {DEVICE}…")
    model = LSTMClassifier(len(FEATURE_COLS)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
    model.eval()

    ds = ParquetSequenceDataset(TEST_DATES)
    loader = DataLoader(
        ds,
        batch_size=None,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    y_true, y_prob = [], []
    for Xb, yb in tqdm(loader, desc="Evaluating", unit="batch", ncols=80):
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        with torch.no_grad():
            probs = torch.sigmoid(model(Xb)).cpu().numpy()
        y_true.extend(yb.cpu().numpy().tolist())
        y_prob.extend(probs.tolist())

    y_true = np.array(y_true, int)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    print("\n──────────  METRICS  ──────────")
    print("Accuracy       :", accuracy_score(y_true, y_pred))
    print("Balanced Acc   :", balanced_accuracy_score(y_true, y_pred))
    print("AUROC          :", roc_auc_score(y_true, y_prob))
    print(classification_report(y_true, y_pred, digits=3))

# 5) ---------- run ------------------------------------------------------------
if __name__ == "__main__":
    evaluate()
