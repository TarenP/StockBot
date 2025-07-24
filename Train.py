#!/usr/bin/env python
# train_topselector_lstm.py – LSTM version of Top‑Selector tuned for recall
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
    roc_auc_score, classification_report
)
import matplotlib.pyplot as plt

torch.multiprocessing.set_start_method("spawn", force=True)

# Configuration – tuned for better minority recall
PARQUET_PATH = Path("MasterDS/Master_cleaned.parquet")
RETURN_COL   = "ret_5d_future"
BATCH_SIZE   = 512
EPOCHS       = 15
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW       = 10
POS_RATIO    = 0.50                # increased to 50% positives per batch
NUM_WORKERS  = max(1, min(6, os.cpu_count() - 2))

# -----------------------------------------------------------
# Metadata & scalers – unchanged
# -----------------------------------------------------------
pf = pq.ParquetFile(PARQUET_PATH)
ALL_COLS = pf.schema_arrow.names
FEATURE_COLS = [c for c in ALL_COLS if c not in {RETURN_COL, "date", "ticker"}]

dates = sorted({d for rg in range(pf.num_row_groups)
                for d in pf.read_row_group(rg, ["date"]).column(0).to_pylist()})
split = int(len(dates) * 0.8)
TRAIN_DATES = set(dates[:split])
TEST_DATES  = set(dates[split:])

# Scaler fit on training sample
samples = []
for i in range(min(20, pf.num_row_groups)):
    df0 = pf.read_row_group(i).to_pandas()
    df0 = df0[df0["date"].isin(TRAIN_DATES)]
    if not df0.empty:
        samples.append(df0.sample(3000, random_state=42)[FEATURE_COLS])
SCALER = StandardScaler().fit(pd.concat(samples).values)

# 95th percentile cutoff per date
d_cutoff = defaultdict(list)
for batch in pf.iter_batches(batch_size=500_000):
    df1 = batch.to_pandas()[["date", RETURN_COL]]
    for d,g in df1.groupby("date"): d_cutoff[d].extend(g[RETURN_COL].values)
CUTOFF_MAP = {d: np.quantile(v, 0.95) for d,v in d_cutoff.items()}

# Total training steps
total_rows = sum(pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups))
TRAIN_STEPS = int((total_rows * 0.8) // BATCH_SIZE)
print(f"{TRAIN_STEPS} training batches per epoch")

# -----------------------------------------------------------
# Dataset – unchanged
# -----------------------------------------------------------
class ParquetSequenceDataset(IterableDataset):
    def __init__(self, dates, pos_ratio):
        super().__init__()
        self.dates = set(dates)
        self.pos_ratio = pos_ratio

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        nwrk = info.num_workers if info else 1

        pf_local = pq.ParquetFile(PARQUET_PATH)
        history = defaultdict(lambda: deque(maxlen=WINDOW))
        buf_X, buf_y, buf_meta = [], [], []

        for rg in range(wid, pf_local.num_row_groups, nwrk):
            df = pf_local.read_row_group(rg).to_pandas()
            df = df[df["date"].isin(self.dates)]
            if df.empty: continue
            df.sort_values(["ticker","date"], inplace=True)
            df["label"] = (df[RETURN_COL] >= df["date"].map(CUTOFF_MAP)).astype(int)

            for row in df.itertuples(index=False):
                feats = np.array([getattr(row,c) for c in FEATURE_COLS], np.float32)
                hist = history[row.ticker]
                hist.append(feats)
                if len(hist) < WINDOW: continue
                buf_X.append(np.stack(hist))
                buf_y.append(float(row.label))
                buf_meta.append((row.date, row.ticker))
                if len(buf_y) >= BATCH_SIZE:
                    batch = self._process_batch(buf_X, buf_y, buf_meta)
                    buf_X, buf_y, buf_meta = [], [], []
                    yield batch

        # flush
        if buf_y:
            yield self._process_batch(buf_X, buf_y, buf_meta)

    def _process_batch(self, X, y, meta):
        X_arr = np.array(X); y_arr = np.array(y, dtype=np.float32)
        # oversample positives
        if self.pos_ratio:
            pos = np.where(y_arr==1)[0]; neg = np.where(y_arr==0)[0]
            if pos.size and neg.size:
                n_pos = int(BATCH_SIZE*self.pos_ratio)
                n_neg = BATCH_SIZE - n_pos
                sel = np.concatenate([
                    np.random.choice(pos,n_pos,replace=pos.size<n_pos),
                    np.random.choice(neg,n_neg,replace=neg.size<n_neg)
                ])
                np.random.shuffle(sel)
                X_arr = X_arr[sel]; y_arr = y_arr[sel]
        else:
            if len(y_arr)>BATCH_SIZE:
                idx = np.random.choice(len(y_arr),BATCH_SIZE,replace=False)
                X_arr = X_arr[idx]; y_arr = y_arr[idx]

        flat = SCALER.transform(X_arr.reshape(-1,X_arr.shape[-1]))
        Xs = np.clip(flat,-5,5).reshape(X_arr.shape)
        return (
            torch.from_numpy(Xs.astype(np.float32)),
            torch.from_numpy(y_arr),
            meta
        )

# -----------------------------------------------------------
# Model – LSTM + Attention with increased capacity
# -----------------------------------------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim,
                 hidden_dim=256,  # increased hidden size
                 num_layers=3,    # deeper LSTM
                 bidirectional=True,
                 dropout=0.3):    # higher dropout
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers>1 else 0
        )
        D = hidden_dim * (2 if bidirectional else 1)
        self.attn = nn.Linear(D, D)
        self.norm = nn.LayerNorm(D)
        self.fc   = nn.Linear(D, 1)

    def forward(self, x):
        h, _ = self.lstm(x)
        scores = torch.matmul(self.attn(h), h.transpose(-1,-2)) / math.sqrt(h.size(-1))
        a = torch.softmax(scores, dim=-1)
        h2 = torch.matmul(a, h)
        v  = h2.mean(dim=1)
        return self.fc(self.norm(v)).squeeze(1)

# -----------------------------------------------------------
# Focal Loss – unchanged
# -----------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.):
        super().__init__(); self.a, self.g = alpha, gamma
    def forward(self, logit, y):
        b = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
        p = torch.sigmoid(logit)
        pt = y*p + (1-y)*(1-p)
        return (self.a * (1-pt).pow(self.g) * b).mean()

# -----------------------------------------------------------
# Training loop with tuned LR and scheduler
# -----------------------------------------------------------
def train():
    ds = ParquetSequenceDataset(TRAIN_DATES, pos_ratio=POS_RATIO)
    loader = DataLoader(ds, batch_size=None, num_workers=NUM_WORKERS,
                        pin_memory=True, persistent_workers=bool(NUM_WORKERS),
                        prefetch_factor=2 if NUM_WORKERS else 0)

    model = LSTMClassifier(len(FEATURE_COLS)).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=2, threshold=5e-4
    )
    scaler = GradScaler()
    crit   = FocalLoss().to(DEVICE)

    epoch_losses = []
    for ep in range(1, EPOCHS+1):
        model.train(); run_loss=0.0; steps=0
        pbar = tqdm(total=TRAIN_STEPS, desc=f"Epoch {ep}/{EPOCHS}", ncols=100)
        for xb,yb,_ in loader:
            if steps>=TRAIN_STEPS: break
            xb,yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with autocast(device_type=DEVICE.type, dtype=torch.float16):
                loss = crit(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(opt); scaler.update()
            run_loss += loss.item(); steps+=1
            pbar.update(1)
        pbar.close()

        epoch_loss = run_loss / max(steps,1)
        epoch_losses.append(epoch_loss)
        print(f"→ Epoch {ep} mean loss: {epoch_loss:.4f}")
        scheduler.step(epoch_loss)
        torch.save(model.state_dict(), f"models/TopSelector_epoch{ep}.pt")

    torch.save(model.state_dict(), "models/TopSelector_final.pt")
    return model, epoch_losses

# -----------------------------------------------------------
# Evaluation unchanged
# -----------------------------------------------------------
def evaluate(model):
    ds = ParquetSequenceDataset(TEST_DATES, pos_ratio=None)
    loader = DataLoader(ds, batch_size=None, num_workers=0)

    y_true,y_prob = [], []
    model.eval(); pbar = tqdm(desc="Evaluating", unit="batch")
    with torch.no_grad():
        for xb,yb,_ in loader:
            xb = xb.to(DEVICE)
            probs = torch.sigmoid(model(xb)).cpu().numpy()
            y_true.extend(yb.numpy().tolist())
            y_prob.extend(probs.tolist())
            pbar.update(1)
    pbar.close()

    y_t = np.array(y_true,int); y_p = np.array(y_prob)
    y_pred = (y_p>=0.5).astype(int)
    print("\nEvaluation Metrics")
    print("Accuracy       :", accuracy_score(y_t,y_pred))
    print("Balanced Acc   :", balanced_accuracy_score(y_t,y_pred))
    print("AUROC          :", roc_auc_score(y_t,y_p))
    print(classification_report(y_t,y_pred,digits=3))

# -----------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------
def main():
    os.makedirs("models", exist_ok=True)
    print(f"Using {DEVICE} | workers={NUM_WORKERS}")
    model, losses = train()
    plt.figure(figsize=(6,4)); plt.plot(range(1,EPOCHS+1), losses, marker='o')
    plt.title("Mean Loss per Epoch"); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.grid(True); plt.tight_layout(); plt.savefig("training_loss.png")
    evaluate(model)

if __name__ == "__main__":
    main()

# Exports
torch.manual_seed(42)
