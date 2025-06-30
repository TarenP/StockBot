import os
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------
# Configuration
# ---------------------------------------
parquet_path = 'Master_cleaned.parquet'
return_col   = 'ret_5d'
batch_size   = 2048
n_epochs     = 10
patience     = 3
device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------
# Prepare ParquetFile
# ---------------------------------------
print("[DEBUG] Opening Parquet file for streaming...")
pf = pq.ParquetFile(parquet_path)
all_cols = pf.schema_arrow.names
exclude  = {return_col, 'date', 'ticker'}
feature_cols = [c for c in all_cols if c not in exclude]
print(f"[DEBUG] Detected feature columns: {len(feature_cols)}")

# ---------------------------------------
# Compute date-wise cutoff map
# ---------------------------------------
print("[DEBUG] Computing date-wise cutoff map...")
cutoff_lists = {}
all_dates = set()
for i, batch in enumerate(pf.iter_batches(batch_size=1_000_000)):
    dfb = batch.to_pandas()[['date', return_col]]
    for date, grp in dfb.groupby('date'):
        all_dates.add(date)
        cutoff_lists.setdefault(date, []).extend(grp[return_col].values)
    if (i + 1) % 10 == 0:
        print(f"[DEBUG] Processed {i+1} partitions for cutoff map...")

cutoff_map = {date: np.quantile(vals, 0.95) for date, vals in cutoff_lists.items()}
print(f"[DEBUG] Computed cutoff for {len(cutoff_map)} dates.")

# Chronological train/val split
all_dates = sorted(all_dates)
split_idx = int(len(all_dates) * 0.8)
train_dates = set(all_dates[:split_idx])
val_dates   = set(all_dates[split_idx:])
print(f"[DEBUG] Train dates: {len(train_dates)}, Validation dates: {len(val_dates)}")

# Fit a global StandardScaler on a small sample
print("[DEBUG] Fitting StandardScaler on first row group...")
sample = pf.read_row_group(0).to_pandas()[feature_cols]
scaler = StandardScaler().fit(sample.values)
print("[DEBUG] Scaler mean/scale calculated.")

# ---------------------------------------
# Define an Iterable Dataset generator
# ---------------------------------------
def parquet_generator(dates_subset, split_name="train"):
    local_pf = pq.ParquetFile(parquet_path)
    print(f"[DEBUG] Starting generator for {split_name}, dates_subset size = {len(dates_subset)}")
    for i, batch in enumerate(local_pf.iter_batches(batch_size=batch_size)):
        df = batch.to_pandas()
        df = df[df['date'].isin(dates_subset)]
        if df.empty:
            continue
        # Label on the fly
        df['label'] = (df[return_col] >= df['date'].map(cutoff_map)).astype(np.int8)
        # Extract and clean features/labels
        X_raw = df[feature_cols].values.astype(np.float32)
        y_raw = df['label'].values.astype(np.float32)
        mask = np.isfinite(X_raw).all(axis=1)
        if not mask.any():
            continue
        X = scaler.transform(X_raw[mask])
        y = y_raw[mask]
        # Yield mini-batches
        for j in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[j:j+batch_size]).to(device)
            yb = torch.from_numpy(y[j:j+batch_size]).to(device)
            if j == 0:
                print(f"[DEBUG] {split_name} partition {i}: yielding {len(xb)} samples")
            yield xb, yb

# ---------------------------------------
# Define the MLP model
# ---------------------------------------
class SelectorMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

model = SelectorMLP(len(feature_cols)).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Estimate steps per epoch
total_rows = sum(pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups))
train_steps = int(total_rows * 0.8) // batch_size
val_steps   = int(total_rows * 0.2) // batch_size
print(f"[DEBUG] Steps per epoch: train={train_steps}, val={val_steps}")

# ---------------------------------------
# Training with early stopping on validation AUC
# ---------------------------------------
best_val_auc = 0.0
epochs_no_improve = 0

for epoch in range(1, n_epochs + 1):
    print(f"\n[DEBUG] Starting epoch {epoch}/{n_epochs}")
    # Training
    model.train()
    train_losses = []
    gen = parquet_generator(train_dates, split_name="train")
    for batch_idx, (X_batch, y_batch) in enumerate(gen):
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        if (batch_idx + 1) % 50 == 0:
            print(f"[DEBUG] Epoch {epoch} Train batch {batch_idx+1}/{train_steps}, loss={loss.item():.4f}")
        if batch_idx+1 >= train_steps:
            break
    avg_train_loss = np.mean(train_losses)

    # Validation
    model.eval()
    val_losses, y_trues, y_scores = [], [], []
    gen_val = parquet_generator(val_dates, split_name="val")
    for batch_idx, (X_batch, y_batch) in enumerate(gen_val):
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        val_losses.append(loss.item())
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        y_scores.append(probs)
        y_trues.append(y_batch.detach().cpu().numpy())
        if (batch_idx + 1) % 20 == 0:
            print(f"[DEBUG] Epoch {epoch} Val batch {batch_idx+1}/{val_steps}, loss={loss.item():.4f}")
        if batch_idx+1 >= val_steps:
            break
    avg_val_loss = np.mean(val_losses)
    y_trues = np.concatenate(y_trues)
    y_scores = np.concatenate(y_scores)
    val_auc = roc_auc_score(y_trues, y_scores)

    print(f"[DEBUG] Epoch {epoch} summary: Train loss {avg_train_loss:.4f}, Val loss {avg_val_loss:.4f}, Val AUC {val_auc:.4f}")

    # Early stopping
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'best_selector.pt')
        print(f"[DEBUG] New best model saved with AUC {val_auc:.4f}")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("[DEBUG] Early stopping triggered.")
            break

# ---------------------------------------
# Save final model
# ---------------------------------------
model.load_state_dict(torch.load('best_selector.pt'))
torch.save(model.state_dict(), 'selector_mlp.pth')
print("Done—model saved to selector_mlp.pth")
