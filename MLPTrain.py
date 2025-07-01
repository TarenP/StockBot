import os
import pyarrow.parquet as pq
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm

# ---------------------------------------
# Configuration
# ---------------------------------------
parquet_path = 'Master_cleaned.parquet'
return_col   = 'ret_5d'
batch_size   = 2048
n_epochs     = 10
patience     = 3
device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ---------------------------------------
# Prepare ParquetFile and feature columns
# ---------------------------------------
pf = pq.ParquetFile(parquet_path)
all_cols = pf.schema_arrow.names
exclude  = {return_col, 'date', 'ticker'}
feature_cols = [c for c in all_cols if c not in exclude]

# ---------------------------------------
# Precompute date-wise cutoff map
# ---------------------------------------
cutoff_lists, all_dates = {}, set()
for batch in pf.iter_batches(batch_size=1_000_000):
    dfb = batch.to_pandas()[['date', return_col]]
    for date, grp in dfb.groupby('date'):
        all_dates.add(date)
        cutoff_lists.setdefault(date, []).extend(grp[return_col].values)
cutoff_map = {date: np.quantile(vals, 0.95) for date, vals in cutoff_lists.items()}
all_dates = sorted(all_dates)
split_idx = int(len(all_dates) * 0.8)
train_dates, val_dates = set(all_dates[:split_idx]), set(all_dates[split_idx:])

# ---------------------------------------
# Fit scaler on a sample row group
# ---------------------------------------
sample = pf.read_row_group(0).to_pandas()[feature_cols]
scaler = StandardScaler().fit(sample.values)

# ---------------------------------------
# Define IterableDataset for streaming
# ---------------------------------------
class ParquetDataset(IterableDataset):
    def __init__(self, dates_subset):
        self.dates_subset = dates_subset

    def __iter__(self):
        local_pf = pq.ParquetFile(parquet_path)
        for batch in local_pf.iter_batches(batch_size=batch_size):
            df = batch.to_pandas()
            df = df[df['date'].isin(self.dates_subset)]
            if df.empty:
                continue
            df['label'] = (df[return_col] >= df['date'].map(cutoff_map)).astype(np.int8)
            X_raw = df[feature_cols].values.astype(np.float32)
            y_raw = df['label'].values.astype(np.float32)
            mask = np.isfinite(X_raw).all(axis=1)
            if not mask.any():
                continue
            X = scaler.transform(X_raw[mask])
            y = y_raw[mask]
            for i in range(0, len(X), batch_size):
                xb = torch.from_numpy(X[i:i+batch_size]).to(device)
                yb = torch.from_numpy(y[i:i+batch_size]).to(device)
                yield xb, yb

# ---------------------------------------
# Create DataLoaders without multiprocessing
# ---------------------------------------
train_ds = ParquetDataset(train_dates)
val_ds   = ParquetDataset(val_dates)
train_loader = DataLoader(train_ds, batch_size=None, num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=None, num_workers=0)

# ---------------------------------------
# Define the MLP model
# ---------------------------------------
class SelectorMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),        nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

model = SelectorMLP(len(feature_cols)).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Estimate steps per epoch
total_rows   = sum(pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups))
train_steps  = int(total_rows * 0.8) // batch_size
val_steps    = int(total_rows * 0.2) // batch_size

# ---------------------------------------
# Training loop with tqdm progress bars
# ---------------------------------------
best_val_auc, epochs_no_improve = 0.0, 0

for epoch in range(1, n_epochs + 1):
    # Training
    model.train()
    train_loss = 0.0
    pbar_train = tqdm(train_loader, total=train_steps, desc=f"Epoch {epoch}/{n_epochs} Train", dynamic_ncols=True)
    for batch_idx, (X_batch, y_batch) in enumerate(pbar_train, 1):
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pbar_train.set_postfix(loss=train_loss / batch_idx)
        
    avg_train_loss = train_loss / train_steps

    # Validation
    model.eval()
    val_loss = 0.0
    y_trues, y_scores = [], []
    pbar_val = tqdm(val_loader, total=val_steps, desc=f"Epoch {epoch}/{n_epochs} Val", dynamic_ncols=True)
    with torch.no_grad():
        for batch_idx, (X_batch, y_batch) in enumerate(pbar_val, 1):
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            val_loss += loss.item()
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            y_scores.append(probs)
            y_trues.append(y_batch.cpu().numpy())
            pbar_val.set_postfix(loss=val_loss / batch_idx)
            
    avg_val_loss = val_loss / val_steps
    y_trues = np.concatenate(y_trues)
    y_scores = np.concatenate(y_scores)
    val_auc = roc_auc_score(y_trues, y_scores)

    print(f"Epoch {epoch}/{n_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val AUC: {val_auc:.4f}")

    # Early stopping
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'best_selector_state_dict.pt')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

# ---------------------------------------
# Save final model and state dict
# ---------------------------------------
# Save only the state dict
torch.jit.script(model.state_dict(), 'selector_mlp_state_dict.pt')
# Save entire model object for easy loading
torch.jit.script(model, 'selector_mlp_model.pt')

print("Training complete. Saved:")
print(" - State dict -> selector_mlp_state_dict.pt")
print(" - Full model -> selector_mlp_model.pt")
