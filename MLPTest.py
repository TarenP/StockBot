import os
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from torch.utils.data import TensorDataset, DataLoader
from torch.serialization import safe_globals


# Temp
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

# ---------------------------------------
# Configuration
# ---------------------------------------
parquet_path   = 'Master_cleaned.parquet'
return_col     = 'ret_5d'
batch_size     = 1_000_000   # streaming size for Parquet
test_fraction  = 0.2         # last 20% of dates for test
device         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# ---------------------------------------
# Build X_test, y_test (same as before)
# ---------------------------------------
pf = pq.ParquetFile(parquet_path)
all_cols = pf.schema_arrow.names
exclude  = {return_col, 'date', 'ticker'}
feature_cols = [c for c in all_cols if c not in exclude]

cutoff_lists, all_dates = {}, set()
for batch in pf.iter_batches(batch_size=batch_size):
    dfb = batch.to_pandas()[['date', return_col]]
    for date, grp in dfb.groupby('date'):
        all_dates.add(date)
        cutoff_lists.setdefault(date, []).extend(grp[return_col].values)
cutoff_map = {d: np.quantile(vals, 0.95) for d, vals in cutoff_lists.items()}
all_dates = sorted(all_dates)

# split_idx  = int(len(all_dates) * (1 - test_fraction))
# test_dates = set(all_dates[split_idx:])
split_idx = next(i for i, d in enumerate(all_dates) if str(d) >= '2025-06-15')
test_dates = set(all_dates[split_idx:])

sample = pf.read_row_group(0).to_pandas()[feature_cols]
scaler = StandardScaler().fit(sample.values)

chunks = []
for batch in pf.iter_batches(batch_size=batch_size):
    df = batch.to_pandas()
    df = df[df['date'].isin(test_dates)]
    if not df.empty:
        chunks.append(df)
test_df = pd.concat(chunks, ignore_index=True)

test_df['label'] = (test_df[return_col] >= test_df['date'].map(cutoff_map)).astype(np.int8)
mask = np.isfinite(test_df[feature_cols].values).all(axis=1)
test_df = test_df.loc[mask].reset_index(drop=True)

X_test = scaler.transform(test_df[feature_cols].values.astype(np.float32))
y_test = test_df['label'].values.astype(np.int8)
print("X_test shape:", X_test.shape, "y_test dist:", np.bincount(y_test))

# ---------------------------------------
# Load the TorchScript model
# ---------------------------------------
with safe_globals([SelectorMLP]):
    scripted_model = torch.load(
        'selector_mlp_model.pt',
        map_location=device,
        weights_only=False
    )
scripted_model.to(device).eval()

# ---------------------------------------
# Batched inference & metrics
# ---------------------------------------
test_ds     = TensorDataset(torch.from_numpy(X_test).float(),
                            torch.from_numpy(y_test).long())
test_loader = DataLoader(test_ds, batch_size=2048, shuffle=False, num_workers=0)

all_probs, all_preds, all_labels = [], [], []
with torch.no_grad():
    for Xb, yb in test_loader:
        Xb = Xb.to(device)
        logits = scripted_model(Xb)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.5).astype(np.int8)

        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(yb.numpy())

all_probs  = np.concatenate(all_probs)
all_preds  = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

auc      = roc_auc_score(all_labels, all_probs)
accuracy = accuracy_score(all_labels, all_preds)

print(f"\nTest AUC:      {auc:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n",
      classification_report(all_labels, all_preds, digits=4))


test_df['prob'] = all_probs
test_df['pred'] = (all_probs >= 0.5).astype(np.int8)
selected = test_df[test_df['pred'] == 1]
print(selected[['date', 'ticker', 'prob']].head())