#!/usr/bin/env python
# test_topselector_lstm.py - Simplified version using training dataset class
# -----------------------------------------------------------------------------
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    roc_auc_score, classification_report
)
from tqdm.auto import tqdm
import numpy as np

# Import everything from training script to ensure consistency
from Train import (
    LSTMClassifier,
    ParquetSequenceDataset,
    TEST_DATES,  # Use the same test dates definition
    FEATURE_COLS,
    RETURN_COL,
    BATCH_SIZE,
    WINDOW,
    DEVICE
)

# Configuration - can override training params if needed
MODEL_PATH = "models/TopSelector_epoch4.pt"
NUM_WORKERS = max(1, min(6, os.cpu_count() - 2))

def evaluate():
    print(f"Loading model on {DEVICE}...")
    model = LSTMClassifier(len(FEATURE_COLS)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # Use the EXACT SAME dataset class as training (just with pos_ratio=None)
    test_ds = ParquetSequenceDataset(TEST_DATES, pos_ratio=None)
    test_loader = DataLoader(
        test_ds,
        batch_size=None,  # Let dataset handle batching
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=bool(NUM_WORKERS)
    )

    y_true, y_prob = [], []
    with torch.no_grad():
        for xb, yb, _ in tqdm(test_loader, desc="Evaluating", unit="batch"):
            xb = xb.to(DEVICE)
            probs = torch.sigmoid(model(xb)).cpu().numpy()
            y_true.extend(yb.numpy().tolist())
            y_prob.extend(probs.tolist())

    y_true = np.array(y_true, dtype=int)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    print("\nEvaluation Metrics:")
    print("Accuracy       :", accuracy_score(y_true, y_pred))
    print("Balanced Acc   :", balanced_accuracy_score(y_true, y_pred))
    print("AUROC          :", roc_auc_score(y_true, y_prob))
    print(classification_report(y_true, y_pred, digits=3))

if __name__ == "__main__":
    evaluate()