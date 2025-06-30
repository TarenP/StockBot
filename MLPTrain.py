import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1) Load your master DataFrame
master_df = pd.read_parquet('Master.parquet')
if 'date' in master_df.columns and 'ticker' in master_df.columns:
    master_df = master_df.set_index(['date', 'ticker'])

# 2) Define return and feature columns
return_col = 'ret_5d'
feature_cols = [c for c in master_df.columns if c not in (return_col, 'label')]

# 3) Label top 5% tickers per day
def label_top_percentile(df, return_col, percentile=0.95):
    df = df.copy()
    labels = pd.Series(0, index=df.index)
    for date, group in df.groupby(level='date'):
        cutoff = group[return_col].quantile(percentile)
        mask = group[return_col] >= cutoff
        labels.loc[mask.index[mask]] = 1
    df['label'] = labels
    return df

labeled_df = label_top_percentile(master_df, return_col)

# 4) Split into train/validation
val_frac = 0.2
n = len(labeled_df)
indices = np.arange(n)
np.random.shuffle(indices)
val_size = int(n * val_frac)
val_idx = indices[:val_size]
train_idx = indices[val_size:]

train_df = labeled_df.iloc[train_idx]
val_df   = labeled_df.iloc[val_idx]

# 5) Optional: standardize features using training scaler
scaler = StandardScaler()
train_df_features = scaler.fit_transform(train_df[feature_cols])
val_df_features   = scaler.transform(val_df[feature_cols])
train_df.loc[:, feature_cols] = train_df_features
val_df.loc[:, feature_cols]   = val_df_features

# 6) Define batch generator
def batch_generator(df, feature_cols, label_col, batch_size=1024):
    num_rows = len(df)
    while True:
        df_shuffled = df.sample(frac=1)
        for start in range(0, num_rows, batch_size):
            batch = df_shuffled.iloc[start:start + batch_size]
            X_batch = batch[feature_cols].values.astype(np.float32)
            y_batch = batch[label_col].values.astype(np.float32)
            yield X_batch, y_batch

batch_size = 1024
train_steps = len(train_df) // batch_size
val_steps   = len(val_df) // batch_size

# 7) Create tf.data.Datasets
train_dataset = tf.data.Dataset.from_generator(
    lambda: batch_generator(train_df, feature_cols, 'label', batch_size),
    output_signature=(
        tf.TensorSpec(shape=(None, len(feature_cols)), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    )
)
val_dataset = tf.data.Dataset.from_generator(
    lambda: batch_generator(val_df, feature_cols, 'label', batch_size),
    output_signature=(
        tf.TensorSpec(shape=(None, len(feature_cols)), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    )
)

# 8) Build and compile model
input_dim = len(feature_cols)
model = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=[tf.keras.metrics.AUC(name='auc')]
)

# 9) Train using streaming datasets
model.fit(
    train_dataset,
    epochs=10,
    steps_per_epoch=train_steps,
    validation_data=val_dataset,
    validation_steps=val_steps,
    callbacks=[
        callbacks.EarlyStopping(
            monitor='val_auc',
            patience=3,
            restore_best_weights=True
        )
    ],
    verbose=1
)

# 10) Save the trained model
model.save('selector_mlp.keras')
print("Model saved to selector_mlp.keras")
