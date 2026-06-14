"""
Variant 20 — LSTM-AE: Temporal autoencoder over per-timestep aggregate statistics.
Trains on mean feature vectors of the 34 training time steps (sliding windows T=10).
Scores each test transaction by the reconstruction error of its time step.
Reference: Malhotra et al., ICML Anomaly Detection Workshop 2016; course Module 4.
"""
import sys
sys.path.insert(0, "..")

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_all
from src.preprocessing import get_labeled, get_feature_cols, TRAIN_STEPS, TEST_STEPS
from src.lstm_ae import build_timestep_sequences, train_lstm_ae, score_timesteps
from src.autoencoder import f1_optimal_threshold, evt_threshold
from src.evaluation import evaluate as eval_metrics

WINDOW = 10

print("Loading data...")
df, edges = load_all()

# Exclude time_step from model features — it's the grouping key, not a transaction feature
all_cols    = get_feature_cols(df.drop(columns=["label"], errors="ignore"))
feature_cols = [c for c in all_cols if c != "time_step"]
print(f"Feature dims: {len(feature_cols)}  (excluded time_step from scaling)")

# Scale on training time steps only (prevent leakage)
train_rows = df[df["time_step"].isin(TRAIN_STEPS)]
scaler = StandardScaler().fit(train_rows[feature_cols].values)
df_scaled = df.copy()
df_scaled[feature_cols] = scaler.transform(df[feature_cols].values)
# time_step column is preserved as original integers in df_scaled

print(f"Building timestep sequences (window_size={WINDOW})...")
windows, step_means, step_ids = build_timestep_sequences(df_scaled, feature_cols, window_size=WINDOW)

# Only windows that fall entirely within training steps (steps 1-34)
n_train_steps = sum(1 for t in step_ids if t in TRAIN_STEPS)
train_windows = windows[: max(1, n_train_steps - WINDOW + 1)]
print(f"Total steps: {len(step_ids)}  Train steps: {n_train_steps}")
print(f"Train windows: {len(train_windows)}  shape: {train_windows.shape}")

input_dim = train_windows.shape[2]
print(f"\nTraining LSTM-AE (input_dim={input_dim}, hidden=128, 2 layers, epochs=300)")
print("-" * 60)
model = train_lstm_ae(
    train_windows,
    input_dim=input_dim,
    hidden_dim=128,
    num_layers=2,
    dropout=0.2,
    epochs=300,
    batch_size=16,
    lr=1e-3,
)

# ── score all time steps ───────────────────────────────────────────────────────
print("\nScoring all time steps...")
ts_scores    = score_timesteps(model, step_means, window_size=WINDOW)
step_score_map = {t: ts_scores[i] for i, t in enumerate(step_ids)}

print("Top-5 most anomalous time steps:")
for t in sorted(step_score_map, key=step_score_map.get, reverse=True)[:5]:
    print(f"  step {t:2d}  score={step_score_map[t]:.6f}")

# ── map scores to transactions ─────────────────────────────────────────────────
labeled = get_labeled(df)

train_labeled = labeled[labeled["time_step"].isin(TRAIN_STEPS)].copy()
y_train       = train_labeled["label"].values
train_scores  = train_labeled["time_step"].map(step_score_map).values

test_labeled = labeled[labeled["time_step"].isin(TEST_STEPS)].copy()
y_test       = test_labeled["label"].values
test_scores  = test_labeled["time_step"].map(step_score_map).values

# ── threshold options ──────────────────────────────────────────────────────────
t_f1  = f1_optimal_threshold(train_scores, y_train)
t_evt = evt_threshold(train_scores, tail_quantile=0.90, exceedance_prob=0.065)

for tag, t in [("@F1opt", t_f1), ("@EVT", t_evt)]:
    y_pred = (test_scores >= t).astype(int)
    res = eval_metrics(y_test, y_pred, y_score=test_scores, name=f"LSTM-AE{tag}")
    print(f"  threshold={t:.4f}  F1={res['f1']:.4f}")

torch.save(model.state_dict(), "../models/lstm_ae.pt")
print("\nSaved: models/lstm_ae.pt")
