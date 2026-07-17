"""
Variant — LSTM-AE per-transaction scoring. Closes the "LSTM-AE per-transaction
variant" gap from HANDOFF.md.

Baseline (lstm_autoencoder.py) scores every transaction in a time step with the SAME
step-level reconstruction error (AUC=0.446 — cannot discriminate individual
illicit vs licit within a step). This variant reuses the identical trained LSTM-AE
but scores each transaction against its own step's reconstructed feature template
(src.lstm_ae.score_transactions_per_tx) — a real per-transaction signal.
"""
import sys
sys.path.insert(0, "..")

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_all
from src.preprocessing import get_labeled, get_feature_cols, TRAIN_STEPS, TEST_STEPS
from src.lstm_ae import build_timestep_sequences, train_lstm_ae, score_transactions_per_tx
from src.autoencoder import f1_optimal_threshold, evt_threshold
from src.evaluation import evaluate as eval_metrics

WINDOW = 10

print("Loading data...")
df, edges = load_all()

all_cols = get_feature_cols(df.drop(columns=["label"], errors="ignore"))
feature_cols = [c for c in all_cols if c != "time_step"]
print(f"Feature dims: {len(feature_cols)}  (excluded time_step from scaling)")

train_rows = df[df["time_step"].isin(TRAIN_STEPS)]
scaler = StandardScaler().fit(train_rows[feature_cols].values)
df_scaled = df.copy()
df_scaled[feature_cols] = scaler.transform(df[feature_cols].values)

print(f"Building timestep sequences (window_size={WINDOW})...")
windows, step_means, step_ids = build_timestep_sequences(df_scaled, feature_cols, window_size=WINDOW)

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

# ── per-transaction scores (uses ALL steps' windows for reconstruction templates,
#    same as baseline's use of step_means for scoring; only train windows for fitting) ──
print("\nScoring individual transactions against their step's reconstruction template...")
tx_scores_all = score_transactions_per_tx(model, df_scaled, feature_cols, step_means, step_ids, window_size=WINDOW)
df_scaled["_pertx_score"] = tx_scores_all

labeled = get_labeled(df_scaled)
train_labeled = labeled[labeled["time_step"].isin(TRAIN_STEPS)]
test_labeled  = labeled[labeled["time_step"].isin(TEST_STEPS)]

y_train = train_labeled["label"].values
y_test  = test_labeled["label"].values
train_scores = train_labeled["_pertx_score"].values
test_scores  = test_labeled["_pertx_score"].values

t_f1  = f1_optimal_threshold(train_scores, y_train)
t_evt = evt_threshold(train_scores, tail_quantile=0.90, exceedance_prob=0.065)

print("\n=== Per-Transaction LSTM-AE Results ===")
for tag, t in [("@F1opt", t_f1), ("@EVT", t_evt)]:
    y_pred = (test_scores >= t).astype(int)
    res = eval_metrics(y_test, y_pred, y_score=test_scores, name=f"LSTM-AE-PerTx{tag}")
    print(f"  threshold={t:.4f}  F1={res['f1']:.4f}  AUC={res['roc_auc']:.4f}  AP={res['ap']:.4f}")

print("\nBaseline (step-level, all txs in a step share one score): F1=0.122  AUC=0.446")

torch.save(model.state_dict(), "../models/lstm_ae_pertx.pt")
print("\nSaved: models/lstm_ae_pertx.pt")
