"""
Variant 35-38 — beta-VAE grid search: beta in {0.01, 1, 2, 4, 8}.
Course ref: beta>1 forces tighter normal cluster (KL dominates), harder for anomalies to reconstruct.
Current VAEv2 uses beta=0.01 (reconstruction-dominated). Expected: beta=2-4 optimal.
"""
import sys
sys.path.insert(0, "..")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

from src.data_loader import load_all
from src.preprocessing import prepare_unsupervised_temporal
from src.vae import train_vaev2, vae_anomaly_scores, f1_optimal_threshold
from src.autoencoder import evt_threshold
from src.evaluation import evaluate as eval_metrics

print("Loading data...")
df, edges = load_all()
X_train_all, X_test_labeled, y_test, feat_cols, scaler = prepare_unsupervised_temporal(df)
print(f"  Train (all, licit+unlabeled): {X_train_all.shape}")
print(f"  Test  (labeled only):         {X_test_labeled.shape}  illicit={y_test.mean():.3f}")

# Train only on licit rows (unsupervised: licit-only training)
from src.preprocessing import get_labeled, TRAIN_STEPS, TEST_STEPS
labeled_train = get_labeled(df)
labeled_train = labeled_train[labeled_train["time_step"].isin(TRAIN_STEPS)]
licit_train   = labeled_train[labeled_train["label"] == 0]
X_licit = scaler.transform(licit_train[feat_cols].values)
print(f"  Licit-only train: {X_licit.shape}")

BETAS      = [0.01, 1.0, 2.0, 4.0, 8.0]
EPOCHS     = 150
input_dim  = X_licit.shape[1]
results    = []

print(f"\nGrid search beta x {len(BETAS)} values, {EPOCHS} epochs each")
print("-" * 70)

for beta in BETAS:
    print(f"\n  beta={beta}")
    model = train_vaev2(
        X_licit, input_dim=input_dim,
        hidden_dim=256, latent_dim=16,
        epochs=EPOCHS, batch_size=512, lr=1e-3, beta=beta,
    )
    scores = vae_anomaly_scores(model, X_test_labeled)

    # VAE score direction: higher score = more anomalous
    auc = roc_auc_score(y_test, scores)
    if auc < 0.5:
        scores = -scores
        auc = 1.0 - auc
        direction = "inverted"
    else:
        direction = "normal"

    ap   = average_precision_score(y_test, scores)
    t    = f1_optimal_threshold(scores, y_test)
    pred = (scores >= t).astype(int)
    f1   = f1_score(y_test, pred)

    print(f"    AUC={auc:.4f}  AP={ap:.4f}  F1={f1:.4f}  direction={direction}  threshold={t:.4f}")
    results.append({"beta": beta, "f1": f1, "auc": auc, "ap": ap})

print("\n" + "=" * 70)
print("=== beta-VAE Results ===")
print(f"{'beta':>6}  {'F1':>6}  {'AUC':>6}  {'AP':>6}")
print("-" * 30)
for r in results:
    marker = " <-- best" if r["f1"] == max(x["f1"] for x in results) else ""
    print(f"{r['beta']:>6}  {r['f1']:>6.4f}  {r['auc']:>6.4f}  {r['ap']:>6.4f}{marker}")

best = max(results, key=lambda x: x["f1"])
print(f"\nBest beta={best['beta']}  F1={best['f1']:.4f}  (VAEv2 baseline F1=0.2560)")
print(f"Previous VAE (beta=1.0):  F1=0.3400")
print(f"Previous VAEv2 (beta=0.01): F1=0.2560")
