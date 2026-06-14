"""
Round 2 improvements for GNN + neural models:
  1. GATv2         — 3-layer, heads=2, residual, LayerNorm, dropout=0.5
  2. GraphSAGEv2   — 3-layer, hidden=256, LayerNorm, self-loops
  3. DenoisingAE   — latent=8, noise corruption, deeper encoder, F1-optimal threshold
  4. VAEv2         — beta=0.01, deeper 256-hidden, ELBO score, F1-optimal threshold

Validation set for threshold tuning: labeled steps 30-34 (within train period).
"""
import sys
sys.path.insert(0, '..')

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score
)
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_all
from src.preprocessing import get_labeled, get_feature_cols, TRAIN_STEPS, TEST_STEPS
from src.graph_utils import build_graph
from src.gnn import GraphSAGEv2, train_epoch as sage_train, evaluate as sage_eval, class_weights
from src.gat import GATv2, train_epoch as gat_train, evaluate as gat_eval
from src.autoencoder import (
    train_denoising_autoencoder, reconstruction_errors,
    f1_optimal_threshold as ae_f1_thresh, DEVICE as AE_DEV
)
from src.vae import (
    train_vaev2, elbo_anomaly_scores,
    f1_optimal_threshold as vae_f1_thresh, DEVICE as VAE_DEV
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}  ({torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else 'CPU'})")

# ── load data ──────────────────────────────────────────────────────────────────
print("\nLoading data...")
df, edges = load_all()

labeled      = get_labeled(df)
feat_cols    = get_feature_cols(labeled)
VAL_STEPS    = range(30, 35)   # validation slice for threshold tuning

labeled_train = get_labeled(df[df["time_step"].isin(TRAIN_STEPS)])
labeled_val   = get_labeled(df[df["time_step"].isin(VAL_STEPS)])
labeled_test  = get_labeled(df[df["time_step"].isin(TEST_STEPS)])

# scaler fit on all train labeled (licit only for AE/VAE)
scaler = StandardScaler().fit(labeled_train[labeled_train["label"]==0][feat_cols].values)
X_licit_train = scaler.transform(labeled_train[labeled_train["label"]==0][feat_cols].values)
X_val         = scaler.transform(labeled_val[feat_cols].values)
X_test        = scaler.transform(labeled_test[feat_cols].values)
y_val         = labeled_val["label"].values
y_test        = labeled_test["label"].values

print(f"Train licit: {len(X_licit_train):,}  Val labeled: {len(X_val):,}  Test labeled: {len(X_test):,}")
print(f"Val illicit rate: {y_val.mean():.3f}  Test illicit rate: {y_test.mean():.3f}")

results   = []
prev_rows = {
    "GAT":         {"F1": 0.3592, "ROC-AUC": 0.8469, "Avg-Prec": 0.3891},
    "GraphSAGE":   {"F1": 0.5409, "ROC-AUC": 0.8883, "Avg-Prec": 0.5487},
    "Autoencoder": {"F1": 0.3564, "ROC-AUC": 0.7814, "Avg-Prec": 0.3355},
    "VAE":         {"F1": 0.3398, "ROC-AUC": 0.7780, "Avg-Prec": 0.3098},
}

def record(name, y_true, y_pred, y_score, prev_name=None):
    r = {
        "Model":     name,
        "F1":        round(f1_score(y_true, y_pred), 4),
        "ROC-AUC":   round(roc_auc_score(y_true, y_score), 4),
        "Avg-Prec":  round(average_precision_score(y_true, y_score), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred), 4),
    }
    results.append(r)
    prev = prev_rows.get(prev_name or name, {})
    delta = f"  delta F1={r['F1']-prev.get('F1',r['F1']):+.4f}" if prev else ""
    print(f"  {name:25s}  F1={r['F1']:.4f}  AUC={r['ROC-AUC']:.4f}  AP={r['Avg-Prec']:.4f}{delta}")
    return r

# ── build graph (shared for GNN models) ───────────────────────────────────────
print("\nBuilding graph...")
data, _, _ = build_graph(df, edges)
data = data.to(DEVICE)
input_dim    = data.x.shape[1]
test_mask_np = data.test_mask.cpu().numpy().astype(bool)
wts          = class_weights(data.y[data.train_mask], DEVICE)
criterion    = torch.nn.CrossEntropyLoss(weight=wts)

# ── 1. GATv2 ──────────────────────────────────────────────────────────────────
print("\n[1/4] GATv2 — 3-layer, heads=2, residual, dropout=0.5")
gatv2 = GATv2(input_dim=input_dim, hidden_dim=64, output_dim=2,
              heads=2, dropout=0.5).to(DEVICE)
print(f"  Params: {sum(p.numel() for p in gatv2.parameters()):,}")
opt_gat = torch.optim.Adam(gatv2.parameters(), lr=1e-3, weight_decay=5e-4)
sch_gat = torch.optim.lr_scheduler.CosineAnnealingLR(opt_gat, T_max=300)

for ep in range(1, 301):
    loss = gat_train(gatv2, data, opt_gat, criterion, DEVICE)
    sch_gat.step()
    if ep % 75 == 0:
        print(f"  epoch {ep}/300  loss={loss:.4f}")

y_pred_gat, y_score_gat, y_true_gat = gat_eval(gatv2, data, test_mask_np, DEVICE)
record("GATv2", y_true_gat, y_pred_gat, y_score_gat, prev_name="GAT")
torch.save(gatv2.state_dict(), "../models/gatv2.pt")

# ── 2. GraphSAGEv2 ────────────────────────────────────────────────────────────
print("\n[2/4] GraphSAGEv2 — 3-layer, hidden=256, LayerNorm, self-loops")
sagev2 = GraphSAGEv2(input_dim=input_dim, hidden_dim=256, output_dim=2,
                     dropout=0.3).to(DEVICE)
print(f"  Params: {sum(p.numel() for p in sagev2.parameters()):,}")
opt_sage = torch.optim.Adam(sagev2.parameters(), lr=1e-3, weight_decay=5e-4)
sch_sage = torch.optim.lr_scheduler.CosineAnnealingLR(opt_sage, T_max=250)

for ep in range(1, 251):
    loss = sage_train(sagev2, data, opt_sage, criterion, DEVICE)
    sch_sage.step()
    if ep % 50 == 0:
        print(f"  epoch {ep}/250  loss={loss:.4f}")

y_pred_s2, y_score_s2, y_true_s2 = sage_eval(sagev2, data, test_mask_np, DEVICE)
record("GraphSAGEv2", y_true_s2, y_pred_s2, y_score_s2, prev_name="GraphSAGE")
torch.save(sagev2.state_dict(), "../models/graphsagev2.pt")

# ── 3. Denoising Autoencoder ──────────────────────────────────────────────────
print(f"\n[3/4] DenoisingAE — latent=8, noise=0.1, F1-optimal threshold (device={AE_DEV})")
dae = train_denoising_autoencoder(X_licit_train, input_dim=X_licit_train.shape[1],
                                  latent_dim=8, noise_std=0.1, epochs=120)

dae_errors_val  = reconstruction_errors(dae, X_val)
dae_errors_test = reconstruction_errors(dae, X_test)

# direction detection
if roc_auc_score(y_val, -dae_errors_val) > roc_auc_score(y_val, dae_errors_val):
    print("  Score inverted on val set")
    dae_scores_val  = -dae_errors_val
    dae_scores_test = -dae_errors_test
else:
    dae_scores_val  = dae_errors_val
    dae_scores_test = dae_errors_test

opt_t_dae  = ae_f1_thresh(dae_scores_val, y_val)
y_pred_dae = (dae_scores_test >= opt_t_dae).astype(int)
print(f"  F1-optimal threshold (val): {opt_t_dae:.5f}")
record("DenoisingAE", y_test, y_pred_dae, dae_scores_test, prev_name="Autoencoder")
torch.save(dae.state_dict(), "../models/denoising_ae.pt")

# ── 4. VAEv2 ──────────────────────────────────────────────────────────────────
print(f"\n[4/4] VAEv2 — beta=0.01, hidden=256, latent=16, ELBO score (device={VAE_DEV})")
vaev2 = train_vaev2(X_licit_train, input_dim=X_licit_train.shape[1],
                    hidden_dim=256, latent_dim=16, epochs=120, beta=0.01)

elbo_val  = elbo_anomaly_scores(vaev2, X_val)
elbo_test = elbo_anomaly_scores(vaev2, X_test)

if roc_auc_score(y_val, -elbo_val) > roc_auc_score(y_val, elbo_val):
    print("  Score inverted on val set")
    elbo_val  = -elbo_val
    elbo_test = -elbo_test

opt_t_vae  = vae_f1_thresh(elbo_val, y_val)
y_pred_vae = (elbo_test >= opt_t_vae).astype(int)
print(f"  F1-optimal threshold (val): {opt_t_vae:.5f}")
record("VAEv2", y_test, y_pred_vae, elbo_test, prev_name="VAE")
torch.save(vaev2.state_dict(), "../models/vaev2.pt")

# ── delta table ────────────────────────────────────────────────────────────────
print("\n" + "="*72)
print(f"{'Improved':<20} {'Previous':<20} {'Prev F1':>8} {'New F1':>8} {'Delta':>8} {'Delta AUC':>10}")
print("-"*72)
pairs = [
    ("GATv2",       "GAT"),
    ("GraphSAGEv2", "GraphSAGE"),
    ("DenoisingAE", "Autoencoder"),
    ("VAEv2",       "VAE"),
]
for new_name, old_name in pairs:
    nr = next(r for r in results if r["Model"] == new_name)
    pr = prev_rows[old_name]
    df1 = nr["F1"]   - pr["F1"]
    da  = nr["ROC-AUC"] - pr["ROC-AUC"]
    print(f"{new_name:<20} {old_name:<20} {pr['F1']:>8.4f} {nr['F1']:>8.4f} {df1:>+8.4f} {da:>+10.4f}")
print("="*72)

# ── save results ───────────────────────────────────────────────────────────────
df_res = pd.DataFrame(results)
df_res.to_csv("../reports/deep_improved_results.csv", index=False)

# ── comparison plot ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

metrics = ["F1", "ROC-AUC", "Avg-Prec"]
x  = np.arange(len(pairs))
w  = 0.28

for ax, metric in zip(axes, ["F1", "ROC-AUC"]):
    prev_vals = [prev_rows[old][metric] for _, old in pairs]
    new_vals  = [next(r for r in results if r["Model"]==new)[metric] for new, _ in pairs]
    labels    = [f"{new}\nvs {old}" for new, old in pairs]

    b1 = ax.bar(x - w/2, prev_vals, w, label="Previous", color="#95a5a6", edgecolor="white")
    b2 = ax.bar(x + w/2, new_vals,  w, label="Improved", color="#e74c3c", edgecolor="white")
    for bar, v in zip(b2, new_vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f"{v:.3f}", ha="center", fontsize=8.5, fontweight="bold")
    for bar, v in zip(b1, prev_vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f"{v:.3f}", ha="center", fontsize=8)

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 1.08); ax.set_ylabel(metric)
    ax.set_title(f"{metric}: Previous vs Improved")
    ax.legend()

plt.suptitle("Round 2 Improvements — GNN & Neural Models\n"
             "Test set: steps 35-49  |  Threshold tuned on val steps 30-34",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("../reports/deep_improvement_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved: reports/deep_improvement_comparison.png")
print("Saved: reports/deep_improved_results.csv")
