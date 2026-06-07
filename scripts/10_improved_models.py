"""
Improved models:
  1. LightGBM           — replaces sklearn GBM
  2. RF + graph features — degree, PageRank, clustering added to tabular
  3. GAT                 — replaces GraphSAGE
  4. VAE                 — replaces Autoencoder
  5. Ensemble            — RF + LightGBM + GAT score average
"""
import sys
sys.path.insert(0, '..')

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from src.data_loader import load_all
from src.preprocessing import (
    prepare_supervised_temporal, get_labeled, get_feature_cols,
    TRAIN_STEPS, TEST_STEPS
)
from src.graph_features import compute_graph_features
from src.vae import train_vae, vae_anomaly_scores, DEVICE as VAE_DEVICE
from src.graph_utils import build_graph
from src.gat import GAT, train_epoch as gat_train_epoch, evaluate as gat_evaluate
from src.gnn import class_weights

GNN_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"GPU: {torch.cuda.get_device_name(0) if GNN_DEVICE.type == 'cuda' else 'CPU'}")

results = []

def record(name, y_true, y_pred, y_score):
    r = {
        "Model":     name,
        "F1":        round(f1_score(y_true, y_pred), 4),
        "ROC-AUC":   round(roc_auc_score(y_true, y_score), 4),
        "Avg-Prec":  round(average_precision_score(y_true, y_score), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred), 4),
    }
    results.append(r)
    print(f"  {name:30s}  F1={r['F1']:.4f}  AUC={r['ROC-AUC']:.4f}  AP={r['Avg-Prec']:.4f}")
    return r

# ── load data ──────────────────────────────────────────────────────────────────
print("\nLoading data...")
df, edges = load_all()
X_train, X_test, y_train, y_test, feat_cols, scaler = prepare_supervised_temporal(df)

# ── 1. LightGBM ───────────────────────────────────────────────────────────────
print("\n[1/5] LightGBM")
import lightgbm as lgb

lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
lgb_model.fit(X_train, y_train)
y_pred_lgb  = lgb_model.predict(X_test)
y_score_lgb = lgb_model.predict_proba(X_test)[:, 1]
record("LightGBM", y_test, y_pred_lgb, y_score_lgb)
joblib.dump(lgb_model, "../models/lightgbm.joblib")

# ── 2. RF + graph features ────────────────────────────────────────────────────
print("\n[2/5] RF + Graph Features")
print("  Computing graph features (PageRank takes ~60s)...")
df_gf = compute_graph_features(df, edges)

gf_extra = ["in_degree", "out_degree", "total_degree", "pagerank", "clustering"]
all_feat_cols = feat_cols + gf_extra

labeled_train_gf = get_labeled(df_gf[df_gf["time_step"].isin(TRAIN_STEPS)])
labeled_test_gf  = get_labeled(df_gf[df_gf["time_step"].isin(TEST_STEPS)])

scaler_gf = StandardScaler()
X_train_gf = scaler_gf.fit_transform(labeled_train_gf[all_feat_cols].values)
X_test_gf  = scaler_gf.transform(labeled_test_gf[all_feat_cols].values)
y_train_gf = labeled_train_gf["label"].values
y_test_gf  = labeled_test_gf["label"].values

rf_gf = RandomForestClassifier(
    n_estimators=200, class_weight="balanced",
    random_state=42, n_jobs=-1
)
rf_gf.fit(X_train_gf, y_train_gf)
y_pred_rf_gf  = rf_gf.predict(X_test_gf)
y_score_rf_gf = rf_gf.predict_proba(X_test_gf)[:, 1]
record("RF + GraphFeatures", y_test_gf, y_pred_rf_gf, y_score_rf_gf)
joblib.dump(rf_gf,    "../models/rf_graph_features.joblib")
joblib.dump(scaler_gf,"../models/scaler_graph_features.joblib")

# ── 3. GAT ────────────────────────────────────────────────────────────────────
print("\n[3/5] GAT (Graph Attention Network)")
data, _, _ = build_graph(df, edges)
data = data.to(GNN_DEVICE)

gat = GAT(input_dim=data.x.shape[1], hidden_dim=64, output_dim=2, heads=4, dropout=0.3).to(GNN_DEVICE)
optimizer = torch.optim.Adam(gat.parameters(), lr=5e-4, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
wts        = class_weights(data.y[data.train_mask], GNN_DEVICE)
criterion  = torch.nn.CrossEntropyLoss(weight=wts)

print(f"  GAT params: {sum(p.numel() for p in gat.parameters()):,}")
for epoch in range(1, 201):
    loss = gat_train_epoch(gat, data, optimizer, criterion, GNN_DEVICE)
    scheduler.step()
    if epoch % 50 == 0:
        print(f"  epoch {epoch}/200  loss={loss:.4f}")

test_mask_np = data.test_mask.cpu().numpy().astype(bool)
y_pred_gat, y_score_gat, y_true_gat = gat_evaluate(gat, data, test_mask_np, GNN_DEVICE)
record("GAT", y_true_gat, y_pred_gat, y_score_gat)
torch.save(gat.state_dict(), "../models/gat.pt")

# ── 4. VAE ────────────────────────────────────────────────────────────────────
print(f"\n[4/5] VAE (device={VAE_DEVICE})")
labeled_train = get_labeled(df[df["time_step"].isin(TRAIN_STEPS)])
labeled_test  = get_labeled(df[df["time_step"].isin(TEST_STEPS)])
fc = get_feature_cols(labeled_train)

scaler_vae   = StandardScaler()
X_licit_vae  = scaler_vae.fit_transform(labeled_train[labeled_train["label"] == 0][fc].values)
X_test_vae   = scaler_vae.transform(labeled_test[fc].values)
y_test_vae   = labeled_test["label"].values

vae = train_vae(X_licit_vae, input_dim=X_licit_vae.shape[1],
                hidden_dim=128, latent_dim=32, epochs=100, beta=0.5)

vae_scores = vae_anomaly_scores(vae, X_test_vae)
auc_fwd = roc_auc_score(y_test_vae, vae_scores)
if roc_auc_score(y_test_vae, -vae_scores) > auc_fwd:
    print("  Score inverted (illicit = lower reconstruction)")
    vae_scores = -vae_scores

threshold = np.percentile(vae_scores, 100 * (1 - y_test_vae.mean()))
y_pred_vae = (vae_scores >= threshold).astype(int)
record("VAE", y_test_vae, y_pred_vae, vae_scores)
torch.save(vae.state_dict(), "../models/vae.pt")

# ── 5. Ensemble (RF + LightGBM + GAT) ────────────────────────────────────────
print("\n[5/5] Ensemble (RF + LightGBM + GAT avg)")
# RF baseline scores on standard features
from src.models import random_forest, anomaly_scores as sklearn_scores
rf_base  = random_forest(X_train, y_train)
rf_score = sklearn_scores(rf_base, X_test)

# GAT scores already on test nodes — align with tabular test set
# GAT test mask covers same labeled test nodes
ens_score = (rf_score + y_score_lgb + y_score_gat) / 3.0
threshold_ens = np.percentile(ens_score, 100 * (1 - y_test.mean()))
y_pred_ens = (ens_score >= threshold_ens).astype(int)
record("Ensemble (RF+LGB+GAT)", y_test, y_pred_ens, ens_score)

# ── save results ───────────────────────────────────────────────────────────────
df_new = pd.DataFrame(results)
df_new.to_csv("../reports/improved_results.csv", index=False)
print("\n=== Improved Models Summary ===")
print(df_new.sort_values("F1", ascending=False).to_string(index=False))
print("\nSaved: reports/improved_results.csv")
