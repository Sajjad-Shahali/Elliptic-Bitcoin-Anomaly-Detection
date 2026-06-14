"""
Variant 21 — Ensemble: soft-vote of GBM (Optuna) + LightGBM + GraphSAGEv2.
Loads all three saved models from models/. Aligns predictions via txId.
Expected best: F1 > 0.824 (current GBM solo).
"""
import sys
sys.path.insert(0, "..")

import numpy as np
import torch
import joblib
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_all
from src.preprocessing import get_labeled, get_feature_cols, TRAIN_STEPS, TEST_STEPS
from src.graph_utils import build_graph
from src.gnn import GraphSAGEv2, evaluate as gnn_evaluate
from src.autoencoder import f1_optimal_threshold
from src.evaluation import evaluate as eval_metrics, plot_confusion_matrix, plot_pr_curve

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── load and split data ────────────────────────────────────────────────────────
print("Loading data...")
df, edges = load_all()
labeled   = get_labeled(df)
feat_cols = get_feature_cols(labeled)

# Steps 30-34 = held-out validation for threshold calibration
# (avoids threshold collapse when calibrating on supervised model train probs)
VAL_STEPS   = list(range(30, 35))
PURE_TRAIN  = [s for s in TRAIN_STEPS if s not in VAL_STEPS]

train_df = labeled[labeled["time_step"].isin(TRAIN_STEPS)].reset_index(drop=True)
val_df   = labeled[labeled["time_step"].isin(VAL_STEPS)].reset_index(drop=True)
test_df  = labeled[labeled["time_step"].isin(TEST_STEPS)].reset_index(drop=True)
y_train  = train_df["label"].values
y_val    = val_df["label"].values
y_test   = test_df["label"].values

print(f"  Train: {len(train_df):,} labeled  Val (steps 30-34): {len(val_df):,}  Test: {len(test_df):,}")
print(f"  Illicit rate — train={y_train.mean():.3f}  val={y_val.mean():.3f}  test={y_test.mean():.3f}")

# Scale on train only
scaler = StandardScaler().fit(train_df[feat_cols].values)
X_train_sc = scaler.transform(train_df[feat_cols].values)
X_val_sc   = scaler.transform(val_df[feat_cols].values)
X_test_sc  = scaler.transform(test_df[feat_cols].values)

# ── tabular model probabilities ────────────────────────────────────────────────
print("-" * 60)
print("Loading GBM and LightGBM...")
gbm  = joblib.load("../models/gb_tuned.joblib")
lgbm = joblib.load("../models/lightgbm.joblib")
print(f"  GBM loaded   ({type(gbm).__name__})")
print(f"  LGBM loaded  ({type(lgbm).__name__})")

print("Scoring val + test sets with GBM / LightGBM...")
gbm_val    = gbm.predict_proba(X_val_sc)[:, 1]
lgbm_val   = lgbm.predict_proba(X_val_sc)[:, 1]
gbm_test   = gbm.predict_proba(X_test_sc)[:, 1]
lgbm_test  = lgbm.predict_proba(X_test_sc)[:, 1]
print(f"  GBM  val  illicit-prob mean={gbm_val.mean():.4f}  max={gbm_val.max():.4f}")
print(f"  LGBM val  illicit-prob mean={lgbm_val.mean():.4f}  max={lgbm_val.max():.4f}")
print(f"  GBM  solo test F1={f1_score(y_test, gbm.predict(X_test_sc)):.4f}")
print(f"  LGBM solo test F1={f1_score(y_test, lgbm.predict(X_test_sc)):.4f}")

# ── GraphSAGEv2 probabilities (aligned via txId) ───────────────────────────────
print("-" * 60)
print("Loading GraphSAGEv2 and building graph...")
data, node_ids, _ = build_graph(df, edges)
data = data.to(DEVICE)
input_dim = data.x.shape[1]
print(f"  Graph: {data.num_nodes:,} nodes  {data.num_edges:,} edges  input_dim={input_dim}")

sage = GraphSAGEv2(input_dim=input_dim, hidden_dim=256, output_dim=2, dropout=0.3).to(DEVICE)
sage.load_state_dict(torch.load("../models/graphsagev2.pt", map_location=DEVICE))
print("  GraphSAGEv2 weights loaded from models/graphsagev2.pt")

train_mask_np = data.train_mask.cpu().numpy().astype(bool)
test_mask_np  = data.test_mask.cpu().numpy().astype(bool)
print(f"  Running inference: train={train_mask_np.sum():,} nodes  test={test_mask_np.sum():,} nodes")

_, sage_train_raw, sage_y_train = gnn_evaluate(sage, data, train_mask_np, DEVICE)
_, sage_test_raw,  sage_y_test  = gnn_evaluate(sage, data, test_mask_np,  DEVICE)
print(f"  SAGEv2 solo test F1={f1_score(sage_y_test, (sage_test_raw >= 0.5).astype(int)):.4f}")
print(f"  SAGEv2 test illicit-prob mean={sage_test_raw.mean():.4f}  max={sage_test_raw.max():.4f}")

# Build txId → prob lookup then align to DataFrame row order
sage_train_map = dict(zip(node_ids[train_mask_np], sage_train_raw))
sage_test_map  = dict(zip(node_ids[test_mask_np],  sage_test_raw))

# Val nodes: steps 30-34 which are inside train_mask
sage_val   = np.array([sage_train_map.get(t, gbm_val[i])   for i, t in enumerate(val_df["txId"].values)])
sage_train = np.array([sage_train_map.get(t, 0.5)          for t in train_df["txId"].values])
sage_test  = np.array([sage_test_map.get(t,  0.5)          for t in test_df["txId"].values])
missing = sum(1 for t in test_df["txId"].values if t not in sage_test_map)
if missing:
    print(f"  WARNING: {missing} test txIds not in graph — using prob=0.5 fallback")

# ── soft voting ────────────────────────────────────────────────────────────────
print("-" * 60)
print("Computing ensemble (equal weights: 1/3 each)...")
ens_val  = (gbm_val  + lgbm_val  + sage_val)  / 3.0
ens_test = (gbm_test + lgbm_test + sage_test) / 3.0
print(f"  Ensemble val  prob mean={ens_val.mean():.4f}  max={ens_val.max():.4f}")
print(f"  Ensemble test prob mean={ens_test.mean():.4f}  max={ens_test.max():.4f}")

# Threshold calibrated on validation set (steps 30-34) — avoids overfitting to train probs
print("Finding optimal threshold on val set (steps 30-34)...")
best_t, best_f1_val = 0.5, 0.0
for t in np.arange(0.05, 0.95, 0.005):
    f = f1_score(y_val, (ens_val >= t).astype(int), zero_division=0)
    if f > best_f1_val:
        best_f1_val, best_t = f, t
print(f"  Optimal threshold: {best_t:.3f}  (val F1={best_f1_val:.4f})")
t_opt  = best_t
y_pred = (ens_test >= t_opt).astype(int)

# ── results ────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"=== Individual model F1 on test ===")
print(f"GBM (Optuna):    {f1_score(y_test, gbm.predict(X_test_sc)):.4f}")
print(f"LightGBM:        {f1_score(y_test, lgbm.predict(X_test_sc)):.4f}")
print(f"GraphSAGEv2:     {f1_score(sage_y_test, (sage_test_raw >= 0.5).astype(int)):.4f}")

print(f"\n=== Ensemble (threshold={t_opt:.4f}) ===")
res = eval_metrics(y_test, y_pred, y_score=ens_test, name="Ensemble (GBM+LGBM+SAGEv2)")

plot_confusion_matrix(y_test, y_pred, name="Ensemble", save=True)
plot_pr_curve(
    y_test,
    {"Ensemble": ens_test, "GBM": gbm_test, "LightGBM": lgbm_test, "SAGEv2": sage_test},
    save=True,
)
print(f"\nEnsemble F1={res['f1']:.4f}  AUC={res['roc_auc']:.4f}")
