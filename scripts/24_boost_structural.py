"""
Test LightGBM + XGBoost on 172 structural features (165 tabular + 7 topology).
GBM+Structural = F1=0.8265 (current best). Can LightGBM/XGBoost beat it?
All models use n_jobs=-1 (all cores).
"""
import sys
sys.path.insert(0, "..")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import networkx as nx
import joblib
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from src.data_loader import load_all
from src.preprocessing import get_labeled, get_feature_cols, TRAIN_STEPS, TEST_STEPS
from src.evaluation import evaluate as eval_metrics

STRUCT_COLS = ["in_degree", "out_degree", "total_degree", "clustering",
               "avg_nb_deg", "ego_density", "oddball"]

print("Loading data...")
df, edges = load_all()

print("Building graph + structural features...")
G = nx.DiGraph()
G.add_nodes_from(df["txId"].values)
G.add_edges_from(zip(edges["txId1"], edges["txId2"]))
G_und = G.to_undirected()

in_deg  = dict(G.in_degree())
out_deg = dict(G.out_degree())
tot_deg = {n: in_deg[n] + out_deg[n] for n in in_deg}
clust   = nx.clustering(G_und)
avg_nb  = nx.average_neighbor_degree(G_und)
ego_density, oddball_score = {}, {}
for node in G_und.nodes():
    nb = list(G_und.neighbors(node))
    ni = len(nb) + 1
    sub = G_und.subgraph([node] + nb)
    ei  = sub.number_of_edges()
    ego_density[node]   = (2 * ei) / (ni * (ni - 1)) if ni > 1 else 0.0
    oddball_score[node] = abs(ei - ni**1.5) / (ni**1.5 + 1e-8)

struct_df = pd.DataFrame({
    "txId":         list(in_deg.keys()),
    "in_degree":    [in_deg[n]             for n in in_deg],
    "out_degree":   [out_deg[n]            for n in in_deg],
    "total_degree": [tot_deg[n]            for n in in_deg],
    "clustering":   [clust.get(n, 0)       for n in in_deg],
    "avg_nb_deg":   [avg_nb.get(n, 0)      for n in in_deg],
    "ego_density":  [ego_density.get(n, 0) for n in in_deg],
    "oddball":      [oddball_score.get(n, 0) for n in in_deg],
})

df_s    = df.merge(struct_df, on="txId", how="left").fillna(0)
labeled = get_labeled(df_s)

feat_cols_tab = get_feature_cols(labeled)
# Replicate script-18 layout: merged cols (includes struct) + explicit struct repeat
ALL_FEAT = feat_cols_tab + STRUCT_COLS   # 173 + 7 = 180 to match saved scaler

train_df = labeled[labeled["time_step"].isin(TRAIN_STEPS)].reset_index(drop=True)
test_df  = labeled[labeled["time_step"].isin(TEST_STEPS)].reset_index(drop=True)
y_train  = train_df["label"].values
y_test   = test_df["label"].values

scaler     = StandardScaler().fit(train_df[ALL_FEAT].values)
X_train_sc = scaler.transform(train_df[ALL_FEAT].values)
X_test_sc  = scaler.transform(test_df[ALL_FEAT].values)

print(f"  Train: {len(train_df):,}  Test: {len(test_df):,}  Features: {len(ALL_FEAT)}")

# ── baselines (original 166 features: time_step + 165 tabular, no structural) ─
# Use unmerged df to get original col list
labeled_orig = get_labeled(df)
orig_feat_cols = get_feature_cols(labeled_orig)  # time_step + f0..f164 = 166 cols
train_orig = labeled_orig[labeled_orig["time_step"].isin(TRAIN_STEPS)].reset_index(drop=True)
test_orig  = labeled_orig[labeled_orig["time_step"].isin(TEST_STEPS)].reset_index(drop=True)

gbm_base  = joblib.load("../models/gb_tuned.joblib")
lgbm_base = joblib.load("../models/lightgbm.joblib")
scaler_orig = StandardScaler().fit(train_orig[orig_feat_cols].values)
X_train_orig = scaler_orig.transform(train_orig[orig_feat_cols].values)
X_test_orig  = scaler_orig.transform(test_orig[orig_feat_cols].values)
print(f"\nBaselines ({len(orig_feat_cols)} feat):")
print(f"  GBM   baseline: F1={f1_score(test_orig['label'].values, gbm_base.predict(X_test_orig)):.4f}")
print(f"  LGBM  baseline: F1={f1_score(test_orig['label'].values, lgbm_base.predict(X_test_orig)):.4f}")

# ── [1] LightGBM + structural (172 feat, same params as saved model) ──────────
print("\n[1/3] LightGBM + structural features (same params, n_jobs=-1)...")
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
lgbm_s = lgb.LGBMClassifier(
    n_estimators=500, max_depth=8, learning_rate=0.05,
    num_leaves=63, subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=pos_weight, random_state=42,
    n_jobs=-1, verbose=-1,
)
lgbm_s.fit(X_train_sc, y_train)
lgbm_s_f1  = f1_score(y_test, lgbm_s.predict(X_test_sc))
lgbm_s_auc = roc_auc_score(y_test, lgbm_s.predict_proba(X_test_sc)[:, 1])
lgbm_s_ap  = average_precision_score(y_test, lgbm_s.predict_proba(X_test_sc)[:, 1])
print(f"  LGBM+structural: F1={lgbm_s_f1:.4f}  AUC={lgbm_s_auc:.4f}  AP={lgbm_s_ap:.4f}")
res_lgbm = eval_metrics(y_test, lgbm_s.predict(X_test_sc),
                        y_score=lgbm_s.predict_proba(X_test_sc)[:,1],
                        name="LightGBM+Structural")
joblib.dump(lgbm_s, "../models/lgbm_structural.joblib")

# ── [2] LightGBM + structural with more trees + lower lr (shrinkage) ──────────
print("\n[2/3] LightGBM + structural — shrinkage (n=1000, lr=0.025)...")
lgbm_sh = lgb.LGBMClassifier(
    n_estimators=1000, max_depth=8, learning_rate=0.025,
    num_leaves=63, subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=pos_weight, random_state=42,
    n_jobs=-1, verbose=-1,
)
lgbm_sh.fit(X_train_sc, y_train)
lgbm_sh_f1  = f1_score(y_test, lgbm_sh.predict(X_test_sc))
lgbm_sh_auc = roc_auc_score(y_test, lgbm_sh.predict_proba(X_test_sc)[:, 1])
lgbm_sh_ap  = average_precision_score(y_test, lgbm_sh.predict_proba(X_test_sc)[:, 1])
print(f"  LGBM+shrinkage:  F1={lgbm_sh_f1:.4f}  AUC={lgbm_sh_auc:.4f}  AP={lgbm_sh_ap:.4f}")
res_lgbm_sh = eval_metrics(y_test, lgbm_sh.predict(X_test_sc),
                            y_score=lgbm_sh.predict_proba(X_test_sc)[:,1],
                            name="LightGBM+Structural+Shrinkage")
joblib.dump(lgbm_sh, "../models/lgbm_structural_shrinkage.joblib")

# ── [3] XGBoost + structural ──────────────────────────────────────────────────
print("\n[3/3] XGBoost + structural features...")
try:
    import xgboost as xgb
    xgb_s = xgb.XGBClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        tree_method="hist", device="cpu",
        n_jobs=-1, random_state=42,
        eval_metric="logloss", verbosity=0,
    )
    xgb_s.fit(X_train_sc, y_train)
    xgb_s_f1  = f1_score(y_test, xgb_s.predict(X_test_sc))
    xgb_s_auc = roc_auc_score(y_test, xgb_s.predict_proba(X_test_sc)[:, 1])
    xgb_s_ap  = average_precision_score(y_test, xgb_s.predict_proba(X_test_sc)[:, 1])
    print(f"  XGBoost+structural: F1={xgb_s_f1:.4f}  AUC={xgb_s_auc:.4f}  AP={xgb_s_ap:.4f}")
    eval_metrics(y_test, xgb_s.predict(X_test_sc),
                 y_score=xgb_s.predict_proba(X_test_sc)[:,1],
                 name="XGBoost+Structural")
    joblib.dump(xgb_s, "../models/xgb_structural.joblib")
    xgb_ok = True
except ImportError:
    print("  XGBoost not installed — skipping")
    xgb_s_f1 = xgb_s_auc = xgb_s_ap = None
    xgb_ok = False

# ── summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("=== Boost+Structural Results ===")
print(f"Previous best (GBM+Structural):    F1=0.8265  AUC=0.9243")
print(f"LGBM baseline ({len(orig_feat_cols)} feat):          F1={f1_score(test_orig['label'].values, lgbm_base.predict(X_test_orig)):.4f}")
print(f"LGBM+Structural (500 trees):       F1={lgbm_s_f1:.4f}  AUC={lgbm_s_auc:.4f}  AP={lgbm_s_ap:.4f}  delta={lgbm_s_f1-0.8170:+.4f}")
print(f"LGBM+Structural+Shrinkage (1000t): F1={lgbm_sh_f1:.4f}  AUC={lgbm_sh_auc:.4f}  AP={lgbm_sh_ap:.4f}  delta={lgbm_sh_f1-0.8170:+.4f}")
if xgb_ok:
    print(f"XGBoost+Structural (500 trees):    F1={xgb_s_f1:.4f}  AUC={xgb_s_auc:.4f}  AP={xgb_s_ap:.4f}")
new_best = max([lgbm_s_f1, lgbm_sh_f1] + ([xgb_s_f1] if xgb_ok and xgb_s_f1 else []))
if new_best > 0.8265:
    print(f"\nNEW BEST: F1={new_best:.4f} beats GBM+Structural (0.8265)")
else:
    print(f"\nNo improvement over GBM+Structural (0.8265). Best here: F1={new_best:.4f}")
