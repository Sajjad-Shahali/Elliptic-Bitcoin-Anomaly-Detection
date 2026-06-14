"""
Variant 39 — Structural graph features + IsolationForest.
Course ref: Doc 6 §7 — degree, clustering coeff, avg-neighbor-degree, ego-net density, OddBall law.
Pure topology signal: no node attributes used. Checks whether illicit nodes have anomalous connectivity.
Also: structural features concatenated to tabular features for GBM (variant 40).
"""
import sys
sys.path.insert(0, "..")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
import joblib

from src.data_loader import load_all
from src.preprocessing import get_labeled, get_feature_cols, TRAIN_STEPS, TEST_STEPS
from src.autoencoder import f1_optimal_threshold
from src.evaluation import evaluate as eval_metrics

print("Loading data...")
df, edges = load_all()

# ── build graph and compute structural features ────────────────────────────────
print("Building NetworkX graph...")
G = nx.DiGraph()
G.add_nodes_from(df["txId"].values)
G.add_edges_from(zip(edges["txId1"], edges["txId2"]))
G_und = G.to_undirected()
print(f"  Nodes: {G.number_of_nodes():,}  Edges: {G.number_of_edges():,}")

print("Computing degree features...")
in_deg  = dict(G.in_degree())
out_deg = dict(G.out_degree())
tot_deg = {n: in_deg[n] + out_deg[n] for n in in_deg}

print("Computing clustering coefficients (~30s)...")
clust = nx.clustering(G_und)

print("Computing average neighbor degree...")
avg_nb_deg = nx.average_neighbor_degree(G_und)

print("Computing ego-net density and OddBall features...")
ego_density   = {}
oddball_score = {}
for node in G_und.nodes():
    neighbors = list(G_und.neighbors(node))
    ni = len(neighbors) + 1          # ego-net nodes = node + neighbors
    subgraph = G_und.subgraph([node] + neighbors)
    ei = subgraph.number_of_edges()  # ego-net edges
    ego_density[node]   = (2 * ei) / (ni * (ni - 1)) if ni > 1 else 0.0
    expected_ei         = ni ** 1.5
    oddball_score[node] = abs(ei - expected_ei) / (expected_ei + 1e-8)

print("Assembling structural feature DataFrame...")
struct_df = pd.DataFrame({
    "txId":         list(in_deg.keys()),
    "in_degree":    [in_deg[n]        for n in in_deg],
    "out_degree":   [out_deg[n]       for n in in_deg],
    "total_degree": [tot_deg[n]       for n in in_deg],
    "clustering":   [clust.get(n, 0)  for n in in_deg],
    "avg_nb_deg":   [avg_nb_deg.get(n, 0) for n in in_deg],
    "ego_density":  [ego_density.get(n, 0) for n in in_deg],
    "oddball":      [oddball_score.get(n, 0) for n in in_deg],
})
STRUCT_COLS = ["in_degree", "out_degree", "total_degree", "clustering",
               "avg_nb_deg", "ego_density", "oddball"]

df_struct = df.merge(struct_df, on="txId", how="left").fillna(0)
labeled   = get_labeled(df_struct)
train_df  = labeled[labeled["time_step"].isin(TRAIN_STEPS)].reset_index(drop=True)
test_df   = labeled[labeled["time_step"].isin(TEST_STEPS)].reset_index(drop=True)
y_train   = train_df["label"].values
y_test    = test_df["label"].values

# ── Variant 39: IsoForest on structural features only ─────────────────────────
print("\n[1/2] IsolationForest on structural features only...")
scaler_s = StandardScaler().fit(train_df[STRUCT_COLS].values)
X_train_s = scaler_s.transform(train_df[STRUCT_COLS].values)
X_test_s  = scaler_s.transform(test_df[STRUCT_COLS].values)

# Also train on ALL train-step nodes (including unlabeled) for better density estimate
all_train = df_struct[df_struct["time_step"].isin(TRAIN_STEPS)]
X_all_s   = scaler_s.transform(all_train[STRUCT_COLS].values)

iso = IsolationForest(n_estimators=200, contamination=0.065, random_state=42, n_jobs=-1)
iso.fit(X_all_s)

scores_s       = -iso.score_samples(X_test_s)
train_scores_s = -iso.score_samples(X_train_s)
auc_s = roc_auc_score(y_test, scores_s)
ap_s  = average_precision_score(y_test, scores_s)
t_s   = f1_optimal_threshold(train_scores_s, y_train)
f1_s = f1_score(y_test, (scores_s >= t_s).astype(int))
print(f"  IsoForest (structural only): F1={f1_s:.4f}  AUC={auc_s:.4f}  AP={ap_s:.4f}")
res_iso = eval_metrics(y_test, (scores_s >= t_s).astype(int), y_score=scores_s,
                       name="IsoForest-StructuralFeatures")

# ── Variant 40: GBM on 165 tabular + 7 structural features ────────────────────
print("\n[2/2] GBM on tabular + structural features (165+7=172 features)...")
feat_cols_tab = get_feature_cols(labeled)
all_feat_cols = feat_cols_tab + STRUCT_COLS

scaler_c = StandardScaler().fit(train_df[all_feat_cols].values)
X_train_c = scaler_c.transform(train_df[all_feat_cols].values)
X_test_c  = scaler_c.transform(test_df[all_feat_cols].values)

gbm = joblib.load("../models/gb_tuned.joblib")
gbm_aug = GradientBoostingClassifier(**gbm.get_params())
gbm_aug.fit(X_train_c, y_train)
f1_c = f1_score(y_test, gbm_aug.predict(X_test_c))
auc_c = roc_auc_score(y_test, gbm_aug.predict_proba(X_test_c)[:, 1])
ap_c  = average_precision_score(y_test, gbm_aug.predict_proba(X_test_c)[:, 1])
print(f"  GBM+Structural: F1={f1_c:.4f}  AUC={auc_c:.4f}  AP={ap_c:.4f}")
res_gbm = eval_metrics(y_test, gbm_aug.predict(X_test_c),
                       y_score=gbm_aug.predict_proba(X_test_c)[:, 1],
                       name="GBM+StructuralFeatures")
joblib.dump(gbm_aug,   "../models/gbm_structural.joblib")
joblib.dump(scaler_c,  "../models/scaler_structural.joblib")

print("\n=== Structural Features Summary ===")
print(f"IsoForest (tabular only, baseline):    F1=0.0210")
print(f"IsoForest (structural only):           F1={f1_s:.4f}  AUC={auc_s:.4f}")
print(f"GBM (tabular only, Optuna):            F1=0.8241")
print(f"GBM (tabular+structural):              F1={f1_c:.4f}  AUC={auc_c:.4f}")
