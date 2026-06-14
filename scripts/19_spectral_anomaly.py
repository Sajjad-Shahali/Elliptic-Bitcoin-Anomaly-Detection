"""
Variant 41 — Spectral embedding + IsolationForest/OCSVM.
Course ref: Doc 6 §8 — L_sym eigenvectors encode graph structure; anomalous nodes have unusual spectral coords.
Pure topology: no node features. Cheeger inequality: small lambda_2 signals sparse cut / suspicious substructures.
"""
import sys
sys.path.insert(0, "..")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

from src.data_loader import load_all
from src.preprocessing import get_labeled, TRAIN_STEPS, TEST_STEPS
from src.autoencoder import f1_optimal_threshold
from src.evaluation import evaluate as eval_metrics

K_EIGENVECTORS = 50

print("Loading data...")
df, edges = load_all()

# ── build sparse adjacency matrix ─────────────────────────────────────────────
print("Building sparse adjacency matrix...")
node_ids   = df["txId"].values
id_to_idx  = {txid: i for i, txid in enumerate(node_ids)}
n          = len(node_ids)

valid = edges["txId1"].isin(id_to_idx) & edges["txId2"].isin(id_to_idx)
e     = edges[valid]
src   = e["txId1"].map(id_to_idx).values
dst   = e["txId2"].map(id_to_idx).values

# Undirected: add both directions
rows = np.concatenate([src, dst])
cols = np.concatenate([dst, src])
data = np.ones(len(rows), dtype=np.float32)
A    = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
A.data = np.ones_like(A.data)          # binarize (remove duplicates weight >1)
A    = (A > 0).astype(np.float32)
print(f"  Adjacency: {n:,} x {n:,}  nnz={A.nnz:,}")

# ── compute L_sym = I - D^{-1/2} A D^{-1/2} ──────────────────────────────────
print("Computing normalised Laplacian L_sym...")
deg     = np.asarray(A.sum(axis=1)).flatten()
deg_inv = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
D_inv   = sp.diags(deg_inv)
A_norm  = D_inv @ A @ D_inv                     # D^{-1/2} A D^{-1/2}
# Use A_norm directly for eigsh (largest eigs of A_norm = smallest of L_sym)

# ── compute top-K eigenvectors of normalised adjacency (ARPACK) ───────────────
print(f"Computing top-{K_EIGENVECTORS} eigenvectors (ARPACK, may take 2-5 min)...")
eigenvalues, eigenvectors = eigsh(A_norm, k=K_EIGENVECTORS, which="LM")
# Sort by descending eigenvalue (largest first)
order       = np.argsort(-eigenvalues)
eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]

print(f"  Eigenvalue range: [{eigenvalues[-1]:.4f}, {eigenvalues[0]:.4f}]")
print(f"  Embedding shape: {eigenvectors.shape}")

# Fiedler value: second-smallest eigenvalue of L_sym = 1 - second-largest of A_norm
fiedler = 1.0 - eigenvalues[-2] if len(eigenvalues) >= 2 else None
if fiedler is not None:
    print(f"  Fiedler value (lambda_2 of L_sym): {fiedler:.6f}  (small = sparse cut exists)")

# ── spectral node embeddings ───────────────────────────────────────────────────
Z = eigenvectors  # (n, K) — spectral coordinates per node

# ── align with labeled sets ───────────────────────────────────────────────────
labeled   = get_labeled(df)
train_df  = labeled[labeled["time_step"].isin(TRAIN_STEPS)].reset_index(drop=True)
test_df   = labeled[labeled["time_step"].isin(TEST_STEPS)].reset_index(drop=True)
y_train   = train_df["label"].values
y_test    = test_df["label"].values

train_idx = train_df["txId"].map(id_to_idx).values
test_idx  = test_df["txId"].map(id_to_idx).values

Z_train = Z[train_idx]
Z_test  = Z[test_idx]

scaler  = StandardScaler().fit(Z_train)
Z_train_sc = scaler.transform(Z_train)
Z_test_sc  = scaler.transform(Z_test)

# ── Variant 41a: IsoForest on spectral embedding ───────────────────────────────
print("\n[1/2] IsolationForest on spectral embedding...")
# Use all train-step nodes for density fit (not just labeled)
all_train_idx = np.array([id_to_idx[t] for t in
                           df[df["time_step"].isin(TRAIN_STEPS)]["txId"].values
                           if t in id_to_idx])
Z_all_train = scaler.transform(Z[all_train_idx])

iso = IsolationForest(n_estimators=200, contamination=0.065, random_state=42, n_jobs=-1)
iso.fit(Z_all_train)

scores_iso = -iso.score_samples(Z_test_sc)
auc_iso    = roc_auc_score(y_test, scores_iso)
ap_iso     = average_precision_score(y_test, scores_iso)
train_scores_iso = -iso.score_samples(Z_train_sc)
t_iso = f1_optimal_threshold(train_scores_iso, y_train)
f1_iso = f1_score(y_test, (scores_iso >= t_iso).astype(int))
print(f"  IsoForest (spectral): F1={f1_iso:.4f}  AUC={auc_iso:.4f}  AP={ap_iso:.4f}")
res_iso = eval_metrics(y_test, (scores_iso >= t_iso).astype(int),
                       y_score=scores_iso, name="IsoForest-Spectral")

# ── Variant 41b: OCSVM on spectral embedding (licit-only train) ───────────────
print("\n[2/2] OneClassSVM on spectral embedding (licit-only train)...")
licit_train = train_df[train_df["label"] == 0]
licit_idx   = licit_train["txId"].map(id_to_idx).values
Z_licit     = scaler.transform(Z[licit_idx])

ocsvm = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
ocsvm.fit(Z_licit)

scores_ocsvm = -ocsvm.score_samples(Z_test_sc)
auc_ocsvm    = roc_auc_score(y_test, scores_ocsvm)
ap_ocsvm     = average_precision_score(y_test, scores_ocsvm)
train_scores_ocsvm = -ocsvm.score_samples(Z_train_sc)
t_ocsvm = f1_optimal_threshold(train_scores_ocsvm, y_train)
f1_ocsvm = f1_score(y_test, (scores_ocsvm >= t_ocsvm).astype(int))
print(f"  OCSVM (spectral):     F1={f1_ocsvm:.4f}  AUC={auc_ocsvm:.4f}  AP={ap_ocsvm:.4f}")
res_ocsvm = eval_metrics(y_test, (scores_ocsvm >= t_ocsvm).astype(int),
                         y_score=scores_ocsvm, name="OCSVM-Spectral")

print("\n=== Spectral Anomaly Detection Summary ===")
print(f"IsoForest (tabular, baseline):    F1=0.0210  AUC=0.172")
print(f"OCSVM    (tabular, baseline):     F1=0.0230  AUC=0.235")
print(f"DOMINANT (GCN-AE, this session):  F1=0.2120  AUC=0.669")
print(f"IsoForest (spectral K={K_EIGENVECTORS}):      F1={f1_iso:.4f}  AUC={auc_iso:.4f}")
print(f"OCSVM    (spectral K={K_EIGENVECTORS}):       F1={f1_ocsvm:.4f}  AUC={auc_ocsvm:.4f}")
print(f"\nFiedler value: {fiedler:.6f} — {'sparse cut detected' if fiedler is not None and fiedler < 0.01 else 'well-connected graph'}")
