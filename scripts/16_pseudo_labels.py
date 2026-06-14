"""
Variant 34 — Semi-supervised pseudo-labels.
GBM (saved) scores 157k unlabeled nodes → high-confidence become pseudo-labels → retrain GBM + GraphSAGEv2.
Course ref: manifold learning — unlabeled points constrain structure even without labels.
"""
import sys
sys.path.insert(0, "..")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import joblib
import torch
import torch.nn.functional as F
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_all
from src.preprocessing import get_labeled, get_feature_cols, TRAIN_STEPS, TEST_STEPS
from src.graph_utils import build_graph
from src.gnn import GraphSAGEv2, train_epoch, evaluate, class_weights
from src.evaluation import evaluate as eval_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── data ───────────────────────────────────────────────────────────────────────
print("Loading data...")
df, edges = load_all()
labeled    = get_labeled(df)
feat_cols  = get_feature_cols(labeled)

train_df = labeled[labeled["time_step"].isin(TRAIN_STEPS)].reset_index(drop=True)
test_df  = labeled[labeled["time_step"].isin(TEST_STEPS)].reset_index(drop=True)
y_train  = train_df["label"].values
y_test   = test_df["label"].values

scaler     = StandardScaler().fit(train_df[feat_cols].values)
X_train_sc = scaler.transform(train_df[feat_cols].values)
X_test_sc  = scaler.transform(test_df[feat_cols].values)

print(f"  Labeled train: {len(train_df):,}  (illicit={y_train.mean():.3f})")
print(f"  Labeled test:  {len(test_df):,}   (illicit={y_test.mean():.3f})")

# ── load saved GBM, score unlabeled ───────────────────────────────────────────
print("\nLoading saved GBM (gb_tuned.joblib)...")
gbm = joblib.load("../models/gb_tuned.joblib")
print(f"  GBM baseline F1={f1_score(y_test, gbm.predict(X_test_sc)):.4f}")

# Unlabeled nodes in training time steps
unknown_train = df[(df["time_step"].isin(TRAIN_STEPS)) & (~df["class"].isin(["1", "2"]))].copy()
print(f"\nScoring {len(unknown_train):,} unlabeled train-step nodes...")
X_unk_sc = scaler.transform(unknown_train[feat_cols].values)
unk_probs = gbm.predict_proba(X_unk_sc)[:, 1]
print(f"  Unlabeled illicit-prob: mean={unk_probs.mean():.4f}  p95={np.percentile(unk_probs,95):.4f}  max={unk_probs.max():.4f}")

# ── pseudo-label thresholds ───────────────────────────────────────────────────
for p_ill, p_lic in [(0.90, 0.05), (0.95, 0.02)]:
    pseudo_ill = (unk_probs > p_ill).sum()
    pseudo_lic = (unk_probs < p_lic).sum()
    print(f"  Thresholds p_ill>{p_ill} p_lic<{p_lic}: pseudo-illicit={pseudo_ill:,}  pseudo-licit={pseudo_lic:,}")

P_ILL, P_LIC = 0.90, 0.05
mask_ill = unk_probs > P_ILL
mask_lic = unk_probs < P_LIC

pseudo_X   = np.vstack([X_unk_sc[mask_ill], X_unk_sc[mask_lic]])
pseudo_y   = np.concatenate([np.ones(mask_ill.sum()), np.zeros(mask_lic.sum())])
print(f"\nPseudo-labels added: {mask_ill.sum():,} illicit + {mask_lic.sum():,} licit = {len(pseudo_y):,} total")

X_aug = np.vstack([X_train_sc, pseudo_X])
y_aug = np.concatenate([y_train, pseudo_y])
print(f"Augmented train size: {len(y_aug):,}  (illicit={y_aug.mean():.3f})")

# ── retrain GBM on augmented data ─────────────────────────────────────────────
print("\n[1/2] Retraining GBM on augmented data...")
saved_params = gbm.get_params()
gbm_aug = GradientBoostingClassifier(**saved_params)
gbm_aug.fit(X_aug, y_aug)
gbm_aug_f1 = f1_score(y_test, gbm_aug.predict(X_test_sc))
print(f"  GBM baseline F1 = {f1_score(y_test, gbm.predict(X_test_sc)):.4f}")
print(f"  GBM+pseudo   F1 = {gbm_aug_f1:.4f}  delta={gbm_aug_f1 - f1_score(y_test, gbm.predict(X_test_sc)):+.4f}")
res_gbm = eval_metrics(y_test, gbm_aug.predict(X_test_sc),
                       y_score=gbm_aug.predict_proba(X_test_sc)[:, 1],
                       name="GBM+PseudoLabels")
joblib.dump(gbm_aug, "../models/gbm_pseudo.joblib")

# ── retrain GraphSAGEv2 with pseudo-labeled graph nodes ───────────────────────
print("\n[2/2] Retraining GraphSAGEv2 with pseudo-labeled graph nodes...")
data, node_ids, _ = build_graph(df, edges)
print(f"  Graph: {data.num_nodes:,} nodes  {data.num_edges:,} edges")

# Expand labels: set pseudo-labeled nodes in graph
node_id_to_idx = {txid: i for i, txid in enumerate(node_ids)}
y_expanded = data.y.clone()  # -1 for unknown, 0/1 for labeled

# pseudo-illicit
unk_txids = unknown_train["txId"].values
for txid, is_ill, is_lic in zip(unk_txids, mask_ill, mask_lic):
    idx = node_id_to_idx.get(txid)
    if idx is not None:
        if is_ill:
            y_expanded[idx] = 1
        elif is_lic:
            y_expanded[idx] = 0

# New train mask: original labeled train + pseudo-labeled
orig_train_mask  = data.train_mask.clone()
pseudo_train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
for txid, is_ill, is_lic in zip(unk_txids, mask_ill, mask_lic):
    idx = node_id_to_idx.get(txid)
    if idx is not None and (is_ill or is_lic):
        pseudo_train_mask[idx] = True

aug_train_mask = orig_train_mask | pseudo_train_mask
print(f"  Original train mask: {orig_train_mask.sum():,}")
print(f"  Pseudo-labeled:      {pseudo_train_mask.sum():,}")
print(f"  Augmented train:     {aug_train_mask.sum():,}")

data.y          = y_expanded
data.train_mask = aug_train_mask
data = data.to(DEVICE)

EPOCHS = 250
model  = GraphSAGEv2(input_dim=data.x.shape[1], hidden_dim=256, output_dim=2, dropout=0.3).to(DEVICE)
wts    = class_weights(data.y[data.train_mask], DEVICE)
crit   = torch.nn.CrossEntropyLoss(weight=wts)
opt    = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
sch    = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

print(f"  Training {EPOCHS} epochs...")
for ep in range(1, EPOCHS + 1):
    loss = train_epoch(model, data, opt, crit, DEVICE)
    sch.step()
    if ep % 50 == 0:
        print(f"    Epoch {ep:3d}/{EPOCHS}  loss={loss:.4f}")

test_mask_np = data.test_mask.cpu().numpy().astype(bool)
y_pred, y_score, y_true = evaluate(model, data, test_mask_np, DEVICE)
sage_f1 = f1_score(y_true, y_pred)
print(f"  SAGEv2 baseline F1      = 0.6980")
print(f"  SAGEv2+pseudo   F1      = {sage_f1:.4f}  delta={sage_f1-0.698:+.4f}")
res_sage = eval_metrics(y_true, y_pred, y_score=y_score, name="SAGEv2+PseudoLabels")
torch.save(model.state_dict(), "../models/graphsagev2_pseudo.pt")

print("\n=== Round 3 — Pseudo-label Summary ===")
print(f"GBM baseline:       F1=0.8241")
print(f"GBM+pseudo:         F1={res_gbm['f1']:.4f}")
print(f"SAGEv2 baseline:    F1=0.6980")
print(f"SAGEv2+pseudo:      F1={res_sage['f1']:.4f}")
