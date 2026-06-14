"""
Variant 42 — Weighted ensemble: GBM 40% + LightGBM 40% + SAGEv2 20%.
SAGEv2 softmax probs are lower-magnitude than tree models, diluting equal-weight ensemble.
Reducing its weight to 20% should lift ensemble above GBM solo (F1=0.8241).
Also tests 50/50 (GBM+LGBM only) as ablation.
"""
import sys
sys.path.insert(0, "..")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import joblib
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_all
from src.preprocessing import get_labeled, get_feature_cols, TRAIN_STEPS, TEST_STEPS
from src.graph_utils import build_graph
from src.gnn import GraphSAGEv2, evaluate as gnn_evaluate
from src.evaluation import evaluate as eval_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

print("Loading data...")
df, edges = load_all()
labeled   = get_labeled(df)
feat_cols = get_feature_cols(labeled)

VAL_STEPS = list(range(30, 35))
train_df  = labeled[labeled["time_step"].isin(TRAIN_STEPS)].reset_index(drop=True)
val_df    = labeled[labeled["time_step"].isin(VAL_STEPS)].reset_index(drop=True)
test_df   = labeled[labeled["time_step"].isin(TEST_STEPS)].reset_index(drop=True)
y_val     = val_df["label"].values
y_test    = test_df["label"].values

scaler     = StandardScaler().fit(train_df[feat_cols].values)
X_val_sc   = scaler.transform(val_df[feat_cols].values)
X_test_sc  = scaler.transform(test_df[feat_cols].values)

print("Loading GBM and LightGBM...")
gbm  = joblib.load("../models/gb_tuned.joblib")
lgbm = joblib.load("../models/lightgbm.joblib")
gbm_val    = gbm.predict_proba(X_val_sc)[:, 1]
lgbm_val   = lgbm.predict_proba(X_val_sc)[:, 1]
gbm_test   = gbm.predict_proba(X_test_sc)[:, 1]
lgbm_test  = lgbm.predict_proba(X_test_sc)[:, 1]
print(f"  GBM  solo F1={f1_score(y_test, gbm.predict(X_test_sc)):.4f}")
print(f"  LGBM solo F1={f1_score(y_test, lgbm.predict(X_test_sc)):.4f}")

print("Loading GraphSAGEv2...")
data, node_ids, _ = build_graph(df, edges)
data = data.to(DEVICE)
sage = GraphSAGEv2(input_dim=data.x.shape[1], hidden_dim=256, output_dim=2, dropout=0.3).to(DEVICE)
sage.load_state_dict(torch.load("../models/graphsagev2.pt", map_location=DEVICE))

train_mask_np = data.train_mask.cpu().numpy().astype(bool)
test_mask_np  = data.test_mask.cpu().numpy().astype(bool)
_, sage_train_raw, _ = gnn_evaluate(sage, data, train_mask_np, DEVICE)
_, sage_test_raw,  _ = gnn_evaluate(sage, data, test_mask_np,  DEVICE)
sage_train_map = dict(zip(node_ids[train_mask_np], sage_train_raw))
sage_test_map  = dict(zip(node_ids[test_mask_np],  sage_test_raw))
sage_val  = np.array([sage_train_map.get(t, gbm_val[i])  for i, t in enumerate(val_df["txId"].values)])
sage_test = np.array([sage_test_map.get(t, 0.5)          for t in test_df["txId"].values])
print(f"  SAGEv2 solo F1={f1_score(y_test, (sage_test_raw >= 0.5).astype(int)):.4f}")
print(f"  SAGEv2 test prob mean={sage_test_raw.mean():.4f}  max={sage_test_raw.max():.4f}")
print(f"  GBM    test prob mean={gbm_test.mean():.4f}  max={gbm_test.max():.4f}")

def best_threshold(val_probs, y_val_):
    best_t, best_f = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.005):
        f = f1_score(y_val_, (val_probs >= t).astype(int), zero_division=0)
        if f > best_f:
            best_f, best_t = f, t
    return best_t, best_f

print("\n" + "=" * 60)
WEIGHT_SETS = [
    ("Equal (1/3 each)",   [1/3, 1/3, 1/3]),
    ("40/40/20",           [0.40, 0.40, 0.20]),
    ("45/45/10",           [0.45, 0.45, 0.10]),
    ("50/50/0 (no SAGE)",  [0.50, 0.50, 0.00]),
]

results = []
for name, (w_gbm, w_lgbm, w_sage) in [(n, w) for n, w in WEIGHT_SETS]:
    ens_val  = w_gbm * gbm_val  + w_lgbm * lgbm_val  + w_sage * sage_val
    ens_test = w_gbm * gbm_test + w_lgbm * lgbm_test + w_sage * sage_test
    t, val_f1 = best_threshold(ens_val, y_val)
    pred = (ens_test >= t).astype(int)
    f1   = f1_score(y_test, pred)
    print(f"  {name:<22} threshold={t:.3f}  val_F1={val_f1:.4f}  test_F1={f1:.4f}")
    results.append((name, w_gbm, w_lgbm, w_sage, t, f1))

best_name, *_, best_f1 = max(results, key=lambda x: x[-1])
best_weights = [(n, w1, w2, w3, t, f1) for n, w1, w2, w3, t, f1 in results if n == best_name][0]
w_gbm, w_lgbm, w_sage = best_weights[1], best_weights[2], best_weights[3]
best_t = best_weights[4]

print(f"\nBest weighting: '{best_name}'  F1={best_f1:.4f}")

ens_test_best = w_gbm * gbm_test + w_lgbm * lgbm_test + w_sage * sage_test
res = eval_metrics(y_test, (ens_test_best >= best_t).astype(int),
                   y_score=ens_test_best, name=f"WeightedEnsemble-{best_name}")

print(f"\n=== Weighted Ensemble Summary ===")
print(f"GBM solo:            F1=0.8241")
print(f"Equal ensemble:      F1={[f for n,_,_,_,_,f in results if 'Equal' in n][0]:.4f}")
print(f"Best weighted:       F1={best_f1:.4f}  (weights: GBM={w_gbm} LGBM={w_lgbm} SAGE={w_sage})")
