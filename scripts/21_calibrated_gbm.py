"""
Variant 43 — Calibrated GBM probabilities (Platt scaling + isotonic regression).
GBM predict_proba is often overconfident. Calibration may improve:
 1. Ensemble quality (better-calibrated probs from each component)
 2. Standalone F1 (threshold selection more stable when probs are well-calibrated)
Calibration fit on val steps 30-34 to avoid leaking test.
"""
import sys
sys.path.insert(0, "..")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_all
from src.preprocessing import get_labeled, get_feature_cols, TRAIN_STEPS, TEST_STEPS
from src.evaluation import evaluate as eval_metrics

print("Loading data...")
df, edges = load_all()
labeled   = get_labeled(df)
feat_cols = get_feature_cols(labeled)

VAL_STEPS  = list(range(30, 35))
train_df   = labeled[labeled["time_step"].isin(TRAIN_STEPS)].reset_index(drop=True)
val_df     = labeled[labeled["time_step"].isin(VAL_STEPS)].reset_index(drop=True)
test_df    = labeled[labeled["time_step"].isin(TEST_STEPS)].reset_index(drop=True)
y_train    = train_df["label"].values
y_val      = val_df["label"].values
y_test     = test_df["label"].values

scaler     = StandardScaler().fit(train_df[feat_cols].values)
X_train_sc = scaler.transform(train_df[feat_cols].values)
X_val_sc   = scaler.transform(val_df[feat_cols].values)
X_test_sc  = scaler.transform(test_df[feat_cols].values)

print("Loading GBM...")
gbm = joblib.load("../models/gb_tuned.joblib")

gbm_train_probs = gbm.predict_proba(X_train_sc)[:, 1]
gbm_val_probs   = gbm.predict_proba(X_val_sc)[:, 1]
gbm_test_probs  = gbm.predict_proba(X_test_sc)[:, 1]
gbm_test_preds  = gbm.predict(X_test_sc)
gbm_f1          = f1_score(y_test, gbm_test_preds)
gbm_brier       = brier_score_loss(y_test, gbm_test_probs)
print(f"  GBM baseline: F1={gbm_f1:.4f}  Brier={gbm_brier:.4f}  AUC={roc_auc_score(y_test, gbm_test_probs):.4f}")
print(f"  GBM test prob: mean={gbm_test_probs.mean():.4f}  p95={np.percentile(gbm_test_probs,95):.4f}  max={gbm_test_probs.max():.4f}")

def best_threshold_val(val_probs, y_val_):
    best_t, best_f = 0.5, 0.0
    for t in np.arange(0.01, 0.99, 0.005):
        f = f1_score(y_val_, (val_probs >= t).astype(int), zero_division=0)
        if f > best_f:
            best_f, best_t = f, t
    return best_t, best_f

def apply_platt(val_raw, test_raw, y_val_):
    """Fit sigmoid (Platt scaling) on val raw probs, apply to test."""
    lr = LogisticRegression(C=1.0, max_iter=1000)
    lr.fit(val_raw.reshape(-1, 1), y_val_)
    return lr.predict_proba(test_raw.reshape(-1, 1))[:, 1], \
           lr.predict_proba(val_raw.reshape(-1, 1))[:, 1]

def apply_isotonic(val_raw, test_raw, y_val_):
    """Fit isotonic regression on val raw probs, clip test to [0,1]."""
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    ir.fit(val_raw, y_val_)
    return ir.predict(test_raw), ir.predict(val_raw)

print("\n[1/2] Platt scaling (sigmoid)...")
sig_probs, sig_val_pr = apply_platt(gbm_val_probs, gbm_test_probs, y_val)
t_sig, _ = best_threshold_val(sig_val_pr, y_val)
sig_f1   = f1_score(y_test, (sig_probs >= t_sig).astype(int))
sig_auc  = roc_auc_score(y_test, sig_probs)
sig_ap   = average_precision_score(y_test, sig_probs)
sig_brier = brier_score_loss(y_test, sig_probs)
print(f"  Platt:     F1={sig_f1:.4f}  AUC={sig_auc:.4f}  AP={sig_ap:.4f}  Brier={sig_brier:.4f}  threshold={t_sig:.3f}")
print(f"  Prob mean={sig_probs.mean():.4f}  p95={np.percentile(sig_probs,95):.4f}  max={sig_probs.max():.4f}")
res_sig = eval_metrics(y_test, (sig_probs >= t_sig).astype(int),
                       y_score=sig_probs, name="GBM-PlattCalibrated")

print("\n[2/2] Isotonic regression...")
iso_probs, iso_val_pr = apply_isotonic(gbm_val_probs, gbm_test_probs, y_val)
t_iso, _ = best_threshold_val(iso_val_pr, y_val)
iso_f1   = f1_score(y_test, (iso_probs >= t_iso).astype(int))
iso_auc  = roc_auc_score(y_test, iso_probs)
iso_ap   = average_precision_score(y_test, iso_probs)
iso_brier = brier_score_loss(y_test, iso_probs)
print(f"  Isotonic:  F1={iso_f1:.4f}  AUC={iso_auc:.4f}  AP={iso_ap:.4f}  Brier={iso_brier:.4f}  threshold={t_iso:.3f}")
print(f"  Prob mean={iso_probs.mean():.4f}  p95={np.percentile(iso_probs,95):.4f}  max={iso_probs.max():.4f}")
res_iso = eval_metrics(y_test, (iso_probs >= t_iso).astype(int),
                       y_score=iso_probs, name="GBM-IsotonicCalibrated")

best_method = "sigmoid" if sig_f1 >= iso_f1 else "isotonic"
print(f"\nBest calibration: {best_method}")

print(f"\n=== Calibration Summary ===")
print(f"GBM uncalibrated:    F1={gbm_f1:.4f}  AUC={roc_auc_score(y_test, gbm_test_probs):.4f}  Brier={gbm_brier:.4f}")
print(f"GBM Platt scaling:   F1={sig_f1:.4f}  AUC={sig_auc:.4f}  Brier={sig_brier:.4f}")
print(f"GBM Isotonic:        F1={iso_f1:.4f}  AUC={iso_auc:.4f}  Brier={iso_brier:.4f}")
print(f"Best method: {best_method}  (lower Brier = better calibrated probabilities)")
