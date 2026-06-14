"""
Variant 44 — Temporal rolling features per transaction.
Current 165 features are per-transaction snapshots. Rolling statistics across time steps
capture behavioral drift: e.g., a node's aggregate feature mean over past 5 time steps.
Approach: for each time step t, compute rolling mean/std of all-transactions aggregate
across steps [t-W, t-1], merge onto each transaction as extra context features.
This adds global temporal context without leaking future data.
"""
import sys
sys.path.insert(0, "..")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_all
from src.preprocessing import get_labeled, get_feature_cols, TRAIN_STEPS, TEST_STEPS
from src.evaluation import evaluate as eval_metrics

WINDOW = 5  # rolling window in time steps

print("Loading data...")
df, edges = load_all()
labeled   = get_labeled(df)
feat_cols = get_feature_cols(labeled)

print(f"Building temporal rolling features (window={WINDOW})...")
# time_step is in feat_cols from get_feature_cols() — exclude for aggregation
agg_cols = [c for c in feat_cols if c != "time_step"]
all_steps = sorted(df["time_step"].unique())
step_means = (
    df.groupby("time_step")[agg_cols].mean()
    .reindex(all_steps)
    .reset_index()
)

# Rolling mean and std over window W for each feature
roll_mean = step_means[agg_cols].rolling(window=WINDOW, min_periods=1).mean()
roll_std  = step_means[agg_cols].rolling(window=WINDOW, min_periods=1).std().fillna(0)

# Rename columns
roll_mean.columns = [f"roll_mean_{c}" for c in agg_cols]
roll_std.columns  = [f"roll_std_{c}"  for c in agg_cols]

roll_df = pd.concat([step_means[["time_step"]], roll_mean, roll_std], axis=1)
ROLL_COLS = list(roll_mean.columns) + list(roll_std.columns)
print(f"  Rolling feature matrix: {roll_df.shape}  ({len(ROLL_COLS)} new features)")

# Merge rolling features onto labeled set by time_step
labeled_roll = labeled.merge(roll_df, on="time_step", how="left")
ALL_FEAT = feat_cols + ROLL_COLS

print(f"  Total features: {len(ALL_FEAT)} ({len(feat_cols)} original + {len(ROLL_COLS)} rolling)")

train_df = labeled_roll[labeled_roll["time_step"].isin(TRAIN_STEPS)].reset_index(drop=True)
test_df  = labeled_roll[labeled_roll["time_step"].isin(TEST_STEPS)].reset_index(drop=True)
y_train  = train_df["label"].values
y_test   = test_df["label"].values

scaler     = StandardScaler().fit(train_df[ALL_FEAT].values)
X_train_sc = scaler.transform(train_df[ALL_FEAT].values)
X_test_sc  = scaler.transform(test_df[ALL_FEAT].values)

print("\n[1/2] GBM baseline (165 features, no rolling)...")
gbm_base = joblib.load("../models/gb_tuned.joblib")
X_train_base = scaler.transform(train_df[feat_cols].values) if False else StandardScaler().fit_transform(
    train_df[feat_cols].values
)
X_test_base = StandardScaler().fit(train_df[feat_cols].values).transform(test_df[feat_cols].values)
f1_base = f1_score(y_test, gbm_base.predict(X_test_base))
print(f"  GBM baseline (165 feat): F1={f1_base:.4f}")

print("\n[2/2] GBM with rolling features...")
gbm_roll = GradientBoostingClassifier(**gbm_base.get_params())
gbm_roll.fit(X_train_sc, y_train)
roll_preds  = gbm_roll.predict(X_test_sc)
roll_probs  = gbm_roll.predict_proba(X_test_sc)[:, 1]
f1_roll     = f1_score(y_test, roll_preds)
auc_roll    = roc_auc_score(y_test, roll_probs)
ap_roll     = average_precision_score(y_test, roll_probs)
print(f"  GBM+rolling ({len(ALL_FEAT)} feat): F1={f1_roll:.4f}  AUC={auc_roll:.4f}  AP={ap_roll:.4f}")
res = eval_metrics(y_test, roll_preds, y_score=roll_probs, name="GBM+TemporalRolling")
joblib.dump(gbm_roll, "../models/gbm_temporal_rolling.joblib")

# Feature importance: which rolling features matter most?
importances = gbm_roll.feature_importances_
imp_df = pd.DataFrame({"feature": ALL_FEAT, "importance": importances})
imp_df = imp_df.sort_values("importance", ascending=False)
top_roll = imp_df[imp_df["feature"].str.startswith("roll_")].head(10)
print(f"\nTop-10 rolling features by importance:")
for _, row in top_roll.iterrows():
    print(f"  {row['feature']:<35}  {row['importance']:.5f}")

print(f"\n=== Temporal Rolling Feature Summary ===")
print(f"GBM baseline (165 feat):     F1={f1_base:.4f}  AUC={roc_auc_score(y_test, gbm_base.predict_proba(X_test_base)[:,1]):.4f}")
print(f"GBM+rolling ({len(ALL_FEAT)} feat):  F1={f1_roll:.4f}  AUC={auc_roll:.4f}  AP={ap_roll:.4f}")
print(f"Delta: {f1_roll - f1_base:+.4f}")
