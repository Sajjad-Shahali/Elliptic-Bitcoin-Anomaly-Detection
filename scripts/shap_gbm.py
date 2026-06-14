"""
Generate SHAP beeswarm for GBM (Optuna params) — the best tabular model.
Output: reports/shap_gbm_beeswarm.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import shap

ROOT    = os.path.join(os.path.dirname(__file__), "..")
REPORTS = os.path.join(ROOT, "reports")
DATA    = os.path.join(ROOT, "data", "elliptic_bitcoin_dataset")

# ─── load data ───────────────────────────────────────────────────────────────
print("Loading data...")
feat_path  = os.path.join(DATA, "elliptic_txs_features.csv")
class_path = os.path.join(DATA, "elliptic_txs_classes.csv")

feat_names = ["txId", "time_step"] + [f"lf_{i}" if i <= 93
              else f"af_{i-93}" for i in range(1, 166)]
feat = pd.read_csv(feat_path, header=None, names=feat_names)
cls  = pd.read_csv(class_path)

df = feat.merge(cls, on="txId")
df = df[df["class"].isin(["1", "2"])].copy()
df["label"] = (df["class"] == "1").astype(int)
ts = df["time_step"].astype(int)

feature_cols = feat_names[2:]   # lf_1 ... af_71
X_all = df[feature_cols].values
y_all = df["label"].values

X_train, y_train = X_all[ts <= 34], y_all[ts <= 34]
X_test,  y_test  = X_all[ts >= 35], y_all[ts >= 35]

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ─── train GBM (Optuna best params) ─────────────────────────────────────────
print("Training GBM (Optuna params)...")
gbm = GradientBoostingClassifier(
    n_estimators=245, max_depth=8, learning_rate=0.121,
    subsample=0.918, random_state=42
)
gbm.fit(X_train, y_train)

# ─── SHAP ────────────────────────────────────────────────────────────────────
print("Computing SHAP values (test set)...")
explainer = shap.TreeExplainer(gbm)

# subsample test for speed (SHAP on 16k rows is slow for GBM)
rng = np.random.default_rng(42)
idx = rng.choice(len(X_test), size=min(3000, len(X_test)), replace=False)
X_sample = X_test[idx]
y_sample = y_test[idx]

shap_values = explainer.shap_values(X_sample)

# GradientBoostingClassifier returns a 2D array (one per class for multiclass,
# but for binary it returns 1D). Wrap safely.
if isinstance(shap_values, list):
    sv = shap_values[1]   # class 1 = illicit
else:
    sv = shap_values

# ─── beeswarm ────────────────────────────────────────────────────────────────
print("Plotting beeswarm...")
shap_exp = shap.Explanation(
    values=sv,
    base_values=np.full(len(X_sample), explainer.expected_value
                        if not isinstance(explainer.expected_value, list)
                        else explainer.expected_value[1]),
    data=X_sample,
    feature_names=feature_cols,
)

fig, ax = plt.subplots(figsize=(8, 7))
plt.sca(ax)
shap.plots.beeswarm(shap_exp, max_display=20, show=False)
plt.title("SHAP Beeswarm — GBM (Optuna, F1=0.824)\nTop-20 features by mean |SHAP|",
          fontsize=10, fontweight="bold", pad=8)
plt.tight_layout()

out = os.path.join(REPORTS, "shap_gbm_beeswarm.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")
