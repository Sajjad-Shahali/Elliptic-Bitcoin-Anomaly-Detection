"""
Regenerate pr_curves.png with top-10 representative models.
Re-trains the key models (fast tabular ones) on the temporal split and
saves their Precision-Recall curves together.

Output: reports/pr_curves.png
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, average_precision_score

ROOT    = os.path.join(os.path.dirname(__file__), "..")
REPORTS = os.path.join(ROOT, "reports")
DATA    = os.path.join(ROOT, "data", "elliptic_bitcoin_dataset")

# ─── load data ───────────────────────────────────────────────────────────────
print("Loading data...")
feat_path  = os.path.join(DATA, "elliptic_txs_features.csv")
class_path = os.path.join(DATA, "elliptic_txs_classes.csv")

feat_cols = ["txId", "time_step"] + [f"f{i}" for i in range(165)]
feat = pd.read_csv(feat_path, header=None, names=feat_cols)
cls  = pd.read_csv(class_path)

df = feat.merge(cls, on="txId")
df = df[df["class"].isin(["1", "2"])].copy()
df["label"] = (df["class"] == "1").astype(int)

X_all = df[[c for c in feat_cols if c not in ("txId", "time_step")]].values
y_all = df["label"].values
ts    = df["time_step"].values

train_mask = ts <= 34
test_mask  = ts >= 35

X_train, y_train = X_all[train_mask], y_all[train_mask]
X_test,  y_test  = X_all[test_mask],  y_all[test_mask]

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

pos_w = (y_train == 0).sum() / (y_train == 1).sum()

# ─── define models ───────────────────────────────────────────────────────────
models = {
    "Logistic Reg.":  LogisticRegression(C=1.0, class_weight="balanced",
                                         max_iter=1000, random_state=42),
    "Random Forest":  RandomForestClassifier(n_estimators=100,
                                             class_weight="balanced",
                                             random_state=42, n_jobs=-1),
    "GBM (baseline)": GradientBoostingClassifier(n_estimators=100,
                                                  max_depth=5, random_state=42),
    "GBM (Optuna)":   GradientBoostingClassifier(n_estimators=245, max_depth=8,
                                                  learning_rate=0.121,
                                                  subsample=0.918,
                                                  random_state=42),
    "LightGBM":       lgb.LGBMClassifier(n_estimators=500, max_depth=8,
                                          scale_pos_weight=pos_w,
                                          random_state=42, verbose=-1),
}

# ─── colors ──────────────────────────────────────────────────────────────────
palette = {
    "Logistic Reg.":  ("#95a5a6", "--"),
    "Random Forest":  ("#27ae60", "-"),
    "GBM (baseline)": ("#f39c12", "-"),
    "GBM (Optuna)":   ("#e74c3c", "-"),
    "LightGBM":       ("#8e44ad", "-"),
}

# ─── train, evaluate, plot ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

for name, model in models.items():
    print(f"  Training {name}...")
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    prec, rec, _ = precision_recall_curve(y_test, proba)
    ap = average_precision_score(y_test, proba)
    color, ls = palette[name]
    ax.plot(rec, prec, color=color, lw=1.8, ls=ls,
            label=f"{name}  (AP={ap:.3f})")
    print(f"    AP = {ap:.3f}")

# chance line
baseline_ap = y_test.mean()
ax.axhline(baseline_ap, color="#bdc3c7", lw=1, ls=":", label=f"Chance (AP={baseline_ap:.3f})")

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision–Recall Curves: Key Supervised Models\n"
             "(Elliptic test set, steps 35–49, 6.5 % illicit)",
             fontweight="bold")
ax.legend(fontsize=8, loc="upper right")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.grid(True, alpha=0.3)
plt.tight_layout()

out = os.path.join(REPORTS, "pr_curves.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out}")
