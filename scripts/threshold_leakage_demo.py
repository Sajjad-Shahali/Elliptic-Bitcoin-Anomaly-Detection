"""
Threshold-leakage demo (submission add-on #3, HANDOFF "What to Add for Submission").

Claim (previously stated in HANDOFF/linkedin drafts but never reproduced by a script):
tuning the classification threshold by maximizing F1 on a model's OWN TRAINING
predictions collapses the threshold to an extreme value and inflates the reported
F1 to near-1.0 -- a false signal that does not transfer to real held-out data.

Protocol:
  - Fit GBM (gb_tuned hyperparams) on steps 1-29 ONLY (holds out 30-34 genuinely,
    unlike the main pipeline where "val" 30-34 is a subset of the 1-34 train fit).
  - LEAKY: sweep thresholds on the model's OWN steps-1-29 training predictions,
    pick the F1-maximizing threshold, report the (inflated) "train F1" and apply
    that threshold to the untouched test set (steps 35-49).
  - CORRECT: sweep thresholds on steps 30-34 (never seen during model fitting),
    pick the F1-maximizing threshold there, apply it to the same test set.
  - Compare: leaky "train F1" vs. real test F1 under each threshold choice.
"""
import sys
sys.path.insert(0, "..")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

from src.data_loader import load_all
from src.preprocessing import get_labeled, get_feature_cols
from config import REPORTS_DIR
import os

FIT_STEPS  = range(1, 30)   # 1-29: what the model is actually fit on
VAL_STEPS  = range(30, 35)  # 30-34: genuinely held out from the fit
TEST_STEPS = range(35, 50)  # 35-49: final evaluation, untouched throughout

print("Loading data...")
df, edges = load_all()
labeled   = get_labeled(df)
feat_cols = get_feature_cols(labeled)

fit_df  = labeled[labeled["time_step"].isin(FIT_STEPS)].reset_index(drop=True)
val_df  = labeled[labeled["time_step"].isin(VAL_STEPS)].reset_index(drop=True)
test_df = labeled[labeled["time_step"].isin(TEST_STEPS)].reset_index(drop=True)
print(f"  Fit (1-29):  {len(fit_df):,}  (illicit={fit_df['label'].mean():.3f})")
print(f"  Val (30-34): {len(val_df):,}  (illicit={val_df['label'].mean():.3f})  -- held out from fit")
print(f"  Test (35-49):{len(test_df):,}  (illicit={test_df['label'].mean():.3f})")

scaler = StandardScaler().fit(fit_df[feat_cols].values)
X_fit  = scaler.transform(fit_df[feat_cols].values)
X_val  = scaler.transform(val_df[feat_cols].values)
X_test = scaler.transform(test_df[feat_cols].values)
y_fit, y_val, y_test = fit_df["label"].values, val_df["label"].values, test_df["label"].values

print("\nFitting GBM (gb_tuned hyperparams) on steps 1-29 only...")
saved_params = joblib.load("../models/gb_tuned.joblib").get_params()
gbm = GradientBoostingClassifier(**saved_params)
gbm.fit(X_fit, y_fit)

fit_scores  = gbm.predict_proba(X_fit)[:, 1]
val_scores  = gbm.predict_proba(X_val)[:, 1]
test_scores = gbm.predict_proba(X_test)[:, 1]

def sweep_threshold(scores, y_true, n=200):
    """Sweep thresholds over the full [0,1] probability range, return (best_t, best_f1, curve)."""
    thresholds = np.linspace(0.0, 1.0, n)
    f1s = np.array([f1_score(y_true, (scores >= t).astype(int), zero_division=0) for t in thresholds])
    best_i = f1s.argmax()
    return thresholds[best_i], f1s[best_i], thresholds, f1s

# ── LEAKY: threshold tuned on the model's own training predictions ────────────
t_leaky, f1_leaky_train, th_grid, f1_curve_leaky = sweep_threshold(fit_scores, y_fit)
f1_leaky_on_test = f1_score(y_test, (test_scores >= t_leaky).astype(int))

# ── CORRECT: threshold tuned on genuinely held-out val (steps 30-34) ──────────
t_correct, f1_correct_val, _, f1_curve_correct = sweep_threshold(val_scores, y_val)
f1_correct_on_test = f1_score(y_test, (test_scores >= t_correct).astype(int))

# ── default 0.5 threshold, for reference ───────────────────────────────────────
f1_default_test = f1_score(y_test, (test_scores >= 0.5).astype(int))

print("\n=== Threshold Leakage Demo ===")
print(f"{'Protocol':<28s}{'Tuned on':<14s}{'Chosen t':>10s}{'\"F1\" at tune time':>20s}{'Real test F1':>15s}")
print(f"{'LEAKY (train self-tune)':<28s}{'train (1-29)':<14s}{t_leaky:>10.4f}{f1_leaky_train:>20.4f}{f1_leaky_on_test:>15.4f}")
print(f"{'CORRECT (held-out val)':<28s}{'val (30-34)':<14s}{t_correct:>10.4f}{f1_correct_val:>20.4f}{f1_correct_on_test:>15.4f}")
print(f"{'Default (no tuning)':<28s}{'--':<14s}{0.5:>10.4f}{'--':>20s}{f1_default_test:>15.4f}")

gap_train_reported_vs_test = f1_leaky_train - f1_leaky_on_test
print(f"\nLeaky protocol reports F1={f1_leaky_train:.4f} at tune time but only achieves "
      f"{f1_leaky_on_test:.4f} on real test data -- a {gap_train_reported_vs_test:.4f} F1 illusion.")
print(f"Correct protocol's val F1 ({f1_correct_val:.4f}) is close to its real test F1 "
      f"({f1_correct_on_test:.4f}), delta={abs(f1_correct_val-f1_correct_on_test):.4f} -- honest estimate.")

# ── save results + plot ────────────────────────────────────────────────────────
summary = pd.DataFrame([
    {"protocol": "leaky_train_self_tune", "tuned_on": "train (1-29)", "threshold": t_leaky,
     "f1_at_tune_time": f1_leaky_train, "real_test_f1": f1_leaky_on_test},
    {"protocol": "correct_held_out_val", "tuned_on": "val (30-34)", "threshold": t_correct,
     "f1_at_tune_time": f1_correct_val, "real_test_f1": f1_correct_on_test},
    {"protocol": "default_0.5", "tuned_on": "--", "threshold": 0.5,
     "f1_at_tune_time": None, "real_test_f1": f1_default_test},
])
summary.to_csv(os.path.join(REPORTS_DIR, "threshold_leakage_demo.csv"), index=False)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

ax = axes[0]
ax.plot(th_grid, f1_curve_leaky, color="#d6604d", label="F1 curve on TRAIN (leaky tune set)")
ax.plot(th_grid, f1_curve_correct, color="#2166ac", label="F1 curve on VAL (correct tune set)")
ax.axvline(t_leaky, color="#d6604d", linestyle="--", alpha=0.7)
ax.axvline(t_correct, color="#2166ac", linestyle="--", alpha=0.7)
ax.scatter([t_leaky], [f1_leaky_train], color="#d6604d", zorder=5, s=60)
ax.scatter([t_correct], [f1_correct_val], color="#2166ac", zorder=5, s=60)
ax.set_xlabel("Decision threshold")
ax.set_ylabel("F1 on tuning set")
ax.set_title("Where Each Protocol Picks Its Threshold")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
labels = ["Leaky\n(train self-tune)", "Correct\n(held-out val)", "Default\n(t=0.5)"]
reported = [f1_leaky_train, f1_correct_val, np.nan]
real_test = [f1_leaky_on_test, f1_correct_on_test, f1_default_test]
x = np.arange(3)
w = 0.35
ax.bar(x - w/2, reported, w, label="F1 reported at tune time", color="#f4a582")
ax.bar(x + w/2, real_test, w, label="Real test F1", color="#4393c3")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("F1")
ax.set_title("Reported vs. Real F1, by Protocol")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(REPORTS_DIR, "threshold_leakage_demo.png"), dpi=150)
plt.close(fig)

print(f"\nSaved: reports/threshold_leakage_demo.csv, reports/threshold_leakage_demo.png")
