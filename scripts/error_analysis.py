"""
Error analysis (submission add-on, HANDOFF §"What to Add for Submission").
Which illicit transactions does the best leakage-free model (GBM Optuna, tabular) miss,
and are they structurally different from the ones it catches?

Checks:
  1. Confusion matrix breakdown (TP/FP/FN/TN)
  2. Are misses concentrated in specific time steps? (concept-drift connection)
  3. Are false negatives near-threshold (borderline) or confidently wrong?
  4. Do missed illicit txs have lower graph degree than caught ones? (isolated launderers
     vs hub-like ones -- degree is a proxy for how much "guilt by association" signal exists)
  5. Top-3 robust SHAP features (lf_53, lf_90, af_70) -- do they separate FN from TP?
"""
import sys
sys.path.insert(0, "..")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import networkx as nx
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

from src.data_loader import load_all
from src.preprocessing import get_labeled, get_feature_cols, TRAIN_STEPS, TEST_STEPS
from config import REPORTS_DIR
import os

print("Loading data...")
df, edges = load_all()
labeled  = get_labeled(df)
feat_cols = get_feature_cols(labeled)
train_df = labeled[labeled["time_step"].isin(TRAIN_STEPS)].reset_index(drop=True)
test_df  = labeled[labeled["time_step"].isin(TEST_STEPS)].reset_index(drop=True)
y_train, y_test = train_df["label"].values, test_df["label"].values

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(train_df[feat_cols].values)
X_test_sc = scaler.transform(test_df[feat_cols].values)

gb = joblib.load("../models/gb_tuned.joblib")
y_pred  = gb.predict(X_test_sc)
y_prob  = gb.predict_proba(X_test_sc)[:, 1]

test_df = test_df.copy()
test_df["y_true"] = y_test
test_df["y_pred"] = y_pred
test_df["y_prob"] = y_prob

def outcome(row):
    if row.y_true == 1 and row.y_pred == 1: return "TP"
    if row.y_true == 1 and row.y_pred == 0: return "FN"
    if row.y_true == 0 and row.y_pred == 1: return "FP"
    return "TN"
test_df["outcome"] = test_df.apply(outcome, axis=1)

counts = test_df["outcome"].value_counts()
print(f"\n=== Confusion Matrix (GBM Optuna, test steps 35-49) ===")
print(counts.to_string())
tp, fn, fp = counts.get("TP", 0), counts.get("FN", 0), counts.get("FP", 0)
print(f"Precision={tp/(tp+fp):.4f}  Recall={tp/(tp+fn):.4f}")

# ── 1. temporal concentration of FN/FP ─────────────────────────────────────────
print("\n=== FN/FP by time step ===")
ts_illicit = test_df[test_df.y_true == 1].groupby("time_step").size()
ts_fn      = test_df[test_df.outcome == "FN"].groupby("time_step").size().reindex(ts_illicit.index, fill_value=0)
ts_fn_rate = (ts_fn / ts_illicit).fillna(0)
for t in ts_illicit.index:
    print(f"  step {t:2d}: illicit={ts_illicit[t]:3d}  FN={ts_fn.get(t,0):3d}  FN_rate={ts_fn_rate.get(t,0):.3f}")

worst_steps = ts_fn_rate.sort_values(ascending=False).head(5)
print(f"\nWorst 5 steps by FN rate: {dict(worst_steps.round(3))}")

# ── 2. probability distribution: borderline vs confidently-wrong FN ───────────
fn_probs = test_df[test_df.outcome == "FN"]["y_prob"]
tp_probs = test_df[test_df.outcome == "TP"]["y_prob"]
print(f"\n=== FN probability distribution (threshold=0.5) ===")
print(f"  FN probs: mean={fn_probs.mean():.4f}  median={fn_probs.median():.4f}  "
      f"borderline (>0.3)={100*(fn_probs > 0.3).mean():.1f}%  confidently-wrong (<0.1)={100*(fn_probs < 0.1).mean():.1f}%")
print(f"  TP probs: mean={tp_probs.mean():.4f}  median={tp_probs.median():.4f}")

# ── 3. graph degree: are missed illicit txs more isolated? ────────────────────
print("\nComputing node degree from edge list...")
G = nx.DiGraph()
G.add_nodes_from(df["txId"].values)
G.add_edges_from(zip(edges["txId1"], edges["txId2"]))
deg = dict(G.degree())
test_df["total_degree"] = test_df["txId"].map(deg).fillna(0)

fn_deg = test_df[test_df.outcome == "FN"]["total_degree"]
tp_deg = test_df[test_df.outcome == "TP"]["total_degree"]
u_stat, p_deg = mannwhitneyu(fn_deg, tp_deg, alternative="two-sided")
print(f"\n=== Graph degree: FN vs TP (illicit only) ===")
print(f"  FN degree: mean={fn_deg.mean():.2f}  median={fn_deg.median():.1f}")
print(f"  TP degree: mean={tp_deg.mean():.2f}  median={tp_deg.median():.1f}")
print(f"  Mann-Whitney U p-value: {p_deg:.4g}  {'(significant)' if p_deg < 0.05 else '(not significant)'}")

# ── 4. robust SHAP features: lf_53, lf_90, af_70 ───────────────────────────────
print("\n=== Robust cross-model features: FN vs TP (raw, unscaled) ===")
feat_summary = []
for f in ["lf_53", "lf_90", "af_70"]:
    if f not in test_df.columns:
        continue
    fn_vals = test_df[test_df.outcome == "FN"][f]
    tp_vals = test_df[test_df.outcome == "TP"][f]
    u, p = mannwhitneyu(fn_vals, tp_vals, alternative="two-sided")
    print(f"  {f}: FN mean={fn_vals.mean():.4f}  TP mean={tp_vals.mean():.4f}  p={p:.4g}")
    feat_summary.append({"feature": f, "fn_mean": fn_vals.mean(), "tp_mean": tp_vals.mean(), "p_value": p})

# ── save summary CSV + plots ───────────────────────────────────────────────────
summary = pd.DataFrame([
    {"metric": "TP", "value": tp}, {"metric": "FN", "value": fn}, {"metric": "FP", "value": fp},
    {"metric": "precision", "value": tp/(tp+fp)}, {"metric": "recall", "value": tp/(tp+fn)},
    {"metric": "fn_prob_mean", "value": fn_probs.mean()}, {"metric": "tp_prob_mean", "value": tp_probs.mean()},
    {"metric": "fn_degree_mean", "value": fn_deg.mean()}, {"metric": "tp_degree_mean", "value": tp_deg.mean()},
    {"metric": "degree_mannwhitney_p", "value": p_deg},
])
summary.to_csv(os.path.join(REPORTS_DIR, "error_analysis_summary.csv"), index=False)
pd.DataFrame(feat_summary).to_csv(os.path.join(REPORTS_DIR, "error_analysis_features.csv"), index=False)

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

ax = axes[0]
ax.bar(ts_fn_rate.index, ts_fn_rate.values, color="#d6604d")
ax.set_xlabel("Time step (test, 35-49)")
ax.set_ylabel("FN rate (missed / illicit)")
ax.set_title("Miss Rate by Time Step")
ax.grid(axis="y", alpha=0.3)

ax = axes[1]
ax.hist(tp_probs, bins=30, alpha=0.6, label=f"TP (n={len(tp_probs)})", color="#2166ac", density=True)
ax.hist(fn_probs, bins=30, alpha=0.6, label=f"FN (n={len(fn_probs)})", color="#d6604d", density=True)
ax.axvline(0.5, color="k", linestyle="--", linewidth=1)
ax.set_xlabel("P(illicit)")
ax.set_ylabel("Density")
ax.set_title("Predicted Probability: TP vs FN")
ax.legend(fontsize=8)

ax = axes[2]
ax.boxplot([tp_deg.clip(upper=tp_deg.quantile(0.95)), fn_deg.clip(upper=fn_deg.quantile(0.95))],
           tick_labels=["TP", "FN"], patch_artist=True,
           boxprops=dict(facecolor="#92c5de"))
ax.set_ylabel("Graph degree (clipped at p95)")
ax.set_title(f"Node Degree: TP vs FN\n(Mann-Whitney p={p_deg:.3f})")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(REPORTS_DIR, "error_analysis.png"), dpi=150)
plt.close(fig)

print("\nSaved: reports/error_analysis_summary.csv, reports/error_analysis_features.csv, reports/error_analysis.png")
