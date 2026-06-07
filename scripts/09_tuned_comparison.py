"""
Post-tuning comparison — loads saved tuned models, compares against baseline.
Run after 08_tune_and_save.py completes.
"""
import sys
sys.path.insert(0, '..')

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import torch

from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, precision_recall_curve, roc_curve
)

from src.data_loader import load_all
from src.preprocessing import prepare_supervised_temporal
from src.graph_utils import build_graph
from src.gnn import GraphSAGE, evaluate as gnn_evaluate

# ── load data + tuned models ──────────────────────────────────────────────────
print("Loading data and tuned models...")
df, edges = load_all()
X_train, X_test, y_train, y_test, feat_cols, scaler = prepare_supervised_temporal(df)

rf_tuned = joblib.load("../models/rf_tuned.joblib")
gb_tuned = joblib.load("../models/gb_tuned.joblib")

GNN_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data, _, _ = build_graph(df, edges)
data = data.to(GNN_DEVICE)
# infer hidden_dim from saved weights
state = torch.load("../models/graphsage_tuned.pt", map_location=GNN_DEVICE)
hidden_dim = state["conv1.lin_l.weight"].shape[0]
gnn_tuned = GraphSAGE(input_dim=data.x.shape[1], hidden_dim=hidden_dim, output_dim=2).to(GNN_DEVICE)
gnn_tuned.load_state_dict(state)

# baseline models (default params)
from src.models import random_forest, gradient_boosting
from src.gnn import class_weights
rf_base = random_forest(X_train, y_train)
gb_base = gradient_boosting(X_train, y_train)

# retrain baseline GNN
gnn_base = GraphSAGE(data.x.shape[1], 128, 2, 0.3).to(GNN_DEVICE)
opt  = torch.optim.Adam(gnn_base.parameters(), lr=1e-3, weight_decay=5e-4)
sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
wts  = class_weights(data.y[data.train_mask], GNN_DEVICE)
crit = torch.nn.CrossEntropyLoss(weight=wts)
gnn_base.train()
for _ in range(200):
    opt.zero_grad()
    out = gnn_base(data.x, data.edge_index)
    loss = crit(out[data.train_mask], data.y[data.train_mask])
    loss.backward(); opt.step(); sch.step()

test_mask_np = data.test_mask.cpu().numpy().astype(bool)

# ── collect results ────────────────────────────────────────────────────────────
def row(name, y_true, y_pred, y_score, tuned):
    return {
        "Model":     name,
        "Tuned":     tuned,
        "ROC-AUC":   round(roc_auc_score(y_true, y_score), 4),
        "Avg-Prec":  round(average_precision_score(y_true, y_score), 4),
        "F1":        round(f1_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred), 4),
        "_y_true":   y_true,
        "_y_score":  y_score,
    }

records = []
# RF baseline vs tuned
records.append(row("RandomForest",       y_test, rf_base.predict(X_test), rf_base.predict_proba(X_test)[:,1], "Baseline"))
records.append(row("RandomForest",       y_test, rf_tuned.predict(X_test), rf_tuned.predict_proba(X_test)[:,1], "Tuned"))
# GBM baseline vs tuned
records.append(row("GradientBoosting",   y_test, gb_base.predict(X_test), gb_base.predict_proba(X_test)[:,1], "Baseline"))
records.append(row("GradientBoosting",   y_test, gb_tuned.predict(X_test), gb_tuned.predict_proba(X_test)[:,1], "Tuned"))
# GNN baseline vs tuned
y_pred_b, y_score_b, y_true_b = gnn_evaluate(gnn_base,  data, test_mask_np, GNN_DEVICE)
y_pred_t, y_score_t, y_true_t = gnn_evaluate(gnn_tuned, data, test_mask_np, GNN_DEVICE)
records.append(row("GraphSAGE", y_true_b, y_pred_b, y_score_b, "Baseline"))
records.append(row("GraphSAGE", y_true_t, y_pred_t, y_score_t, "Tuned"))

# ── table ─────────────────────────────────────────────────────────────────────
df_res = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")} for r in records])
print("\n=== Baseline vs Tuned ===")
print(df_res.to_string(index=False))
df_res.to_csv("../reports/tuned_leaderboard.csv", index=False)

# delta table
models = ["RandomForest", "GradientBoosting", "GraphSAGE"]
print("\n=== Delta (Tuned - Baseline) ===")
for m in models:
    b = df_res[(df_res["Model"]==m) & (df_res["Tuned"]=="Baseline")].iloc[0]
    t = df_res[(df_res["Model"]==m) & (df_res["Tuned"]=="Tuned")].iloc[0]
    print(f"  {m:22s}  F1: {b['F1']:.4f} -> {t['F1']:.4f} ({t['F1']-b['F1']:+.4f})"
          f"  AUC: {b['ROC-AUC']:.4f} -> {t['ROC-AUC']:.4f} ({t['ROC-AUC']-b['ROC-AUC']:+.4f})")

# ── plots ──────────────────────────────────────────────────────────────────────
COLORS   = {"Baseline": "#95a5a6", "Tuned": "#e74c3c"}
MODEL_COLORS = {"RandomForest": "#2980b9", "GradientBoosting": "#27ae60", "GraphSAGE": "#8e44ad"}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# PR curves baseline vs tuned per model
for ax, m in zip(axes, models):
    for r in records:
        if r["Model"] != m or r["Model"] == "LOF":
            continue
        p, rec, _ = precision_recall_curve(r["_y_true"], r["_y_score"])
        ap = r["Avg-Prec"]
        ls = "-" if r["Tuned"] == "Tuned" else "--"
        ax.plot(rec, p, ls=ls, label=f"{r['Tuned']} (AP={ap:.3f})",
                color=MODEL_COLORS[m])
    ax.set_title(m); ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.legend(fontsize=9)

plt.suptitle("Baseline vs Tuned — Precision-Recall Curves", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("../reports/tuned_pr_curves.png", dpi=150, bbox_inches="tight")
plt.show()

# F1 delta bar chart
fig, ax = plt.subplots(figsize=(9, 5))
x  = np.arange(len(models))
w  = 0.35
for i, tuned in enumerate(["Baseline", "Tuned"]):
    vals = [df_res[(df_res["Model"]==m) & (df_res["Tuned"]==tuned)].iloc[0]["F1"] for m in models]
    bars = ax.bar(x + i*w - w/2, vals, w, label=tuned,
                  color=[COLORS[tuned]]*3, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f"{v:.3f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(models)
ax.set_ylabel("F1 — illicit class"); ax.set_ylim(0, 1.05)
ax.set_title("F1 Score: Baseline vs Tuned (Optuna, 40 trials)")
ax.legend()
plt.tight_layout()
plt.savefig("../reports/tuned_f1_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nSaved: reports/tuned_leaderboard.csv")
print("Saved: reports/tuned_pr_curves.png")
print("Saved: reports/tuned_f1_comparison.png")
