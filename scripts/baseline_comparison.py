"""
Final model comparison — all methods on the same test set (steps 35-49).
Produces leaderboard table + PR/ROC/F1 bar charts.
"""
import sys
sys.path.insert(0, '..')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, precision_recall_curve, roc_curve
)
import torch

from src.data_loader import load_all
from src.preprocessing import (
    prepare_supervised_temporal, prepare_unsupervised_temporal,
    get_labeled, get_feature_cols, TRAIN_STEPS, TEST_STEPS
)
from src.models import (
    isolation_forest, local_outlier_factor, one_class_svm,
    random_forest, gradient_boosting, logistic_regression,
    predict_anomaly, anomaly_scores
)
from src.autoencoder import train_autoencoder, reconstruction_errors, DEVICE as AE_DEVICE
from src.graph_utils import build_graph
from src.gnn import GraphSAGE, evaluate as gnn_evaluate, class_weights
from sklearn.preprocessing import StandardScaler

# ── load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
df, edges = load_all()

# supervised split
X_tr_sup, X_te_sup, y_train, y_test, feat_cols, scaler_sup = prepare_supervised_temporal(df)

# unsupervised split
X_tr_uns, X_te_uns, y_te_uns, _, scaler_uns = prepare_unsupervised_temporal(df)

# licit-only train for OCSVM / AE
labeled_train = get_labeled(df[df['time_step'].isin(TRAIN_STEPS)])
X_licit_sup   = scaler_sup.transform(labeled_train[labeled_train['label'] == 0][feat_cols].values)

labeled_test  = get_labeled(df[df['time_step'].isin(TEST_STEPS)])
# AE uses its own scaler fit on licit-only
scaler_ae = StandardScaler().fit(labeled_train[labeled_train['label'] == 0][feat_cols].values)
X_licit_ae = scaler_ae.transform(labeled_train[labeled_train['label'] == 0][feat_cols].values)
X_te_ae    = scaler_ae.transform(labeled_test[feat_cols].values)

print(f"Test set: {len(y_test):,} transactions  illicit={y_test.mean():.3f}\n")

results = []

def record(name, y_true, y_pred, y_score, category):
    results.append({
        "Model":     name,
        "Category":  category,
        "ROC-AUC":   round(roc_auc_score(y_true, y_score), 4),
        "Avg-Prec":  round(average_precision_score(y_true, y_score), 4),
        "F1":        round(f1_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred), 4),
        "_y_true":   y_true,
        "_y_score":  y_score,
    })
    print(f"  {name:25s}  AUC={results[-1]['ROC-AUC']:.4f}  AP={results[-1]['Avg-Prec']:.4f}  F1={results[-1]['F1']:.4f}")

# ── 1. Isolation Forest ────────────────────────────────────────────────────────
print("[1/8] Isolation Forest")
iforest = isolation_forest(X_tr_uns, contamination=0.1)
y_pred  = predict_anomaly(iforest, X_te_uns)
y_score = anomaly_scores(iforest, X_te_uns)
record("IsolationForest", y_te_uns, y_pred, y_score, "Unsupervised")

# ── 2. One-Class SVM ──────────────────────────────────────────────────────────
print("[2/8] One-Class SVM")
ocsvm  = one_class_svm(X_licit_sup, nu=0.05)
y_pred = predict_anomaly(ocsvm, X_te_sup)
y_score= anomaly_scores(ocsvm, X_te_sup)
record("OneClassSVM", y_test, y_pred, y_score, "Unsupervised")

# ── 3. LOF ────────────────────────────────────────────────────────────────────
print("[3/8] LOF (transductive)")
y_pred_lof, _ = local_outlier_factor(X_te_uns, contamination=y_test.mean())
# LOF has no continuous score in transductive mode — use pred as score
record("LOF", y_te_uns, y_pred_lof, y_pred_lof.astype(float), "Unsupervised")

# ── 4. Logistic Regression ────────────────────────────────────────────────────
print("[4/8] Logistic Regression")
lr     = logistic_regression(X_tr_sup, y_train)
y_pred = predict_anomaly(lr, X_te_sup)
y_score= anomaly_scores(lr, X_te_sup)
record("LogisticRegression", y_test, y_pred, y_score, "Supervised")

# ── 5. Random Forest ──────────────────────────────────────────────────────────
print("[5/8] Random Forest")
rf     = random_forest(X_tr_sup, y_train)
y_pred = predict_anomaly(rf, X_te_sup)
y_score= anomaly_scores(rf, X_te_sup)
record("RandomForest", y_test, y_pred, y_score, "Supervised")

# ── 6. Gradient Boosting ──────────────────────────────────────────────────────
print("[6/8] Gradient Boosting")
gb     = gradient_boosting(X_tr_sup, y_train)
y_pred = predict_anomaly(gb, X_te_sup)
y_score= anomaly_scores(gb, X_te_sup)
record("GradientBoosting", y_test, y_pred, y_score, "Supervised")

# ── 7. Autoencoder ────────────────────────────────────────────────────────────
print(f"[7/8] Autoencoder (device={AE_DEVICE})")
ae = train_autoencoder(X_licit_ae, input_dim=X_licit_ae.shape[1], latent_dim=32, epochs=100)
te_errors = reconstruction_errors(ae, X_te_ae)
# auto-detect score direction
auc_fwd = roc_auc_score(y_test, te_errors)
ae_score = -te_errors if roc_auc_score(y_test, -te_errors) > auc_fwd else te_errors
from sklearn.metrics import roc_auc_score as _auc
threshold = np.percentile(ae_score, 100 * (1 - y_test.mean()))
y_pred_ae = (ae_score >= threshold).astype(int)
record("Autoencoder", y_test, y_pred_ae, ae_score, "Neural")

# ── 8. GraphSAGE ──────────────────────────────────────────────────────────────
print("[8/8] GraphSAGE")
GNN_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data, node_ids, _ = build_graph(df, edges)
data = data.to(GNN_DEVICE)

gnn = GraphSAGE(input_dim=data.x.shape[1], hidden_dim=128, output_dim=2, dropout=0.3).to(GNN_DEVICE)
optimizer  = torch.optim.Adam(gnn.parameters(), lr=1e-3, weight_decay=5e-4)
scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
weights    = class_weights(data.y[data.train_mask], GNN_DEVICE)
criterion  = torch.nn.CrossEntropyLoss(weight=weights)

gnn.train()
for epoch in range(1, 201):
    optimizer.zero_grad()
    out  = gnn(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward(); optimizer.step(); scheduler.step()
    if epoch % 50 == 0:
        print(f"  epoch {epoch}/200  loss={loss.item():.4f}")

test_mask_np = data.test_mask.cpu().numpy().astype(bool)
y_pred_gnn, y_score_gnn, y_true_gnn = gnn_evaluate(gnn, data, test_mask_np, GNN_DEVICE)
record("GraphSAGE", y_true_gnn, y_pred_gnn, y_score_gnn, "GNN")

# ── leaderboard table ──────────────────────────────────────────────────────────
print("\n" + "="*70)
df_results = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in results])
df_results = df_results.sort_values("F1", ascending=False).reset_index(drop=True)
df_results.index += 1
print(df_results.to_string())
df_results.to_csv("../reports/leaderboard.csv", index=True)
print("\nSaved: reports/leaderboard.csv")

# ── plots ──────────────────────────────────────────────────────────────────────
COLORS = {
    "Unsupervised": "#e67e22",
    "Supervised":   "#2980b9",
    "Neural":       "#8e44ad",
    "GNN":          "#27ae60",
}
CAT_ORDER = ["Unsupervised", "Supervised", "Neural", "GNN"]

fig = plt.figure(figsize=(20, 16))
gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.3)

# 1. PR curves
ax1 = fig.add_subplot(gs[0, 0])
for r in results:
    if r["Model"] == "LOF":
        continue  # no continuous score
    p, rec, _ = precision_recall_curve(r["_y_true"], r["_y_score"])
    ap  = r["Avg-Prec"]
    cat = r["Category"]
    ax1.plot(rec, p, label=f"{r['Model']} (AP={ap:.3f})", color=COLORS[cat])
ax1.set_xlabel("Recall"); ax1.set_ylabel("Precision")
ax1.set_title("Precision-Recall Curves"); ax1.legend(fontsize=7.5)

# 2. ROC curves
ax2 = fig.add_subplot(gs[0, 1])
for r in results:
    if r["Model"] == "LOF":
        continue
    fpr, tpr, _ = roc_curve(r["_y_true"], r["_y_score"])
    ax2.plot(fpr, tpr, label=f"{r['Model']} (AUC={r['ROC-AUC']:.3f})", color=COLORS[r["Category"]])
ax2.plot([0,1],[0,1],"k--",lw=0.8)
ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR")
ax2.set_title("ROC Curves"); ax2.legend(fontsize=7.5)

# 3. F1 bar chart
ax3 = fig.add_subplot(gs[1, 0])
df_sorted = df_results.sort_values("F1")
colors_bar = [COLORS[c] for c in df_sorted["Category"]]
bars = ax3.barh(df_sorted["Model"], df_sorted["F1"], color=colors_bar)
ax3.set_xlabel("F1 — illicit class"); ax3.set_title("F1 Score Comparison")
ax3.axvline(0, color="gray", lw=0.5)
for bar, val in zip(bars, df_sorted["F1"]):
    ax3.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
             f"{val:.3f}", va="center", fontsize=8)
# legend
from matplotlib.patches import Patch
ax3.legend(handles=[Patch(color=COLORS[c], label=c) for c in CAT_ORDER],
           fontsize=8, loc="lower right")

# 4. Multi-metric radar-style grouped bar
ax4 = fig.add_subplot(gs[1, 1])
metrics = ["ROC-AUC", "Avg-Prec", "F1", "Precision", "Recall"]
x = np.arange(len(metrics))
width = 0.8 / len(df_results)
for i, (_, row) in enumerate(df_results.iterrows()):
    vals = [row[m] for m in metrics]
    ax4.bar(x + i*width - 0.4 + width/2, vals, width,
            label=row["Model"], color=COLORS[row["Category"]], alpha=0.85)
ax4.set_xticks(x); ax4.set_xticklabels(metrics, fontsize=9)
ax4.set_ylim(0, 1.05); ax4.set_title("All Metrics — All Models")
ax4.legend(fontsize=6.5, ncol=2)

plt.suptitle("Elliptic Bitcoin Anomaly Detection — Final Model Comparison\n"
             "Test set: time steps 35-49  |  Illicit rate: 6.5%",
             fontsize=13, fontweight="bold", y=1.01)
plt.savefig("../reports/final_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: reports/final_comparison.png")
