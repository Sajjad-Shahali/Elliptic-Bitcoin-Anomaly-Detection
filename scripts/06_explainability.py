import sys
sys.path.insert(0, '..')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import torch

from src.data_loader import load_all
from src.preprocessing import prepare_supervised_temporal
from src.models import random_forest, gradient_boosting

# ── data ──────────────────────────────────────────────────────────────────────
df, edges = load_all()
X_train, X_test, y_train, y_test, feat_cols, scaler = prepare_supervised_temporal(df)
feat_names = feat_cols  # list of strings

print(f"Train: {X_train.shape}  Test: {X_test.shape}")
print(f"Illicit in test: {y_test.sum()} / {len(y_test)}")

# ── train models ──────────────────────────────────────────────────────────────
print("\nTraining Random Forest...")
rf = random_forest(X_train, y_train)

print("Training Gradient Boosting...")
gb = gradient_boosting(X_train, y_train)

# ── 1. Random Forest — TreeExplainer ──────────────────────────────────────────
print("\nComputing SHAP values (RF)...")
rf_explainer = shap.TreeExplainer(rf)

# use test set, subsample for speed
rng = np.random.default_rng(42)
idx = rng.choice(len(X_test), size=min(2000, len(X_test)), replace=False)
X_sample = X_test[idx]
y_sample = y_test[idx]

rf_shap = rf_explainer.shap_values(X_sample)
# SHAP >= 0.40: may be ndarray (n_samples, n_features, n_classes) or list
if isinstance(rf_shap, list):
    rf_shap_illicit = rf_shap[1]
else:
    rf_shap_illicit = rf_shap[..., 1] if rf_shap.ndim == 3 else rf_shap

# summary bar
shap.summary_plot(rf_shap_illicit, X_sample, feature_names=feat_names,
                  plot_type="bar", max_display=20, show=False)
plt.title("RF — Mean |SHAP| for illicit class (top 20 features)")
plt.tight_layout()
plt.savefig("../reports/shap_rf_bar.png", dpi=120, bbox_inches="tight")
plt.show()

# beeswarm (shows direction + magnitude)
shap.summary_plot(rf_shap_illicit, X_sample, feature_names=feat_names,
                  max_display=20, show=False)
plt.title("RF — SHAP beeswarm (illicit class)")
plt.tight_layout()
plt.savefig("../reports/shap_rf_beeswarm.png", dpi=120, bbox_inches="tight")
plt.show()

# ── 2. Gradient Boosting — TreeExplainer ──────────────────────────────────────
print("\nComputing SHAP values (GBM)...")
gb_explainer = shap.TreeExplainer(gb)
gb_shap = gb_explainer.shap_values(X_sample)
if isinstance(gb_shap, list):
    gb_shap = gb_shap[1]
elif gb_shap.ndim == 3:
    gb_shap = gb_shap[..., 1]

shap.summary_plot(gb_shap, X_sample, feature_names=feat_names,
                  plot_type="bar", max_display=20, show=False)
plt.title("GBM — Mean |SHAP| for illicit class (top 20 features)")
plt.tight_layout()
plt.savefig("../reports/shap_gbm_bar.png", dpi=120, bbox_inches="tight")
plt.show()

# ── 3. Top-feature dependence plots (RF) ──────────────────────────────────────
top_features = np.argsort(np.abs(rf_shap_illicit).mean(axis=0))[::-1][:3]
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, fi in zip(axes, top_features):
    fname = feat_names[fi]
    vals  = X_sample[:, fi]
    sv    = rf_shap_illicit[:, fi]
    sc = ax.scatter(vals, sv, c=y_sample, cmap="RdYlGn_r", alpha=0.4, s=8)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel(fname)
    ax.set_ylabel("SHAP value")
    ax.set_title(f"Dependence: {fname}")
plt.colorbar(sc, ax=axes[-1], label="label (1=illicit)")
plt.suptitle("RF — SHAP dependence plots (top 3 features)", y=1.02)
plt.tight_layout()
plt.savefig("../reports/shap_rf_dependence.png", dpi=120, bbox_inches="tight")
plt.show()

# ── 4. Waterfall for individual predictions ────────────────────────────────────
# pick one correctly-predicted illicit transaction
illicit_idx = np.where((y_sample == 1))[0]
if len(illicit_idx):
    i = illicit_idx[0]
    exp = shap.Explanation(
        values=rf_shap_illicit[i],
        base_values=rf_explainer.expected_value[1] if hasattr(rf_explainer.expected_value, '__len__') else rf_explainer.expected_value,
        data=X_sample[i],
        feature_names=feat_names,
    )
    shap.plots.waterfall(exp, max_display=15, show=False)
    plt.title("RF — Waterfall: single illicit transaction")
    plt.tight_layout()
    plt.savefig("../reports/shap_rf_waterfall_illicit.png", dpi=120, bbox_inches="tight")
    plt.show()

# ── 5. Feature overlap RF vs GBM ──────────────────────────────────────────────
rf_top20  = set(np.argsort(np.abs(rf_shap_illicit).mean(axis=0))[::-1][:20])
gbm_top20 = set(np.argsort(np.abs(gb_shap).mean(axis=0))[::-1][:20])
overlap   = rf_top20 & gbm_top20
print(f"\nTop-20 feature overlap RF vs GBM: {len(overlap)}/20")
print("Shared:", [feat_names[i] for i in sorted(overlap)])

# ── 6. GraphSAGE gradient attribution ─────────────────────────────────────────
print("\nComputing gradient attribution (GraphSAGE)...")
from src.graph_utils import build_graph
from src.gnn import GraphSAGE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data, node_ids, _ = build_graph(df, edges)
data = data.to(DEVICE)

input_dim = data.x.shape[1]
gnn_model = GraphSAGE(input_dim=input_dim, hidden_dim=128, output_dim=2).to(DEVICE)
gnn_model.load_state_dict(torch.load("../models/graphsage.pt", map_location=DEVICE))
gnn_model.eval()

# gradient of illicit logit w.r.t. input features, averaged over test nodes
data.x.requires_grad_(True)
out = gnn_model(data.x, data.edge_index)
illicit_logits = out[:, 1]  # illicit class logit

test_mask = data.test_mask
illicit_logits[test_mask].mean().backward()

grad = data.x.grad[test_mask].abs().cpu().detach().numpy()
mean_grad = grad.mean(axis=0)

top_gnn_idx = np.argsort(mean_grad)[::-1][:20]
top_gnn_feats = [feat_names[i] for i in top_gnn_idx]
top_gnn_vals  = mean_grad[top_gnn_idx]

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(top_gnn_feats, top_gnn_vals, color="steelblue")
ax.set_xticklabels(top_gnn_feats, rotation=45, ha="right")
ax.set_ylabel("Mean |gradient|")
ax.set_title("GraphSAGE — Input gradient attribution (top 20 features)")
plt.tight_layout()
plt.savefig("../reports/gnn_gradient_attribution.png", dpi=120)
plt.show()

# ── 7. Cross-model feature agreement ──────────────────────────────────────────
gnn_top20 = set(top_gnn_idx[:20])
triple_overlap = rf_top20 & gbm_top20 & gnn_top20
print(f"\nTriple overlap RF + GBM + GNN top-20: {len(triple_overlap)}/20")
print("Agreed features:", [feat_names[i] for i in sorted(triple_overlap)])
