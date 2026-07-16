"""
GNNExplainer node-level attribution — closes the "GNNExplainer for node-level
graph attribution" gap from HANDOFF.md.

Explains individual predictions of the best supervised GNN (GraphSAGEv2 + pseudo-labels,
F1=0.7293) on a sample of correctly-classified illicit test nodes: which input features
and which neighboring edges drove the "illicit" prediction. Cross-checks against the
three cross-model robust features already identified via SHAP/gradient attribution
(lf_53, lf_90, af_70 — see README "Explainability").
"""
import sys
sys.path.insert(0, "..")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.explain import Explainer, GNNExplainer

from src.data_loader import load_all
from src.graph_utils import build_graph
from src.gnn import GraphSAGEv2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_NODES_TO_EXPLAIN = 15
ROBUST_FEATURES = {"lf_53", "lf_90", "af_70"}

print(f"Device: {DEVICE}")
print("Loading data...")
df, edges = load_all()
data, node_ids, _ = build_graph(df, edges)
data = data.to(DEVICE)
input_dim = data.x.shape[1]

# feature names match build_graph's feat_cols: everything except txId/class/label
feat_cols = [c for c in df.columns if c not in {"txId", "class", "label"}]
assert len(feat_cols) == input_dim, f"{len(feat_cols)} vs {input_dim}"

print("Loading GraphSAGEv2 + PseudoLabels (models/graphsagev2_pseudo.pt)...")
model = GraphSAGEv2(input_dim=input_dim, hidden_dim=256, output_dim=2, dropout=0.3).to(DEVICE)
model.load_state_dict(torch.load("../models/graphsagev2_pseudo.pt", map_location=DEVICE))
model.eval()

with torch.no_grad():
    out = model(data.x, data.edge_index)
    probs = F.softmax(out, dim=1)[:, 1]
    preds = out.argmax(dim=1)

y = data.y
test_mask = data.test_mask
true_positive = test_mask & (y == 1) & (preds == 1)
tp_idx = true_positive.nonzero(as_tuple=True)[0]
tp_probs = probs[tp_idx]
top_idx = tp_idx[tp_probs.argsort(descending=True)[:N_NODES_TO_EXPLAIN]].tolist()

print(f"True-positive illicit test nodes available: {true_positive.sum().item()}")
print(f"Explaining top-{len(top_idx)} highest-confidence true positives...")

explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type="model",
    node_mask_type="attributes",
    edge_mask_type="object",
    model_config=dict(
        mode="multiclass_classification",
        task_level="node",
        return_type="raw",
    ),
)

rows = []
feature_importance_sum = np.zeros(input_dim)

for node_idx in top_idx:
    explanation = explainer(data.x, data.edge_index, index=node_idx, target=preds)

    node_feat_mask = explanation.node_mask[node_idx].detach().cpu().numpy()
    feature_importance_sum += node_feat_mask
    top_feat_ids = np.argsort(node_feat_mask)[::-1][:10]
    top_feat_names = [feat_cols[i] for i in top_feat_ids]

    edge_mask = explanation.edge_mask.detach().cpu().numpy()
    top_edge_ids = np.argsort(edge_mask)[::-1][:5]
    ei = data.edge_index.cpu().numpy()
    neighbor_txids = []
    for e in top_edge_ids:
        src, dst = ei[0, e], ei[1, e]
        other = dst if src == node_idx else src
        neighbor_txids.append(int(node_ids[other]))

    hit_robust = sorted(ROBUST_FEATURES.intersection(top_feat_names))
    rows.append({
        "node_idx": node_idx,
        "txId": int(node_ids[node_idx]),
        "pred_prob_illicit": float(probs[node_idx].item()),
        "top_features": ",".join(top_feat_names),
        "robust_features_present": ",".join(hit_robust) if hit_robust else "",
        "top_neighbor_txids": ",".join(str(t) for t in neighbor_txids),
    })
    print(f"  node {node_idx} (txId={int(node_ids[node_idx])})  "
          f"p={probs[node_idx].item():.3f}  robust_hit={hit_robust or 'none'}")

summary_df = pd.DataFrame(rows)
summary_df.to_csv("../reports/gnn_explainer_summary.csv", index=False)
print("\nSaved: reports/gnn_explainer_summary.csv")

# aggregate feature importance across the sample
avg_importance = feature_importance_sum / len(top_idx)
top20_idx = np.argsort(avg_importance)[::-1][:20]
top20_names = [feat_cols[i] for i in top20_idx]
top20_vals = avg_importance[top20_idx]

fig, ax = plt.subplots(figsize=(8, 6))
colors = ["crimson" if n in ROBUST_FEATURES else "steelblue" for n in top20_names]
ax.barh(range(len(top20_names))[::-1], top20_vals, color=colors)
ax.set_yticks(range(len(top20_names))[::-1])
ax.set_yticklabels(top20_names)
ax.set_xlabel("Avg GNNExplainer feature-mask importance")
ax.set_title(f"GNNExplainer — top-20 features across {len(top_idx)} illicit test nodes\n(red = cross-model robust feature)")
plt.tight_layout()
fig.savefig("../reports/gnn_explainer_feature_importance.png", dpi=120)
print("Saved: reports/gnn_explainer_feature_importance.png")

n_hits = sum(1 for r in rows if r["robust_features_present"])
print("\n=== GNNExplainer Summary ===")
print(f"Explained {len(top_idx)} true-positive illicit nodes.")
print(f"Robust features (lf_53/lf_90/af_70) appeared in top-10 for {n_hits}/{len(top_idx)} explained nodes.")
print(f"Overall top-5 by avg importance: {top20_names[:5]}")
