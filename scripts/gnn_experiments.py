"""
GNN Experiments — Full progression report.

Trains every GNN variant from scratch in one script:
  Exp 1 : GraphSAGE  2-layer  h=64   baseline architecture
  Exp 2 : GraphSAGE  2-layer  h=128  wider hidden
  Exp 3 : GraphSAGE  2-layer  h=256  even wider
  Exp 4 : GraphSAGE  2-layer  h=128  + longer training (300 ep)
  Exp 5 : GraphSAGE  3-layer  h=128  deeper
  Exp 6 : GraphSAGE  3-layer  h=256  + LayerNorm + self-loops  (GraphSAGEv2)
  Exp 7 : GraphSAGE  3-layer  h=256  + Optuna best params      (best SAGE)
  Exp 8 : GAT        2-layer  4 heads  baseline
  Exp 9 : GAT        2-layer  2 heads  fewer heads
  Exp 10: GAT        3-layer  2 heads  + residual + LayerNorm  (GATv2)

Outputs:
  reports/gnn_experiments_table.csv
  reports/gnn_experiments_progression.png
  reports/gnn_experiments_radar.png
"""
import sys
sys.path.insert(0, '..')

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.utils import add_self_loops
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, precision_score, recall_score

from src.data_loader import load_all
from src.graph_utils import build_graph
from src.gnn import class_weights

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU   : {torch.cuda.get_device_name(0)}")

# ── data ──────────────────────────────────────────────────────────────────────
print("\nBuilding graph...")
df, edges = load_all()
data, _, _ = build_graph(df, edges)
data = data.to(DEVICE)
INPUT_DIM    = data.x.shape[1]
TEST_MASK    = data.test_mask.cpu().numpy().astype(bool)
WTS          = class_weights(data.y[data.train_mask], DEVICE)
CRITERION    = nn.CrossEntropyLoss(weight=WTS)

# ── model definitions ──────────────────────────────────────────────────────────

class SAGE(nn.Module):
    def __init__(self, in_dim, hidden, n_layers=2, dropout=0.3, layernorm=False, selfloops=False):
        super().__init__()
        self.convs      = nn.ModuleList()
        self.norms      = nn.ModuleList()
        self.dropout    = dropout
        self.layernorm  = layernorm
        self.selfloops  = selfloops
        dims = [in_dim] + [hidden] * n_layers
        for i in range(n_layers):
            self.convs.append(SAGEConv(dims[i], dims[i+1]))
            self.norms.append(nn.LayerNorm(dims[i+1]) if layernorm else nn.Identity())
        self.clf = nn.Linear(hidden, 2)

    def forward(self, x, edge_index):
        ei = edge_index
        if self.selfloops:
            ei, _ = add_self_loops(ei, num_nodes=x.size(0))
        for conv, norm in zip(self.convs, self.norms):
            x = norm(F.relu(conv(x, ei)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.clf(x)


class GATNet(nn.Module):
    def __init__(self, in_dim, hidden, n_layers=2, heads=4, dropout=0.3, residual=False, layernorm=False):
        super().__init__()
        self.convs     = nn.ModuleList()
        self.norms     = nn.ModuleList()
        self.residuals = nn.ModuleList()
        self.dropout   = dropout
        self.residual  = residual
        # layer 0: in_dim -> hidden (concat=False to keep dim stable)
        self.convs.append(GATConv(in_dim, hidden, heads=heads, dropout=dropout, concat=False))
        self.norms.append(nn.LayerNorm(hidden) if layernorm else nn.Identity())
        self.residuals.append(nn.Linear(in_dim, hidden, bias=False))
        for _ in range(n_layers - 1):
            self.convs.append(GATConv(hidden, hidden, heads=heads, dropout=dropout, concat=False))
            self.norms.append(nn.LayerNorm(hidden) if layernorm else nn.Identity())
            self.residuals.append(nn.Identity())
        self.clf = nn.Linear(hidden, 2)

    def forward(self, x, edge_index):
        for conv, norm, res in zip(self.convs, self.norms, self.residuals):
            x_in = x
            x = F.elu(conv(x, edge_index))
            if self.residual:
                x = x + res(x_in)
            x = norm(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.clf(x)


# ── training + eval helpers ───────────────────────────────────────────────────

@torch.no_grad()
def eval_model(model):
    model.eval()
    out    = model(data.x, data.edge_index)
    probs  = F.softmax(out, dim=1)[:, 1].cpu().numpy()
    preds  = out.argmax(dim=1).cpu().numpy()
    labels = data.y.cpu().numpy()
    y_true, y_pred, y_score = labels[TEST_MASK], preds[TEST_MASK], probs[TEST_MASK]
    return {
        "F1":        round(f1_score(y_true, y_pred), 4),
        "ROC-AUC":   round(roc_auc_score(y_true, y_score), 4),
        "Avg-Prec":  round(average_precision_score(y_true, y_score), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred), 4),
    }


def run(exp_id, name, model, lr=1e-3, wd=5e-4, epochs=200, log_every=50):
    model = model.to(DEVICE)
    print(f"\n[Exp {exp_id:02d}] {name}")
    print(f"         Params={sum(p.numel() for p in model.parameters()):,}  "
          f"lr={lr}  epochs={epochs}")
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    model.train()
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        out  = model(data.x, data.edge_index)
        loss = CRITERION(out[data.train_mask], data.y[data.train_mask])
        loss.backward(); opt.step(); sch.step()
        if ep % log_every == 0:
            print(f"         ep {ep}/{epochs}  loss={loss.item():.4f}")
    metrics = eval_model(model)
    row = {"Exp": exp_id, "Name": name, **metrics}
    print(f"         F1={metrics['F1']:.4f}  AUC={metrics['ROC-AUC']:.4f}  AP={metrics['Avg-Prec']:.4f}")
    torch.save(model.state_dict(), f"../models/gnn_exp{exp_id:02d}.pt")
    return row


# ── experiments ───────────────────────────────────────────────────────────────
records = []

records.append(run(1,  "GraphSAGE  2L h=64  baseline",
    SAGE(INPUT_DIM, 64,  2, dropout=0.3), epochs=200))

records.append(run(2,  "GraphSAGE  2L h=128",
    SAGE(INPUT_DIM, 128, 2, dropout=0.3), epochs=200))

records.append(run(3,  "GraphSAGE  2L h=256",
    SAGE(INPUT_DIM, 256, 2, dropout=0.3), epochs=200))

records.append(run(4,  "GraphSAGE  2L h=128  300ep",
    SAGE(INPUT_DIM, 128, 2, dropout=0.3), epochs=300))

records.append(run(5,  "GraphSAGE  3L h=128",
    SAGE(INPUT_DIM, 128, 3, dropout=0.3), epochs=200))

records.append(run(6,  "GraphSAGEv2  3L h=256 LN+SL",
    SAGE(INPUT_DIM, 256, 3, dropout=0.3, layernorm=True, selfloops=True), epochs=250))

records.append(run(7,  "GraphSAGE  Optuna best  3L h=256",
    SAGE(INPUT_DIM, 256, 3, dropout=0.45, layernorm=True, selfloops=True),
    lr=0.0038, wd=1.83e-5, epochs=274))

records.append(run(8,  "GAT  2L  4heads baseline",
    GATNet(INPUT_DIM, 64, 2, heads=4, dropout=0.3), lr=5e-4, epochs=200))

records.append(run(9,  "GAT  2L  2heads",
    GATNet(INPUT_DIM, 64, 2, heads=2, dropout=0.4), lr=1e-3, epochs=200))

records.append(run(10, "GATv2  3L  2heads residual+LN",
    GATNet(INPUT_DIM, 64, 3, heads=2, dropout=0.5, residual=True, layernorm=True),
    lr=1e-3, epochs=300))

# ── results table ──────────────────────────────────────────────────────────────
df_res = pd.DataFrame(records)
df_res.to_csv("../reports/gnn_experiments_table.csv", index=False)

print("\n" + "="*80)
print(f"{'Exp':>4}  {'Name':<42}  {'F1':>6}  {'AUC':>6}  {'AP':>6}  {'Prec':>6}  {'Rec':>6}")
print("-"*80)
for _, r in df_res.iterrows():
    marker = " <-- BEST" if r["F1"] == df_res["F1"].max() else ""
    print(f"{int(r['Exp']):>4}  {r['Name']:<42}  {r['F1']:>6.4f}  "
          f"{r['ROC-AUC']:>6.4f}  {r['Avg-Prec']:>6.4f}  "
          f"{r['Precision']:>6.4f}  {r['Recall']:>6.4f}{marker}")
print("="*80)

# ── plot 1: F1 progression ─────────────────────────────────────────────────────
SAGE_COLOR = "#2980b9"
GAT_COLOR  = "#e74c3c"
is_sage = [r["Name"].startswith(("GraphSAGE", "GraphSAGEv2")) for _, r in df_res.iterrows()]

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# F1 bar progression
ax = axes[0]
colors = [SAGE_COLOR if s else GAT_COLOR for s in is_sage]
bars = ax.bar(df_res["Exp"], df_res["F1"], color=colors, edgecolor="white", width=0.7)
for bar, val in zip(bars, df_res["F1"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.006,
            f"{val:.3f}", ha="center", fontsize=8, fontweight="bold")
ax.set_xticks(df_res["Exp"])
ax.set_xticklabels([f"Exp {i}" for i in df_res["Exp"]], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("F1 — illicit class")
ax.set_ylim(0, 1.0)
ax.set_title("GNN Experiments — F1 Progression")
ax.axhline(df_res["F1"].max(), color="gold", lw=1.5, ls="--", label=f"Best F1={df_res['F1'].max():.3f}")
ax.legend(handles=[
    mpatches.Patch(color=SAGE_COLOR, label="GraphSAGE family"),
    mpatches.Patch(color=GAT_COLOR,  label="GAT family"),
], loc="upper left")
ax.legend(fontsize=9)

# 3-metric grouped bar: F1 / AUC / AP per experiment
ax = axes[1]
x   = np.arange(len(df_res))
w   = 0.25
m1  = ax.bar(x - w,   df_res["F1"],       w, label="F1",       color="#2ecc71", edgecolor="white")
m2  = ax.bar(x,       df_res["ROC-AUC"],  w, label="ROC-AUC",  color="#3498db", edgecolor="white")
m3  = ax.bar(x + w,   df_res["Avg-Prec"], w, label="Avg-Prec", color="#9b59b6", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels([f"E{i}" for i in df_res["Exp"]], fontsize=8)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_title("All Metrics per Experiment")
ax.legend(fontsize=9)

plt.suptitle("GNN Experiments — Elliptic Bitcoin Anomaly Detection\n"
             "10 experiments across GraphSAGE and GAT families",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("../reports/gnn_experiments_progression.png", dpi=150, bbox_inches="tight")
plt.show()

# ── plot 2: radar chart (top experiments) ─────────────────────────────────────
from matplotlib.patches import FancyArrowPatch

top = df_res.nlargest(5, "F1")
metrics_radar = ["F1", "ROC-AUC", "Avg-Prec", "Precision", "Recall"]
N = len(metrics_radar)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig_r, ax_r = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
palette = ["#e74c3c","#2980b9","#27ae60","#f39c12","#8e44ad"]

for (_, row), color in zip(top.iterrows(), palette):
    vals = [row[m] for m in metrics_radar] + [row[metrics_radar[0]]]
    ax_r.plot(angles, vals, "o-", lw=2, color=color, label=f"E{int(row['Exp'])}: {row['Name'][:28]}")
    ax_r.fill(angles, vals, alpha=0.08, color=color)

ax_r.set_xticks(angles[:-1])
ax_r.set_xticklabels(metrics_radar, fontsize=11)
ax_r.set_ylim(0, 1)
ax_r.set_title("Top-5 GNN Experiments — Radar Chart", fontsize=13, fontweight="bold", pad=20)
ax_r.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)
plt.tight_layout()
plt.savefig("../reports/gnn_experiments_radar.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nSaved:")
print("  reports/gnn_experiments_table.csv")
print("  reports/gnn_experiments_progression.png")
print("  reports/gnn_experiments_radar.png")
print(f"\nBest: Exp {df_res.loc[df_res['F1'].idxmax(),'Exp']}  "
      f"{df_res.loc[df_res['F1'].idxmax(),'Name']}  "
      f"F1={df_res['F1'].max():.4f}")
