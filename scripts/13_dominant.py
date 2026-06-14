"""
Variant 19 — DOMINANT: GCN-based attributed-graph autoencoder for unsupervised
node anomaly detection. Jointly reconstructs node features and graph structure.
Reference: Ding et al., "Deep Anomaly Detection on Attributed Networks," SDM 2019.
"""
import sys
sys.path.insert(0, "..")

import torch
import numpy as np

from src.data_loader import load_all
from src.graph_utils import build_graph
from src.dominant import DOMINANT, dominant_loss, dominant_anomaly_scores
from src.autoencoder import f1_optimal_threshold, evt_threshold
from src.evaluation import evaluate as eval_metrics, plot_confusion_matrix, plot_pr_curve

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS   = 200
LR       = 1e-3
HIDDEN   = 128
LATENT   = 64
ALPHA    = 0.5   # attribute vs structure weight (tune if needed)
DROPOUT  = 0.3

print(f"Device: {DEVICE}")
print("Loading data...")
df, edges = load_all()
data, node_ids, _ = build_graph(df, edges)
data = data.to(DEVICE)

input_dim = data.x.shape[1]
model     = DOMINANT(input_dim, HIDDEN, LATENT, DROPOUT).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
print(f"Graph: {data.num_nodes:,} nodes  {data.num_edges:,} edges")
print(f"Training DOMINANT {EPOCHS} epochs  alpha={ALPHA}  (unsupervised — no labels used)")
print("-" * 60)

for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()
    z, x_hat = model(data.x, data.edge_index)
    loss = dominant_loss(data.x, x_hat, z, data.edge_index, data.num_nodes, alpha=ALPHA)
    loss.backward()
    optimizer.step()
    scheduler.step()
    if epoch % 10 == 0:
        lr_now = scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={loss.item():.6f}  lr={lr_now:.2e}")

# ── anomaly scores ─────────────────────────────────────────────────────────────
print("\nComputing anomaly scores...")
scores_all = dominant_anomaly_scores(model, data.x, data.edge_index, alpha=ALPHA)

y_all          = data.y.cpu().numpy()
train_mask_np  = data.train_mask.cpu().numpy().astype(bool)
test_mask_np   = data.test_mask.cpu().numpy().astype(bool)
train_labeled  = train_mask_np & (y_all >= 0)
test_labeled   = test_mask_np  & (y_all >= 0)

# ── threshold options ──────────────────────────────────────────────────────────
# (1) Optimal F1 on train labeled (oracle-like; for upper-bound comparison)
t_f1  = f1_optimal_threshold(scores_all[train_labeled], y_all[train_labeled])

# (2) EVT-GPD threshold calibrated on train labeled scores
t_evt = evt_threshold(scores_all[train_labeled], tail_quantile=0.90, exceedance_prob=0.065)

y_true = y_all[test_labeled]

for tag, t in [("@F1opt", t_f1), ("@EVT",  t_evt)]:
    y_pred = (scores_all[test_labeled] >= t).astype(int)
    res = eval_metrics(y_true, y_pred, y_score=scores_all[test_labeled],
                       name=f"DOMINANT{tag}")
    print(f"  threshold={t:.4f}  F1={res['f1']:.4f}")

# ── save best (F1-optimal threshold) ──────────────────────────────────────────
y_pred_best = (scores_all[test_labeled] >= t_f1).astype(int)
plot_confusion_matrix(y_true, y_pred_best, name="DOMINANT", save=True)
plot_pr_curve(y_true, {"DOMINANT": scores_all[test_labeled]}, save=True)

torch.save(model.state_dict(), "../models/dominant.pt")
print("\nSaved: models/dominant.pt")
