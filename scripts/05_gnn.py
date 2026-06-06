import sys
sys.path.insert(0, '..')

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.data_loader import load_all
from src.graph_utils import build_graph
from src.gnn import GraphSAGE, train_epoch, evaluate, class_weights
from src.evaluation import evaluate as eval_metrics, plot_confusion_matrix, plot_pr_curve

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS  = 200
LR      = 1e-3
HIDDEN  = 128

print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# --- build graph ---
df, edges = load_all()
data, node_ids, scaler = build_graph(df, edges)

# move to device incrementally (full graph fits in 12 GB VRAM)
data = data.to(DEVICE)

# --- model + optimizer ---
input_dim = data.x.shape[1]
model = GraphSAGE(input_dim=input_dim, hidden_dim=HIDDEN, output_dim=2, dropout=0.3).to(DEVICE)

y_train = data.y[data.train_mask]
weights = class_weights(y_train, DEVICE)
criterion = torch.nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

print(f"\nModel params: {sum(p.numel() for p in model.parameters()):,}")
print(f"Class weights: licit={weights[0]:.3f}  illicit={weights[1]:.3f}")
print(f"\nTraining GraphSAGE for {EPOCHS} epochs...")

# --- training loop ---
losses = []
for epoch in range(1, EPOCHS + 1):
    loss = train_epoch(model, data, optimizer, criterion, DEVICE)
    scheduler.step()
    losses.append(loss)
    if epoch % 40 == 0:
        tr_mask = data.train_mask.cpu().numpy().astype(bool)
        te_mask = data.test_mask.cpu().numpy().astype(bool)
        _, tr_probs, tr_labels = evaluate(model, data, tr_mask, DEVICE)
        _, te_probs, te_labels = evaluate(model, data, te_mask, DEVICE)
        from sklearn.metrics import roc_auc_score
        tr_auc = roc_auc_score(tr_labels, tr_probs)
        te_auc = roc_auc_score(te_labels, te_probs)
        print(f"  Epoch {epoch:3d}  loss={loss:.4f}  train_auc={tr_auc:.4f}  test_auc={te_auc:.4f}")

# --- loss curve ---
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(losses)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("GraphSAGE training loss")
plt.tight_layout()
plt.savefig("../reports/gnn_loss.png", dpi=120)
plt.show()

# --- final evaluation ---
test_mask_np = data.test_mask.cpu().numpy().astype(bool)
y_pred, y_score, y_true = evaluate(model, data, test_mask_np, DEVICE)

res = eval_metrics(y_true, y_pred, y_score, name="GraphSAGE")
plot_confusion_matrix(y_true, y_pred, name="GraphSAGE", save=True)
plot_pr_curve(y_true, {"GraphSAGE": y_score}, save=True)

# --- save ---
torch.save(model.state_dict(), "../models/graphsage.pt")
print("\nModel saved to models/graphsage.pt")
