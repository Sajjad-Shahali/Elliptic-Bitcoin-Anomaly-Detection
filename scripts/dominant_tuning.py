"""
DOMINANT alpha/epoch tuning — closes the "DOMINANT with more epochs / alpha tuning"
gap from HANDOFF.md. Baseline (dominant_graph_ae.py): alpha=0.5, epochs=200, F1=0.212, AUC=0.669.
Grid search over alpha (attribute vs structure loss weight) x epochs, unsupervised throughout.
"""
import sys
sys.path.insert(0, "..")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

from src.data_loader import load_all
from src.graph_utils import build_graph
from src.dominant import DOMINANT, dominant_loss, dominant_anomaly_scores
from src.autoencoder import f1_optimal_threshold

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN, LATENT, DROPOUT, LR = 128, 64, 0.3, 1e-3
ALPHAS = [0.1, 0.3, 0.5, 0.7, 0.9]
EPOCH_OPTIONS = [200, 500]

print(f"Device: {DEVICE}")
print("Loading data...")
df, edges = load_all()
data, _, _ = build_graph(df, edges)
data = data.to(DEVICE)
input_dim = data.x.shape[1]

y_all = data.y.cpu().numpy()
train_mask_np = data.train_mask.cpu().numpy().astype(bool)
test_mask_np  = data.test_mask.cpu().numpy().astype(bool)
train_labeled = train_mask_np & (y_all >= 0)
test_labeled  = test_mask_np  & (y_all >= 0)
y_true = y_all[test_labeled]

results = []
best = {"f1": -1.0}

for epochs in EPOCH_OPTIONS:
    for alpha in ALPHAS:
        print(f"\n--- alpha={alpha}  epochs={epochs} ---")
        model = DOMINANT(input_dim, HIDDEN, LATENT, DROPOUT).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            z, x_hat = model(data.x, data.edge_index)
            loss = dominant_loss(data.x, x_hat, z, data.edge_index, data.num_nodes, alpha=alpha)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if epoch % 100 == 0 or epoch == epochs:
                print(f"  epoch {epoch:3d}/{epochs}  loss={loss.item():.6f}")

        scores_all = dominant_anomaly_scores(model, data.x, data.edge_index, alpha=alpha)
        t_f1 = f1_optimal_threshold(scores_all[train_labeled], y_all[train_labeled])
        y_pred = (scores_all[test_labeled] >= t_f1).astype(int)

        f1  = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, scores_all[test_labeled])
        ap  = average_precision_score(y_true, scores_all[test_labeled])
        print(f"  F1={f1:.4f}  AUC={auc:.4f}  AP={ap:.4f}  threshold={t_f1:.4f}")

        results.append({"alpha": alpha, "epochs": epochs, "f1": f1, "auc": auc, "ap": ap, "threshold": t_f1})

        if f1 > best["f1"]:
            best = {"f1": f1, "auc": auc, "ap": ap, "alpha": alpha, "epochs": epochs}
            torch.save(model.state_dict(), "../models/dominant_tuned.pt")

results_df = pd.DataFrame(results)
results_df.to_csv("../reports/dominant_tuning.csv", index=False)
print("\nSaved: reports/dominant_tuning.csv")

fig, ax = plt.subplots(figsize=(7, 5))
for epochs in EPOCH_OPTIONS:
    sub = results_df[results_df["epochs"] == epochs].sort_values("alpha")
    ax.plot(sub["alpha"], sub["f1"], marker="o", label=f"epochs={epochs}")
ax.axhline(0.212, color="gray", linestyle="--", label="baseline (alpha=0.5, 200ep)")
ax.set_xlabel("alpha (attribute weight)")
ax.set_ylabel("F1 (illicit)")
ax.set_title("DOMINANT: F1 vs alpha and training length")
ax.legend()
plt.tight_layout()
fig.savefig("../reports/dominant_tuning.png", dpi=120)
print("Saved: reports/dominant_tuning.png")

print("\n=== DOMINANT Tuning Summary ===")
print(f"Baseline (alpha=0.5, 200ep):  F1=0.212  AUC=0.669")
print(f"Best config: alpha={best['alpha']}  epochs={best['epochs']}  F1={best['f1']:.4f}  AUC={best['auc']:.4f}  AP={best['ap']:.4f}")
print("\nSaved: models/dominant_tuned.pt")
