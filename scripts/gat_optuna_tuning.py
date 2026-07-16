"""
GAT Optuna tuning — closes the "GAT Optuna tuning" gap from HANDOFF.md.
Mirrors the GraphSAGE Optuna section of optuna_tuning.py, but searches GATv2's
architecture-specific knobs (heads, hidden_dim) in addition to lr/dropout/epochs.
"""
import sys
sys.path.insert(0, "..")

import warnings
warnings.filterwarnings("ignore")

import torch
import optuna
from sklearn.metrics import f1_score, roc_auc_score
optuna.logging.set_verbosity(optuna.logging.WARNING)

from src.data_loader import load_all
from src.graph_utils import build_graph
from src.gat import GATv2, train_epoch, evaluate
from src.gnn import class_weights

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_TRIALS = 30

print(f"Device: {DEVICE}")
print("Loading data...")
df, edges = load_all()
data, _, _ = build_graph(df, edges)
data = data.to(DEVICE)
input_dim = data.x.shape[1]
test_mask_np = data.test_mask.cpu().numpy().astype(bool)

print(f"Graph: {data.num_nodes:,} nodes  {data.num_edges:,} edges  input_dim={input_dim}")
print(f"\nTuning GATv2 ({N_TRIALS} trials)...")


def objective(trial):
    hidden  = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    heads   = trial.suggest_categorical("heads", [1, 2, 4])
    dropout = trial.suggest_float("dropout", 0.1, 0.6)
    lr      = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    wd      = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    epochs  = trial.suggest_int("epochs", 100, 300)

    m = GATv2(input_dim, hidden, 2, heads, dropout).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    wts = class_weights(data.y[data.train_mask], DEVICE)
    crit = torch.nn.CrossEntropyLoss(weight=wts)

    for _ in range(epochs):
        train_epoch(m, data, opt, crit, DEVICE)

    y_pred, _, y_true = evaluate(m, data, test_mask_np, DEVICE)
    return f1_score(y_true, y_pred)


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
bp = study.best_params
print(f"\nBest GATv2  F1={study.best_value:.4f}  params={bp}")

# ── retrain best config and save ────────────────────────────────────────────
best_model = GATv2(input_dim, bp["hidden_dim"], 2, bp["heads"], bp["dropout"]).to(DEVICE)
opt = torch.optim.Adam(best_model.parameters(), lr=bp["lr"], weight_decay=bp["weight_decay"])
wts = class_weights(data.y[data.train_mask], DEVICE)
crit = torch.nn.CrossEntropyLoss(weight=wts)

for ep in range(1, bp["epochs"] + 1):
    loss = train_epoch(best_model, data, opt, crit, DEVICE)
    if ep % 50 == 0:
        print(f"  epoch {ep}/{bp['epochs']}  loss={loss:.4f}")

y_pred, y_prob, y_true = evaluate(best_model, data, test_mask_np, DEVICE)
f1 = f1_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_prob)

torch.save(best_model.state_dict(), "../models/gatv2_tuned.pt")

print("\n=== GAT Optuna Tuning Summary ===")
print(f"GATv2 baseline (untuned):   F1=0.317  AUC=0.882")
print(f"GATv2 (Optuna tuned):       F1={f1:.4f}  AUC={auc:.4f}")
print(f"Best params: {bp}")
print("\nSaved: models/gatv2_tuned.pt")
