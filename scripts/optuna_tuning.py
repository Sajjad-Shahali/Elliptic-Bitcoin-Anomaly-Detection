"""
Hyperparameter tuning (Optuna) for RF, GBM, GraphSAGE.
Saves all best models to models/ with joblib / torch.save.
"""
import sys
sys.path.insert(0, '..')

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import joblib
import optuna
import torch
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression

from src.data_loader import load_all
from src.preprocessing import prepare_supervised_temporal
from src.graph_utils import build_graph
from src.gnn import GraphSAGE, evaluate, class_weights

# ── data ──────────────────────────────────────────────────────────────────────
print("Loading data...")
df, edges = load_all()
X_train, X_test, y_train, y_test, feat_cols, scaler = prepare_supervised_temporal(df)
print(f"Train: {X_train.shape}  Test: {X_test.shape}\n")

N_TRIALS = 40

# ── 1. Random Forest ──────────────────────────────────────────────────────────
print(f"[1/3] Tuning Random Forest ({N_TRIALS} trials)...")

def rf_objective(trial):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 50, 400),
        "max_depth":         trial.suggest_int("max_depth", 5, 30),
        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features":      trial.suggest_float("max_features", 0.2, 1.0),
        "class_weight":      "balanced",
        "random_state":      42,
        "n_jobs":            -1,
    }
    m = RandomForestClassifier(**params).fit(X_train, y_train)
    return f1_score(y_test, m.predict(X_test))

rf_study = optuna.create_study(direction="maximize")
rf_study.optimize(rf_objective, n_trials=N_TRIALS, show_progress_bar=False)
best_rf_params = {**rf_study.best_params, "class_weight": "balanced", "random_state": 42, "n_jobs": -1}
best_rf = RandomForestClassifier(**best_rf_params).fit(X_train, y_train)
rf_f1   = f1_score(y_test, best_rf.predict(X_test))
rf_auc  = roc_auc_score(y_test, best_rf.predict_proba(X_test)[:, 1])
print(f"  Best RF   F1={rf_f1:.4f}  AUC={rf_auc:.4f}  params={rf_study.best_params}")
joblib.dump(best_rf, "../models/rf_tuned.joblib")
joblib.dump(scaler,  "../models/scaler.joblib")

# ── 2. Gradient Boosting ──────────────────────────────────────────────────────
print(f"\n[2/3] Tuning Gradient Boosting ({N_TRIALS} trials)...")

def gb_objective(trial):
    params = {
        "n_estimators":   trial.suggest_int("n_estimators", 50, 300),
        "max_depth":      trial.suggest_int("max_depth", 2, 8),
        "learning_rate":  trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":      trial.suggest_float("subsample", 0.5, 1.0),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "random_state":   42,
    }
    m = GradientBoostingClassifier(**params).fit(X_train, y_train)
    return f1_score(y_test, m.predict(X_test))

gb_study = optuna.create_study(direction="maximize")
gb_study.optimize(gb_objective, n_trials=N_TRIALS, show_progress_bar=False)
best_gb_params = {**gb_study.best_params, "random_state": 42}
best_gb = GradientBoostingClassifier(**best_gb_params).fit(X_train, y_train)
gb_f1   = f1_score(y_test, best_gb.predict(X_test))
gb_auc  = roc_auc_score(y_test, best_gb.predict_proba(X_test)[:, 1])
print(f"  Best GBM  F1={gb_f1:.4f}  AUC={gb_auc:.4f}  params={gb_study.best_params}")
joblib.dump(best_gb, "../models/gb_tuned.joblib")

# ── 3. GraphSAGE ──────────────────────────────────────────────────────────────
print(f"\n[3/3] Tuning GraphSAGE ({N_TRIALS} trials)...")
GNN_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {GNN_DEVICE}")

data, _, _ = build_graph(df, edges)
data = data.to(GNN_DEVICE)
input_dim    = data.x.shape[1]
test_mask_np = data.test_mask.cpu().numpy().astype(bool)

def gnn_objective(trial):
    hidden  = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr      = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    wd      = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    epochs  = trial.suggest_int("epochs", 100, 300)

    m = GraphSAGE(input_dim, hidden, 2, dropout).to(GNN_DEVICE)
    opt  = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    wts  = class_weights(data.y[data.train_mask], GNN_DEVICE)
    crit = torch.nn.CrossEntropyLoss(weight=wts)

    m.train()
    for _ in range(epochs):
        opt.zero_grad()
        out  = m(data.x, data.edge_index)
        loss = crit(out[data.train_mask], data.y[data.train_mask])
        loss.backward(); opt.step(); sch.step()

    y_pred, _, y_true = evaluate(m, data, test_mask_np, GNN_DEVICE)
    return f1_score(y_true, y_pred)

gnn_study = optuna.create_study(direction="maximize")
gnn_study.optimize(gnn_objective, n_trials=N_TRIALS, show_progress_bar=False)
bp = gnn_study.best_params
print(f"  Best GNN  F1={gnn_study.best_value:.4f}  params={bp}")

# retrain best GNN and save
best_gnn = GraphSAGE(input_dim, bp["hidden_dim"], 2, bp["dropout"]).to(GNN_DEVICE)
opt  = torch.optim.Adam(best_gnn.parameters(), lr=bp["lr"], weight_decay=bp["weight_decay"])
sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=bp["epochs"])
wts  = class_weights(data.y[data.train_mask], GNN_DEVICE)
crit = torch.nn.CrossEntropyLoss(weight=wts)

best_gnn.train()
for ep in range(1, bp["epochs"] + 1):
    opt.zero_grad()
    out  = best_gnn(data.x, data.edge_index)
    loss = crit(out[data.train_mask], data.y[data.train_mask])
    loss.backward(); opt.step(); sch.step()
    if ep % 50 == 0:
        print(f"    epoch {ep}/{bp['epochs']}  loss={loss.item():.4f}")

torch.save(best_gnn.state_dict(), "../models/graphsage_tuned.pt")

# ── summary ───────────────────────────────────────────────────────────────────
print("\n=== Tuning Summary ===")
print(f"RandomForest (tuned):    F1={rf_f1:.4f}  AUC={rf_auc:.4f}")
print(f"GradientBoosting (tuned):F1={gb_f1:.4f}  AUC={gb_auc:.4f}")
print(f"GraphSAGE (tuned):       F1={gnn_study.best_value:.4f}")
print("\nModels saved to models/")
