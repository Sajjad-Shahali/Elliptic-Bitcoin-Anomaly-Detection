import sys
sys.path.insert(0, '..')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import load_all
from src.preprocessing import prepare_supervised_temporal
from src.models import random_forest, gradient_boosting, logistic_regression, predict_anomaly, anomaly_scores
from src.evaluation import evaluate, plot_confusion_matrix, plot_pr_curve, plot_roc_curve

df, edges = load_all()
X_train, X_test, y_train, y_test, feat_cols, scaler = prepare_supervised_temporal(df)

print(f"Train (steps 1-34): {X_train.shape}  illicit: {y_train.mean():.3f}")
print(f"Test  (steps 35-49): {X_test.shape}  illicit: {y_test.mean():.3f}")

# --- Random Forest ---
print("\n[1/3] Random Forest...")
rf = random_forest(X_train, y_train)
y_pred_rf  = predict_anomaly(rf, X_test)
y_score_rf = anomaly_scores(rf, X_test)
res_rf = evaluate(y_test, y_pred_rf, y_score_rf, name='RandomForest')
plot_confusion_matrix(y_test, y_pred_rf, name='RandomForest', save=True)

# --- Gradient Boosting ---
print("\n[2/3] Gradient Boosting...")
gb = gradient_boosting(X_train, y_train)
y_pred_gb  = predict_anomaly(gb, X_test)
y_score_gb = anomaly_scores(gb, X_test)
res_gb = evaluate(y_test, y_pred_gb, y_score_gb, name='GradientBoosting')
plot_confusion_matrix(y_test, y_pred_gb, name='GradientBoosting', save=True)

# --- Logistic Regression ---
print("\n[3/3] Logistic Regression...")
lr = logistic_regression(X_train, y_train)
y_pred_lr  = predict_anomaly(lr, X_test)
y_score_lr = anomaly_scores(lr, X_test)
res_lr = evaluate(y_test, y_pred_lr, y_score_lr, name='LogisticRegression')
plot_confusion_matrix(y_test, y_pred_lr, name='LogisticRegression', save=True)

# --- comparison ---
results = pd.DataFrame([res_rf, res_gb, res_lr]).set_index('name')
print("\n=== Summary ===")
print(results.sort_values('f1', ascending=False))

plot_pr_curve(y_test, {
    'RandomForest': y_score_rf,
    'GradientBoosting': y_score_gb,
    'LogisticRegression': y_score_lr,
}, save=True)

plot_roc_curve(y_test, {
    'RandomForest': y_score_rf,
    'GradientBoosting': y_score_gb,
    'LogisticRegression': y_score_lr,
}, save=True)

# --- feature importance ---
importances = rf.feature_importances_
top_idx   = np.argsort(importances)[::-1][:20]
top_feats = [feat_cols[i] for i in top_idx]
top_vals  = importances[top_idx]

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(top_feats, top_vals)
ax.set_xticklabels(top_feats, rotation=45, ha='right')
ax.set_title('Top 20 feature importances — Random Forest')
plt.tight_layout()
plt.savefig('../reports/feature_importance.png', dpi=120)
plt.show()
