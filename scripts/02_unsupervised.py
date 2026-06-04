import sys
sys.path.insert(0, '..')

from src.data_loader import load_all
from src.preprocessing import prepare_unsupervised_temporal
from src.models import isolation_forest, local_outlier_factor, one_class_svm, predict_anomaly, anomaly_scores
from src.evaluation import evaluate, plot_confusion_matrix, plot_pr_curve

df, edges = load_all()
X_train, X_test, y_test, feat_cols, scaler = prepare_unsupervised_temporal(df)

print(f"Train (all, steps 1-34): {X_train.shape}")
print(f"Test  (labeled, steps 35-49): {X_test.shape}")
print(f"Illicit rate in test: {y_test.mean():.3f}")

# --- Isolation Forest ---
print("\n[1/3] Isolation Forest...")
iforest = isolation_forest(X_train, contamination=0.1)
y_pred_if  = predict_anomaly(iforest, X_test)
y_score_if = anomaly_scores(iforest, X_test)
res_if = evaluate(y_test, y_pred_if, y_score_if, name='IsolationForest')
plot_confusion_matrix(y_test, y_pred_if, name='IsolationForest', save=True)

# --- One-Class SVM (train on licit-only train rows) ---
print("\n[2/3] One-Class SVM...")
# use licit rows from train for OCSVM — needs label info
from src.preprocessing import get_labeled, get_feature_cols, TRAIN_STEPS
import pandas as pd
labeled_train = get_labeled(df[df['time_step'].isin(TRAIN_STEPS)])
X_licit_train = scaler.transform(labeled_train[labeled_train['label'] == 0][feat_cols].values)

ocsvm = one_class_svm(X_licit_train, nu=0.05)
y_pred_svm  = predict_anomaly(ocsvm, X_test)
y_score_svm = anomaly_scores(ocsvm, X_test)
res_svm = evaluate(y_test, y_pred_svm, y_score_svm, name='OneClassSVM')
plot_confusion_matrix(y_test, y_pred_svm, name='OneClassSVM', save=True)

# --- LOF (on test set only — transductive) ---
print("\n[3/3] Local Outlier Factor...")
y_pred_lof, _ = local_outlier_factor(X_test, contamination=0.1)
res_lof = evaluate(y_test, y_pred_lof, name='LOF')
plot_confusion_matrix(y_test, y_pred_lof, name='LOF', save=True)

# --- comparison ---
import pandas as pd
results = pd.DataFrame([res_if, res_svm, res_lof]).set_index('name')
print("\n=== Summary ===")
print(results.sort_values('f1', ascending=False))

plot_pr_curve(y_test, {
    'IsolationForest': y_score_if,
    'OneClassSVM': y_score_svm,
}, save=True)
