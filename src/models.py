from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.linear_model import LogisticRegression
import numpy as np


# --- Unsupervised anomaly detectors ---

def isolation_forest(X_train, contamination=0.1, random_state=42):
    model = IsolationForest(contamination=contamination, random_state=random_state, n_jobs=-1)
    model.fit(X_train)
    return model


def local_outlier_factor(X, contamination=0.1, n_neighbors=20):
    model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, n_jobs=-1)
    labels = model.fit_predict(X)
    # LOF: -1=outlier → 1(illicit), 1=inlier → 0(licit)
    return (labels == -1).astype(int), model


def one_class_svm(X_train, nu=0.1, kernel="rbf"):
    model = OneClassSVM(nu=nu, kernel=kernel)
    model.fit(X_train)
    return model


# --- Supervised classifiers ---

def random_forest(X_train, y_train, random_state=42):
    model = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                   random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def gradient_boosting(X_train, y_train, random_state=42):
    model = GradientBoostingClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def logistic_regression(X_train, y_train, random_state=42):
    model = LogisticRegression(class_weight="balanced", max_iter=1000,
                               random_state=random_state)
    model.fit(X_train, y_train)
    return model


def predict_anomaly(model, X):
    """Unified predict: returns binary labels (1=anomaly/illicit, 0=normal/licit)."""
    if hasattr(model, "predict_proba"):
        return model.predict(X)
    # IsolationForest / OneClassSVM: -1=anomaly, 1=normal
    raw = model.predict(X)
    return (raw == -1).astype(int)


def anomaly_scores(model, X):
    """Return continuous anomaly score (higher = more anomalous)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return -model.decision_function(X)
    if hasattr(model, "score_samples"):
        return -model.score_samples(X)
    raise ValueError("Model has no scoring method")
