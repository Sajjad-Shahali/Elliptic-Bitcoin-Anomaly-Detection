import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import LABEL_MAP, RANDOM_STATE, TEST_SIZE
from src.data_loader import load_all


def get_labeled(df: pd.DataFrame) -> pd.DataFrame:
    """Return only labeled rows with binary label column."""
    labeled = df[df["class"].isin(["1", "2"])].copy()
    labeled["label"] = labeled["class"].map(LABEL_MAP)
    return labeled


def get_feature_cols(df: pd.DataFrame) -> list:
    drop = {"txId", "class", "label"}
    return [c for c in df.columns if c not in drop]


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test), scaler


def prepare_supervised(df: pd.DataFrame):
    """
    Returns X_train, X_test, y_train, y_test, feature_cols.
    Uses only labeled rows. Drops time_step from features.
    """
    labeled = get_labeled(df)
    feature_cols = get_feature_cols(labeled)
    X = labeled[feature_cols].values
    y = labeled["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)
    return X_train_sc, X_test_sc, y_train, y_test, feature_cols, scaler


def prepare_unsupervised(df: pd.DataFrame):
    """
    Returns X_all (scaled, all rows including unknown), X_labeled, y_labeled.
    Useful for unsupervised anomaly detectors evaluated on labeled subset.
    """
    feature_cols = get_feature_cols(df.drop(columns=["label"], errors="ignore"))
    X_all = df[feature_cols].values

    scaler = StandardScaler()
    X_all_sc = scaler.fit_transform(X_all)

    labeled = get_labeled(df)
    labeled_idx = df[df["class"].isin(["1", "2"])].index
    X_labeled_sc = X_all_sc[df.index.isin(labeled_idx)]
    y_labeled = labeled["label"].values

    return X_all_sc, X_labeled_sc, y_labeled, feature_cols, scaler
