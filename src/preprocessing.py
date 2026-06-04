import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import LABEL_MAP, RANDOM_STATE, TEST_SIZE

# Elliptic paper split: train steps 1-34, test steps 35-49
TRAIN_STEPS = range(1, 35)
TEST_STEPS  = range(35, 50)


def get_labeled(df: pd.DataFrame) -> pd.DataFrame:
    labeled = df[df["class"].isin(["1", "2"])].copy()
    labeled["label"] = labeled["class"].map(LABEL_MAP)
    return labeled


def get_feature_cols(df: pd.DataFrame) -> list:
    drop = {"txId", "class", "label"}
    return [c for c in df.columns if c not in drop]


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test), scaler


def prepare_supervised_temporal(df: pd.DataFrame):
    """
    Temporal split matching Elliptic paper: train steps 1-34, test steps 35-49.
    Only labeled rows used. Scaler fit on train only.
    """
    labeled = get_labeled(df)
    feature_cols = get_feature_cols(labeled)

    train = labeled[labeled["time_step"].isin(TRAIN_STEPS)]
    test  = labeled[labeled["time_step"].isin(TEST_STEPS)]

    X_train = train[feature_cols].values
    y_train = train["label"].values
    X_test  = test[feature_cols].values
    y_test  = test["label"].values

    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)
    return X_train_sc, X_test_sc, y_train, y_test, feature_cols, scaler


def prepare_unsupervised_temporal(df: pd.DataFrame):
    """
    Unsupervised variant of temporal split.
    Scaler fit on all train-step rows (including unknown).
    Evaluated on labeled test-step rows only.
    """
    feature_cols = get_feature_cols(df.drop(columns=["label"], errors="ignore"))

    train_all = df[df["time_step"].isin(TRAIN_STEPS)]
    test_all  = df[df["time_step"].isin(TEST_STEPS)]

    scaler = StandardScaler()
    X_train_all = scaler.fit_transform(train_all[feature_cols].values)

    labeled_test = get_labeled(test_all)
    X_test_labeled = scaler.transform(labeled_test[feature_cols].values)
    y_test = labeled_test["label"].values

    return X_train_all, X_test_labeled, y_test, feature_cols, scaler
