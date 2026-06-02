import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "elliptic_bitcoin_dataset")


def load_features() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "elliptic_txs_features.csv")
    df = pd.read_csv(path, header=None)
    cols = ["txId", "time_step"] + [f"lf_{i}" for i in range(1, 94)] + [f"af_{i}" for i in range(1, 73)]
    df.columns = cols
    return df


def load_classes() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, "elliptic_txs_classes.csv"))


def load_edgelist() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, "elliptic_txs_edgelist.csv"))


def load_all():
    features = load_features()
    classes = load_classes()
    edges = load_edgelist()
    df = features.merge(classes, on="txId", how="left")
    return df, edges
