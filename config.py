import os

ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT, "data", "elliptic_bitcoin_dataset")

FEATURES_CSV = os.path.join(DATA_DIR, "elliptic_txs_features.csv")
CLASSES_CSV  = os.path.join(DATA_DIR, "elliptic_txs_classes.csv")
EDGES_CSV    = os.path.join(DATA_DIR, "elliptic_txs_edgelist.csv")

MODELS_DIR  = os.path.join(ROOT, "models")
REPORTS_DIR = os.path.join(ROOT, "reports")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# label encoding: 1=illicit → 1 (anomaly), 2=licit → 0 (normal)
LABEL_MAP = {"1": 1, "2": 0}

RANDOM_STATE = 42
TEST_SIZE    = 0.2
