import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessing import TRAIN_STEPS, TEST_STEPS, LABEL_MAP


def build_graph(df: pd.DataFrame, edges: pd.DataFrame):
    """
    Build a PyG Data object from the full Elliptic dataset.

    Returns:
        data       - PyG Data with x, edge_index, y, train_mask, test_mask
        node_ids   - array mapping row index → txId
        scaler     - fitted StandardScaler
    """
    # node index: stable ordering by txId
    node_ids = df["txId"].values
    id_to_idx = {txid: i for i, txid in enumerate(node_ids)}

    # node features (drop txId, class, label)
    feat_drop = {"txId", "class", "label"}
    feat_cols = [c for c in df.columns if c not in feat_drop]
    X = df[feat_cols].values.astype(np.float32)

    # scale using train-step rows only
    train_mask_np = df["time_step"].isin(TRAIN_STEPS).values
    scaler = StandardScaler()
    scaler.fit(X[train_mask_np])
    X = scaler.transform(X)

    # edge index — drop edges where either node not in graph
    valid = edges["txId1"].isin(id_to_idx) & edges["txId2"].isin(id_to_idx)
    edges_valid = edges[valid]
    src = edges_valid["txId1"].map(id_to_idx).values
    dst = edges_valid["txId2"].map(id_to_idx).values
    # undirected: add both directions
    edge_index = torch.tensor(
        np.stack([np.concatenate([src, dst]),
                  np.concatenate([dst, src])], axis=0),
        dtype=torch.long,
    )

    # labels: -1 = unknown, 0 = licit, 1 = illicit
    label_series = df["class"].map({"1": 1, "2": 0})
    y = label_series.fillna(-1).astype(int).values

    # masks: labeled nodes only, split by time step
    labeled = y >= 0
    train_mask = labeled & train_mask_np
    test_mask  = labeled & df["time_step"].isin(TEST_STEPS).values

    data = Data(
        x          = torch.tensor(X, dtype=torch.float32),
        edge_index = edge_index,
        y          = torch.tensor(y, dtype=torch.long),
        train_mask = torch.tensor(train_mask, dtype=torch.bool),
        test_mask  = torch.tensor(test_mask,  dtype=torch.bool),
    )

    print(f"Graph: {data.num_nodes:,} nodes  {data.num_edges:,} edges  "
          f"train_labeled={train_mask.sum():,}  test_labeled={test_mask.sum():,}")
    return data, node_ids, scaler
