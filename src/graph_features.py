import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import StandardScaler


def compute_graph_features(df: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """
    Compute node-level graph features and merge onto df by txId.
    Added features: in_degree, out_degree, total_degree, pagerank, clustering
    """
    print("  Building directed graph...")
    G = nx.DiGraph()
    G.add_nodes_from(df["txId"].values)
    G.add_edges_from(zip(edges["txId1"], edges["txId2"]))

    print("  Computing degree features...")
    in_deg  = dict(G.in_degree())
    out_deg = dict(G.out_degree())

    print("  Computing PageRank (may take ~30s)...")
    pr = nx.pagerank(G, max_iter=100, tol=1e-4)

    print("  Computing clustering coefficient...")
    G_und = G.to_undirected()
    clust = nx.clustering(G_und)

    gf = pd.DataFrame({
        "txId":        list(in_deg.keys()),
        "in_degree":   list(in_deg.values()),
        "out_degree":  list(out_deg.values()),
        "total_degree":[in_deg[n] + out_deg[n] for n in in_deg],
        "pagerank":    [pr.get(n, 0.0) for n in in_deg],
        "clustering":  [clust.get(n, 0.0) for n in in_deg],
    })

    return df.merge(gf, on="txId", how="left").fillna(0)
