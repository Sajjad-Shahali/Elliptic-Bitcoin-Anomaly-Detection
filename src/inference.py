"""
Inference pipeline — load best model (GBM+Structural) and score new transactions.

Usage:
    from src.inference import load_pipeline, score_transactions

    pipeline = load_pipeline()
    scores = score_transactions(pipeline, df_new, edges_new)
    # scores: dict {txId -> {"prob_illicit": float, "label": "illicit"|"licit"|"unknown", "risk": "HIGH"|"MEDIUM"|"LOW"}}
"""
import numpy as np
import pandas as pd
import networkx as nx
import joblib
from pathlib import Path

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"

STRUCT_COLS = [
    "in_degree", "out_degree", "total_degree",
    "clustering", "avg_nb_deg", "ego_density", "oddball",
]

HIGH_RISK_THRESHOLD   = 0.70
MEDIUM_RISK_THRESHOLD = 0.30


def _compute_structural_features(df: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Build graph from df+edges and compute 7 structural topology features."""
    G     = nx.DiGraph()
    G.add_nodes_from(df["txId"].values)
    G.add_edges_from(zip(edges["txId1"], edges["txId2"]))
    G_und = G.to_undirected()

    in_deg  = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    tot_deg = {n: in_deg[n] + out_deg[n] for n in in_deg}
    clust   = nx.clustering(G_und)
    avg_nb  = nx.average_neighbor_degree(G_und)

    ego_density   = {}
    oddball_score = {}
    for node in G_und.nodes():
        neighbors = list(G_und.neighbors(node))
        ni = len(neighbors) + 1
        sub = G_und.subgraph([node] + neighbors)
        ei  = sub.number_of_edges()
        ego_density[node]   = (2 * ei) / (ni * (ni - 1)) if ni > 1 else 0.0
        expected_ei         = ni ** 1.5
        oddball_score[node] = abs(ei - expected_ei) / (expected_ei + 1e-8)

    struct_df = pd.DataFrame({
        "txId":         list(in_deg.keys()),
        "in_degree":    [in_deg[n]             for n in in_deg],
        "out_degree":   [out_deg[n]            for n in in_deg],
        "total_degree": [tot_deg[n]            for n in in_deg],
        "clustering":   [clust.get(n, 0)       for n in in_deg],
        "avg_nb_deg":   [avg_nb.get(n, 0)      for n in in_deg],
        "ego_density":  [ego_density.get(n, 0) for n in in_deg],
        "oddball":      [oddball_score.get(n, 0) for n in in_deg],
    })
    return df.merge(struct_df, on="txId", how="left").fillna(0)


def load_pipeline(models_dir: Path = MODELS_DIR) -> dict:
    """
    Load GBM+Structural pipeline from disk.
    Returns dict with keys: model, scaler, feat_cols, struct_cols.
    """
    model  = joblib.load(models_dir / "gbm_structural.joblib")
    scaler = joblib.load(models_dir / "scaler_structural.joblib")

    # Reconstruct feature column list (same order as training: tabular + structural)
    # Tabular: all columns except txId, class, label, time_step (165 features)
    # Structural: STRUCT_COLS (7 features) appended
    feat_cols = None  # resolved at score time from df column order
    return {
        "model":       model,
        "scaler":      scaler,
        "struct_cols": STRUCT_COLS,
    }


def score_transactions(
    pipeline: dict,
    df: pd.DataFrame,
    edges: pd.DataFrame,
    tabular_feat_cols: list = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Score new transactions with the GBM+Structural model.

    Parameters
    ----------
    pipeline : dict returned by load_pipeline()
    df       : DataFrame with columns [txId, time_step, f0..f164] (Elliptic format)
               OR any DataFrame with the same 165 tabular features
    edges    : DataFrame with columns [txId1, txId2]
    tabular_feat_cols : list of 165 feature column names. If None, auto-detected
                        as all columns except txId/class/label/time_step.
    verbose  : print progress

    Returns
    -------
    DataFrame with columns: [txId, prob_illicit, label, risk]
    """
    model       = pipeline["model"]
    scaler      = pipeline["scaler"]
    struct_cols = pipeline["struct_cols"]

    if verbose:
        print(f"Scoring {len(df):,} transactions...")

    # Resolve tabular feature columns
    if tabular_feat_cols is None:
        drop_cols = {"txId", "class", "label", "time_step"}
        tabular_feat_cols = [c for c in df.columns if c not in drop_cols]

    if verbose:
        print(f"  Tabular features: {len(tabular_feat_cols)}")
        print(f"  Building graph ({len(df):,} nodes, {len(edges):,} edges)...")

    # Compute structural features
    df_struct = _compute_structural_features(df, edges)

    # Script 18 used get_feature_cols(labeled) AFTER merging structural onto df,
    # so structural cols were already included in feat_cols_tab, then STRUCT_COLS
    # was appended again — 173 + 7 = 180 features. Replicate exact same layout.
    merged_feat_cols = [c for c in df_struct.columns
                        if c not in {"txId", "class", "label"}]
    all_feat_cols = merged_feat_cols + struct_cols  # structural appear twice → 180
    if verbose:
        print(f"  Total features: {len(all_feat_cols)} "
              f"({len(merged_feat_cols)} from merged df + {len(struct_cols)} structural repeat = 180)")

    X = df_struct[all_feat_cols].values
    X_sc = scaler.transform(X)

    probs  = model.predict_proba(X_sc)[:, 1]
    labels = np.where(probs >= HIGH_RISK_THRESHOLD, "illicit",
             np.where(probs >= MEDIUM_RISK_THRESHOLD, "unknown", "licit"))
    risk   = np.where(probs >= HIGH_RISK_THRESHOLD, "HIGH",
             np.where(probs >= MEDIUM_RISK_THRESHOLD, "MEDIUM", "LOW"))

    results = pd.DataFrame({
        "txId":         df["txId"].values,
        "prob_illicit": probs,
        "label":        labels,
        "risk":         risk,
    }).sort_values("prob_illicit", ascending=False).reset_index(drop=True)

    if verbose:
        n_high   = (risk == "HIGH").sum()
        n_medium = (risk == "MEDIUM").sum()
        n_low    = (risk == "LOW").sum()
        print(f"\n  Results:")
        print(f"    HIGH risk   (p >= {HIGH_RISK_THRESHOLD}):   {n_high:,}  ({100*n_high/len(df):.1f}%)")
        print(f"    MEDIUM risk (p >= {MEDIUM_RISK_THRESHOLD}): {n_medium:,}  ({100*n_medium/len(df):.1f}%)")
        print(f"    LOW risk    (p <  {MEDIUM_RISK_THRESHOLD}): {n_low:,}  ({100*n_low/len(df):.1f}%)")
        print(f"\n  Top-10 highest-risk transactions:")
        print(results.head(10).to_string(index=False))

    return results


def score_from_csv(
    features_csv: str,
    edgelist_csv: str,
    output_csv: str = None,
    models_dir: Path = MODELS_DIR,
) -> pd.DataFrame:
    """
    Convenience wrapper: load CSVs, score, optionally save results.

    Parameters
    ----------
    features_csv : path to elliptic_txs_features.csv (or equivalent)
    edgelist_csv : path to elliptic_txs_edgelist.csv
    output_csv   : if set, save scored results to this path
    """
    print(f"Loading features from {features_csv} ...")
    df = pd.read_csv(features_csv, header=None)

    # Assign column names matching Elliptic format
    n_feat = df.shape[1] - 2  # minus txId and time_step
    cols = ["txId", "time_step"] + [f"f{i}" for i in range(n_feat)]
    df.columns = cols

    print(f"Loading edges from {edgelist_csv} ...")
    edges = pd.read_csv(edgelist_csv)
    if list(edges.columns) != ["txId1", "txId2"]:
        edges.columns = ["txId1", "txId2"]

    pipeline = load_pipeline(models_dir)
    tabular_cols = [f"f{i}" for i in range(n_feat)]
    results = score_transactions(pipeline, df, edges, tabular_feat_cols=tabular_cols)

    if output_csv:
        results.to_csv(output_csv, index=False)
        print(f"\nSaved scored results to {output_csv}")

    return results
