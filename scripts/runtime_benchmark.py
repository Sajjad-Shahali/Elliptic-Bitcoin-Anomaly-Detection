"""
Runtime / scalability benchmark (submission add-on, HANDOFF §"What to Add for Submission").

Measures per-transaction inference latency for the models discussed in the report:
  - GBM Optuna-tuned      (tabular, 165 feat)
  - GBM + Structural      (172 feat -- separates one-time graph-feature cost from
                            per-tx model inference cost, since the former is
                            amortized/recomputed at graph-refresh cadence, not per tx)
  - GraphSAGEv2            (full-graph forward pass, GPU)

Reports: model-only inference (ms/tx, batch-amortized), and for GBM+Structural,
the one-time structural feature computation cost for the whole graph.
"""
import sys
sys.path.insert(0, "..")
import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
import networkx as nx
import joblib
import torch

from src.data_loader import load_all
from src.preprocessing import prepare_supervised_temporal, get_labeled, TRAIN_STEPS, TEST_STEPS
from src.graph_utils import build_graph
from src.gnn import GraphSAGEv2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_REPEATS = 20

print("Loading data...")
df, edges = load_all()
X_train_sc, X_test_sc, y_train, y_test, feat_cols, scaler = prepare_supervised_temporal(df)
n_test = len(y_test)
print(f"Test set: {n_test:,} rows")

def bench_model(model, X, n_repeats=N_REPEATS, name=""):
    # warm-up
    model.predict_proba(X[:100])
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        model.predict_proba(X)
        times.append(time.perf_counter() - t0)
    times = np.array(times)
    ms_per_tx = 1000 * times / len(X)
    print(f"  {name:22s}  batch={len(X):,}  total={times.mean()*1000:.2f}ms (+/-{times.std()*1000:.2f})  "
          f"=> {ms_per_tx.mean():.5f} ms/tx")
    return ms_per_tx.mean(), ms_per_tx.std()

# ── 1. GBM Optuna (tabular, 165 feat) ──────────────────────────────────────────
print("\n[1/3] GBM Optuna-tuned (tabular, 165 feat)...")
gb_tuned = joblib.load("../models/gb_tuned.joblib")
gbm_ms, gbm_std = bench_model(gb_tuned, X_test_sc, name="GBM-Optuna")

# ── 2. GBM + Structural (172 feat): split graph-feature cost vs model cost ────
print("\n[2/3] GBM + Structural (172 feat)...")
print("  Timing one-time structural feature computation (whole graph)...")
t0 = time.perf_counter()
G = nx.DiGraph()
G.add_nodes_from(df["txId"].values)
G.add_edges_from(zip(edges["txId1"], edges["txId2"]))
G_und = G.to_undirected()
in_deg  = dict(G.in_degree())
out_deg = dict(G.out_degree())
tot_deg = {n: in_deg[n] + out_deg[n] for n in in_deg}
clust      = nx.clustering(G_und)
avg_nb_deg = nx.average_neighbor_degree(G_und)
ego_density, oddball_score = {}, {}
for node in G_und.nodes():
    neighbors = list(G_und.neighbors(node))
    ni = len(neighbors) + 1
    sub = G_und.subgraph([node] + neighbors)
    ei  = sub.number_of_edges()
    ego_density[node]   = (2 * ei) / (ni * (ni - 1)) if ni > 1 else 0.0
    expected_ei          = ni ** 1.5
    oddball_score[node] = abs(ei - expected_ei) / (expected_ei + 1e-8)
struct_time = time.perf_counter() - t0
n_nodes = G.number_of_nodes()
print(f"  Structural features: {struct_time:.2f}s for {n_nodes:,} nodes "
      f"=> {1000*struct_time/n_nodes:.5f} ms/tx (one-time, amortized over graph refresh interval)")

struct_df = pd.DataFrame({
    "txId":         list(in_deg.keys()),
    "in_degree":    [in_deg[n]        for n in in_deg],
    "out_degree":   [out_deg[n]       for n in in_deg],
    "total_degree": [tot_deg[n]       for n in in_deg],
    "clustering":   [clust.get(n, 0)  for n in in_deg],
    "avg_nb_deg":   [avg_nb_deg.get(n, 0) for n in in_deg],
    "ego_density":  [ego_density.get(n, 0) for n in in_deg],
    "oddball":      [oddball_score.get(n, 0) for n in in_deg],
})
STRUCT_COLS = ["in_degree", "out_degree", "total_degree", "clustering",
               "avg_nb_deg", "ego_density", "oddball"]
df_struct = df.merge(struct_df, on="txId", how="left").fillna(0)
labeled_struct = get_labeled(df_struct)
test_df_s = labeled_struct[labeled_struct["time_step"].isin(TEST_STEPS)].reset_index(drop=True)
feat_cols_tab = [c for c in test_df_s.columns if c not in {"txId", "class", "label"}]
all_feat_cols = feat_cols_tab + STRUCT_COLS
scaler_struct = joblib.load("../models/scaler_structural.joblib")
X_test_struct = scaler_struct.transform(test_df_s[all_feat_cols].values)

gbm_struct = joblib.load("../models/gbm_structural.joblib")
struct_ms, struct_std = bench_model(gbm_struct, X_test_struct, name="GBM+Structural (model only)")

# ── 3. GraphSAGEv2 (GPU, full-graph forward pass) ──────────────────────────────
print(f"\n[3/3] GraphSAGEv2 (full-graph forward, device={DEVICE})...")
data, node_ids, _ = build_graph(df, edges)
model = GraphSAGEv2(input_dim=data.x.shape[1], hidden_dim=256, output_dim=2, dropout=0.3).to(DEVICE)
model.load_state_dict(torch.load("../models/graphsagev2.pt", map_location=DEVICE))
model.eval()
data = data.to(DEVICE)

with torch.no_grad():
    for _ in range(5):  # extra warm-up: PyG pure-python fallback + CUDA JIT settle
        _ = model(data.x, data.edge_index)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        _ = model(data.x, data.edge_index)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
times = np.array(times)
n_scored = data.num_nodes
print(f"  Per-call times (ms): {np.round(times*1000, 2).tolist()}")
med_s = np.median(times)
sage_ms = 1000 * med_s / n_scored
print(f"  Full graph: {n_scored:,} nodes  median={med_s*1000:.2f}ms  mean={times.mean()*1000:.2f}ms (+/-{times.std()*1000:.2f}, outlier-skewed)  "
      f"=> {sage_ms:.5f} ms/tx (median, amortized -- single forward pass scores all nodes at once)")

print("\n=== Runtime Summary (ms/transaction) ===")
print(f"  GBM Optuna (tabular)              : {gbm_ms:.5f} ms/tx")
print(f"  GBM + Structural, model only       : {struct_ms:.5f} ms/tx")
print(f"  GBM + Structural, incl. graph feat : {struct_ms + 1000*struct_time/n_nodes:.5f} ms/tx "
      f"(if recomputed per scoring pass, no caching)")
print(f"  GraphSAGEv2 (amortized full graph) : {sage_ms:.5f} ms/tx")
print(f"\nNote: GraphSAGE cost is amortized (all {n_scored:,} nodes scored per forward pass);")
print(f"a single new transaction still requires materializing its full k-hop neighborhood")
print(f"subgraph before inference in an online setting, which this benchmark does not isolate.")

results = pd.DataFrame([
    {"model": "GBM-Optuna", "ms_per_tx": gbm_ms, "std_ms_per_tx": gbm_std, "note": "tabular, 165 feat"},
    {"model": "GBM+Structural (model only)", "ms_per_tx": struct_ms, "std_ms_per_tx": struct_std, "note": "172 feat"},
    {"model": "GBM+Structural (incl. graph feat, no cache)", "ms_per_tx": struct_ms + 1000*struct_time/n_nodes,
     "std_ms_per_tx": struct_std, "note": "one-time graph feature cost amortized over all nodes"},
    {"model": "GraphSAGEv2", "ms_per_tx": sage_ms, "std_ms_per_tx": np.nan,
     "note": f"full-graph forward pass amortized over {n_scored:,} nodes, {DEVICE.type}"},
])
results.to_csv("../reports/runtime_benchmark.csv", index=False)
print("\nSaved: reports/runtime_benchmark.csv")
