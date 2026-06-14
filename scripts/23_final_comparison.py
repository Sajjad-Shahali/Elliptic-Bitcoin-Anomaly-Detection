"""
Final comparison — all 48 experiments across 5 rounds.
Produces:
  reports/final_leaderboard.csv      — full ranked table
  reports/final_top15_f1.png         — horizontal bar chart top-15 by F1
  reports/final_tier_comparison.png  — grouped bar: best per tier/category
  reports/final_round_progression.png — F1 progression across rounds for key models
"""
import sys
sys.path.insert(0, "..")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── all 48 experiment results ──────────────────────────────────────────────────
RESULTS = [
    # (rank, name, category, round_, f1, auc, ap)
    (1,  "GBM + Structural (172 feat)",      "Supervised",        "Rnd 4",  0.8265, 0.9243, 0.8031),
    (2,  "GradientBoosting (Optuna)",         "Supervised",        "Optuna", 0.8241, 0.9204, None),
    (3,  "GBM + PseudoLabels",               "Semi-supervised",   "Rnd 4",  0.8235, 0.9240, 0.8040),
    (4,  "Ensemble (GBM+LGBM+SAGEv2)",       "Ensemble",          "Rnd 3",  0.8210, 0.9240, 0.8020),
    (5,  "Weighted Ensemble (equal 1/3)",     "Ensemble",          "Rnd 5",  0.8209, 0.9238, 0.8019),
    (6,  "LightGBM",                         "Supervised",        "Rnd 1",  0.8170, 0.9300, 0.8040),
    (7,  "GBM + Temporal Rolling (496 feat)","Supervised",        "Rnd 5",  0.8179, 0.9300, 0.8017),
    (8,  "RF + Graph Features",              "Supervised",        "Rnd 1",  0.8070, 0.9420, 0.8000),
    (9,  "GBM Platt-calibrated",             "Supervised",        "Rnd 5",  0.8006, 0.9204, 0.8000),
    (10, "Random Forest (baseline)",         "Supervised",        "Base",   0.8010, 0.9350, 0.7940),
    (11, "Random Forest (Optuna)",           "Supervised",        "Optuna", 0.7970, 0.9290, None),
    (12, "GradientBoosting (baseline)",      "Supervised",        "Base",   0.7660, 0.9140, 0.7850),
    (13, "Ensemble (RF+LGB+GAT)",            "Ensemble",          "Rnd 1",  0.7420, 0.9220, 0.7890),
    (14, "SAGEv2 + PseudoLabels",           "Semi-supervised",   "Rnd 4",  0.7293, 0.8901, 0.7517),
    (15, "GBM Isotonic-calibrated",          "Supervised",        "Rnd 5",  0.7340, 0.8753, 0.7682),
    (16, "GraphSAGEv2",                      "GNN",               "Rnd 2",  0.6980, 0.9130, 0.7200),
    (17, "GraphSAGE (Optuna)",               "GNN",               "Optuna", 0.6880, None,   None),
    (18, "GNN Exp 6 — 3L h=256 LN+SL",      "GNN",               "GNN-exp",0.6830, 0.9060, 0.7040),
    (19, "GNN Exp 5 — 3L h=128",             "GNN",               "GNN-exp",0.6150, 0.8990, 0.6110),
    (20, "GNN Exp 4 — 2L h=128 300ep",       "GNN",               "GNN-exp",0.5970, 0.8970, 0.6070),
    (21, "GNN Exp 3 — 2L h=256",             "GNN",               "GNN-exp",0.5800, 0.8950, 0.5950),
    (22, "GNN Exp 2 — 2L h=128",             "GNN",               "GNN-exp",0.5770, 0.8960, 0.6030),
    (23, "GraphSAGE (baseline)",             "GNN",               "Base",   0.5410, 0.8880, 0.5490),
    (24, "GNN Exp 1 — 2L h=64",              "GNN",               "GNN-exp",0.5280, 0.8870, 0.5190),
    (25, "beta-VAE (beta=0.01)",             "Neural AE",         "Rnd 4",  0.4065, 0.7832, 0.3651),
    (26, "GNN Exp 10 — GATv2 3L residual",  "GNN",               "GNN-exp",0.3820, 0.8870, 0.5470),
    (27, "DenoisingAE",                      "Neural AE",         "Rnd 2",  0.3610, 0.7920, 0.3660),
    (28, "GAT (baseline)",                   "GNN",               "Rnd 1",  0.3590, 0.8470, 0.3890),
    (29, "Autoencoder (baseline)",           "Neural AE",         "Base",   0.3560, 0.7810, 0.3360),
    (30, "beta-VAE (beta=1.0)",              "Neural AE",         "Rnd 4",  0.3717, 0.7760, 0.3384),
    (31, "GNN Exp 9 — GAT 2L 2heads",        "GNN",               "GNN-exp",0.3410, 0.8640, 0.4190),
    (32, "VAE",                              "Neural AE",         "Rnd 1",  0.3400, 0.7780, 0.3100),
    (33, "beta-VAE (beta=8.0)",              "Neural AE",         "Rnd 4",  0.3496, 0.7625, 0.2993),
    (34, "GATv2",                            "GNN",               "Rnd 2",  0.3170, 0.8820, 0.3760),
    (35, "beta-VAE (beta=4.0)",              "Neural AE",         "Rnd 4",  0.3127, 0.7535, 0.2867),
    (36, "GNN Exp 8 — GAT 2L 4heads",        "GNN",               "GNN-exp",0.3070, 0.8240, 0.3660),
    (37, "Logistic Regression",              "Supervised",        "Base",   0.3020, 0.8810, 0.2910),
    (38, "beta-VAE (beta=2.0)",              "Neural AE",         "Rnd 4",  0.2798, 0.7455, 0.2108),
    (39, "VAEv2",                            "Neural AE",         "Rnd 2",  0.2560, 0.7320, 0.1610),
    (40, "DOMINANT",                         "Graph AE (unsup)",  "Rnd 3",  0.2120, 0.6690, 0.1230),
    (41, "LSTM-AE",                          "Neural AE",         "Rnd 3",  0.1220, 0.4460, 0.0640),
    (42, "IsoForest (structural only)",      "Unsupervised",      "Rnd 4",  0.0941, 0.4576, 0.0556),
    (43, "LOF",                              "Unsupervised",      "Base",   0.0760, 0.5060, 0.0660),
    (44, "OCSVM (spectral K=50)",            "Unsupervised",      "Rnd 4",  0.0613, 0.3909, 0.0537),
    (45, "IsoForest (spectral K=50)",        "Unsupervised",      "Rnd 4",  0.0562, 0.4071, 0.0521),
    (46, "One-Class SVM",                    "Unsupervised",      "Base",   0.0230, 0.2350, 0.0390),
    (47, "Isolation Forest",                 "Unsupervised",      "Base",   0.0210, 0.1720, 0.0360),
    (48, "GNN Exp 7 (Optuna best)",          "GNN",               "GNN-exp",0.6900, 0.9010, 0.7060),
]

cols = ["Rank", "Model", "Category", "Round", "F1", "AUC", "AP"]
df = pd.DataFrame(RESULTS, columns=cols)
df = df.sort_values("F1", ascending=False).reset_index(drop=True)
df["Rank"] = range(1, len(df) + 1)
df.to_csv("../reports/final_leaderboard.csv", index=False)
print(f"Saved reports/final_leaderboard.csv ({len(df)} experiments)")

# ── colour palette per category ───────────────────────────────────────────────
COLORS = {
    "Supervised":       "#2196F3",
    "Semi-supervised":  "#00BCD4",
    "Ensemble":         "#9C27B0",
    "GNN":              "#4CAF50",
    "Neural AE":        "#FF9800",
    "Graph AE (unsup)": "#F44336",
    "Unsupervised":     "#9E9E9E",
}

# ── Plot 1: horizontal bar — top 15 by F1 ─────────────────────────────────────
top15 = df.head(15).sort_values("F1")
fig, ax = plt.subplots(figsize=(11, 7))
bar_colors = [COLORS.get(c, "#607D8B") for c in top15["Category"]]
bars = ax.barh(top15["Model"], top15["F1"], color=bar_colors, edgecolor="white", linewidth=0.5)
ax.axvline(x=0.824, color="#B71C1C", linestyle="--", linewidth=1.2, label="Previous best (GBM Optuna 0.824)")
for bar, f1 in zip(bars, top15["F1"]):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
            f"{f1:.4f}", va="center", ha="left", fontsize=8.5)
patches = [mpatches.Patch(color=v, label=k) for k, v in COLORS.items() if k in top15["Category"].values]
ax.legend(handles=patches, loc="lower right", fontsize=8)
ax.set_xlabel("F1 (illicit class)", fontsize=10)
ax.set_title("Top-15 Models — Final Leaderboard (48 experiments)", fontsize=12, fontweight="bold")
ax.set_xlim(0, 0.96)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("../reports/final_top15_f1.png", dpi=150)
plt.close()
print("Saved reports/final_top15_f1.png")

# ── Plot 2: best per tier grouped bar (F1 + AUC) ──────────────────────────────
TIERS = {
    "Supervised\n(tree)",    df[df["Category"].isin(["Supervised"])]["F1"].max(),
    "Semi-\nsupervised",     df[df["Category"] == "Semi-supervised"]["F1"].max(),
    "Ensemble",              df[df["Category"] == "Ensemble"]["F1"].max(),
    "GNN",                   df[df["Category"] == "GNN"]["F1"].max(),
    "Neural AE",             df[df["Category"] == "Neural AE"]["F1"].max(),
    "Graph AE\n(unsup)",     df[df["Category"] == "Graph AE (unsup)"]["F1"].max(),
    "Unsupervised",          df[df["Category"] == "Unsupervised"]["F1"].max(),
}
tier_best = {}
for cat in COLORS:
    sub = df[df["Category"] == cat]
    if len(sub):
        row = sub.loc[sub["F1"].idxmax()]
        tier_best[cat] = {"model": row["Model"], "f1": row["F1"],
                          "auc": row["AUC"] if not pd.isna(row["AUC"]) else 0}

tier_labels  = list(tier_best.keys())
tier_f1      = [tier_best[k]["f1"]  for k in tier_labels]
tier_auc     = [tier_best[k]["auc"] for k in tier_labels]
tier_clrs    = [COLORS[k] for k in tier_labels]
x = np.arange(len(tier_labels))
w = 0.35
fig, ax = plt.subplots(figsize=(12, 6))
b1 = ax.bar(x - w/2, tier_f1,  w, color=tier_clrs, alpha=0.9, label="F1 (illicit)")
b2 = ax.bar(x + w/2, tier_auc, w, color=tier_clrs, alpha=0.45, label="ROC-AUC")
for bar, v in zip(b1, tier_f1):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{v:.3f}", ha="center", fontsize=8.5, fontweight="bold")
for bar, v in zip(b2, tier_auc):
    if v > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", fontsize=7.5)
ax.set_xticks(x)
ax.set_xticklabels(tier_labels, fontsize=9)
ax.set_ylabel("Score", fontsize=10)
ax.set_title("Best Model per Tier — F1 vs ROC-AUC", fontsize=12, fontweight="bold")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("../reports/final_tier_comparison.png", dpi=150)
plt.close()
print("Saved reports/final_tier_comparison.png")

# ── Plot 3: round progression for key models ──────────────────────────────────
progression = {
    "GBM (best)":        [0.766, None, None, 0.824, 0.8265],  # Base, R1, R2, Optuna, R4 (structural)
    "GraphSAGEv2":       [0.541, None, 0.698, 0.688, 0.729],  # Base, R1, R2, Optuna, R4 (pseudo)
    "Neural AE (best)":  [0.356, 0.340, 0.361, None, 0.407],  # AE, VAE, DAE, -, beta-VAE
    "Unsupervised":      [0.021, None, None, None, 0.094],    # IsoForest base, structural
}
round_labels = ["Baseline", "Round 1", "Round 2", "Optuna", "Round 3-5"]
fig, ax = plt.subplots(figsize=(10, 5))
model_colors = {"GBM (best)": "#2196F3", "GraphSAGEv2": "#4CAF50",
                "Neural AE (best)": "#FF9800", "Unsupervised": "#9E9E9E"}
for model, scores in progression.items():
    x_pts = [i for i, v in enumerate(scores) if v is not None]
    y_pts = [v for v in scores if v is not None]
    ax.plot(x_pts, y_pts, "o-", label=model, color=model_colors[model],
            linewidth=2, markersize=7)
    for xi, yi in zip(x_pts, y_pts):
        ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8,
                    color=model_colors[model])
ax.set_xticks(range(len(round_labels)))
ax.set_xticklabels(round_labels, fontsize=9)
ax.set_ylabel("F1 (illicit class)", fontsize=10)
ax.set_title("Model Improvement Across 5 Rounds", fontsize=12, fontweight="bold")
ax.set_ylim(0, 0.95)
ax.legend(fontsize=9, loc="upper left")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("../reports/final_round_progression.png", dpi=150)
plt.close()
print("Saved reports/final_round_progression.png")

# ── Console summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("=== FINAL LEADERBOARD — TOP 15 ===")
print(f"{'Rank':<5} {'Model':<42} {'Category':<18} {'F1':>6}  {'AUC':>6}  {'AP':>6}")
print("-" * 72)
for _, row in df.head(15).iterrows():
    auc = f"{row['AUC']:.4f}" if not pd.isna(row['AUC']) else "   —  "
    ap  = f"{row['AP']:.4f}"  if not pd.isna(row['AP'])  else "   —  "
    print(f"{int(row['Rank']):<5} {row['Model']:<42} {row['Category']:<18} {row['F1']:.4f}  {auc}  {ap}")

print("\n=== BEST PER TIER ===")
for cat, info in tier_best.items():
    auc_s = f"{info['auc']:.4f}" if info['auc'] else "   —  "
    print(f"  {cat:<22} F1={info['f1']:.4f}  AUC={auc_s}  [{info['model'][:38]}]")

print(f"\nTotal experiments: {len(df)}")
print(f"Best model: {df.iloc[0]['Model']} — F1={df.iloc[0]['F1']:.4f}")
print(f"Plots saved to reports/")
