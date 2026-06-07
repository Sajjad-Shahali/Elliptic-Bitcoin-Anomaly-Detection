"""
Compare improved models vs baseline.
Reads baseline from reports/leaderboard.csv, improved from reports/improved_results.csv.
"""
import sys
sys.path.insert(0, '..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── load results ───────────────────────────────────────────────────────────────
baseline = pd.read_csv("../reports/leaderboard.csv")[["Model", "Category", "F1", "ROC-AUC", "Avg-Prec"]]
improved = pd.read_csv("../reports/improved_results.csv")[["Model", "F1", "ROC-AUC", "Avg-Prec"]]

# map improved models to their baseline counterpart
MAPPING = {
    "LightGBM":            ("GradientBoosting", "LightGBM replaces GBM"),
    "RF + GraphFeatures":  ("RandomForest",     "RF + graph node features"),
    "GAT":                 ("GraphSAGE",         "GAT replaces GraphSAGE"),
    "VAE":                 ("Autoencoder",        "VAE replaces Autoencoder"),
    "Ensemble (RF+LGB+GAT)": (None,             "New — no baseline equivalent"),
}

# ── delta table ────────────────────────────────────────────────────────────────
print("="*72)
print(f"{'Model':<28} {'Baseline F1':>12} {'Improved F1':>12} {'Delta F1':>10} {'Delta AUC':>10}")
print("-"*72)

delta_rows = []
for imp_name, (base_name, note) in MAPPING.items():
    imp_row = improved[improved["Model"] == imp_name]
    if imp_row.empty:
        continue
    imp_f1  = imp_row.iloc[0]["F1"]
    imp_auc = imp_row.iloc[0]["ROC-AUC"]
    imp_ap  = imp_row.iloc[0]["Avg-Prec"]

    if base_name:
        base_row = baseline[baseline["Model"] == base_name]
        base_f1  = base_row.iloc[0]["F1"]  if not base_row.empty else None
        base_auc = base_row.iloc[0]["ROC-AUC"] if not base_row.empty else None
        delta_f1  = imp_f1  - base_f1  if base_f1  is not None else None
        delta_auc = imp_auc - base_auc if base_auc is not None else None
        print(f"{imp_name:<28} {base_f1:>12.4f} {imp_f1:>12.4f} "
              f"{delta_f1:>+10.4f} {delta_auc:>+10.4f}  ({note})")
    else:
        print(f"{imp_name:<28} {'--':>12} {imp_f1:>12.4f} {'--':>10} {'--':>10}  ({note})")

    delta_rows.append({
        "Improved Model": imp_name,
        "Replaces":       base_name or "—",
        "Baseline F1":    base_f1 if base_name else None,
        "Improved F1":    imp_f1,
        "Delta F1":       delta_f1 if base_name else None,
        "Baseline AUC":   base_auc if base_name else None,
        "Improved AUC":   imp_auc,
        "Delta AUC":      delta_auc if base_name else None,
        "Avg-Prec":       imp_ap,
        "Note":           note,
    })

print("="*72)
df_delta = pd.DataFrame(delta_rows)
df_delta.to_csv("../reports/improvement_delta.csv", index=False)

# ── full leaderboard: baseline + improved combined ─────────────────────────────
all_models = []
for _, r in baseline.iterrows():
    all_models.append({"Model": r["Model"], "Version": "Baseline", "F1": r["F1"],
                       "ROC-AUC": r["ROC-AUC"], "Avg-Prec": r["Avg-Prec"], "Category": r["Category"]})
cat_map = {"LightGBM": "Supervised", "RF + GraphFeatures": "Supervised",
           "GAT": "GNN", "VAE": "Neural", "Ensemble (RF+LGB+GAT)": "Ensemble"}
for _, r in improved.iterrows():
    all_models.append({"Model": r["Model"], "Version": "Improved", "F1": r["F1"],
                       "ROC-AUC": r["ROC-AUC"], "Avg-Prec": r["Avg-Prec"],
                       "Category": cat_map.get(r["Model"], "Other")})
df_all = pd.DataFrame(all_models).sort_values("F1", ascending=False).reset_index(drop=True)
df_all.index += 1
print("\n=== Full Combined Leaderboard ===")
print(df_all.to_string())
df_all.to_csv("../reports/combined_leaderboard.csv", index=False)

# ── plots ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

# 1. F1 delta bar chart
ax = axes[0]
valid = df_delta.dropna(subset=["Delta F1"])
colors = ["#27ae60" if v >= 0 else "#e74c3c" for v in valid["Delta F1"]]
bars = ax.barh(valid["Improved Model"], valid["Delta F1"], color=colors)
ax.axvline(0, color="black", lw=0.8)
for bar, val in zip(bars, valid["Delta F1"]):
    ax.text(bar.get_width() + (0.002 if val >= 0 else -0.002),
            bar.get_y() + bar.get_height()/2,
            f"{val:+.4f}", va="center", ha="left" if val >= 0 else "right", fontsize=9)
ax.set_xlabel("F1 Delta (Improved − Baseline)")
ax.set_title("F1 Improvement over Baseline")

# 2. Side-by-side F1 for each replacement pair
ax = axes[1]
pairs = [(r["Improved Model"], r["Replaces"], r["Baseline F1"], r["Improved F1"])
         for _, r in df_delta.iterrows() if r["Replaces"] != "—"]
x = np.arange(len(pairs))
w = 0.35
labels = [p[0] for p in pairs]
ax.bar(x - w/2, [p[2] for p in pairs], w, label="Baseline", color="#95a5a6", edgecolor="white")
bars2 = ax.bar(x + w/2, [p[3] for p in pairs], w, label="Improved", color="#2980b9", edgecolor="white")
for bar, val in zip(bars2, [p[3] for p in pairs]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels([f"{p[0]}\nvs {p[1]}" for p in pairs], fontsize=8)
ax.set_ylabel("F1 — illicit class")
ax.set_ylim(0, 1.05)
ax.set_title("Baseline vs Improved — F1 by Pair")
ax.legend()

# 3. Full leaderboard F1 bar (all models, color by version)
ax = axes[2]
df_sorted = df_all.sort_values("F1")
colors3 = ["#e74c3c" if v == "Improved" else "#95a5a6" for v in df_sorted["Version"]]
bars3 = ax.barh(df_sorted["Model"] + " (" + df_sorted["Version"] + ")",
                df_sorted["F1"], color=colors3)
for bar, val in zip(bars3, df_sorted["F1"]):
    ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=7.5)
ax.set_xlabel("F1 — illicit class")
ax.set_title("All Models Combined Leaderboard")
ax.legend(handles=[
    mpatches.Patch(color="#e74c3c", label="Improved"),
    mpatches.Patch(color="#95a5a6", label="Baseline"),
], fontsize=9)

plt.suptitle("Baseline vs Improved Models — Elliptic Bitcoin Anomaly Detection",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("../reports/improvement_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved: reports/improvement_comparison.png")
print("Saved: reports/improvement_delta.csv")
print("Saved: reports/combined_leaderboard.csv")
