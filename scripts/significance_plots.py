"""
Plots for the submission add-ons: bootstrap CI, McNemar agreement, SOTA comparison, runtime.
Reads CSVs produced by statistical_significance.py and runtime_benchmark.py.
Produces:
  reports/bootstrap_f1_ci.png
  reports/mcnemar_agreement.png
  reports/sota_comparison.png
  reports/runtime_benchmark.png
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
from config import REPORTS_DIR
import os

plt.rcParams.update({"font.size": 10})

# ── 1. Bootstrap F1 CI ─────────────────────────────────────────────────────────
boot = pd.read_csv(os.path.join(REPORTS_DIR, "bootstrap_f1_ci.csv"))
fig, ax = plt.subplots(figsize=(6, 3.2))
y = np.arange(len(boot))
err_lo = boot["f1"] - boot["ci_low"]
err_hi = boot["ci_high"] - boot["f1"]
ax.errorbar(boot["f1"], y, xerr=[err_lo, err_hi], fmt="o", color="#2166ac",
            ecolor="#2166ac", elinewidth=2, capsize=5, markersize=8)
ax.set_yticks(y)
ax.set_yticklabels(boot["model"])
ax.set_xlabel("F1 (illicit class)")
ax.set_title("Bootstrap 95% CI, Top-3 Models (1,000 resamples)")
ax.axvspan(boot["ci_low"].min(), boot["ci_high"].max(), color="#2166ac", alpha=0.06)
ax.invert_yaxis()
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(REPORTS_DIR, "bootstrap_f1_ci.png"), dpi=150)
plt.close(fig)
print("Saved bootstrap_f1_ci.png")

# ── 2. McNemar agreement (b vs c per pair) ─────────────────────────────────────
mc = pd.read_csv(os.path.join(REPORTS_DIR, "mcnemar_results.csv"))
fig, ax = plt.subplots(figsize=(6.5, 3.2))
labels = [f"{a}\nvs\n{b}" for a, b in zip(mc["model_a"], mc["model_b"])]
x = np.arange(len(mc))
w = 0.35
ax.bar(x - w/2, mc["b"], w, label="A right, B wrong (b)", color="#4393c3")
ax.bar(x + w/2, mc["c"], w, label="A wrong, B right (c)", color="#d6604d")
for i, p in enumerate(mc["p_value"]):
    ax.text(i, max(mc["b"][i], mc["c"][i]) + 0.5, f"p={p:.2f}", ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Disagreeing test rows (of 16,670)")
ax.set_title("McNemar's Test: Pairwise Disagreement, Top-3 Models")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(REPORTS_DIR, "mcnemar_agreement.png"), dpi=150)
plt.close(fig)
print("Saved mcnemar_agreement.png")

# ── 3. SOTA comparison ──────────────────────────────────────────────────────────
sota = [
    ("Logistic Regression\n(Weber 2019)",        0.481, "prior, transductive"),
    ("GCN\n(Weber 2019)",                         0.628, "prior, transductive"),
    ("Skip-GCN\n(Weber 2019)",                    0.705, "prior, transductive"),
    ("Random Forest\n(Weber 2019)",               0.788, "prior, temporal"),
    ("Augmented GCN\n(Alarab 2020)",              0.740, "prior, transductive"),
    ("GraphSAGE+SSL\n(Lo 2023)",                  0.750, "prior, transductive"),
    ("EvolveGCN\n(Pareja 2020)",                  0.770, "prior, transductive"),
    ("GraphSAGE\n(Maganti 2026)",                 0.294, "prior, transductive"),
    ("Random Forest\n(Maganti 2026)",             0.821, "prior, strict-inductive"),
    ("GraphSAGE\n(Maganti 2026)",                 0.689, "prior, strict-inductive"),
    ("Random Forest\n(ours)",                     0.801, "ours, non-graph"),
    ("GBM Optuna\n(ours)",                        0.824, "ours, non-graph"),
    ("GBM+Structural\n(ours)",                    0.827, "ours, non-graph"),
    ("GraphSAGEv2\n(ours)",                       0.698, "ours, transductive"),
]
names   = [s[0] for s in sota]
f1s     = [s[1] for s in sota]
groups  = [s[2] for s in sota]
color_map = {
    "prior, transductive":       "#bababa",
    "prior, temporal":           "#878787",
    "prior, strict-inductive":   "#1a9850",
    "ours, non-graph":           "#2166ac",
    "ours, transductive":        "#d6604d",
}
colors = [color_map[g] for g in groups]

fig, ax = plt.subplots(figsize=(7.5, 5.5))
y = np.arange(len(names))
ax.barh(y, f1s, color=colors)
ax.set_yticks(y)
ax.set_yticklabels(names, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("F1 (illicit class)")
ax.set_title(r"Published Elliptic F1 vs. This Work, by Evaluation Protocol")
ax.set_xlim(0, 0.9)
ax.grid(axis="x", alpha=0.3)
for yi, f1 in zip(y, f1s):
    ax.text(f1 + 0.01, yi, f"{f1:.3f}", va="center", fontsize=7.5)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in color_map.values()]
ax.legend(handles, color_map.keys(), fontsize=7, loc="lower right")
plt.tight_layout()
fig.savefig(os.path.join(REPORTS_DIR, "sota_comparison.png"), dpi=150)
plt.close(fig)
print("Saved sota_comparison.png")

# ── 4. Runtime benchmark (log scale) ────────────────────────────────────────────
rt = pd.read_csv(os.path.join(REPORTS_DIR, "runtime_benchmark.csv"))
fig, ax = plt.subplots(figsize=(7, 3.5))
colors_rt = ["#2166ac", "#4393c3", "#92c5de", "#d6604d"]
bars = ax.bar(range(len(rt)), rt["ms_per_tx"], color=colors_rt[:len(rt)])
ax.set_yscale("log")
ax.set_xticks(range(len(rt)))
ax.set_xticklabels(rt["model"], fontsize=7.5, rotation=15, ha="right")
ax.set_ylabel("ms / transaction (log scale)")
ax.set_title("Inference Latency by Model")
for i, v in enumerate(rt["ms_per_tx"]):
    ax.text(i, v * 1.15, f"{v:.4f}", ha="center", fontsize=8)
ax.grid(axis="y", alpha=0.3, which="both")
plt.tight_layout()
fig.savefig(os.path.join(REPORTS_DIR, "runtime_benchmark.png"), dpi=150)
plt.close(fig)
print("Saved runtime_benchmark.png")

print("\nAll plots saved to reports/")
