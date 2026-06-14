"""
Fix two plot issues for the LaTeX report:
1. Regenerate GNN progression as a proper 2-row stacked figure from CSV.
2. Crop the partial suptitle from ae_errors_right.png (appears between charts).

Outputs:
  reports/gnn_stacked_top.png    (F1 bar chart, full column width)
  reports/gnn_stacked_bot.png    (3-metric grouped bars, full column width)
  reports/ae_errors_right.png    (overwritten, top suptitle area cropped)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

REPORTS = os.path.join(os.path.dirname(__file__), "..", "reports")
CSV     = os.path.join(REPORTS, "gnn_experiments_table.csv")

# ─── 1. GNN stacked plots ────────────────────────────────────────────────────
df = pd.read_csv(CSV)

SAGE_COLOR = "#2980b9"
GAT_COLOR  = "#e74c3c"
is_sage = df["Name"].str.startswith(("GraphSAGE", "GraphSAGEv2"))

# --- top figure: F1 bar progression ---
fig, ax = plt.subplots(figsize=(9, 4))
colors = [SAGE_COLOR if s else GAT_COLOR for s in is_sage]
bars = ax.bar(df["Exp"], df["F1"], color=colors, edgecolor="white", width=0.7)
for bar, val in zip(bars, df["F1"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.006,
            f"{val:.3f}", ha="center", fontsize=8, fontweight="bold")
ax.set_xticks(df["Exp"])
ax.set_xticklabels([f"Exp {i}" for i in df["Exp"]], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("F1 — illicit class")
ax.set_ylim(0, 1.0)
ax.set_title("GNN Experiments — F1 Progression", fontsize=11, fontweight="bold")
ax.axhline(df["F1"].max(), color="gold", lw=1.5, ls="--",
           label=f"Best F1 = {df['F1'].max():.3f}")
ax.legend(handles=[
    mpatches.Patch(color=SAGE_COLOR, label="GraphSAGE family"),
    mpatches.Patch(color=GAT_COLOR,  label="GAT family"),
    plt.Line2D([0], [0], color="gold", lw=1.5, ls="--", label=f"Best F1={df['F1'].max():.3f}"),
], fontsize=8, loc="upper left")
plt.tight_layout()
out_top = os.path.join(REPORTS, "gnn_stacked_top.png")
plt.savefig(out_top, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out_top}")

# --- bottom figure: 3-metric grouped bars ---
fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(df))
w = 0.25
ax.bar(x - w, df["F1"],       w, label="F1",      color="#2ecc71", edgecolor="white")
ax.bar(x,     df["ROC-AUC"],  w, label="ROC-AUC", color="#3498db", edgecolor="white")
ax.bar(x + w, df["Avg-Prec"], w, label="Avg-Prec", color="#9b59b6", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels([f"E{i}" for i in df["Exp"]], fontsize=8)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_title("All Metrics per Experiment", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
plt.tight_layout()
out_bot = os.path.join(REPORTS, "gnn_stacked_bot.png")
plt.savefig(out_bot, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out_bot}")

# ─── 2. Crop partial suptitle from ae_errors_right.png ───────────────────────
ae_right_path = os.path.join(REPORTS, "ae_errors_right.png")
if not os.path.exists(ae_right_path):
    print(f"  SKIP (not found): ae_errors_right.png")
else:
    img = Image.open(ae_right_path)
    w, h = img.size
    # The partial suptitle text ("ction error distribution") sits at the very
    # top of this image (right half of the original wide figure).
    # Crop it by removing the top 55 pixels.
    CROP_TOP = 55
    if h > 460:  # guard: only crop once (original height ~480)
        img.crop((0, CROP_TOP, w, h)).save(ae_right_path)
        print(f"  Cropped top {CROP_TOP}px from ae_errors_right.png  ({w}x{h} -> {w}x{h-CROP_TOP})")
    else:
        print(f"  SKIP ae_errors_right.png crop (already cropped, h={h})")

print("Done.")
