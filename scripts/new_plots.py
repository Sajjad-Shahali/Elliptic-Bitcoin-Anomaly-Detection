"""
Generate three new report plots and fix ae_errors_left.png partial suptitle.

Outputs:
  reports/concept_drift.png        illicit rate per time step (concept drift)
  reports/beta_f1_curve.png        beta-VAE beta vs F1 curve
  reports/ae_errors_left.png       overwrite: crop partial suptitle from top
  (gnn_experiments_radar.png already exists — no action needed)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT    = os.path.join(os.path.dirname(__file__), "..")
REPORTS = os.path.join(ROOT, "reports")
DATA    = os.path.join(ROOT, "data", "elliptic_bitcoin_dataset")

# ─── 1. Fix ae_errors_left.png  (crop partial suptitle from top) ─────────────
ae_left_path = os.path.join(REPORTS, "ae_errors_left.png")
if os.path.exists(ae_left_path):
    img = Image.open(ae_left_path)
    w, h = img.size
    CROP_TOP = 55
    # Only crop if image still has the original suptitle area (height near 480)
    if h > 460:
        img.crop((0, CROP_TOP, w, h)).save(ae_left_path)
        print(f"  Cropped top {CROP_TOP}px from ae_errors_left.png  ({w}x{h} -> {w}x{h-CROP_TOP})")
    else:
        print(f"  SKIP ae_errors_left.png crop (already cropped, h={h})")
else:
    print("  SKIP: ae_errors_left.png not found")

# ─── 2. Concept drift: illicit rate per time step ────────────────────────────
print("  Building concept drift plot...")

feat_path  = os.path.join(DATA, "elliptic_txs_features.csv")
class_path = os.path.join(DATA, "elliptic_txs_classes.csv")

# features CSV: col0=txId, col1=time_step (no header for features themselves)
feat_cols = ["txId", "time_step"] + [f"f{i}" for i in range(165)]
feat = pd.read_csv(feat_path, header=None, names=feat_cols, usecols=["txId", "time_step"])
cls  = pd.read_csv(class_path)

df = feat.merge(cls, on="txId")
df = df[df["class"].isin(["1", "2"])]   # labeled only
df["time_step"] = df["time_step"].astype(int)
df["illicit"]   = (df["class"] == "1").astype(int)

per_step = df.groupby("time_step").agg(
    total   = ("illicit", "count"),
    illicit = ("illicit", "sum"),
).reset_index()
per_step["rate"] = per_step["illicit"] / per_step["total"]

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.fill_between(per_step["time_step"], per_step["rate"] * 100,
                alpha=0.25, color="#e74c3c")
ax.plot(per_step["time_step"], per_step["rate"] * 100,
        color="#e74c3c", lw=2, marker="o", ms=4, label="Illicit rate (%)")

# train/test boundary
ax.axvline(34.5, color="#2c3e50", lw=1.5, ls="--", label="Train | Test split (step 34/35)")
ax.axvspan(1,  34.5, alpha=0.07, color="#3498db", label="Train (steps 1–34)")
ax.axvspan(34.5, 49, alpha=0.07, color="#e67e22", label="Test  (steps 35–49)")

# annotate average rates — draw dashed mean lines, text near top
train_rate = per_step[per_step["time_step"] <= 34]["rate"].mean() * 100
test_rate  = per_step[per_step["time_step"] >  34]["rate"].mean() * 100

y_max = per_step["rate"].max() * 100
text_y = y_max * 0.92   # near top of chart

ax.axhline(train_rate, xmin=0,    xmax=34.5/49, color="#2980b9",
           lw=1.2, ls=":", alpha=0.7)
ax.axhline(test_rate,  xmin=34.5/49, xmax=1.0,  color="#e67e22",
           lw=1.2, ls=":", alpha=0.7)

ax.annotate(f"Train avg\n{train_rate:.1f}%",
            xy=(17, train_rate), xytext=(17, text_y),
            fontsize=9, fontweight="bold", color="#2980b9",
            ha="center", va="top",
            arrowprops=dict(arrowstyle="-", color="#2980b9",
                            lw=1.0, linestyle="dotted"))
ax.annotate(f"Test avg\n{test_rate:.1f}%",
            xy=(42, test_rate), xytext=(42, text_y),
            fontsize=9, fontweight="bold", color="#e67e22",
            ha="center", va="top",
            arrowprops=dict(arrowstyle="-", color="#e67e22",
                            lw=1.0, linestyle="dotted"))

ax.set_xlabel("Time step")
ax.set_ylabel("Illicit rate (%)")
ax.set_title("Concept Drift: Illicit Transaction Rate per Time Step", fontweight="bold")
ax.set_xlim(1, 49)
ax.set_ylim(0, None)
ax.legend(fontsize=8, loc="upper right")
plt.tight_layout()
out = os.path.join(REPORTS, "concept_drift.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

# ─── 3. beta-VAE: beta vs F1 curve ──────────────────────────────────────────
print("  Building beta-VAE F1 curve...")

# beta values and F1 from final experiments (TABLE VI in report)
betas  = [0.01, 1.0,  2.0,  4.0,  8.0]
f1s    = [0.407, 0.372, 0.280, 0.313, 0.350]
aucs   = [0.783, 0.776, 0.746, 0.754, 0.763]

fig, ax1 = plt.subplots(figsize=(7, 3.5))
ax2 = ax1.twinx()

ax1.plot(betas, f1s, "o-", color="#e74c3c", lw=2, ms=7, label="F1 (illicit)")
ax2.plot(betas, aucs, "s--", color="#3498db", lw=1.5, ms=6, label="ROC-AUC")

# annotate best
best_idx = f1s.index(max(f1s))
ax1.annotate(f"β={betas[best_idx]}\nF1={f1s[best_idx]:.3f}",
             xy=(betas[best_idx], f1s[best_idx]),
             xytext=(0.15, 0.36), fontsize=8, color="#c0392b",
             arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1))

ax1.set_xlabel("β (KL weight in β-VAE ELBO)")
ax1.set_ylabel("F1 — illicit class", color="#e74c3c")
ax2.set_ylabel("ROC-AUC", color="#3498db")
ax1.tick_params(axis="y", labelcolor="#e74c3c")
ax2.tick_params(axis="y", labelcolor="#3498db")
ax1.set_xscale("log")
ax1.set_title("β-VAE Anomaly Detection: Effect of KL Weight β on Performance",
              fontweight="bold")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower right")

ax1.grid(True, alpha=0.3)
plt.tight_layout()
out = os.path.join(REPORTS, "beta_f1_curve.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

print("\nDone.")
