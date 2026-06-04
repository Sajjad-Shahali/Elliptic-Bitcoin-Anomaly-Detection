import sys
sys.path.insert(0, '..')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_all
from src.preprocessing import get_labeled, get_feature_cols, TRAIN_STEPS, TEST_STEPS

sns.set_theme(style='whitegrid')

df, edges = load_all()
print(f"Total transactions: {len(df):,}")
print(f"Columns: {df.shape[1]}")

# --- class distribution ---
counts = df['class'].value_counts()
print("\nClass counts:")
print(counts)

labeled = get_labeled(df)
illicit_pct = (labeled['label'] == 1).mean() * 100
print(f"\nIllicit rate (labeled): {illicit_pct:.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
counts.plot(kind='bar', ax=axes[0], color=['#95a5a6', '#2ecc71', '#e74c3c'])
axes[0].set_title('All transactions')
axes[0].set_xticklabels(['unknown', 'licit', 'illicit'], rotation=0)

labeled['class'].value_counts().plot(kind='bar', ax=axes[1], color=['#2ecc71', '#e74c3c'])
axes[1].set_title('Labeled only')
axes[1].set_xticklabels(['licit', 'illicit'], rotation=0)
plt.tight_layout()
plt.savefig('../reports/class_distribution.png', dpi=120)
plt.show()

# --- temporal split visualization ---
time_class = df.groupby(['time_step', 'class']).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(14, 5))
time_class.plot(ax=ax)
ax.axvline(x=34.5, color='red', linestyle='--', linewidth=2, label='train/test split (step 34|35)')
ax.set_title('Transactions per time step — red line = temporal train/test split')
ax.set_xlabel('Time step')
ax.legend()
plt.tight_layout()
plt.savefig('../reports/temporal_split.png', dpi=120)
plt.show()

train_labeled = labeled[labeled['time_step'].isin(TRAIN_STEPS)]
test_labeled  = labeled[labeled['time_step'].isin(TEST_STEPS)]
print(f"\nTrain (steps 1-34): {len(train_labeled):,} labeled  |  illicit: {train_labeled['label'].mean():.3f}")
print(f"Test  (steps 35-49): {len(test_labeled):,} labeled  |  illicit: {test_labeled['label'].mean():.3f}")

# --- feature distributions ---
feat_cols = get_feature_cols(labeled)
X_illicit = labeled[labeled['label'] == 1][feat_cols]
X_licit   = labeled[labeled['label'] == 0][feat_cols]

sample_feats = feat_cols[1:7]
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for i, col in enumerate(sample_feats):
    ax = axes[i // 3][i % 3]
    ax.hist(X_licit[col].clip(-5, 5), bins=60, alpha=0.6, label='licit', color='#2ecc71', density=True)
    ax.hist(X_illicit[col].clip(-5, 5), bins=60, alpha=0.6, label='illicit', color='#e74c3c', density=True)
    ax.set_title(col)
    ax.legend(fontsize=8)
plt.suptitle('Local features: illicit vs licit (clipped ±5)', y=1.01)
plt.tight_layout()
plt.savefig('../reports/feature_distributions.png', dpi=120)
plt.show()

# --- graph stats ---
print(f"\nEdges: {len(edges):,}")
print(f"Avg degree: {len(edges) * 2 / df['txId'].nunique():.2f}")
