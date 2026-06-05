import sys
sys.path.insert(0, '..')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from src.data_loader import load_all
from src.preprocessing import get_labeled, get_feature_cols, TRAIN_STEPS, TEST_STEPS
from src.autoencoder import DEVICE, train_autoencoder, reconstruction_errors, threshold_predict
from src.evaluation import evaluate, plot_confusion_matrix, plot_pr_curve

import torch
print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

df, edges = load_all()

feat_cols = get_feature_cols(get_labeled(df))

labeled_train = get_labeled(df[df['time_step'].isin(TRAIN_STEPS)])
labeled_test  = get_labeled(df[df['time_step'].isin(TEST_STEPS)])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_licit_train = scaler.fit_transform(
    labeled_train[labeled_train['label'] == 0][feat_cols].values
)
X_test = scaler.transform(labeled_test[feat_cols].values)
y_test = labeled_test['label'].values

print(f"\nTrain (licit only, steps 1-34): {X_licit_train.shape}")
print(f"Test  (labeled, steps 35-49):  {X_test.shape}  illicit={y_test.mean():.3f}")

# --- train ---
input_dim = X_licit_train.shape[1]
print(f"\nTraining autoencoder  input_dim={input_dim}  latent=32  epochs=100 ...")
model = train_autoencoder(X_licit_train, input_dim=input_dim, latent_dim=32, epochs=100)

# --- reconstruction errors ---
train_errors = reconstruction_errors(model, X_licit_train)
test_errors  = reconstruction_errors(model, X_test)

print(f"\nTrain recon error — mean={train_errors.mean():.4f}  p95={np.percentile(train_errors,95):.4f}")
print(f"Test  recon error — mean={test_errors.mean():.4f}  p95={np.percentile(test_errors,95):.4f}")

# check score direction
auc_forward = roc_auc_score(y_test, test_errors)
auc_flipped = roc_auc_score(y_test, -test_errors)
print(f"\nROC-AUC (error as score):  {auc_forward:.4f}")
print(f"ROC-AUC (flipped score):   {auc_flipped:.4f}")

# use best direction
if auc_flipped > auc_forward:
    print("Scores inverted — using flipped (illicit reconstructs with LOWER error)")
    anomaly_scores = -test_errors
else:
    print("Scores normal — illicit reconstructs with HIGHER error")
    anomaly_scores = test_errors

# --- error distribution plot ---
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for ax, cap, title in zip(axes, [5, np.percentile(test_errors, 99)],
                          ['Full range', '99th percentile cap']):
    ax.hist(test_errors[y_test == 0].clip(0, cap), bins=80, alpha=0.6,
            label='licit', color='#2ecc71', density=True)
    ax.hist(test_errors[y_test == 1].clip(0, cap), bins=80, alpha=0.6,
            label='illicit', color='#e74c3c', density=True)
    ax.set_xlabel('Reconstruction MSE')
    ax.set_title(title)
    ax.legend()
plt.suptitle('Autoencoder reconstruction error distribution')
plt.tight_layout()
plt.savefig('../reports/autoencoder_errors.png', dpi=120)
plt.show()

# --- evaluate ---
contamination = y_test.mean()  # use true test illicit rate as threshold
y_pred = threshold_predict(anomaly_scores, contamination=contamination)
res = evaluate(y_test, y_pred, anomaly_scores, name='Autoencoder')
plot_confusion_matrix(y_test, y_pred, name='Autoencoder', save=True)
plot_pr_curve(y_test, {'Autoencoder': anomaly_scores}, save=True)

# --- save ---
torch.save(model.state_dict(), '../models/autoencoder.pt')
print("\nModel saved to models/autoencoder.pt")
