# Elliptic Bitcoin Anomaly Detection

Detecting illicit Bitcoin transactions using the [Elliptic dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) — a real-world graph of 203,769 transactions on the Bitcoin blockchain, partially labeled as illicit (money laundering, scams) or licit.

This project systematically benchmarks **8 anomaly detection methods** across 4 paradigms: unsupervised, supervised, deep learning, and graph neural networks.

---

## Results

> Test set: time steps 35–49 (temporal split). Illicit rate: 6.5%.

| Rank | Model | Category | F1 (illicit) | ROC-AUC | Avg Precision | Precision | Recall |
|------|-------|----------|:------------:|:-------:|:-------------:|:---------:|:------:|
| 1 | **Random Forest** | Supervised | **0.801** | **0.935** | **0.794** | 0.901 | 0.721 |
| 2 | Gradient Boosting | Supervised | 0.766 | 0.914 | 0.785 | 0.815 | 0.722 |
| 3 | GraphSAGE | GNN | 0.541 | 0.888 | 0.549 | 0.478 | 0.623 |
| 4 | Autoencoder | Neural | 0.356 | 0.781 | 0.336 | 0.356 | 0.356 |
| 5 | Logistic Regression | Supervised | 0.302 | 0.881 | 0.291 | 0.182 | 0.872 |
| 6 | LOF | Unsupervised | 0.076 | 0.506 | 0.066 | 0.076 | 0.076 |
| 7 | One-Class SVM | Unsupervised | 0.023 | 0.235 | 0.039 | 0.015 | 0.049 |
| 8 | Isolation Forest | Unsupervised | 0.021 | 0.172 | 0.036 | 0.012 | 0.078 |

**Primary metric: F1 on illicit class.** Accuracy is misleading at 6.5% illicit rate.

### Key Takeaways

- **Graph structure matters.** GraphSAGE (F1 0.54) outperforms the Autoencoder (F1 0.36) using the same features — neighborhood aggregation captures guilt-by-association in transaction chains.
- **Supervised >> Unsupervised** by a wide margin (F1 0.80 vs 0.08). Classical anomaly detectors fail on this dataset: illicit transactions do not form tight, separable clusters in feature space.
- **Autoencoder score inverts.** Illicit transactions reconstruct with *lower* MSE than licit ones — they follow more templated, script-like patterns. High reconstruction error ≠ anomaly here.
- **Three features agree across all model families:** `lf_53`, `lf_90`, `af_70` rank top-20 in Random Forest (SHAP), Gradient Boosting (SHAP), and GraphSAGE (gradient attribution).
- **Concept drift is real.** Illicit rate drops from 11.6% (train steps 1–34) to 6.5% (test steps 35–49). Random splitting inflates all metrics and should not be used.

---

## Dataset

The [Elliptic Data Set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) was published by Elliptic, a blockchain analytics company.

| File | Size | Description |
|------|------|-------------|
| `elliptic_txs_features.csv` | 657 MB | 203,769 transactions × 165 anonymous features |
| `elliptic_txs_classes.csv` | 3.2 MB | Transaction labels |
| `elliptic_txs_edgelist.csv` | 4.3 MB | 234,355 directed edges |

### Labels
| Class | Count | Meaning |
|-------|-------|---------|
| `1` (illicit) | 4,545 | Money laundering, scams, ransomware |
| `2` (licit) | 42,019 | Exchanges, wallets, services |
| `unknown` | 157,205 | Unlabeled |

### Features
- **Column 0:** `txId` (transaction identifier)
- **Column 1:** `time_step` (1–49 discrete time steps)
- **Columns 2–94:** 93 local features (transaction-level — amounts, fees, timestamps)
- **Columns 95–165:** 71 aggregated features (neighborhood statistics)
- Feature names are not published by Elliptic for confidentiality reasons

### Temporal Structure
49 time steps spanning ~2 years of Bitcoin history. Steps 1–43 have labels; 44–49 are unlabeled.

---

## Methodology

### Temporal Train/Test Split

All models use the split from the original Elliptic paper:
- **Train:** steps 1–34
- **Test:** steps 35–49

Random shuffling is explicitly avoided — this is a time-series problem. Shuffling leaks future transaction patterns into training and produces optimistic metrics that do not reflect real-world deployment.

### Evaluation Protocol
- Unsupervised models: trained on all train-step rows (including unknown); evaluated on labeled test rows only
- OCSVM / Autoencoder: trained on licit-only train rows (semi-supervised)
- Supervised models: trained on labeled train rows only
- GraphSAGE: trained on full graph, supervised by labeled train nodes

---

## Methods

### Unsupervised
| Method | Description |
|--------|-------------|
| **Isolation Forest** | Isolates anomalies via random splits. Contamination=0.1. |
| **Local Outlier Factor** | Density-based; transductive — applied directly to test set. |
| **One-Class SVM** | Trained on licit-only transactions. ν=0.05, RBF kernel. |

### Supervised
| Method | Description |
|--------|-------------|
| **Logistic Regression** | L2 regularized, `class_weight='balanced'`. Baseline. |
| **Random Forest** | 100 trees, `class_weight='balanced'`. Best single model. |
| **Gradient Boosting** | 100 trees, sklearn GBM. |

### Neural
| Method | Description |
|--------|-------------|
| **Autoencoder** | PyTorch. Architecture: 166→128→64→32→64→128→166. Trained on licit-only. Anomaly score = reconstruction MSE (inverted — lower = more anomalous). BatchNorm + Dropout. 100 epochs, LR decay, GPU. |

### Graph Neural Network
| Method | Description |
|--------|-------------|
| **GraphSAGE** | 2-layer SAGEConv. Hidden dim 128. Mean aggregation over neighbors. Inverse-frequency class weighting. 200 epochs, cosine LR schedule, GPU. Full graph (203k nodes, 468k edges) fits in VRAM. |

---

## Explainability

SHAP TreeExplainer (RF + GBM) and gradient attribution (GraphSAGE) were used to identify the most predictive features.

### SHAP — Random Forest (top features by mean |SHAP|)
![SHAP bar](reports/shap_rf_bar.png)

### SHAP Beeswarm — direction and magnitude
![SHAP beeswarm](reports/shap_rf_beeswarm.png)

### GraphSAGE Gradient Attribution
![GNN attribution](reports/gnn_gradient_attribution.png)

### Cross-Model Feature Agreement
| Agreement | Features |
|-----------|----------|
| RF ∩ GBM (7/20) | `lf_18`, `lf_47`, `lf_53`, `lf_59`, `lf_76`, `lf_90`, `af_70` |
| RF ∩ GBM ∩ GNN (3/20) | **`lf_53`**, **`lf_90`**, **`af_70`** |

These three features are the most robust signals — they rank top-20 regardless of model family.

---

## Project Structure

```
├── config.py                    # paths, constants, label map, random seed
├── download_data.py             # download dataset via kagglehub
├── requirements.txt
├── HANDOFF.md                   # technical decisions + known issues
│
├── src/
│   ├── data_loader.py           # load_features(), load_classes(), load_all()
│   ├── preprocessing.py         # temporal train/test split pipelines
│   ├── models.py                # sklearn model factories
│   ├── autoencoder.py           # PyTorch autoencoder + train/inference
│   ├── gnn.py                   # GraphSAGE model + train/eval
│   ├── graph_utils.py           # build PyG Data object from raw data
│   └── evaluation.py            # metrics, confusion matrix, PR/ROC plots
│
├── scripts/                     # run in order from project root
│   ├── 01_eda.py                # exploratory analysis
│   ├── 02_unsupervised.py       # IsolationForest, LOF, OCSVM
│   ├── 03_supervised.py         # RF, GBM, Logistic Regression
│   ├── 04_autoencoder.py        # PyTorch autoencoder
│   ├── 05_gnn.py                # GraphSAGE training
│   ├── 06_explainability.py     # SHAP + gradient attribution
│   ├── 07_final_comparison.py   # all models, leaderboard + plots
│   ├── 08_tune_and_save.py      # Optuna tuning, joblib save
│   └── 09_tuned_comparison.py   # baseline vs tuned comparison
│
├── models/                      # saved model weights
│   ├── rf_tuned.joblib
│   ├── gb_tuned.joblib
│   ├── scaler.joblib
│   ├── autoencoder.pt
│   ├── graphsage.pt
│   └── graphsage_tuned.pt
│
├── reports/                     # auto-generated plots (PNG) + CSVs
└── data/                        # gitignored — run download_data.py
```

---

## Setup

### Requirements
- Python 3.10+
- CUDA GPU recommended (RTX 5070 Ti used; CPU fallback works)

### Install

```bash
# create and activate venv
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux/Mac

# install dependencies
pip install -r requirements.txt

# GPU PyTorch (CUDA 12.8 — adjust for your driver)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# PyTorch Geometric
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
```

### Download Dataset

```bash
python download_data.py
```

Requires a [Kaggle account](https://www.kaggle.com/) and `~/.kaggle/kaggle.json` API credentials.

### Run

```bash
python scripts/01_eda.py
python scripts/02_unsupervised.py
python scripts/03_supervised.py
python scripts/04_autoencoder.py
python scripts/05_gnn.py
python scripts/06_explainability.py
python scripts/07_final_comparison.py   # full leaderboard
python scripts/08_tune_and_save.py      # Optuna tuning (~20 min)
python scripts/09_tuned_comparison.py   # baseline vs tuned
```

---

## Visual Results

### Final Leaderboard
![Comparison](reports/final_comparison.png)

### Precision-Recall Curves
Precision-recall is the correct metric for highly imbalanced anomaly detection (6.5% positive rate). Random baseline AP = 0.065.

### Confusion Matrix — Best Model (Random Forest)
![CM RF](reports/cm_RandomForest.png)

---

## Known Issues & Design Notes

| Issue | Detail |
|-------|--------|
| `kagglehub>=1.0.0` import error | Pin to `0.3.6`. Bug: `get_web_endpoint` missing from `kagglesdk`. |
| PyG extension warnings | `pyg-lib`, `torch-scatter`, `torch-sparse` built for torch 2.7, used with 2.11. Non-fatal — falls back to pure Python. |
| LOF transductive | Cannot score unseen data. Applied to test set directly. No continuous anomaly score available. |
| Autoencoder score inversion | Illicit txs have lower reconstruction error. Script auto-detects direction via ROC-AUC comparison. |
| Feature names unknown | Elliptic does not publish feature semantics. Named `lf_1..93` (local) and `af_1..72` (aggregated). |

---

## Environment

| Component | Version |
|-----------|---------|
| Python | 3.12 |
| PyTorch | 2.11.0+cu128 |
| torch_geometric | 2.8.0 |
| scikit-learn | 1.9.0 |
| SHAP | 0.52.0 |
| Optuna | 4.9.0 |
| GPU | NVIDIA RTX 5070 Ti Laptop (12 GB VRAM) |
| CUDA Driver | 592.01 (CUDA 13.1) |

---

## References

- Weber, M. et al. (2019). [Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics](https://arxiv.org/abs/1908.02591). KDD Workshop.
- Elliptic Data Set: [kaggle.com/datasets/ellipticco/elliptic-data-set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
- Hamilton, W. et al. (2017). [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216). NeurIPS. (GraphSAGE)
- Lundberg, S. & Lee, S. (2017). [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874). NeurIPS. (SHAP)
