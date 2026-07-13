# Elliptic Bitcoin Anomaly Detection -- Handoff

## What This Is

Anomaly detection on the Elliptic Bitcoin dataset. Identifies illicit Bitcoin transactions (money laundering, scams) using 8 methods across 4 paradigms: unsupervised, supervised, deep learning, GNN.

---

## Dataset

| File | Size | Description |
|------|------|-------------|
| `elliptic_txs_features.csv` | 657 MB | 203,769 transactions Ã— 165 features (no header) |
| `elliptic_txs_classes.csv` | 3.2 MB | txId -> class label |
| `elliptic_txs_edgelist.csv` | 4.3 MB | directed edges (txId1, txId2) |

**Labels:** `1` = illicit, `2` = licit, `unknown` = unlabeled

**Class breakdown:** 157,205 unknown (77%) . 42,019 licit (21%) . 4,545 illicit (2%)

**Features:** Col 0 = txId, Col 1 = time_step (1--49), Cols 2--94 = 93 local features, Cols 95--165 = 71 aggregated features. Names not published by Elliptic.

**Time steps:** 49 steps. Steps 1--43 labeled; 44--49 unlabeled.

---

## Project Structure

```
config.py                         # paths, constants, label map, random seed
download_data.py                  # re-download dataset via kagglehub
requirements.txt
README.md
HANDOFF.md

src/
    data_loader.py                # load_features(), load_classes(), load_all()
    preprocessing.py              # temporal train/val/test split pipelines
    models.py                     # sklearn model factories
    autoencoder.py                # AE + DenoisingAE, EVT threshold, F1-optimal threshold
    vae.py                        # VAE + VAEv2 + beta-VAE
    dominant.py                   # DOMINANT GCN-AE
    lstm_ae.py                    # LSTM encoder-decoder, sliding windows
    gnn.py                        # GraphSAGE + GraphSAGEv2
    gat.py                        # GAT + GATv2
    graph_features.py             # degree, PageRank, clustering
    graph_utils.py                # build PyG Data object
    inference.py                  # load_pipeline() + score_transactions()
    evaluation.py                 # metrics, confusion matrix, PR/ROC plots

scripts/
    eda.py                        # class distribution, temporal viz, feature dists
    baseline_unsupervised.py      # IsolationForest, LOF, OCSVM
    baseline_supervised.py        # RF, GBM, LogReg + feature importance
    baseline_autoencoder.py       # baseline autoencoder
    baseline_gnn.py               # baseline GraphSAGE
    shap_explainability.py        # SHAP + gradient attribution
    baseline_comparison.py        # all baseline models leaderboard + plots
    optuna_tuning.py              # Optuna 40 trials RF/GBM/GraphSAGE
    tuned_comparison.py           # baseline vs tuned comparison
    improved_models.py            # LightGBM, RF+graph, GAT, VAE (Round 1)
    improved_comparison.py        # Round 1 improvement deltas + plots
    deep_architecture_improvements.py  # GATv2, GraphSAGEv2, DenoisingAE, VAEv2 (Round 2)
    gnn_ablation.py               # 10-experiment GNN ablation study
    dominant_graph_ae.py          # DOMINANT GCN-AE (Round 3)
    lstm_autoencoder.py           # LSTM-AE temporal anomaly detection (Round 3)
    ensemble_soft_vote.py         # GBM+LGBM+SAGEv2 soft-vote ensemble (Round 3)
    pseudo_label_training.py      # semi-supervised GBM+SAGEv2 (Round 4)
    beta_vae_gridsearch.py        # beta-VAE grid search (Round 4)
    structural_graph_features.py  # topology features + GBM -- BEST MODEL (Round 4)
    spectral_anomaly_detection.py # spectral embedding IsoForest/OCSVM (Round 4)
    weighted_ensemble.py          # weighted ensemble sweep (Round 5)
    probability_calibration.py    # Platt + isotonic calibration (Round 5)
    temporal_rolling_features.py  # rolling mean/std features (Round 5)
    lgbm_xgboost_structural.py    # LightGBM/XGBoost on structural features
    final_leaderboard.py          # full leaderboard + 3 comparison plots

models/                           # saved weights
    scaler.joblib / scaler_graph_features.joblib / scaler_structural.joblib
    rf_tuned.joblib / gb_tuned.joblib / lightgbm.joblib
    rf_graph_features.joblib / gbm_structural.joblib  <-- BEST MODEL
    gbm_pseudo.joblib / gbm_temporal_rolling.joblib
    lgbm_structural.joblib / lgbm_structural_shrinkage.joblib
    autoencoder.pt / denoising_ae.pt / dominant.pt / lstm_ae.pt
    vae.pt / vaev2.pt
    graphsage.pt / graphsage_tuned.pt / graphsagev2.pt / graphsagev2_pseudo.pt
    gat.pt / gatv2.pt

reports/                          # auto-generated plots + CSVs
data/                             # gitignored -- run download_data.py
```

---

## How to Run

```bash
# activate venv
.venv\Scripts\activate           # Windows
source .venv/bin/activate        # Linux/Mac

# download data (requires ~/.kaggle/kaggle.json)
python download_data.py

# run pipeline in order
python scripts/eda.py
python scripts/baseline_unsupervised.py
python scripts/baseline_supervised.py
python scripts/baseline_autoencoder.py
python scripts/baseline_gnn.py
python scripts/shap_explainability.py
python scripts/baseline_comparison.py   # full leaderboard
python scripts/optuna_tuning.py      # ~20 min, needs GPU
python scripts/tuned_comparison.py   # run after 08
```

---

## Key Design Decisions

**Temporal split (Elliptic paper standard):**
- Train = steps 1--34 . Test = steps 35--49
- Random split explicitly avoided -- shuffling leaks future into past, inflates all metrics

**Actual split sizes:**
- Supervised: 29,894 train (illicit 11.6%) / 16,670 test (illicit 6.5%)
- Unsupervised: 136,265 train (all rows) / 16,670 test (labeled only)
- AE / OCSVM: trained on licit-only train rows (26,432 txs)

**Label encoding:** `1 -> 1` (illicit/anomaly), `2 -> 0` (licit/normal). In `config.LABEL_MAP`.

**Feature scaling:** `StandardScaler` fit on train only. Returned from both prepare functions. Saved to `models/scaler.joblib`.

**Class imbalance:** RF/GBM use `class_weight='balanced'`. GNN uses inverse-frequency `CrossEntropyLoss` weights. Unsupervised contamination = `0.1`.

**Autoencoder score direction:** Illicit txs reconstruct with *lower* MSE (they follow templated patterns). Anomaly score = `-MSE`. Script auto-detects direction by comparing `roc_auc_score(y, errors)` vs `roc_auc_score(y, -errors)`.

**GraphSAGE graph construction:** Edges made undirected (both directions added). Full graph pushed to VRAM. Train mask = labeled nodes in steps 1--34. Test mask = labeled nodes in steps 35--49.

**Optuna tuning:** 40 trials per model, maximising F1 on illicit class. GNN trials each do a full retrain -- GPU required for reasonable runtime.

---

## Primary Metric

**F1 on illicit class.** Accuracy is meaningless at 6.5% illicit rate. Illicit rate drops from 11.6% (train) to 6.5% (test) -- real concept drift, not a bug.

Use **PR-AUC** to compare models independent of threshold.

---

## Results

### Complete Experiment Ranking -- All 48 Experiments, Best to Worst F1

| Rank | Model | Category | Script | F1 | AUC | Avg-Prec |
|------|-------|----------|--------|:--:|:---:|:--------:|
| 1 | GBM + Structural (165+7 feat) | Supervised | 18 | **0.8265** | 0.9243 | 0.8031 |
| 2 | GradientBoosting Optuna | Supervised | 08 | 0.8241 | 0.920 | -- |
| 3 | GBM + PseudoLabels | Semi-sup | 16 | 0.8235 | 0.9240 | 0.8040 |
| 4 | Ensemble (GBM+LGBM+SAGEv2) | Ensemble | 15 | 0.821 | 0.924 | 0.802 |
| 5 | Weighted Ensemble (equal 1/3) | Ensemble | 20 | 0.8209 | 0.9238 | 0.8019 |
| 6 | LightGBM | Supervised | 10 | 0.817 | 0.930 | 0.804 |
| 7 | GBM + temporal rolling (496 feat) | Supervised | 22 | 0.8179 | 0.9300 | 0.8017 |
| 8 | RF + GraphFeatures | Supervised | 10 | 0.807 | 0.942 | 0.800 |
| 9 | GBM Platt-calibrated | Supervised | 21 | 0.8006 | 0.9204 | 0.8000 |
| 10 | RandomForest baseline | Supervised | baseline | 0.801 | 0.935 | 0.794 |
| 11 | RandomForest Optuna | Supervised | 08 | 0.797 | 0.929 | -- |
| 12 | GradientBoosting baseline | Supervised | baseline | 0.766 | 0.914 | 0.785 |
| 13 | Ensemble RF+LGB+GAT | Ensemble | 10 | 0.742 | 0.922 | 0.789 |
| 14 | SAGEv2 + PseudoLabels | Semi-sup GNN | 16 | 0.7293 | 0.8901 | 0.7517 |
| 15 | GBM Isotonic-calibrated | Supervised | 21 | 0.7340 | 0.8753 | 0.7682 |
| 16 | GraphSAGEv2 | GNN | 12 | 0.698 | 0.913 | 0.720 |
| 17 | GNN Exp 7 -- Optuna 3L h=256 | GNN | gnn_exp | 0.690 | 0.901 | 0.706 |
| 18 | GraphSAGE Optuna | GNN | 08 | 0.688 | -- | -- |
| 19 | GNN Exp 6 -- 3L h=256 LN+SL | GNN | gnn_exp | 0.683 | 0.906 | 0.704 |
| 20 | GNN Exp 5 -- 3L h=128 | GNN | gnn_exp | 0.615 | 0.899 | 0.611 |
| 21 | GNN Exp 4 -- 2L h=128 300ep | GNN | gnn_exp | 0.597 | 0.897 | 0.607 |
| 22 | GNN Exp 3 -- 2L h=256 | GNN | gnn_exp | 0.580 | 0.895 | 0.595 |
| 23 | GNN Exp 2 -- 2L h=128 | GNN | gnn_exp | 0.577 | 0.896 | 0.603 |
| 24 | GraphSAGE baseline | GNN | baseline | 0.541 | 0.888 | 0.549 |
| 25 | GNN Exp 1 -- 2L h=64 | GNN | gnn_exp | 0.528 | 0.887 | 0.519 |
| 26 | beta-VAE (beta=0.01) | Neural-AE | 17 | 0.4065 | 0.7832 | 0.3651 |
| 27 | GNN Exp 10 -- GATv2 3L residual | GNN | gnn_exp | 0.382 | 0.887 | 0.547 |
| 28 | DenoisingAE | Neural-AE | 12 | 0.361 | 0.792 | 0.366 |
| 29 | GAT baseline | GNN | 10 | 0.359 | 0.847 | 0.389 |
| 30 | Autoencoder baseline | Neural-AE | 04/07 | 0.356 | 0.781 | 0.336 |
| 31 | beta-VAE (beta=1.0) | Neural-AE | 17 | 0.3717 | 0.7760 | 0.3384 |
| 32 | GNN Exp 9 -- GAT 2L 2heads | GNN | gnn_exp | 0.341 | 0.864 | 0.419 |
| 33 | VAE | Neural-AE | 10 | 0.340 | 0.778 | 0.310 |
| 34 | beta-VAE (beta=8.0) | Neural-AE | 17 | 0.3496 | 0.7625 | 0.2993 |
| 35 | GATv2 | GNN | 12 | 0.317 | 0.882 | 0.376 |
| 36 | beta-VAE (beta=4.0) | Neural-AE | 17 | 0.3127 | 0.7535 | 0.2867 |
| 37 | GNN Exp 8 -- GAT 2L 4heads | GNN | gnn_exp | 0.307 | 0.824 | 0.366 |
| 38 | LogisticRegression | Supervised | baseline | 0.302 | 0.881 | 0.291 |
| 39 | beta-VAE (beta=2.0) | Neural-AE | 17 | 0.2798 | 0.7455 | 0.2108 |
| 40 | VAEv2 | Neural-AE | 12 | 0.256 | 0.732 | 0.161 |
| 41 | DOMINANT | Graph-AE (unsup) | 13 | 0.212 | 0.669 | 0.123 |
| 42 | LSTM-AE | Temporal-AE (unsup) | 14 | 0.122 | 0.446 | 0.064 |
| 43 | IsoForest (structural only) | Unsupervised | 18 | 0.0941 | 0.4576 | 0.0556 |
| 44 | LOF | Unsupervised | baseline | 0.076 | 0.506 | 0.066 |
| 45 | OCSVM (spectral K=50) | Unsupervised | 19 | 0.0613 | 0.3909 | 0.0537 |
| 46 | IsoForest (spectral K=50) | Unsupervised | 19 | 0.0562 | 0.4071 | 0.0521 |
| 47 | OneClassSVM | Unsupervised | baseline | 0.023 | 0.235 | 0.039 |
| 48 | IsolationForest | Unsupervised | baseline | 0.021 | 0.172 | 0.036 |

Five tiers: Supervised trees (F1 0.74--0.83) > GNN supervised (0.53--0.73) > Neural AE (0.21--0.41) > Unsupervised topology (0.06--0.09) > Classical unsupervised (0.02--0.08)

---

### Final Leaderboard (all versions, all improvements)

| Rank | Model | Version | F1 | ROC-AUC | Avg-Prec |
|------|-------|---------|:--:|:-------:|:--------:|
| 1 | GradientBoosting | Optuna tuned | **0.824** | 0.920 | -- |
| 2 | LightGBM | Improved | 0.817 | 0.930 | 0.804 |
| 3 | RF + GraphFeatures | Improved | 0.807 | 0.942 | 0.800 |
| 4 | RandomForest | Baseline | 0.801 | 0.935 | 0.794 |
| 5 | GraphSAGEv2 | Round-2 improved | 0.698 | 0.913 | 0.720 |
| 6 | GraphSAGE | Optuna tuned | 0.688 | -- | -- |
| 7 | Ensemble (RF+LGB+GAT) | Improved | 0.742 | 0.922 | 0.789 |
| 8 | GraphSAGE | Baseline | 0.541 | 0.888 | 0.549 |
| 9 | DenoisingAE | Round-2 improved | 0.361 | 0.792 | 0.366 |
| 10 | GAT | Improved | 0.359 | 0.847 | 0.389 |
| 11 | Autoencoder | Baseline | 0.356 | 0.781 | 0.336 |
| 12 | VAE | Improved | 0.340 | 0.778 | 0.310 |
| 13 | GATv2 | Round-2 improved | 0.317 | 0.882 | 0.376 |
| 14 | LogisticRegression | Baseline | 0.302 | 0.881 | 0.291 |
| 15 | VAEv2 | Round-2 improved | 0.256 | 0.732 | 0.161 |
| 16 | LOF | Baseline | 0.076 | 0.506 | 0.066 |
| 17 | OneClassSVM | Baseline | 0.023 | 0.235 | 0.039 |
| 18 | IsolationForest | Baseline | 0.021 | 0.172 | 0.036 |

### Key Findings

- **GBM tuned (Optuna) = best model overall at F1=0.824** -- sklearn GBM just needed the right LR + depth
- LightGBM F1=0.817 -- close second, no tuning needed
- **GraphSAGEv2 biggest GNN jump: +15.7 F1** (0.541->0.698) from 3 layers + LayerNorm + self-loops
- GAT consistently underperforms GraphSAGE across all variants -- attention mechanism overfits this graph
- VAEv2 worse than vanilla VAE -- ELBO score adds noise; reconstruction MSE alone is better for anomaly scoring
- DenoisingAE marginal gain (+0.5 F1) -- F1-optimal threshold helps more than denoising itself
- Supervised >> Unsupervised throughout. Classical detectors (IsoForest, OCSVM) never exceeded F1=0.08
- LogReg: high recall (0.87) but precision=0.18 -- unusable in production without heavy post-filtering

### Explainability

- RF vs GBM top-20 SHAP overlap: **7/20 features**
- Triple overlap RF + GBM + GNN gradient: **3 features -- `lf_53`, `lf_90`, `af_70`**
- These are the most robust signals across all model families

---

## What's Done

- [x] Dataset downloaded, 3 CSVs in `data/`
- [x] `src/data_loader.py`
- [x] `src/preprocessing.py` -- temporal supervised + unsupervised pipelines
- [x] `src/models.py` -- IsoForest, LOF, OCSVM, RF, GBM, LR
- [x] `src/evaluation.py` -- metrics + plots
- [x] `src/autoencoder.py` -- PyTorch AE, BatchNorm, Dropout, GPU
- [x] `src/gnn.py` -- GraphSAGE, SAGEConv, class weighting
- [x] `src/graph_utils.py` -- PyG Data builder
- [x] `scripts/eda.py` through `scripts/tuned_comparison.py`
- [x] SHAP TreeExplainer (RF + GBM) + GNN gradient attribution
- [x] Full leaderboard with 4-panel plot (`reports/final_comparison.png`)
- [x] Optuna tuning 40 trials Ã— 3 models, models saved to `models/`
- [x] Baseline vs tuned comparison (`reports/tuned_f1_comparison.png`)
- [x] `README.md` -- full showcase with results, methodology, setup

## Improvement Rounds

### Round 1 -- scripts 10-11 (new model families)

| Improved | Replaces | Prev F1 | New F1 | Delta |
|----------|----------|:-------:|:------:|:-----:|
| LightGBM | GradientBoosting | 0.766 | **0.817** | +0.051 |
| RF + GraphFeatures | RandomForest | 0.801 | 0.807 | +0.006 |
| GAT | GraphSAGE | 0.541 | 0.359 | -0.182 |
| VAE | Autoencoder | 0.356 | 0.340 | -0.017 |

New src: `src/graph_features.py`, `src/vae.py`, `src/gat.py`

New models: `lightgbm.joblib`, `rf_graph_features.joblib`, `gat.pt`, `vae.pt`

### Round 2 -- script 12 (architecture improvements)

| Improved | Replaces | Prev F1 | New F1 | Delta |
|----------|----------|:-------:|:------:|:-----:|
| GraphSAGEv2 | GraphSAGE | 0.541 | **0.698** | **+0.157** |
| DenoisingAE | Autoencoder | 0.356 | 0.361 | +0.005 |
| GATv2 | GAT | 0.359 | 0.317 | -0.042 |
| VAEv2 | VAE | 0.340 | 0.256 | -0.084 |

GraphSAGEv2 changes: 3 layers, hidden=256, LayerNorm, self-loops, 250 epochs

DenoisingAE changes: latent=8 (was 32), Gaussian noise corruption, F1-optimal threshold on val steps 30-34

GAT/VAE: attention and ELBO both hurt -- these architectures don't suit this graph+task

New models: `graphsagev2.pt`, `denoising_ae.pt`, `gatv2.pt`, `vaev2.pt`

### Optuna Tuning -- script 08 (40 trials each)

| Model | Baseline F1 | Tuned F1 | Delta | Best params |
|-------|:-----------:|:--------:|:-----:|-------------|
| GradientBoosting | 0.766 | **0.824** | +0.058 | n=245, depth=8, lr=0.121, sub=0.918 |
| GraphSAGE | 0.541 | 0.688 | +0.147 | hidden=256, dropout=0.45, lr=0.0038 |
| RandomForest | 0.801 | 0.797 | -0.004 | noise -- already near ceiling |

### Round 3 -- scripts 13-15 (course-derived methods, 2026-06-14)

Three new variants derived from course material (DOMINANT, LSTM-AE, Ensemble). All scripts run, results verified.

**New source files:**
- `src/dominant.py` -- DOMINANT GCN-AE (GCNConv encoder + attr/structure decoders)
- `src/lstm_ae.py` -- LSTM-AE (per-timestep aggregate sequences, sliding windows T=10)
- `src/autoencoder.py` -- added `evt_threshold()` (GPD fit to upper tail, course Module 4 §5)

**New scripts:**
- `scripts/dominant_graph_ae.py` -- variant 31: DOMINANT unsupervised graph anomaly detection
- `scripts/lstm_autoencoder.py` -- variant 32: LSTM-AE temporal anomaly detection
- `scripts/ensemble_soft_vote.py` -- variant 33: soft-vote ensemble GBM + LightGBM + GraphSAGEv2

**New saved models:** `models/dominant.pt`, `models/lstm_ae.pt`

| New Variant | Category | F1 | AUC | AP | vs. Previous Best |
|-------------|----------|----|-----|----|-------------------|
| Ensemble (GBM+LGBM+SAGEv2) | Ensemble | 0.821 | 0.924 | 0.802 | -0.003 vs GBM solo |
| DOMINANT | Graph-AE (unsup) | 0.212 | 0.669 | 0.123 | +0.136 vs DenoisingAE |
| LSTM-AE | Temporal-AE (unsup) | 0.122 | 0.446 | 0.064 | +0.046 vs IsoForest |

**Ensemble detail:**
- Threshold calibrated on held-out val set (steps 30-34), NOT training set
- Calibrating on training set causes threshold collapse (supervised models overfit train -> threshold ~0.999)
- Equal-weight soft vote: (GBM_prob + LGBM_prob + SAGEv2_prob) / 3
- SAGEv2 softmax probs are lower-magnitude than GBM/LGBM -- dilutes ensemble slightly
- Previous ensemble (RF+LGB+GAT, F1=0.742) replaced; new ensemble is +7.9 F1
- Weighted vote (40%/40%/20%) would likely beat GBM solo -- not yet run

**DOMINANT detail:**
- 2-layer GCN encoder (avoids over-smoothing; course confirms 2-3L optimal for sparse graphs)
- Joint loss: alpha=0.5 * MSE(attr) + 0.5 * BCE(structure via negative sampling)
- Scores: alpha * norm(attr_error) + (1-alpha) * norm(struct_error) per node
- EVT threshold collapses (too conservative at tail_q=0.90) -- F1-optimal threshold required
- Best unsupervised method for graph anomaly detection; beats all classical detectors
- AUC=0.669 confirms meaningful signal learned purely from graph structure + features

**LSTM-AE detail:**
- Approach: compute mean feature vector per time step (49 x 165 time series), sliding windows T=10
- Bug fixed: `get_feature_cols()` includes `time_step` -- if scaled, grouping column corrupted; fixed by explicitly excluding `time_step` from `feature_cols` (165 features, not 166)
- 25 training windows (steps 1-34, T=10); 300 epochs; loss converges 0.011 -> 0.006
- AUC=0.446 < 0.5 -- signal is inverted AND weak: per-timestep aggregate scoring cannot discriminate individual illicit vs licit transactions within same time step
- Most anomalous time steps detected: 40, 46, 39, 43, 36 (all in test range) -- these are real temporal anomalies caused by concept drift (illicit rate drop from 11.6% to 6.5%), NOT useful as per-transaction classifier
- Correct use: temporal monitoring dashboard (which time periods are suspicious), not transaction scoring

**EVT threshold notes (applies to all AE-based models):**
- `evt_threshold(scores, tail_quantile=0.90, exceedance_prob=0.065)` in `src/autoencoder.py`
- Works well when scores follow heavy-tailed distribution
- On DOMINANT scores (range 0-0.5, dense near 0): GPD fit assigns very high threshold, recalls almost nothing
- On LSTM-AE scores: same issue -- scores tightly clustered, tail too thin for GPD to be useful
- EVT is most useful when reconstruction errors have genuine heavy tails (e.g., Autoencoder/DenoisingAE with inverted -MSE)

### Round 4 -- scripts 16-19 (Tier 2 course-derived, 2026-06-14)

Four new method families: beta-VAE grid search, structural graph features, spectral embedding, semi-supervised pseudo-labels.

**New scripts:**
- `scripts/pseudo_label_training.py` -- variants 34/34b: GBM + SAGEv2 retrained on pseudo-labeled unlabeled nodes
- `scripts/beta_vae_gridsearch.py` -- variants 35-38b: beta-VAE grid search (beta in {0.01, 1, 2, 4, 8})
- `scripts/structural_graph_features.py` -- variants 39/40: structural topology features (degree, clustering, OddBall, ego-density)
- `scripts/spectral_anomaly_detection.py` -- variants 41a/41b: spectral embedding (top-50 eigenvectors of A_norm) + IsoForest/OCSVM

| New Variant | Category | F1 | AUC | AP | vs. Previous Best |
|-------------|----------|----|-----|----|-------------------|
| GBM+Structural (172 feat) | Supervised | **0.8265** | 0.9243 | 0.8031 | **+0.0024 NEW BEST** |
| GBM+PseudoLabels | Semi-sup | 0.8235 | 0.9240 | 0.8040 | -0.0006 vs GBM solo |
| SAGEv2+PseudoLabels | Semi-sup GNN | 0.7293 | 0.8901 | 0.7517 | +0.0313 vs SAGEv2 |
| beta-VAE (beta=0.01) | Neural-AE | 0.4065 | 0.7832 | 0.3651 | +0.046 vs DenoisingAE |
| beta-VAE (beta=1.0) | Neural-AE | 0.3717 | 0.7760 | 0.3384 | +0.011 vs DenoisingAE |
| beta-VAE (beta=8.0) | Neural-AE | 0.3496 | 0.7625 | 0.2993 | -0.011 vs DenoisingAE |
| beta-VAE (beta=4.0) | Neural-AE | 0.3127 | 0.7535 | 0.2867 | -0.048 vs DenoisingAE |
| beta-VAE (beta=2.0) | Neural-AE | 0.2798 | 0.7455 | 0.2108 | -0.081 vs DenoisingAE |
| IsoForest (structural) | Unsupervised | 0.0941 | 0.4576 | 0.0556 | +0.073 vs IsoForest |
| OCSVM (spectral K=50) | Unsupervised | 0.0613 | 0.3909 | 0.0537 | +0.038 vs OCSVM |
| IsoForest (spectral K=50) | Unsupervised | 0.0562 | 0.4071 | 0.0521 | +0.035 vs IsoForest |

**New saved models:** `models/gbm_pseudo.joblib`, `models/graphsagev2_pseudo.pt`, `models/gbm_structural.joblib`, `models/scaler_structural.joblib`

**beta-VAE detail:**
- All beta values invert signal (illicit reconstructs with LOWER error -- templated behavior same as vanilla AE)
- Lower beta = more reconstruction-dominated = stronger signal on this dataset
- beta=0.01 beats DenoisingAE (0.4065 vs 0.361) -- best unsupervised non-graph method
- Higher beta (>1) hurts: KL over-regularizes latent space, compresses both licit and illicit together
- AUC=0.783 confirms solid rank signal even though threshold calibration limits F1

**Structural features detail:**
- 7 pure topology features: in_degree, out_degree, total_degree, clustering, avg_nb_deg, ego_density, OddBall
- IsoForest on structural only (F1=0.094): topology alone weakly separates illicit -- avg degree ~2.3 too sparse for strong signal
- GBM+structural (F1=0.8265): 7 extra features provide marginal improvement over tabular-only (F1=0.8241)
- Marginal gain (+0.0024) confirms 165 tabular features already capture most topology signal via aggregated neighborhood features

**Spectral embedding detail:**
- top-50 eigenvectors of D^{-1/2}AD^{-1/2} via ARPACK eigsh
- Fiedler value = 1.9999 (near maximum of 2): graph is bipartite-like in spectral structure -- no sparse cut exists
- IsoForest + OCSVM in spectral space both below F1=0.07: spectral coordinates carry no anomaly discriminative signal
- Reason: money-laundering chains are embedded WITHIN the spectral "normal" subspace -- they don't form isolated spectral communities
- Spectral methods useful for community detection; not useful for individual node anomaly scoring on this graph

**Pseudo-label detail:**
- GBM scores 106,371 unlabeled train-step nodes; p>0.90 illicit threshold -> 3,653 pseudo-illicit; p<0.05 -> 97,108 pseudo-licit
- Augmented train: 130,655 rows (illicit rate drops 11.6% -> 5.4% -- dominated by pseudo-licit)
- GBM retrain: F1=0.8235, essentially flat (delta=-0.0006) -- GBM already learned optimal boundary from 29k labeled
- SAGEv2 retrain: F1=0.7293 (+0.031) -- graph model benefits more from pseudo-labels because extra nodes populate graph neighborhoods during message passing
- Pseudo-label quality: GBM high-confidence scores on unlabeled are reliable (mean prob=0.057, p95=0.664 -- most unlabeled are licit, small illicit tail)

### Round 5 -- scripts 20-22 (ensemble tuning, calibration, temporal, 2026-06-14)

Three additional variants: weighted ensemble, probability calibration, temporal rolling features.

**New scripts:**
- `scripts/weighted_ensemble.py` -- variant 42: GBM/LGBM/SAGEv2 with weight sweep {1/3, 40/40/20, 45/45/10, 50/50}
- `scripts/probability_calibration.py` -- variants 43a/43b: Platt scaling + isotonic regression on GBM probs
- `scripts/temporal_rolling_features.py` -- variant 44: rolling mean/std of per-timestep feature aggregates (window=5), merged as 330 extra features

| Variant | Method | F1 | AUC | AP | Delta vs GBM solo |
|---------|--------|----|-----|----|-------------------|
| 42 | Weighted ensemble (best: equal 1/3) | 0.8209 | 0.9238 | 0.8019 | -0.003 |
| 43a | GBM + Platt scaling | 0.8006 | 0.9204 | 0.8000 | -0.024 |
| 43b | GBM + Isotonic regression | 0.7340 | 0.8753 | 0.7682 | -0.090 |
| 44 | GBM + temporal rolling (496 feat) | 0.8179 | 0.9300 | 0.8017 | -0.006 |

**Weighted ensemble detail:**
- Tested 4 weight schemes: equal (0.33/0.33/0.33), 40/40/20, 45/45/10, 50/50/0
- Equal weights won with F1=0.8209; heavier GBM/LGBM weights all worse
- Root cause: SAGEv2 probs have same mean as GBM (0.045 vs 0.051) -- downweighting it removes useful graph signal
- Ensemble still below GBM solo (0.8209 < 0.8241) -- individual threshold per model beats joint soft vote on this dataset

**Calibration detail:**
- Calibration fit on val set (steps 30-34) probabilities, not feature space -- avoids rebuilding model
- Platt scaling (sigmoid LR on raw probs): F1=0.8006, Brier=0.0195 -- marginal improvement in calibration quality but F1 drops (more conservative probabilities shift threshold behavior)
- Isotonic regression: F1=0.7340 -- overfits to val set prob distribution, AUC drops from 0.920 to 0.875 (distorts ordering)
- GBM is already well-calibrated (Brier=0.0196 uncalibrated) -- calibration overhead not justified on this dataset

**Temporal rolling features detail:**
- 330 new features: per-timestep rolling mean + rolling std of 165 features over 5-step window
- AUC improves (0.9204 -> 0.9300) but F1 drops (0.8241 -> 0.8179) -- rolling features add global temporal context but dilute per-transaction discriminative power
- Top rolling features by importance: roll_mean_lf_70, roll_std_lf_16, roll_mean_lf_50 -- all low importance (<0.003)
- 165 tabular features already include aggregated neighborhood statistics -- rolling of aggregates over time adds redundant signal at high feature cost
- Higher AUC confirms temporal context is meaningful; F1 loss is threshold sensitivity artifact from 496->165 feature dilution

### GNN Experiment Series -- script `gnn_experiments.py`

Dedicated standalone script running 10 GNN variants sequentially for report purposes.
Outputs: `reports/gnn_experiments_table.csv`, `gnn_experiments_progression.png`, `gnn_experiments_radar.png`

| Exp | Architecture | Config | F1 | AUC | AP | Purpose |
|-----|-------------|--------|:--:|:---:|:--:|---------|
| 1 | GraphSAGE | 2L h=64 200ep | 0.528 | 0.887 | 0.519 | Minimal baseline |
| 2 | GraphSAGE | 2L h=128 200ep | 0.577 | 0.896 | 0.603 | Wider hidden |
| 3 | GraphSAGE | 2L h=256 200ep | 0.580 | 0.895 | 0.595 | Even wider |
| 4 | GraphSAGE | 2L h=128 300ep | 0.597 | 0.897 | 0.607 | Longer training |
| 5 | GraphSAGE | 3L h=128 200ep | 0.615 | 0.899 | 0.611 | Deeper |
| 6 | GraphSAGEv2 | 3L h=256 LN+SL 250ep | 0.683 | 0.906 | 0.704 | Full architecture |
| 7 | GraphSAGE | Optuna 3L h=256 274ep | **0.690** | 0.901 | **0.706** | Best overall |
| 8 | GAT | 2L 4heads lr=5e-4 200ep | 0.307 | 0.824 | 0.366 | GAT baseline |
| 9 | GAT | 2L 2heads lr=1e-3 200ep | 0.341 | 0.864 | 0.419 | Fewer heads |
| 10 | GATv2 | 3L 2heads residual+LN 300ep | 0.382 | 0.887 | 0.547 | Best GAT |

**What this shows for the report:**
- Hidden dim ablation: 64 -> 128 -> 256 (Exps 1-3)
- Depth ablation: 2L -> 3L (Exps 2 vs 5)
- Training duration effect (Exps 2 vs 4)
- Architectural additions: LayerNorm + self-loops effect (Exps 5 vs 6)
- Hyperparameter tuning effect (Exps 6 vs 7)
- GAT family progression (Exps 8-10)
- GraphSAGE vs GAT family comparison throughout

---

## Report Writing Guide

### Narrative for each section

**Introduction / Problem Statement:**
- Bitcoin blockchain is public but pseudonymous -- transaction graph is fully observable
- Elliptic dataset: 203k transactions, 4.5k labeled illicit, 157k unlabeled
- Severe class imbalance (2% illicit among all, 9.7% among labeled)
- Real concept drift: illicit rate drops 11.6% (train) -> 6.5% (test) over time

**Methodology:**
- Temporal split (NOT random) -- cite Elliptic paper; explain why shuffling is wrong
- Validation slice (steps 30-34) for threshold tuning in neural models
- Primary metric: F1 on illicit class; secondary: PR-AUC
- 4 paradigms: unsupervised, supervised, neural, GNN

**Experiments Section 1 -- Unsupervised Baselines:**
- IsoForest, LOF, OCSVM all below F1=0.08
- Conclusion: illicit transactions do not cluster in feature space -- classical anomaly detection fails
- Key insight: illicit txs follow templated patterns, indistinguishable by density alone

**Experiments Section 2 -- Supervised:**
- LR establishes a linear baseline (F1=0.30)
- RF baseline (F1=0.80) -- `class_weight='balanced'` critical for imbalance
- GBM baseline (F1=0.77), Optuna tuned (F1=0.824) -- best overall
- LightGBM (F1=0.817) -- marginal over tuned GBM, no tuning required
- RF + graph features (F1=0.807) -- degree/PageRank add marginal gain; RF already sees aggregated features

**Experiments Section 3 -- Neural Anomaly Detection:**
- AE trained on licit-only; anomaly = reconstruction error
- Counterintuitive: score INVERTS -- illicit reconstructs with lower MSE (templated behavior)
- DenoisingAE: smaller latent (8 vs 32), noise corruption, F1-optimal threshold -> marginal +0.5
- VAE: KL regularization over-smooths latent space -> worse than AE
- Ceiling for neural unsupervised on this dataset: ~F1=0.36

**Experiments Section 4 -- GNN:**
- Graph structure encodes guilt-by-association; illicit clusters via money laundering chains
- Baseline GraphSAGE (2L, h=128): F1=0.541 -- already far above unsupervised
- Ablation study (see GNN experiment table: Exps 1-7):
  - Hidden dim: 64 -> 128 -> 256 -- consistent improvement
  - Depth: 2L -> 3L -- significant gain (captures 3-hop chains)
  - LayerNorm + self-loops -- essential for stability at 3 layers
  - Optuna tuning: best params found (lr=0.0038, dropout=0.45)
- Best SAGE: GraphSAGEv2 F1=0.698
- GAT family (Exps 8-10): attention mechanism overfits sparse graph -- SAGE mean aggregation outperforms throughout
- Key GNN finding: graph depth matters more than graph attention

**Experiments Section 5 -- Explainability:**
- SHAP TreeExplainer on RF and GBM
- Gradient attribution on GraphSAGE
- Triple overlap (3 features agreed across all model families): `lf_53`, `lf_90`, `af_70`
- These are aggregated neighborhood features -- confirms graph signal is real

**Discussion:**
- Best model: GBM tuned (F1=0.824) -- tree models dominate on tabular features
- Best GNN: GraphSAGEv2 (F1=0.698) -- graph structure alone recovers ~86% of tree performance without feature engineering
- AE/VAE ceiling ~0.36: reconstructive approaches limited by illicit tx regularity
- GAT failure mode: attention needs dense graphs; Bitcoin graph is sparse (avg degree ~2.3)

**Conclusion:**
- Supervised + temporal split = essential
- GNN captures complementary signal (graph structure vs feature distributions)
- Ensemble of GBM + GraphSAGEv2 is logical next step
- Three robust features (`lf_53`, `lf_90`, `af_70`) -- future work: identify what they represent

---

## What's Not Done

- [ ] GAT Optuna tuning -- architecture correct, hyperparams still need sweep
- [ ] GNNExplainer for node-level graph attribution
- [x] Semi-supervised approach (pseudo-labels on unlabeled nodes) -- done in pseudo_label_training
- [x] Inference pipeline -- `src/inference.py`: `load_pipeline()` + `score_transactions(pipeline, df, edges)` -> per-tx prob_illicit + HIGH/MEDIUM/LOW risk label. Note: scaler trained on 180 features (173 merged + 7 structural repeated -- matches structural_graph_features training layout)
- [x] Ensemble of top-3: GBM-tuned + LightGBM + GraphSAGEv2 -- done in ensemble_soft_vote (F1=0.821)
- [x] Weighted ensemble -- weighted_ensemble: tested 4 weight schemes; equal (1/3) still best (F1=0.8209); weighting away from SAGEv2 removes useful graph signal
- [x] Calibrated GBM probabilities -- probability_calibration: Platt F1=0.8006, Isotonic F1=0.734; GBM already well-calibrated (Brier=0.0196), calibration not beneficial
- [x] Temporal rolling features -- temporal_rolling_features: 330 rolling features added; AUC improves 0.920->0.930 but F1 drops 0.824->0.818; 165 tabular already capture aggregate signal
- [ ] DOMINANT with more epochs / alpha tuning -- AUC=0.669 suggests more signal available
- [ ] LSTM-AE per-transaction variant -- current timestep-aggregate approach too coarse; need transaction-level temporal features

---

## Publication Viability Assessment (2026-06-14)

### Verdict: workshops / applied tracks YES -- top venue main track NO

**Why NOT ready for KDD/NeurIPS/ICML/ICLR/WWW main track:**
- No novel algorithm -- every method (GBM, GraphSAGE, DOMINANT, beta-VAE, pseudo-labels, EVT) is published elsewhere
- F1=0.8265 is not SOTA -- recent papers on Elliptic hit 0.85--0.92 (EvolveGCN, ROLAND, Bai et al.)
- Elliptic is a saturated benchmark -- dozens of papers exist; reviewers will ask "what's new?"

**What the project DOES have that is valuable:**
1. **Proper temporal validation** -- most published Elliptic papers use random split (inflated metrics by ~5--10 F1). Our strict temporal split + concept drift analysis (illicit rate 11.6%->6.5%) is methodologically correct and underreported
2. **Comprehensive benchmark** -- 48 experiments across 7 paradigms, 5 systematic rounds -- more exhaustive than any single-method Elliptic paper
3. **Publishable negative results** -- spectral null (Fiedler~2, no community signal), calibration failure (GBM already well-calibrated), EVT threshold collapse on dense scores, LGBM val threshold collapse from train-leakage
4. **Threshold leakage finding** -- tuning threshold on supervised model's own training data collapses threshold to 0.01 and inflates val F1 to 1.0 (false signal). This is a methodological trap that affects many published AML papers and is a clean contribution to write up
5. **EVT thresholding applied to AML** -- GPD fit to reconstruction error tails is underexplored in this domain

### Realistic Target Venues

| Venue | Viability | Framing |
|-------|-----------|---------|
| KDD / ICML / NeurIPS workshops (FinML, AML, Graph Learning) | **Yes** | "Systematic benchmark with temporal validation + negative results" |
| IEEE BigData / ICDM | **Yes** | Applied data science track |
| ECML-PKDD Applied Data Science track | **Yes** | Benchmark + methodological pitfalls |
| ACM SIGKDD Applied Data Science track | **Maybe** | Needs stronger novelty framing |
| MDPI Applied Sciences / Electronics (journal) | **Yes** | Survey + benchmark, fast review cycle |
| Top venues main track | **No** | Requires novel algorithm |

### What to Add for Submission

1. [x] **Statistical significance tests** -- `scripts/statistical_significance.py`: bootstrap F1 95% CI (1000 resamples) + McNemar pairwise test, top-3 models (GBM+Structural/Optuna/PseudoLabels). Result: all three overlap almost entirely (CI half-width ±0.018 vs 0.003 point spread), McNemar b/c < 45/16,670 rows disagreeing, no pair significant (p=0.50/1.00/0.75). Written into report §"Statistical Significance of the Top-3 Ranking" -- reframes the 0.827 "best model" claim as noise, not a real ranking. Outputs: `reports/bootstrap_f1_ci.csv`, `reports/mcnemar_results.csv`.
2. [x] **SOTA comparison table** -- researched real published numbers (Weber 2019: LR 0.481/GCN 0.628/Skip-GCN 0.705/RF 0.788; Alarab 2020 augmented-GCN 0.74; EvolveGCN/GraphSAGE-SSL 0.75-0.77 via Maganti 2026 secondary citation). Key find: Maganti 2026 (arXiv:2604.19514) shows virtually all prior Elliptic GNN results are "transductive" (full graph incl. test-period edges visible to encoder during training, only loss masked) -- under their strict-inductive protocol RF hits 0.821 and beats every GNN. **Our own GraphSAGE/GAT/DOMINANT training has this exact same transductive leakage** (`src/gnn.py:train_epoch` runs forward pass on full graph object, masks only the loss) -- disclosed honestly in new report §"Comparison to Published SOTA and a Graph-Leakage Caveat". Our tabular GBM (0.824-0.827) has zero graph exposure and already beats the strict-inductive RF ceiling. Table+figure: `reports/sota_comparison.png`.
3. [x] **Novel claim to anchor the paper** -- `scripts/threshold_leakage_demo.py`: refit GBM on steps 1-29 only (so 30-34 is genuinely held out, unlike the main pipeline where "val" 30-34 is a subset of the 1-34 fit), compared 3 threshold protocols against real test (35-49). Leaky (tune on train's own predictions): threshold collapses to 0.0050, reports F1=1.0000, real test F1 only 0.5484 (0.45 F1 illusion) -- reproduces the exact "collapses to ~0.01, val F1->1.0" claim from HANDOFF/linkedin drafts that was previously undemonstrated by any script. Bonus finding: even the "correct" held-out-val protocol overstates test F1 (0.9701 tune vs 0.7670 real) because val's illicit rate (16.8%) sits closer to train's (10.9%) than to test's post-drift rate (6.5%) -- and the untuned default t=0.5 actually beats both tuned thresholds on real test F1 (0.7801). Sharper framing than originally planned: not just "don't tune on train," but "report the tune-time-vs-real-test gap for any threshold under concept drift." Report §"A Second Leakage Trap: Threshold Tuning on Training Data". Output: `reports/threshold_leakage_demo.csv`, `reports/threshold_leakage_demo.png`.
4. [x] **Runtime / scalability analysis** -- `scripts/runtime_benchmark.py`, real measured numbers (RTX 5070 Ti): GBM tabular 0.0071ms/tx, GBM+Structural 0.0076ms/tx model-only (0.030ms/tx incl. uncached graph-feature pass), GraphSAGEv2 0.0002ms/tx **but this is batch-amortized over all 203,769 nodes in one forward pass (~45ms total)** -- not a true online single-tx cost, flagged explicitly in report since neighborhood assembly for one new tx isn't isolated by this benchmark. Report §"Inference Runtime and Scalability". Output: `reports/runtime_benchmark.csv`, `reports/runtime_benchmark.png`.
5. [x] **Error analysis** -- `scripts/error_analysis.py` on GBM Optuna test predictions (TP=787 FN=296 FP=40 TN=15547, precision=0.952 recall=0.727). Big finding: FN rate is 0-35% for steps 35-42 then jumps to 92-100% for steps 43-49 -- coincides with the documented dark-market-shutdown event at step 43 (external, verified via web search) that also explains the 11.6%->6.5% concept drift. FN are confidently wrong (mean prob 0.014, 95.6% scored <0.10) not borderline -- threshold recalibration won't fix this, needs concept-drift retraining. Graph degree FN vs TP: significant but practically negligible (median 1.0 both, p=0.0053, large-n artifact). The 3 robust cross-model features (lf_53/lf_90/af_70) separate FN from TP with p<1e-24 -- missed post-shutdown illicit txs numerically mimic the licit population on the model's most-trusted features. Report §"Error Analysis: What GBM Misses". Output: `reports/error_analysis.png`, `reports/error_analysis_summary.csv`, `reports/error_analysis_features.csv`.

### Strongest Paper Angle

Frame as: **"A Fair Benchmark for AML on the Elliptic Dataset: Temporal Validation, Threshold Pitfalls, and 48 Experiments Across 7 Paradigms"**

Core argument: published numbers on Elliptic are not comparable because (a) random split leaks future patterns, (b) threshold tuning on training data inflates val F1 to 1.0, (c) no paper has tested >5 methods under identical conditions. This paper does all three.

---

## Dependencies

```
kagglehub==0.3.6         # pinned -- 1.0.x has kagglesdk import bug
pandas . numpy . scikit-learn . matplotlib . seaborn
shap==0.52.0
optuna==4.9.0
joblib                   # bundled with scikit-learn
torch==2.11.0+cu128      # RTX 5070 Ti, CUDA 12.8, driver 592.01
torch_geometric==2.8.0
pyg_lib . torch_scatter . torch_sparse . torch_cluster . torch_spline_conv
  -> installed from pyg.org/whl/torch-2.7.0+cu128 (version mismatch with 2.11 -- non-fatal)
```

---

## Known Issues

| Issue | Fix / Note |
|-------|-----------|
| `kagglehub>=1.0.0` -- `ImportError: get_web_endpoint` | Pin to `0.3.6` |
| PyG extension warnings on torch 2.11 | Built for 2.7, falls back to pure Python. No functional impact. |
| LOF transductive | No continuous score. Applied directly to test set. |
| AE score direction inverted | `-MSE` used. Auto-detected in script via ROC-AUC comparison. |
| `data.y` on CUDA | Always `.cpu().numpy()` before numpy indexing. |
| `elliptic_txs_features.csv` has no header | Column names assigned in `load_features()`. |
| Windows console unicode | Avoid `->` checkmark symbols etc. in print statements -- cp1252 codec breaks. |

---

## Current Status (as of this session)

venv confirmed working: Python 3.12.10, torch 2.11+cu128, CUDA available, sklearn 1.9, PyG 2.8 (pyg-lib/torch-scatter/torch-sparse fall back to pure-python -- non-fatal, known issue above). All 48 experiments' models/reports already on disk from prior sessions.

This session completed all 5 of 5 "What to Add for Submission" items (see checklist above): statistical significance (bootstrap CI + McNemar), SOTA comparison with a disclosed graph-leakage caveat on our own GNN training, a second leakage demo (threshold tuned on train collapses to t=0.005, F1 1.0->0.548 real), runtime benchmark, and error analysis. All written into `Final-report/elliptic_report.tex` (11 pages, compiles clean) and summarized in `README.md` under "Statistical Rigor & Publication-Readiness". New scripts: `scripts/statistical_significance.py`, `scripts/runtime_benchmark.py`, `scripts/error_analysis.py`, `scripts/threshold_leakage_demo.py`, `scripts/significance_plots.py`.

**Submission checklist is now fully done.** Remaining open work is repo hygiene (below) and whatever the user decides next -- no more analysis gaps flagged in the Publication Viability Assessment.

**Repo hygiene not addressed:** untracked LaTeX build junk (`.aux/.log/.nav/.out/.snm/.toc/.synctex.gz`), duplicate report files (`elliptic_report - Copy.tex`, `- PREPRINT.*`, `fixed-report-sajjad-shahali-s340464.*`), and two large archive files (`Elliptic-Bitcoin-Anomaly-Detection.rar`/`.zip`) at repo root are still sitting untracked/uncommitted -- flagged to the user earlier, not yet cleaned or committed.

