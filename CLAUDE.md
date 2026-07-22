# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Machine learning pipeline for predicting surfactant (表面活性剂) critical micelle concentration (pCMC = log CMC) from molecular structure. Two complementary approaches:

1. **PharmHGT GNN** (`pharmhgt/` package) — Heterogeneous Graph Transformer on molecular graphs (atom/bond/pharmacophore views)
2. **522-dim Handcrafted Features** (root scripts) — Feature engineering + 6 regression models (CatBoost, LightGBM, XGBoost, MLP, RNN, Transformer)

## Code Architecture

### PharmHGT GNN Pipeline

```text
pharmhgt/
├── config.py        # PharmHGTConfig dataclass (all hyperparameters)
├── data.py          # SurfactantGraphDataset: SMILES → PyG HeteroData
│                    #   3 node/edge views:
│                    #     - ('atom','bond','atom')  covalent bonds
│                    #     - ('pharm','react','pharm') BRICS reaction edges
│                    #     - ('atom','junc','pharm') fragment membership
│                    #   42-dim atom / 14-dim bond / 194-dim pharm / 34-dim reac features
├── layers.py        # Building blocks: AttentionConv, HeteroGNNLayer, MVMP, GraphGRU
├── model.py         # PharmHGT: proj → MVMP (×depth) → GraphGRU → MLP head
├── train.py         # Training loop with NoamLR, gradient clipping, early stopping
│                    #   Usage: python -m pharmhgt.train [--target pCMC] [--epochs 300]
├── test_example.py  # Optuna hyperparameter search (30 trials) on example_data.csv
└── training_report.md  # Best results: hid_dim=384, depth=5, heads=8 → RMSE=0.508
```

**Data flow**: `SMILES → smiles_to_heterograph() → HeteroData → SurfactantGraphDataset → DataLoader(collate_graphs) → PharmHGT.forward()`

- Atom features: 42-dim one-hot (element, degree, charge, chirality, H count, hybridization, aromatic, mass)
- Bond features: 14-dim (type, conjugation, ring, stereo)
- Pharmacophore features: 194-dim (MACCS + RDKit BaseFeatures)
- Reaction features: 34-dim (BRICS bond rule encoding)
- Junction edges connect each atom to its BRICS fragment node
- Edge index and edge attr are created with reverse edges (bidirectional)
- Gasteiger charges computed during graph construction

### 522-dim Handcrafted Feature Pipeline

```text
Root scripts (standalone):
├── smiles_to_features_pharmhgt.py   # Shared featurization module
│   # Produces 522-dim vector per SMILES (cached under data/features/pharmhgt/)
│   # 220 atom agg + 56 bond agg + 194 MACCS + 34 BRICS + 6 surfactant + 12 descriptors
│
├── train_catboost_use_pharmhgt_features.py    # Optuna 10 trials, 5-fold CV
├── train_lightgbm_use_pharmhgt_features.py    # Optuna 50 trials, 5-fold CV
├── train_xgboost_use_pharmhgt_features.py     # Optuna 200 trials, Top-K holdout
├── train_mlp_use_pharmhgt_features.py         # Fixed params: 4×512 GELU, AdamW
├── train_rnn_use_pharmhgt_features.py         # Fixed params: 3-layer LSTM, 64 hidden
└── train_transformer_use_pharmhgt_features.py # Fixed params: 3-layer encoder, 128 d_model
```

All 6 training scripts follow the same pattern:

1. `load_or_compute_features()` → X/y for train + test
2. `train_test_split(0.125)` → val set
3. [Tree models] Optuna with K-Fold CV
4. Train on full data with best params, evaluate on test
5. Save model to `models/predictor/weights/`, plot to `reports/`

### ophth_pharmhgt_official/ (paper reference implementation)

A git submodule containing the original PharmHGT paper code (DGL-based, not PyG). The `pharmhgt/` package in this repo is an independent PyTorch Geometric reimplementation. Not actively modified.

## Key Dependencies

- **RDKit** — SMILES parsing, fingerprints, descriptors, BRICS decomposition
- **PyTorch + PyTorch Geometric** — GNN (PharmHGT), MLP, RNN, Transformer
- **scikit-learn** — data splits, metrics (RMSE/MAE/R²), preprocessing
- **CatBoost / LightGBM / XGBoost** — gradient boosting models
- **Optuna** — hyperparameter optimization (TPE sampler, MedianPruner)

## Commands

```bash
# Train GNN (PharmHGT, PyTorch Geometric)
python -m pharmhgt.train                              # default config
python -m pharmhgt.train --target pCMC --epochs 500   # custom
python -m pharmhgt.train --lr 1e-3 --hid_dim 512 --depth 6

# Train GNN (PharmHGT official, DGL) with pCMC data
python pharmhgt_official/train_pcmc.py                      # default (20 Optuna trials)
python pharmhgt_official/train_pcmc.py --epochs 80 --trials 30
python pharmhgt_official/train_pcmc.py --lr 5e-4 --hid_dim 384 --trials 0

# Optuna tuning for GNN with example data
python pharmhgt/test_example.py

# Train handcrafted-feature models
python train_catboost_use_pharmhgt_features.py
python train_lightgbm_use_pharmhgt_features.py
python train_xgboost_use_pharmhgt_features.py
python train_mlp_use_pharmhgt_features.py
python train_rnn_use_pharmhgt_features.py
python train_transformer_use_pharmhgt_features.py

# Quick smoke test for featurization
python -c "from smiles_to_features_pharmhgt import smiles_to_features_pharmhgt; print(smiles_to_features_pharmhgt('CCO').shape)"
```

**Note**: No unit tests in this repo. Validation is inline in notebooks/scripts (NaN/Inf checks, shape assertions). `data/` directory is gitignored (contains generated features).

## Key Design Decisions

- **PharmHGT config-driven**: `PharmHGTConfig` dataclass in `pharmhgt/config.py` — override any parameter via argparse
- **Noam LR scheduler**: `warmup_steps * steps_per_epoch` warmup, then inverse sqrt decay (in `pharmhgt/train.py`)
- **PharmHGT official (DGL)**: Legacy paper reference implementation in `pharmhgt_official/`. The streamlined training script `pharmhgt_official/train_pcmc.py` wraps it for pCMC-only prediction (no wandb, no JSON config, internal data split, Optuna support)
- **Optuna pruning**: `MedianPruner` with `n_startup_trials=5` across all tree-model training; GNN uses same pattern in `test_example.py`
- **XGBoost gap penalty**: Penalizes CV score if train-val gap > 0.3 to discourage overfitting
- **Cache-based featurization**: MD5 hash of SMILES detects data changes; `.npy` files + `metadata.json`
- **Surfactant domain knowledge**: DFS-based tail detection, SMARTS head-group matching, counterion exclusion — only used in the 522-dim pipeline, not in the GNN

## Data Sources

- Training: `data/surfpro_imputed.csv` (imputed training set)
- Test: `data/surfpro_test.csv`
- GNN example: `pharmhgt_official/example_data.csv`
- Generated features cached under `data/features/` (gitignored)
- Raw bibliographic data in `data/basic/surfpro_bibliography.bib`
