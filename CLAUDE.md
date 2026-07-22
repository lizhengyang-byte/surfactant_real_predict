# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Machine learning pipeline for predicting surfactant (表面活性剂) critical micelle concentration (pCMC = log CMC) from molecular structure. The pipeline extracts **522-dim handcrafted molecular features** (PharmHGT-style, but as flat vectors rather than graphs) and trains 6 regression models:

1. **Tree models** — CatBoost, LightGBM, XGBoost (Optuna hyperparameter tuning)
2. **Deep learning models** — MLP, RNN-LSTM, Transformer Encoder (PyTorch, fixed architectures)

**Target variable:** pCMC (primary), with auxiliary targets AW_ST_CMC, Gamma_max, Area_min, Pi_CMC, pC20 available in data.

## Code Architecture

### Directory Layout

```text
├── smiles_to_features_pharmhgt.py   # Shared 522-dim featurization (the core module)
├── all_smiles_to_features.py        # Quick smoke test for featurization
├── train_catboost_use_pharmhgt_features.py    # CatBoost + Optuna
├── train_lightgbm_use_pharmhgt_features.py    # LightGBM + Optuna
├── train_xgboost_use_pharmhgt_features.py     # XGBoost + Optuna + holdout
├── train_mlp_use_pharmhgt_features.py         # MLP (4×512 GELU)
├── train_rnn_use_pharmhgt_features.py         # RNN-LSTM (3-layer, 64 hidden)
├── train_transformer_use_pharmhgt_features.py # Transformer Encoder (3-layer, 128 d_model)
├── utils.py                            # Run management: timestamped dirs, metrics, index
├── all_smiles_to_features.py           # Feature pre-computation (run once before training)
├── data/
│   └── surfpro/                   # Raw CSV data
│       ├── surfpro_train.csv      # Training set (with fold column, used by all scripts)
│       ├── surfpro_test.csv       # Test set (used by all scripts)
│       ├── surfpro_imputed.csv    # Imputed training set (not currently used)
│       ├── surfpro_literature.csv # Literature compilation with references
│       └── surfpro_bibliography.bib
├── doc/
│   ├── technical_overview_pharmhgt.md           # English technical doc
│   └── smiles_to_features_pharmhgt_技术文档.md   # Chinese technical doc
└── __pycache__/
```

### Data Flow

All 6 training scripts follow the same pattern:

1. **Featurization:** `load_or_compute_features()` from `smiles_to_features_pharmhgt.py` reads SMILES from CSV, computes 522-dim vectors, caches under `data/features/surfpro/` (`.npy` files + `metadata.json`, cached via MD5 hash of SMILES column)
2. **Split:** `train_test_split(0.125)` → validation set
3. **Tuning (tree models):** Optuna with K-Fold CV, TPE sampler, MedianPruner
4. **Final training:** Full data with best params
5. **Output:** Model, plot, config, metrics, and full log saved to a timestamped directory under `runs/{model}_{timestamp}/`

### Run Management

Every training script uses `utils.py` to create a self-contained run directory:

```text
runs/
├── catboost_20260722_143025/
│   ├── config.json         # Hyperparameters + data config (用于复现)
│   ├── train.log           # 完整运行日志（stdout 重定向，含 Optuna 输出）
│   ├── metrics.json        # 评估指标 test_rmse / test_mae / test_r2
│   ├── pred_vs_true.png    # 预测 vs 真值 + 残差图（仅树模型）
│   └── model.pkl           # 模型权重
├── lightgbm_20260722_150112/
│   └── ...
└── _runs_index.csv         # 所有运行摘要，方便横向对比
```

### 522-dim Feature Breakdown

| Module | Dim | Method |
|--------|-----|--------|
| Atom-level aggregation | 220 | 55-dim atom features × 4 stats (mean/std/min/max) |
| Bond-level aggregation | 56 | 14-dim bond features × 4 stats |
| Pharmacophore | 194 | MACCS keys (padded) |
| Reactivity | 34 | BRICS fragment CRC32 bucket histogram |
| Surfactant type | 4 | anionic/cationic/nonionic/zwitterionic one-hot |
| Head/tail ratio | 2 | head-atom fraction, tail-carbon fraction |
| Molecular descriptors | 12 | Normalized RDKit global descriptors |
| **Total** | **522** | |

**Atom features (55-dim):** element one-hot (16), degree one-hot (6), formal charge (1), implicit H one-hot (5), hybridization one-hot (5), aromatic (1), in-ring (1), mass/100 (1), chiral center (1), radical electrons/2 (1), explicit valence one-hot (4), ring size 3-6 one-hot (4), Gasteiger charge bucket (4), ring ≥7 (1), is N/O (1), H-bond donor (1), H-bond acceptor (1), heavy neighbor count/4 (1).

**Bond features (14-dim):** bond type one-hot (4), conjugated (1), in-ring (1), stereo one-hot (6), aromatic (1), in-ring duplicate (1).

**Surfactant domain knowledge (in `smiles_to_features_pharmhgt.py`):** DFS-based tail detection (longest continuous carbon chain ≥4), SMARTS head-group matching, counterion exclusion from SMILES.

### Model-Specific Details

| Model | Tuning | Architecture / Key Params |
|-------|--------|--------------------------|
| CatBoost | Optuna 10 trials × 5-fold CV | depth [4,10], lr [5e-3, 0.3], iterations [500,3000] |
| LightGBM | Optuna 50 trials × 5-fold CV | boosting [gbdt,dart], depth [3,15], num_leaves [15,255] |
| XGBoost | Optuna 200 trials × Top-K holdout | gap penalty if train-val >0.3, learning_rate [1e-3, 0.3] |
| MLP | Fixed (no Optuna) | 4×512 GELU, LayerNorm, AdamW 1e-3, batch 32, early stopping |
| RNN-LSTM | Fixed (no Optuna) | 3-layer LSTM, 64 hidden, dropout 0.3, AdamW 1e-3 |
| Transformer | Fixed (no Optuna) | 3-layer encoder, 128 d_model, 8 heads, batch 16, AdamW 5e-4 |

## Commands

```bash
# Train any model (all follow the same pattern)
python train_catboost_use_pharmhgt_features.py
python train_lightgbm_use_pharmhgt_features.py
python train_xgboost_use_pharmhgt_features.py
python train_mlp_use_pharmhgt_features.py
python train_rnn_use_pharmhgt_features.py
python train_transformer_use_pharmhgt_features.py

# Quick smoke test for featurization
python all_smiles_to_features.py

# Single-molecule inference
python -c "from smiles_to_features_pharmhgt import smiles_to_features_pharmhgt; print(smiles_to_features_pharmhgt('CCO').shape)"
```

## Key Dependencies

- **RDKit** — SMILES parsing, MACCS keys, BRICS decomposition, molecular descriptors
- **PyTorch** — MLP, RNN-LSTM, Transformer Encoder
- **scikit-learn** — train_test_split, KFold, metrics (RMSE/MAE/R²)
- **CatBoost / LightGBM / XGBoost** — gradient boosting
- **Optuna** — hyperparameter optimization (TPE sampler, MedianPruner, n_startup_trials=5)
- **NumPy, Pandas, Matplotlib/Seaborn** — data handling, plotting
- **joblib** — model serialization

## Key Design Decisions

- **Cache-based featurization:** MD5 hash of SMILES detects data changes; `.npy` files + `metadata.json` under `data/features/surfpro/`. Pass `force_recompute=True` to recompute.
- **Tree models use Optuna; deep models use fixed architectures.** CatBoost/LightGBM/XGBoost each have Optuna search with K-Fold CV. MLP/RNN/Transformer use predefined architectures (no tuning).
- **XGBoost gap penalty:** penalizes CV score if train-val gap > 0.3 to discourage overfitting.
- **Deep models treat 522-dim vector as 522 time steps × 1 feature** (RNN and Transformer treat the feature vector as a sequence).
- **No unit tests.** Validation is inline (NaN/Inf checks, shape assertions).
