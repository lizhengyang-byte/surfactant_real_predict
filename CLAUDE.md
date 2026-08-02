# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Surfactant (表面活性剂) interfacial property prediction from molecular structure. The pipeline computes **522-dim handcrafted molecular features** (PharmHGT-style flat vectors) and trains **10 regression models** across **6 target variables**.

**6 Target Properties (with training-set availability):**

| Target      | Description                     | Train samples | Missing rate |
|-------------|---------------------------------|--------------:|-------------:|
| **pCMC**    | log CMC (primary target)        | 1204          | 9.8%         |
| **pC20**    | Surfactant efficiency           | 564           | 57.8%        |
| **AW_ST_CMC** | Surface tension at CMC        | 843           | 36.9%        |
| **Pi_CMC**  | Surface pressure at CMC         | 631           | 52.7%        |
| **Gamma_max** | Max surface excess            | 628           | 53.0%        |
| **Area_min** | Min area per molecule          | 607           | 54.5%        |

**Key correlations:** pCMC↔pC20 (r=0.76), AW_ST_CMC↔Pi_CMC (r=-0.99, nearly linear), Gamma_max↔Area_min (r=-0.62).

**Gamma_max 特殊处理（README 明确约定）：** 数值极小（~10⁻⁶），训练时自动乘以 `y_scale=1e6`（记录于 `config.json`），预测 API `use_models.py` 已封装自动还原，无需调用方干预。

## Commands

> **重要：所有训练/特征/SHAP 脚本位于 `train/train_{target}_models/`**（每个 target 一套完整副本，根目录无脚本）。**从项目根目录调用**：Python 会把脚本所在目录加入 `sys.path`（相对导入 `from utils import ...`、`from smiles_to_features_pharmhgt import ...` 依赖这一点），且 `utils.py` 按项目根相对路径写 `runs/`。`target` ∈ {`pCMC`, `pC20`, `AW_ST_CMC`, `Pi_CMC`, `Gamma_max`, `Area_min`}。

```bash
# Train any model (pCMC example; 10 个模型脚本名与下面一致)
python train/train_pCMC_models/train_catboost_use_pharmhgt_features.py
python train/train_pCMC_models/train_xgboost_use_pharmhgt_features.py
python train/train_pCMC_models/train_mlp_use_pharmhgt_features.py
# ... train_lightgbm / train_histgb / train_ngboost / train_randomforest / train_cif / train_rnn / train_transformer

# Pre-compute features once per target (speeds up all training scripts)
python train/train_pCMC_models/all_smiles_to_features.py

# Single-molecule prediction (auto-selects best model from _runs_index.csv)
python -c "from use.use_models import SmilesPredict; print(SmilesPredict('CCO'))"

# CLI prediction (also --list to show models, -t to pick target)
python use/use_models.py --smiles "CCCCCCCCCCCCCCOS(=O)(=O)[O-].[Na+]" --model best --target pCMC

# List all trained models
python -c "from use.use_models import list_models; print(list_models())"

# SHAP analysis (one script per tree model, inside the target dir)
python train/train_pCMC_models/shap_catboost.py

# Feature dimension check (importable from project root)
python -c "from smiles_to_features_pharmhgt import smiles_to_features_pharmhgt; print(smiles_to_features_pharmhgt('CCO').shape)"
```

**注意：** 训练脚本以 `python train/train_pCMC_models/train_x.py` 从根调用时，CWD 为项目根 → `runs/` 落在根目录。若 `cd` 进 target 目录再运行，`runs/` 会落在该目录下（与现有 runs/ 结构不符），请勿这样做。

## Code Architecture

### Directory Layout

```text
├── train/
│   ├── train_pCMC_models/             # 每个 target 一套完整脚本副本（结构相同）
│   │   ├── smiles_to_features_pharmhgt.py  # 522-dim 特征提取 + 缓存
│   │   ├── utils.py                   # 运行管理：时间戳目录、stdout tee、指标、双写索引（TARGET_NAME 硬编码）
│   │   ├── all_smiles_to_features.py  # 一次性特征预计算
│   │   ├── train_*.py                 # 10 个训练脚本（模式一致）
│   │   ├── shap_*.py + shap_utils.py  # 7 个 SHAP 脚本 + 1 个跨模型对比
│   ├── train_AW_ST_CMCmodels/
│   ├── train_Gamma_max_models/        # 含 y_scale=1e6 缩放（Gamma_max 数值极小）
│   ├── train_Area_min_models/
│   ├── train_Pi_CMC_models/
│   └── train_pC20_models/
├── use/                               # 预测 API 包
│   ├── __init__.py
│   ├── use_models.py                  # SmilesPredictor, SmilesPredict(), list_models()
│   └── use_demo.py                    # 用法示例
├── test/
│   ├── 01.py                          # MLP batch prediction on test CSV
│   └── 02.py                          # CatBoost single SMILES prediction
├── data/
│   └── surfpro/
│       ├── surfpro_train.csv          # Training set (1335 rows, 10 cols incl. fold)
│       ├── surfpro_test.csv           # Test set (140 rows, 9 cols)
│       ├── surfpro_imputed.csv        # Imputed data (not used by training scripts)
│       ├── surfpro_literature.csv     # Literature compilation with references
│       └── surfpro_bibliography.bib
├── doc/
│   ├── technical_overview_pharmhgt.md
│   ├── smiles_to_features_pharmhgt_技术文档.md  # Chinese tech doc
│   ├── feature_reference_522dim.md             # Complete 522-dim feature index
│   └── report/
│       ├── pdf_style_header.html      # PDF 导出样式模板（pandoc -H 注入，见下节）
│       └── {target}/                  # 模型报告（md + pdf，按 target 分组）
│           └── {model}_report.md/.pdf # 每模型一份人工撰写的中文报告
└── runs/
    ├── _runs_index.csv                # 全局索引（所有 target 混合）
    └── {target}/                      # Organized by target variable
        ├── _runs_index.csv            # target 专有索引
        └── {target}_{model}_{timestamp}/
            ├── config.json            # Reproducible hyperparams + data config
            ├── train.log              # Full stdout (incl. Optuna output)
            ├── metrics.json           # test_rmse / test_mae / test_r2
            ├── model.pkl              # Model weights (joblib or torch.save)
            ├── pred_vs_true.png       # Prediction vs truth + residual plot（深度模型无此图）
            └── shap_*.png / shap_values.npy  # SHAP analysis (generated by shap_*.py)
```

> **图片可用性差异：** 7 个树模型（catboost/cif/histgb/lightgbm/ngboost/rf/xgboost）训练后生成 `pred_vs_true.png`，且 `shap_*.py` 会生成完整 SHAP 图组；**深度模型（MLP/RNN/Transformer）不生成任何图片**——相关报告中需以文字说明代替图引用。

### Data Flow (all 10 training scripts follow the same 5-step pattern)

1. **Featurization:** `load_or_compute_features()` reads SMILES from CSV, computes 522-dim vectors, caches as `.npy` files under `data/features/surfpro/`. Cache key = MD5 hash of SMILES column + target_col name.
2. **Split:** `train_test_split(0.125, random_state=42)` → validation set
3. **Tuning (tree models):** Optuna with K-Fold CV, TPE sampler, MedianPruner. Deep models (MLP/RNN/Transformer) use fixed architectures.
4. **Final training:** Train on full training data with best params
5. **Output:** Timestamped run directory with config, metrics, model, plot

### 522-dim Feature Breakdown

| Module | Dim | Method |
|--------|-----|--------|
| Atom-level aggregation | 220 | 55-dim atom features × 4 stats (mean/std/min/max) |
| Bond-level aggregation | 56 | 14-dim bond features × 4 stats |
| Pharmacophore | 194 | MACCS keys (padded to 194) |
| Reactivity | 34 | BRICS fragment CRC32 bucket histogram |
| Surfactant type | 4 | anionic/cationic/nonionic/zwitterionic one-hot |
| Head/tail ratio | 2 | head-atom fraction, tail-carbon fraction |
| Molecular descriptors | 12 | Normalized RDKit global descriptors (MolWt, LogP, TPSA, etc.) |
| **Total** | **522** | |

**Atom features (55-dim):** element one-hot (16), degree one-hot (6), formal charge, implicit H one-hot (5), hybridization one-hot (5), aromatic, in-ring, mass/100, chiral center, radical electrons/2, explicit valence one-hot (4), ring size 3-6 one-hot (4), Gasteiger charge bucket (4), ring ≥7, is N/O, H-bond donor, H-bond acceptor, heavy neighbor count/4.

**Bond features (14-dim):** bond type one-hot (4), conjugated, in-ring, stereo one-hot (6), aromatic, in-ring duplicate.

## Model Details

**Current best model (pCMC): CIF (ExtraTrees)** — Test RMSE=0.3928, R²=0.8751 (Optuna-tuned: n_estimators=1161, max_depth=21, bootstrap=False).
**Runner-up:** NGBoost — Test RMSE=0.4150, R²=0.8605 (probabilistic, predicts mean+std).
> **注意：** 上述以 `runs/_runs_index.csv`（2026-08-02 运行）为准。README.md 中"MLP R²=0.905"为更早实验的历史记录，当前 runs 索引中 MLP 实际为 Test RMSE=0.4241, R²=0.8543。写报告或对比时以 `runs/_runs_index.csv` 为准。

| Model | Tuning | Architecture / Key Params |
|-------|--------|--------------------------|
| CatBoost | Optuna 50 trials × 5-fold CV | depth [4,10], lr [5e-3, 0.3], iterations [500,3000], l2_leaf_reg [1,50] |
| LightGBM | Optuna 50 trials × 5-fold CV | boosting [gbdt,dart], depth [3,15], num_leaves [15,255] |
| XGBoost | Optuna 200 trials × Top-K holdout | gap penalty (train-val >0.3), multivariate TPE, top-5 holdout filter |
| HistGB | Optuna 50 trials × 5-fold CV | sklearn HistGradientBoostingRegressor |
| NGBoost | Pretuned (no Optuna) | n_estimators=637, lr=0.0337, max_depth=5 (probabilistic: predicts mean+var) |
| RandomForest | Optuna 50 trials × 5-fold CV | n_estimators [200,2000], max_depth [3,30] |
| CIF (ExtraTrees) | Optuna 50 trials × 5-fold CV | Same search space as RF |
| MLP | Optuna 50 trials × 5-fold CV (optional) | Linear→BatchNorm→GELU→Dropout × N layers, Linear→1. Params: n_layers[2,6], hidden_dim[128,1024] |
| RNN-LSTM | Fixed (no Optuna) | 3-layer LSTM, 64 hidden, dropout 0.2, lr 1e-3, 800 epochs |
| Transformer | Fixed (no Optuna) | 3-layer encoder, 128 d_model, 4 heads, FFN 256, GELU, CosineAnnealingLR |

**Deep models treat 522-dim vector as sequence:** RNN and Transformer reshape (batch, 522) → (batch, 522, 1), treating each feature dimension as a time step.

## Caching System

- Feature cache: `data/features/surfpro/{X_train, y_train, X_test, y_test}.npy` + `metadata.json`
- Cache key = MD5 of concatenated SMILES strings
- **IMPORTANT:** Cache stores (X, y) pairs for the SPECIFIC target used during caching. Switching to another target requires recomputation (different `dropna(subset=[target_col])` → different valid molecules).
- For AW_ST_CMC specifically: using pCMC's cached X loses 129/843 training samples (worst case).
- Pass `force_recompute=True` to `load_or_compute_features()` to force re-cache.
- Run `all_smiles_to_features.py` once to warm the cache.

## Prediction API (`use/` package)

`from use import SmilesPredictor, SmilesPredict, quick_predict, list_models`

```python
# Auto-load best model for a target (lowest test_rmse from _runs_index.csv)
predictor = SmilesPredictor(model_name='best', target='pCMC')
pred = predictor.predict('CCO')

# Specific model / other target
predictor = SmilesPredictor(model_name='mlp', target='AW_ST_CMC')
pred = predictor.predict('CCO')

# One-liner (defaults to best model, target pCMC)
SmilesPredict('CCO')
SmilesPredict('CCO', target='Gamma_max')            # 多 target
SmilesPredict(['CCO', 'CCC(=O)O'], return_features=True)  # 批量 + 特征

# List all trained models with metrics
list_models(target='pCMC')
```

CLI 等价：`python use/use_models.py --smiles "CCO" --target pCMC --model best`，`--list [target]` 列出模型。

## SHAP Analysis

Each tree model has a corresponding `shap_{model}.py` script that:
1. Loads cached features + latest run model
2. Computes SHAP values on test set
3. Saves: summary beeswarm, top-20 bar, top-5 dependence plots, best/median/worst waterfall, interaction heatmap
4. All outputs go into the model's run directory

Cross-model comparison: `shap_compare.py` generates `doc/report/shap_cross_model_ranking.png` and `doc/report/shap_feature_agreement.png`.

## Report Generation & PDF Export

`doc/report/{target}/{model}_report.md` 是按模型人工撰写的中文报告（每模型一份），基于该模型运行目录的 `train.log` / `config.json` / `metrics.json` 提炼，统一结构：报告信息 → 概述 → 数据与方法 → 超参数与调优 → 训练过程 → 测试结果 → 特征重要性 → 结论与横向排名。

**生成原则：**

- **逐模型读取日志、逐个撰写**——严禁用脚本批量生成报告正文（用户明确要求保证质量）。
- 图引用用 `<img src="../../../runs/{target}/{model}_{timestamp}/xxx.png" width="720">`，相对路径从 `doc/report/{target}/` 到项目根 `runs/` 需要 `../../../`。
- 深度模型（MLP/RNN/Transformer）无图可引，应写文字说明代替失效链接。

**PDF 导出（markdown → HTML → Edge headless 打印）：**

```bash
# 在 doc/report/{target}/ 目录内执行：
pandoc pCMC_catboost_report.md -o _tmp.html --standalone \
       -H ../pdf_style_header.html --metadata title="pCMC CatBoost 模型报告"
"C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe" \
       --headless=new --disable-gpu --no-pdf-header-footer \
       --user-data-dir="%TEMP%/edge_pdf_profile" \
       --print-to-pdf="pCMC_catboost_report.pdf" "file:///<绝对路径>/_tmp.html"
del _tmp.html
```

- 样式模板：`doc/report/pdf_style_header.html`（微软雅黑中文字体、A4 页边距、表格样式）。
- **坑：** 不能 pandoc 直转 LaTeX（`--pdf-engine=xelatex`），`<img>` 标签会被丢弃导致 PDF 无图；必须经 HTML 中间文件，且相对图片路径是相对 HTML 所在目录解析的，故须在报告子目录内生成 HTML。

## Key Dependencies

- **RDKit** — SMILES parsing, MACCS keys, BRICS decomposition, molecular descriptors
- **PyTorch** — MLP, RNN-LSTM, Transformer Encoder
- **scikit-learn** — train_test_split, KFold, metrics (RMSE/MAE/R²)
- **CatBoost / LightGBM / XGBoost** — gradient boosting
- **Optuna** — hyperparameter optimization (TPE sampler, MedianPruner, n_startup_trials=5)
- **SHAP** — model interpretability (TreeExplainer for tree models, KernelExplainer for NGBoost)
- **NumPy, Pandas, Matplotlib, Seaborn** — data handling, plotting
- **joblib** — sklearn/CatBoost/LightGBM/XGBoost model serialization

## Key Design Decisions

- **All models share the same 522-dim feature vector.** Featurization is decoupled from training in `smiles_to_features_pharmhgt.py`.
- **Target variable is HARDCODED** in each training script (`TARGET_COL = 'pCMC'`). To change target, edit the script or add a cmdline arg.
- **Tree models use Optuna; deep models use fixed architectures** (except MLP, which has optional Optuna).
- **Run management is self-contained:** every run creates `runs/{target}/{model}_{timestamp}/` with config, log, metrics, model — no external experiment tracking.
- **No unit tests.** Validation is inline (NaN/Inf checks, shape assertions).
- **Feature cache is target-specific:** cached X count depends on target's non-null count. Changing target invalidates the cache.
- **XGBoost has the most sophisticated tuning:** 200 trials + Top-K holdout selection + gap penalty against overfitting.
- **Prediction API auto-selects best model** from `_runs_index.csv` by lowest test_rmse.
