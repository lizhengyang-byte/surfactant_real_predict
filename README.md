# SurfPredict — 表面活性剂 pCMC 预测

基于 **522 维手工分子特征（PharmHGT 风格）** 的机器学习 pipeline，预测表面活性剂的 **pCMC**（临界胶束浓度对数值）。

> **当前最佳**: MLP + Optuna 调参 | Test R² = **0.9052** | RMSE = 0.3422

---

## 项目结构

```
├── doc/
│   ├── report/
│   │   ├── summary_pharmhgt_总览报告.md    # 全模型总览（从这里开始）
│   │   ├── mlp_pharmhgt_训练报告.md         # MLP 详细报告
│   │   ├── cif_pharmhgt_训练报告.md         # CIF 详细报告
│   │   ├── catboost_pharmhgt_训练报告.md    # CatBoost 详细报告
│   │   ├── xgboost_pharmhgt_训练报告.md     # XGBoost 详细报告
│   │   ├── ngboost_pharmhgt_训练报告.md     # NGBoost 详细报告
│   │   ├── lightgbm_pharmhgt_训练报告.md    # LightGBM 详细报告
│   │   ├── histgb_pharmhgt_训练报告.md      # HistGB 详细报告
│   │   ├── randomforest_pharmhgt_训练报告.md # RandomForest 详细报告
│   │   └── rnn_pharmhgt_训练报告.md         # RNN-LSTM 详细报告
│   ├── smiles_to_features_pharmhgt_技术文档.md
│   └── technical_overview_pharmhgt.md
├── data/
│   ├── surfpro/          # 原始 CSV 数据 (train/test/literature)
│   └── features/         # 预计算特征缓存 (.npy + metadata.json)
├── runs/
│   ├── _runs_index.csv   # 所有运行摘要，横向对比
│   ├── mlp_20260723_173509/   # 最佳模型 (R²=0.9052)
│   ├── cif_20260723_152733/   # 第二名 (R²=0.8751)
│   ├── catboost_20260722_181638/
│   ├── xgboost_20260722_193605/
│   ├── ngboost_20260723_150347/
│   ├── lightgbm_20260722_180432/
│   ├── histgb_20260723_103854/
│   ├── rf_20260723_151026/
│   └── rnn_20260723_092955/
├── smiles_to_features_pharmhgt.py   # 522 维特征提取（核心模块）
├── train_mlp_use_pharmhgt_features.py
├── train_cif_use_pharmhgt_features.py
├── train_catboost_use_pharmhgt_features.py
├── train_xgboost_use_pharmhgt_features.py
├── train_ngboost_use_pharmhgt_features.py
├── train_lightgbm_use_pharmhgt_features.py
├── train_histgb_use_pharmhgt_features.py
├── train_randomforest_use_pharmhgt_features.py
├── train_rnn_use_pharmhgt_features.py
├── train_transformer_use_pharmhgt_features.py
└── utils.py         # 运行管理：时间戳目录、指标、索引
```

---

## 模型排名

| 排名 | 模型 | R² | RMSE | MAE | 算法类别 | 调参 |
|:---:|------|:---:|:----:|:----:|:--------:|:----:|
| 🥇 | **MLP** 🏆 | **0.9052** | **0.3422** | **0.2403** | 深度学习 | Optuna 50 trials |
| 🥇 | CIF | 0.8751 | 0.3928 | 0.2701 | Bagging\* | Optuna 50 trials |
| 🥇 | CatBoost | 0.8733 | 0.3955 | 0.2669 | Boosting | Optuna 50 trials |
| 4 | XGBoost | 0.8706 | 0.3998 | 0.2695 | Boosting | Optuna 200 trials |
| 5 | NGBoost | 0.8605 | 0.4150 | 0.2853 | 概率 Boosting | Optuna 50 trials |
| 6 | LightGBM | 0.8602 | 0.4155 | 0.2745 | Boosting | Optuna 50 trials |
| 7 | HistGB | 0.8280 | 0.4609 | 0.3094 | Boosting | Optuna 50 trials |
| 8 | RandomForest | 0.8128 | 0.4808 | 0.3355 | Bagging | Optuna 50 trials |
| 9 | RNN-LSTM | 0.7587 | 0.5459 | 0.3619 | 深度学习 | 固定参数 |

> \* CIF 是条件推断森林，基于统计显著性分裂，与标准 RandomForest 有本质不同。

---

## 快速开始

### 单分子预测

```bash
python -c "
from smiles_to_features_pharmhgt import smiles_to_features_pharmhgt
print(smiles_to_features_pharmhgt('CCO').shape)  # (522,)
"
```

### 训练任意模型

```bash
# 所有脚本使用相同的接口
python train_mlp_use_pharmhgt_features.py         # 最佳模型 (GPU)
python train_cif_use_pharmhgt_features.py          # 性价比之选 (CPU)
python train_catboost_use_pharmhgt_features.py     # Boosting 最优 (CPU)
```

每个脚本自动：
1. 从 SMILES 计算 522 维特征（缓存支持）
2. Optuna 超参数优化（50 trials × 5-fold CV）
3. 保存模型/指标/日志到时间戳目录 `runs/{model}_{timestamp}/`

---

## 特征工程

### 522 维特征组成

| 模块 | 维度 | 方法 |
|------|------|------|
| 原子聚合 | 220 | 55 维原子特征 × 4 统计量 (mean/std/min/max) |
| 键聚合 | 56 | 14 维键特征 × 4 统计量 |
| MACCS 指纹 | 194 | 药效团特征（填充） |
| BRICS 碎片 | 34 | 反应性碎片 CRC32 分桶 |
| 表面活性剂类型 | 4 | 阴/阳/非/两性 one-hot |
| 头基/尾链比例 | 2 | head_ratio, tail_ratio |
| 分子描述符 | 12 | RDKit 全局描述符（归一化） |
| **总计** | **522** | |

### 领域知识

- **DFS 尾链检测**: 最长连续碳链 ≥4 → tail_ratio
- **SMARTS 头基匹配**: 阴/阳/非/两性分类
- **反离子排除**: 从 SMILES 排除配对的 counterion

详情见 [`doc/smiles_to_features_pharmhgt_技术文档.md`](doc/smiles_to_features_pharmhgt_技术文档.md)

---

## 运行管理

| 功能 | 说明 |
|------|------|
| 目录结构 | 每次运行创建 `runs/{model}_{timestamp}/` |
| 保存内容 | `config.json` + `metrics.json` + `model.pkl` + `train.log` |
| 全局索引 | `runs/_runs_index.csv` — 所有运行指标横向对比 |
| 特征缓存 | `data/features/surfpro/` — MD5 校验自动检测数据变更 |

---

## 关键依赖

| 用途 | 库 |
|------|-----|
| 特征工程 | RDKit（SMILES 解析、MACCS、BRICS、描述符） |
| 树模型 | CatBoost, LightGBM, XGBoost, scikit-learn |
| 深度学习 | PyTorch（MLP, RNN-LSTM） |
| 超参数优化 | Optuna（TPE 采样器, MedianPruner） |
| 数据处理 | NumPy, Pandas |

---

## 报告入口

| 文档 | 说明 |
|------|------|
| [`summary_pharmhgt_总览报告.md`](doc/report/summary_pharmhgt_总览报告.md) | 全模型总览、排名、推荐 |
| 9 份详细训练报告 | 各模型的调参过程、特征重要性、收敛曲线 |
| [`technical_overview_pharmhgt.md`](doc/technical_overview_pharmhgt.md) | 技术原理（英文） |
| [`smiles_to_features_pharmhgt_技术文档.md`](doc/smiles_to_features_pharmhgt_技术文档.md) | 特征工程原理（中文） |
