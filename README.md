# SurfPredict — 表面活性剂多性质预测

基于 **522 维手工分子特征（PharmHGT 风格）** 的机器学习 pipeline，预测表面活性剂的多种界面性质。

> **当前最佳（pCMC）**: MLP + Optuna 调参 | Test R² = **0.9052** | RMSE = 0.3422

---

## 支持的目标性质

| 目标 | 描述 | 训练样本 | 最佳模型 |
|------|------|---------|---------|
| **pCMC** | log CMC（临界胶束浓度） | 1204 | MLP (R²=0.905) |
| **AW_ST_CMC** | 临界胶束浓度时的表面张力 | 843 | — |

> 更多目标（Gamma_max, Area_min, Pi_CMC, pC20）待扩展。

---

## 项目结构

```
├── train/
│   ├── train_pCMC_models/          # pCMC 预测（完整 10 模型）
│   │   ├── smiles_to_features_pharmhgt.py  # 522 维特征提取
│   │   ├── utils.py                        # 运行管理
│   │   ├── shap_utils.py                   # SHAP 分析工具
│   │   ├── train_*.py                      # 10 个训练脚本
│   │   ├── shap_*.py                       # 7 个 SHAP 分析脚本
│   │   └── all_smiles_to_features.py       # 特征预运算
│   └── train_AW_ST_CMCmodels/      # AW_ST_CMC 预测（同上结构）
├── data/
│   └── surfpro/                    # 原始 CSV 数据
│       ├── surfpro_train.csv       # 训练集 (1335 行, 含 fold 列)
│       ├── surfpro_test.csv        # 测试集 (140 行)
│       ├── surfpro_imputed.csv     # 插补数据（未使用）
│       └── surfpro_literature.csv  # 文献数据汇编
├── data/features/surfpro/
│   ├── pCMC/                       # pCMC 特征缓存
│   └── AW_ST_CMC/                  # AW_ST_CMC 特征缓存
├── runs/
│   ├── _runs_index.csv             # 全局索引（所有 target 混合）
│   ├── pCMC/                       # pCMC 训练结果
│   │   ├── _runs_index.csv         # pCMC 专有索引
│   │   └── pCMC_{model}_{timestamp}/
│   │       ├── config.json         # 超参数配置
│   │       ├── train.log           # 完整运行日志
│   │       ├── metrics.json        # 评估指标
│   │       ├── model.pkl           # 模型权重
│   │       ├── pred_vs_true.png    # 预测 vs 真值
│   │       └── shap_*.png          # SHAP 分析图（如有）
│   └── AW_ST_CMC/                  # AW_ST_CMC 训练结果（同上结构）
├── doc/
│   ├── report/                     # 训练报告 + SHAP 对比图
│   ├── smiles_to_features_pharmhgt_技术文档.md
│   └── technical_overview_pharmhgt.md
├── use/                            # 预测 API 包
│   ├── __init__.py
│   └── use_models.py               # SmilesPredictor, SmilesPredict()
├── test/                           # 快速测试脚本
└── README.md
```

---

## 快速开始

### 训练模型

```bash
# pCMC 预测
cd train/train_pCMC_models
python train_mlp_use_pharmhgt_features.py         # 最佳模型 (GPU)
python train_cif_use_pharmhgt_features.py          # 性价比之选 (CPU)
python train_catboost_use_pharmhgt_features.py     # Boosting 最优 (CPU)

# AW_ST_CMC 预测
cd train/train_AW_ST_CMCmodels
python train_lightgbm_use_pharmhgt_features.py
```

### 单分子预测

```bash
# 使用训练好的最佳模型
python -c "
from use.use_models import SmilesPredict
print(SmilesPredict('CCO'))              # 默认 MLP
print(SmilesPredict('CCO', model_name='catboost'))
"

# 特征提取验证
python -c "
from smiles_to_features_pharmhgt import smiles_to_features_pharmhgt
print(smiles_to_features_pharmhgt('CCO').shape)  # (522,)
"
```

---

## 模型排名（pCMC）

| 排名 | 模型 | R² | RMSE | MAE | 算法类别 | 调参 |
|:---:|------|:---:|:----:|:----:|:--------:|:----:|
| 🥇 | **MLP** 🏆 | **0.9052** | **0.3422** | **0.2403** | 深度学习 | Optuna 50 trials |
| 🥇 | CIF | 0.8751 | 0.3928 | 0.2701 | Bagging | Optuna 50 trials |
| 🥇 | CatBoost | 0.8733 | 0.3955 | 0.2669 | Boosting | Optuna 50 trials |
| 4 | XGBoost | 0.8706 | 0.3998 | 0.2695 | Boosting | Optuna 200 trials |
| 5 | NGBoost | 0.8605 | 0.4150 | 0.2853 | 概率 Boosting | Optuna 50 trials |
| 6 | LightGBM | 0.8602 | 0.4155 | 0.2745 | Boosting | Optuna 50 trials |
| 7 | HistGB | 0.8280 | 0.4609 | 0.3094 | Boosting | Optuna 50 trials |
| 8 | RandomForest | 0.8128 | 0.4808 | 0.3355 | Bagging | Optuna 50 trials |
| 9 | RNN-LSTM | 0.7587 | 0.5459 | 0.3619 | 深度学习 | 固定参数 |
| — | Transformer | — | — | — | 深度学习 | 固定参数 |

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
| 目录结构 | `runs/{target}/{target}_{model}_{timestamp}/` |
| 保存内容 | `config.json` + `metrics.json` + `model.pkl` + `train.log` + `pred_vs_true.png` |
| 双写索引 | `runs/{target}/_runs_index.csv`（专有）+ `runs/_runs_index.csv`（全局）|
| 特征缓存 | `data/features/surfpro/{target}/` — MD5 校验自动检测变更 |
| 日志重定向 | stdout 同时输出到终端和 `train.log`（崩溃自动恢复）|

---

## 关键依赖

| 用途 | 库 |
|------|-----|
| 特征工程 | RDKit（SMILES 解析、MACCS、BRICS、描述符） |
| 树模型 | CatBoost, LightGBM, XGBoost, scikit-learn |
| 深度学习 | PyTorch（MLP, RNN-LSTM, Transformer） |
| 超参数优化 | Optuna（TPE 采样器, MedianPruner） |
| 模型解释 | SHAP |
| 数据处理 | NumPy, Pandas, Matplotlib, Seaborn |

---

## 报告入口

| 文档 | 说明 |
|------|------|
| [`summary_pharmhgt_总览报告.md`](doc/report/summary_pharmhgt_总览报告.md) | 全模型总览、排名、推荐（pCMC）|
| 9 份详细训练报告 | 各模型的调参过程、特征重要性、收敛曲线 |
| [`technical_overview_pharmhgt.md`](doc/technical_overview_pharmhgt.md) | 技术原理（英文） |
| [`smiles_to_features_pharmhgt_技术文档.md`](doc/smiles_to_features_pharmhgt_技术文档.md) | 特征工程原理（中文） |
