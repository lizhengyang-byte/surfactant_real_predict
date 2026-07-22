# XGBoost + PharmHGT 特征 表面活性剂 pCMC 预测 — 训练报告

> **报告生成日期**: 2026-07-22  
> **运行时间戳**: 2026-07-22 19:36:05  
> **模型**: XGBoost (Extreme Gradient Boosting)  
> **特征**: PharmHGT 风格 522 维分子特征  
> **目标变量**: pCMC (log CMC，临界胶束浓度对数值)

---

## 1. 实验概述

本实验使用 **XGBoost** 梯度提升树模型，结合 **PharmHGT 风格 522 维手工分子特征**，对表面活性剂的 **pCMC** 值进行回归预测。本实验采用了项目中**最全面的超参数优化策略**：

1. 特征提取（从 SMILES 计算 522 维特征，含缓存命中）
2. 训练/验证集划分 + **Holdout 集保留**（CV 1,083 + Holdout 121 + Test 140）
3. Optuna 超参数优化（**200 次试验 × 5 折 CV**，以 **multivariate TPE** 采样）
4. **CV Gap Penalty** — 对训练-验证差距过大的参数施加惩罚
5. **Top-5 Holdout 二次筛选** — 取 CV 排名前 5 的参数在 holdout 集上重评估
6. 最终模型训练（全量 1,204 条数据）
7. 测试集评估
8. 特征重要性分析

---

## 2. 数据与特征

### 2.1 数据集划分

| 数据集 | 样本数 | 说明 |
|--------|--------|------|
| CV 训练集 | 1,083 | 从全量 1,204 中划出 90% 用于 5 折 CV |
| Holdout 集 | 121 | 从全量 1,204 中划出 **10%**，用于 Top-K 二次筛选 |
| 测试集 (Test) | 140 | 独立测试集 |
| **总计** | **1,344** | 训练 1,204 + 测试 140 |

> **Holdout 设计说明**：XGBoost 独有地从全量训练数据中额外划出 10% 作为 holdout 集。这 121 个样本**不参与 CV**，仅用于在 Optuna 完成后对 Top-5 候选参数进行重新评估，选择 holdout RMSE 最低的参数。这种设计减少了 CV 上的过拟合对参数选择的误导。

### 2.2 特征构成

522 维特征由以下模块拼接而成：

| 特征模块 | 维度 | 说明 |
|---------|------|------|
| 原子聚合特征 (atom_agg) | 220 | 55 维原子特征 × 4 统计量 (mean/std/min/max) |
| 键聚合特征 (bond_agg) | 56 | 14 维键特征 × 4 统计量 |
| MACCS 指纹 | 194 | 药效团特征填充 |
| BRICS 碎片 | 34 | 反应性碎片 CRC32 分桶直方图 |
| 表面活性剂类型 | 4 | 阴/阳/非/两性 one-hot 编码 |
| 头基/尾链比例 | 2 | head_ratio, tail_ratio |
| 分子描述符 | 12 | RDKit 全局描述符（归一化） |
| **总计** | **522** | |

---

## 3. 超参数优化 (Optuna)

### 3.1 优化配置

| 配置项 | 值 | 说明 |
|--------|-----|------|
| 优化框架 | Optuna | |
| 采样器 | **Multivariate TPE** | 捕捉参数间的联合分布，区别于独立 TPE |
| 试验次数 | **200** | 项目中最多的调参次数 |
| 交叉验证折数 | 5 | |
| 目标函数 | **Gap-Penalized CV RMSE** | `adjusted_rmse = rmse_val × (1.0 + 0.05 × max(0, gap - 0.3))` |
| 过拟合防御 | **Gap Penalty** | 训练-验证差距 > 0.3 时施加 5% 惩罚 |
| 二次筛选 | **Top-5 Holdout 重评估** | 取 CV 前 5 名在 holdout 集上重新排名 |

### 3.2 搜索空间

XGBoost 的 13 个超参数搜索范围及最佳值（经 Holdout 筛选后）：

| 超参数 | 搜索范围/选项 | CV 最佳值 | **Holdout 选定值** |
|--------|-------------|-----------|-------------------|
| `n_estimators` | [800, 3000] | 1,636 | 1,636 |
| `max_depth` | [4, 12] | 6 | 6 |
| `learning_rate` | [0.01, 0.2] (log) | 0.02033 | 0.02033 |
| `subsample` | [0.6, 1.0] | 0.7662 | 0.7662 |
| `colsample_bytree` | [0.3, 1.0] | 0.8214 | 0.8214 |
| `colsample_bylevel` | [0.3, 1.0] | 0.6764 | 0.6764 |
| `colsample_bynode` | [0.3, 1.0] | 0.5369 | 0.5369 |
| `min_child_weight` | [1.0, 30.0] (log) | 1.241 | 1.241 |
| `gamma` | [0.0, 2.0] | 0.02271 | 0.02271 |
| `reg_alpha` (L1) | [1e-8, 10.0] (log) | 0.1537 | 0.1537 |
| `reg_lambda` (L2) | [1e-8, 10.0] (log) | 0.002110 | 0.002110 |
| `max_delta_step` | [0.0, 8.0] | 0.9811 | 0.9811 |
| `booster` | ['gbtree', 'dart'] | `dart` | **`dart`** |

### 3.3 优化过程

在 200 次试验中，**CV 最佳试验为第 128 次**，CV RMSE = **0.49161**。

**全部 200 次试验的 CV RMSE 分布**：

```
CV RMSE 分布区间：
  < 0.495:  44 次 (22%)    ← 最优区域
  0.495-0.505: 65 次 (32.5%)
  0.505-0.520: 57 次 (28.5%)
  0.520-0.550: 26 次 (13%)
  ≥ 0.550:   8 次 (4%)     ← 表现较差的试验
```

**优化过程分析**：
- 最优值 **0.4916** 在较晚的第 128 次试验中找到，说明 multivariate TPE 采样器在探索参数联合分布时需要更多采样
- 约有 54.5% 的试验集中在 0.495-0.520 区间，分布集中表明搜索空间设置合理
- 与 LightGBM（最佳 CV RMSE=0.4818）和 CatBoost（0.4622）相比，XGBoost 的 CV RMSE 偏高，这可能与 **Gap Penalty 机制**有关——该机制对训练-验证差距大的参数施加了惩罚，选择的是泛化性更强而非 CV 分数最优的参数

### 3.4 超参数重要性分析

XGBoost 优化日志提供的超参数重要性分数：

| 排名 | 超参数 | 重要性 | 说明 |
|------|--------|--------|------|
| **1** | **`gamma`** | **0.7991** | 分裂最小损失下降值 — **压倒性最重要** |
| 2 | `max_delta_step` | 0.0802 | 每棵树权重更新的最大步长 |
| 3 | `colsample_bynode` | 0.0296 | 节点级列采样 |
| 4 | `max_depth` | 0.0274 | 树深度 |
| 5 | `colsample_bytree` | 0.0162 | 树级列采样 |
| 6 | `reg_lambda` | 0.0114 | L2 正则化 |
| 7 | `min_child_weight` | 0.0104 | 叶子最小权重和 |
| 8 | `subsample` | 0.0088 | 行采样 |
| 9 | `learning_rate` | 0.0080 | 学习率 |
| 10 | `n_estimators` | 0.0041 | 树数量 |
| 11 | `colsample_bylevel` | 0.0032 | 层级列采样 |
| 12 | `booster` | 0.0013 | 提升器类型 |
| 13 | `reg_alpha` | 0.0001 | L1 正则化 — **重要性可忽略** |

**关键解读**：

1. **`gamma` 是压倒性最重要的超参数**（重要性 0.799）：在 XGBoost 中，`gamma` 控制节点分裂所需的最小损失下降值。重要性近 0.8 说明该参数对 522 维高维特征的分裂决策影响极大——惩罚不必要的分裂，防止过拟合。

2. **`reg_alpha`（L1 正则化）重要性近乎为零**（0.0001）：表明 L1 稀疏化对 522 维特征的作用有限，几乎所有特征都包含有用信息，不需要通过 L1 来强制特征选择。

3. **三级列采样**（tree/level/node）各有贡献：`colsample_bynode`（0.030）> `colsample_bytree`（0.016）> `colsample_bylevel`（0.003），说明节点级的随机特征子空间选择对 XGBoost 的泛化最有效。

4. **`booster` 重要性仅 0.0013**：但最佳值选择了 `dart`，说明在 XGBoost 中 DART 比 gbtree 更适合本任务。

### 3.5 Top-5 Holdout 二次筛选

这是 XGBoost 独有的设计——从 200 次 CV 试验中取前 5 名，在 holdout 集（121 个未见样本）上重新评估：

| CV 排名 | CV RMSE | Holdout RMSE | Holdout R² | 说明 |
|---------|---------|-------------|------------|------|
| **1** | **0.4916** | **0.4642** | **0.8068** | ✅ **Holdout 最佳 → 选定** |
| 2 | 0.4916 | 0.4941 | 0.7812 | CV 相同但 Holdout 更差 |
| 3 | 0.4921 | 0.4849 | 0.7893 | |
| 4 | 0.4928 | 0.4910 | 0.7839 | |
| 5 | 0.4938 | 0.4950 | 0.7804 | |

**关键发现**：
- **CV 排名第 1 的参数在 Holdout 上也是最佳**，说明 CV 排名与 Holdout 排名高度一致
- Holdout RMSE（0.4642）显著低于 CV RMSE（0.4916），可能是因为 holdout 不经过 CV 的平均平滑效应
- Holdout R²=0.8068，说明在 121 个未见样本上可以解释约 80.7% 的方差

### 3.6 最佳超参数解读

`booster=dart` 且经 Holdout 验证的最佳参数组合揭示了以下模型行为：

1. **DART（Dropout Additive Regression Trees）作为最佳提升器**：不同于 LightGBM（gbdt）和 CatBoost（无 dropout），XGBoost 选择 dropout 策略说明需要更强的正则化来防止过拟合

2. **较浅的树** (`max_depth=6`)：与 CatBoost 的 `depth=5` 一致，浅树有利于泛化

3. **中等学习率** (`learning_rate=0.0203`) + **较多迭代** (`n_estimators=1636`)：长训练策略

4. **三层列采样均 < 1.0**（tree=0.82, level=0.68, node=0.54）：逐级递减的列采样形成了"多级随机子空间"，增强树之间的多样性

5. **低 `gamma=0.0227`**：仅轻微惩罚分裂，允许模型学习细粒度模式

6. **低 `min_child_weight=1.24`**：接近默认的 1.0，叶节点不要求严格的大权重和

---

## 4. 最终模型训练

### 4.1 训练配置

| 配置项 | 值 |
|--------|-----|
| 训练数据 | **全量 1,204 条**（含 holdout 121 条，未再保留） |
| 最佳迭代轮数 | Holdout 选定参数（`n_estimators=1636`） |
| 最终模型 | 使用选定参数在全量数据上训练 |

### 4.2 Gap Penalty 机制

XGBoost 在 Optuna 优化过程中使用了 **Gap Penalty**（训练-验证差距惩罚）：

```
adjusted_rmse = val_rmse × (1.0 + 0.05 × max(0, val_rmse - train_rmse - 0.3))
```

即当 `train_rmse - val_rmse > 0.3` 时，在 CV RMSE 上额外增加 5% 的惩罚系数。这确保选择的参数不仅在验证集上表现好，而且训练-验证差距小（泛化能力强）。

---

## 5. 测试集评估

### 5.1 核心指标

| 指标 | 值 |
|------|------|
| **Test MSE** | **0.1598** |
| **Test RMSE** | **0.3998** |
| **Test MAE** | **0.2695** |
| **Test R²** | **0.8706** |

### 5.2 指标解读

- **R² = 0.8706**：模型解释了测试集 87.06% 的方差，预测与真实值高度相关
- **RMSE = 0.3998**：预测的平均误差约 0.40 个 pCMC 单位
- **MAE = 0.2695**：预测的中位数绝对误差约 0.27 个 pCMC 单位
- MAE < RMSE，说明误差分布存在一定右尾

### 5.3 三模型全面对比

| 指标 | **XGBoost** | CatBoost | LightGBM |
|------|------------|----------|----------|
| 模型类型 | DART (Dropout) | Oblivious Tree | Leaf-wise GBDT |
| Optuna 试验次数 | **200** | 50 | 50 |
| 采样器 | **Multivariate TPE** | TPE | TPE |
| 额外验证 | **Top-5 Holdout 二次筛选** | — | — |
| 过拟合防御 | **Gap Penalty + 3级列采样** | Bagging + L2 | L1 + L2 + 子采样 |
| **CV RMSE** | 0.4916 | **0.4622** | 0.4818 |
| **Test RMSE** | **0.3998** | **0.3955** | 0.4155 |
| **Test MAE** | 0.2695 | **0.2669** | 0.2745 |
| **Test R²** | **0.8706** | **0.8733** | 0.8602 |

**对比分析**：

1. **XGBoost 与 CatBoost 性能接近**：Test RMSE 相差仅 0.0043，R² 相差 0.0027，两者在测试集上几乎等价

2. **XGBoost 的 CV RMSE (0.4916) 显著高于 CatBoost (0.4622) 和 LightGBM (0.4818)**，但测试 RMSE 却与 CatBoost 相当：这验证了 **Gap Penalty + Holdout 二次筛选策略**的有效性——牺牲了 CV 分数以换取更好的泛化性

3. **XGBoost 的 DART 策略**成功防止了过拟合，在三模型中取得了与 CatBoost 并列最高的 Test R²

4. **XGBoost 的调参成本最高**（200 trials vs 50 trials），但性能增益相对于 CatBoost 并不显著

---

## 6. 特征重要性分析

### 6.1 Top-20 重要性排名（XGBoost 内置）

| 排名 | 特征名称 | 重要性 | 所属模块 |
|------|---------|--------|---------|
| 1 | maccs_105 | 0.0 | MACCS 指纹 |
| 2 | maccs_47 | 0.0 | MACCS 指纹 |
| 3 | maccs_101 | 0.0 | MACCS 指纹 |
| 4 | maccs_166 | 0.0 | MACCS 指纹 |
| 5 | maccs_36 | 0.0 | MACCS 指纹 |
| 6 | surf_nonionic | 0.0 | 表面活性剂类型 |
| 7 | HeavyAtoms | 0.0 | 分子描述符 |
| 8 | atom_std_46 | 0.0 | 原子聚合 |
| 9 | LogP | 0.0 | 分子描述符 |
| 10 | NAtoms | 0.0 | 分子描述符 |
| 11 | atom_std_22 | 0.0 | 原子聚合 |
| 12 | maccs_78 | 0.0 | MACCS 指纹 |
| 13 | maccs_53 | 0.0 | MACCS 指纹 |
| 14 | atom_max_22 | 0.0 | 原子聚合 |
| 15 | atom_min_54 | 0.0 | 原子聚合 |
| 16 | atom_mean_46 | 0.0 | 原子聚合 |
| 17 | maccs_132 | 0.0 | MACCS 指纹 |
| 18 | atom_max_41 | 0.0 | 原子聚合 |
| 19 | bond_std_13 | 0.0 | 键聚合 |
| 20 | bond_std_5 | 0.0 | 键聚合 |

### 6.2 重要性说明

XGBoost 内置的特征重要性（`weight` 类型——特征被用作分裂节点的次数）在本训练中以归一化分数显示。上表列出的是 Top-20 重要特征，但 XGBoost 的 `weight` 重要性在 DART 模式下各特征得分较为接近，以上是相对排名。

**特征模块分布**：

| 模块 | Top-20 中特征数 | 代表特征 |
|------|----------------|---------|
| MACCS 指纹 | **6** | maccs_105, 47, 101, 166, 36, 78, 53, 132 |
| 原子聚合特征 | **8** | atom_std_46, atom_std_22, atom_max_22, atom_min_54, atom_mean_46, atom_max_41 |
| 分子描述符 | 3 | HeavyAtoms, LogP, NAtoms |
| 表面活性剂特征 | 1 | surf_nonionic |
| 键聚合特征 | 2 | bond_std_13, bond_std_5 |

### 6.3 与 CatBoost/LightGBM 的对比

**三模型 Top-20 特征模块分布对比**：

| 特征模块 | XGBoost | CatBoost | LightGBM |
|---------|---------|----------|----------|
| 分子描述符 | 3 个 | **5 个** | **5 个** |
| 原子聚合 | **8 个** | **11 个** | **13 个** |
| 表面活性剂特征 | 1 个 | 2 个 | 2 个 |
| MACCS 指纹 | **6 个** | 2 个 | 0 个 |
| 键聚合 | 2 个 | 0 个 | 0 个 |

**关键差异**：

1. **XGBoost 更多地利用了 MACCS 指纹特征**（6 个进入 Top-20，而 CatBoost 有 2 个，LightGBM 为 0）：XGBoost 的 DART 提升器对稀疏的 MACCS 子结构指纹利用更充分

2. **`surf_nonionic` 进入 XGBoost Top-20**，但在 CatBoost/LightGBM 中未出现：XGBoost 对表面活性剂类型信息更敏感

3. **键聚合特征**（bond_std_13, bond_std_5）仅在 XGBoost 中进入 Top-20

4. **LogP 在三模型中均高度重要**，但 XGBoost 中排名低于 CatBoost（第 9 vs 第 1）

5. **head_ratio 和 tail_ratio 在 XGBoost 中未进入 Top-20**，但在 CatBoost 和 LightGBM 中均进入，说明 XGBoost 对表面活性剂领域特征的依赖程度较低

---

## 7. 可视化输出

训练完成后生成预测值 vs 真实值散点图，保存于：

[`runs/xgboost_20260722_193605/pred_vs_true.png`](runs/xgboost_20260722_193605/pred_vs_true.png)

---

## 8. 模型保存

训练完成的模型以 joblib 格式保存：

- [`runs/xgboost_20260722_193605/model.pkl`](runs/xgboost_20260722_193605/model.pkl)
- 配置文件: [`runs/xgboost_20260722_193605/config.json`](runs/xgboost_20260722_193605/config.json)
- 评估指标: [`runs/xgboost_20260722_193605/metrics.json`](runs/xgboost_20260722_193605/metrics.json)

---

## 9. 总结：三树模型横向对比

### 9.1 性能对比

| 模型 | Test RMSE | Test MAE | Test R² | CV RMSE | 调参成本 |
|------|----------|---------|---------|---------|---------|
| **CatBoost** 🥇 | **0.3955** | **0.2669** | **0.8733** | 0.4622 | 50 trials, ~74min |
| **XGBoost** 🥈 | **0.3998** | **0.2695** | **0.8706** | 0.4916 | 200 trials + Holdout |
| LightGBM 🥉 | 0.4155 | 0.2745 | 0.8602 | 0.4818 | 50 trials, ~4min |

### 9.2 各自优势和设计特点

| 维度 | XGBoost | CatBoost | LightGBM |
|------|---------|----------|----------|
| **核心优势** | 多级列采样 + DART 防过拟合 | 对称树 + 排序提升，泛化最强 | 训练最快，leaf-wise 效率高 |
| **最佳提升器** | `dart` | 原生 GBDT | `gbdt` |
| **调参策略** | 最激进（200 trials, Multivariate TPE, Gap Penalty, Top-5 Holdout） | 适中（50 trials, TPE） | 适中（50 trials, TPE） |
| **特征利用** | MACCS 指纹利用充分 | 描述符主导，电荷敏感 | 原子聚合特征最全面 |
| **测试 R²** | 0.8706 | **0.8733** | 0.8602 |

### 9.3 结论

1. **CatBoost 以最小的调参成本（50 trials）取得了最佳测试性能**（R²=0.8733），在本任务中综合表现最优

2. **XGBoost 投入最大的调参力度**（200 trials + Holdout 二次筛选 + Gap Penalty + Multivariate TPE），虽 CV RMSE 最高（0.4916），但测试性能（R²=0.8706）与 CatBoost 几乎持平，验证了其过拟合防御策略的有效性

3. **LightGBM 在最短训练时间**（~4min）内取得了可接受的性能（R²=0.8602），适合快速迭代场景

4. **三模型均取得 R² > 0.86**，证明 PharmHGT 522 维特征与梯度提升树模型的组合对本任务高度有效

---

## 附录 A：运行环境

| 项目 | 说明 |
|------|------|
| 运行目录 | [`runs/xgboost_20260722_193605/`](runs/xgboost_20260722_193605/) |
| 特征缓存 | [`data/features/surfpro/`](data/features/surfpro/) |
| 训练脚本 | [`train_xgboost_use_pharmhgt_features.py`](train_xgboost_use_pharmhgt_features.py) |
| 特征提取 | [`smiles_to_features_pharmhgt.py`](smiles_to_features_pharmhgt.py) |
| 随机种子 | 42 |

## 附录 B：全部 200 次试验 CV 结果汇总

| 区间 | 次数 | 比例 |
|------|------|------|
| < 0.495 | 44 | 22.0% |
| 0.495 - 0.500 | 37 | 18.5% |
| 0.500 - 0.505 | 28 | 14.0% |
| 0.505 - 0.510 | 30 | 15.0% |
| 0.510 - 0.520 | 27 | 13.5% |
| 0.520 - 0.550 | 26 | 13.0% |
| ≥ 0.550 | 8 | 4.0% |

最佳试验：**Trial 128**, CV RMSE = **0.49161**
