# 补充实验：容量匹配消融（实验一）与 特征排列随机化（实验四）

本文件夹严格按照《论文修改清单与补充实验方案（含代码）》实现并执行
**实验一（容量匹配消融，必做）** 与 **实验四（特征排列随机化，强烈建议）**。

> **只读复用项目原有文件，不修改任何原始文件**（train/*、use/*、runs/*、data/features/* 均不改动）。
> 唯一的外部依赖是复用 `train/train_pCMC_models/smiles_to_features_pharmhgt.py` 的
> 522 维特征提取函数（import 方式，只读）。

## 运行顺序

```bash
# 1) 统一数据准备（一次即可，产出 data/prepared_data.npz，与 CWD 无关）
python supp_experiments/prepare_data.py

# 2) 实验一：容量匹配消融（6 模型 × 6 目标 = 36 组 per-target 训练）
python supp_experiments/experiment1.py

# 3) 实验四：特征排列随机化（基线复用实验一的 RNN/Transformer，新增 5×2×6=60 组）
python supp_experiments/experiment4.py --permutations 5

# 4) 报告图表
python supp_experiments/make_report_figures.py
```

设备：自动检测 CUDA（本机为 NVIDIA T1200 GPU）。

## 目录

```
supp_experiments/
├── prepare_data.py        # 统一多目标数据集构建（522 维特征 × 全部分子，目标列 NaN 掩码）
├── common.py              # 模型定义、容量搜索、训练/评估协议
├── experiment1.py         # 实验一：容量匹配消融
├── experiment4.py         # 实验四：特征排列随机化
├── make_report_figures.py # 报告图表（R² 热力图、ΔR² 柱状图）
├── data/prepared_data.npz # 统一数据集（含掩码）
├── results/
│   ├── exp1_capacity.csv / exp1_capacity.json / exp1_capacity_pivot.csv
│   ├── exp4_permutation.json / exp4_summary.csv / exp4_verdict.json
│   └── exp1_training.log / exp4_training.log
└── report_实验一_实验四_补充实验.md   # 最终报告（markdown）
```

## 关键口径（与论文/文档对齐）

- 特征：522 维 PharmHGT 风格（`build_feature_vector`），与论文原管线完全一致，
  并经 `prepare_data.py` 与论文 pCMC 特征缓存逐元素交叉校验。
- 目标顺序（文档要求）：`pCMC, gamma_CMC(=AW_ST_CMC), Gamma_max, A_min(=Area_min), Pi_CMC, pC20`；
  `Gamma_max` 已乘 `YSCALE_GMAX=1e6`（同论文约定），评估时还原为原始 1e-6 尺度。
- 训练协议：**per-target** 建模（文档矩阵即 6×6=36 组 / 2×6×6=72 组），
  与论文一致采用 12.5% 验证集（seed 42）早停，max 300 epoch，Adam lr=1e-3，
  ReduceLROnPlateau，patience 30。
- 模型：MLP-original(5层×488)、MLP-small(3层×64)、RNN-original(3层×64)、
  RNN-large(容量≈MLP-original)、Transformer-original(3层×d64)、Transformer-large(容量≈MLP)，
  全部定义照抄文档附录代码（输出维改为 1）。