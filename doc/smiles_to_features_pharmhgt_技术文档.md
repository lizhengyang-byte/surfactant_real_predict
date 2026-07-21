# smiles_to_features_pharmhgt — PharmHGT 风格 522 维分子特征提取技术文档

## 1. 概述

### 1.1 文件定位

`smiles_to_features_pharmhgt.py` 是一个**共享特征提取模块**，供 CatBoost、LightGBM、XGBoost 等树模型训练脚本调用。其核心功能是将 SMILES 字符串转换为固定长度的 **522 维数值特征向量**，并提供了完整的缓存机制以避免重复计算。

### 1.2 设计动机

- 传统的 ECFP/MACCS 指纹无法充分捕获药效团和化学反应信息
- 直接使用 GNN（如图神经网络）计算开销大，且不利于树模型集成
- 借鉴 PharmHGT 异构图 Transformer 思想，将多尺度分子信息编码为手工特征，兼顾信息丰富度与计算效率

### 1.3 依赖关系

```
catboost/lighgbm/xgboost 训练脚本
        │
        ├── load_or_compute_features()  → 批量加载或计算
        └── smiles_to_features_pharmhgt() → 单分子推理
                │
                └── build_feature_vector()
                        ├── get_atom_features()        55 维
                        ├── get_bond_features()        14 维
                        ├── get_pharmacophore_features()  194 维 (MACCS)
                        ├── get_reaction_features()   34 维 (BRICS)
                        └── detect_surfactant()       头基/尾链检测
```

### 1.4 核心常量

| 常量 | 值 | 说明 |
|------|-----|------|
| `ATOM_FEAT_DIM` | 55 | 单原子特征维度 |
| `BOND_FEAT_DIM` | 14 | 单化学键特征维度 |
| `PHARM_FEAT_DIM` | 194 | 药效团（MACCS）特征维度 |
| `REACT_FEAT_DIM` | 34 | 反应性（BRICS）特征维度 |
| `FEATURE_DIM` | 522 | 最终特征向量总维度 |
| `CACHE_DIR` | `data/features/pharmhgt/` | 特征缓存目录 |

---

## 2. 整体架构：522 维特征总览

### 2.1 模块组成

| 序号 | 模块 | 维度 | 计算方式 | 子维度构成 |
|------|------|------|---------|-----------|
| 1 | 原子级聚合特征 | 220 | `get_atom_features` 对每个原子输出 55 维 → 在全分子上做 **mean/std/min/max** 四类统计 | 55 × 4 |
| 2 | 键级聚合特征 | 56 | `get_bond_features` 对每个化学键输出 14 维 → **mean/std/min/max** 四类统计 | 14 × 4 |
| 3 | 药效团特征 | 194 | `MACCSkeys.GenMACCSKeys` 生成 166 位 MACCS 键 → 填充至 194 维 | 194 |
| 4 | 反应性特征 | 34 | `BRICS.BRICSDecompose` 分解碎片 → CRC32 哈希分桶 → 归一化直方图 | 34 |
| 5 | 表面活性剂类型 | 4 | 基于 SMARTS 子结构匹配判定阴/阳/非/两性离子 → one-hot 编码 | 4 |
| 6 | 头基/尾链比例 | 2 | 检测到的头基原子数 / 总原子数, 尾链碳原子数 / 总原子数 | 2 |
| 7 | 分子描述符 | 12 | RDKit 计算的 12 种基本描述符（均经归一化处理） | 12 |
| **总计** | | **522** | | |

### 2.2 数据流

```
SMILES 字符串
    │
    ├─→ [RDKit Mol 解析] ─→ 若无效则返回 None
    │
    ├─→ Gasteiger 电荷计算 (ComputeGasteigerCharges)
    │
    ├─→ 1. 原子特征提取    ──→ N_atoms × 55 ──→ mean/std/min/max → [220]
    ├─→ 2. 键特征提取      ──→ N_bonds × 14 ──→ mean/std/min/max → [56]
    ├─→ 3. MACCS 指纹      ──→ 194 维                                   → [194]
    ├─→ 4. BRICS 碎片直方图 ──→ 34 维                                    → [34]
    ├─→ 5. 表面活性剂检测   ──→ one-hot(4) + head_ratio + tail_ratio     → [6]
    └─→ 6. 分子描述符      ──→ 12 维                                    → [12]
                                                                            │
                                                                    ┌───────┘
                                                                    ▼
                                                            522 维特征向量
```

---

## 3. 模块详解

### 3.1 原子级特征 (Atom Features) — 55 维

函数：`get_atom_features(atom: Chem.Atom) -> np.ndarray`

对分子中**每个原子**提取 55 维特征向量，然后在整个分子上聚合为 4 个统计量（mean/std/min/max），得到 `55 × 4 = 220` 维。

#### 55 维逐位编码表

| 索引范围 | 维度 | 内容 | 编码方式 |
|---------|------|------|---------|
| `[0:16)` | 16 | 原子序数类型 | one-hot，支持 16 种元素：`[1, 3, 5, 6, 7, 8, 9, 11, 14, 15, 16, 17, 19, 35, 53, 79]`（H, Li, B, C, N, O, F, Na, Si, P, S, Cl, K, Br, I, Au），未知原子映射到最后一个 |
| `[16:22)` | 6 | 度数（连接原子数） | one-hot：0-5+，`min(degree, 5)` 作为索引 |
| `[22]` | 1 | 形式电荷 | 标量：`clip(charge, -2, 2) / 2.0`，归一到 [-1, 1] |
| `[23:28)` | 5 | 隐氢数 | one-hot：0-4+，`min(implict_Hs, 4)` 作为索引 |
| `[28:33)` | 5 | 杂化方式 | one-hot：SP, SP2, SP3, SP3D, SP3D2，未知映射到索引 0 |
| `[33]` | 1 | 是否芳香 | 0/1 二值 |
| `[34]` | 1 | 是否在环中 | 0/1 二值 |
| `[35]` | 1 | 原子质量 | 标量：`mass / 100.0` |
| `[36]` | 1 | 是否为手性中心 | 0/1 二值（`CHI_UNSPECIFIED` 为 0） |
| `[37]` | 1 | 自由基电子数 | 标量：`min(radical_electrons, 2) / 2.0` |
| `[38:42)` | 4 | 显式化合价 | one-hot：1-5+（5+ 映射到索引 3） |
| **注意**：此处代码 `feat[37 + val]` 存在偏移重叠问题，显式化合价 1-4 落在索引 [38, 41]，≥5 落在索引 41 |
| `[42:46)` | 4 | 环大小 | one-hot：3, 4, 5, 6 元环 |
| `[46:50)` | 4 | Gasteiger 电荷分桶 | 计算 `(gc + 1.0) / 0.5` 的整数部分，clip 到 [0, 3]，对应 4 个桶 |
| `[50]` | 1 | 环大小 ≥ 7 | 0/1 二值 |
| `[51]` | 1 | 是否为 N 或 O 原子 | 0/1 二值（`atomic_num in (7, 8)`） |
| `[52]` | 1 | 是否为 H 键供体 | 0/1 二值：N 或 O 且含有 H 原子 |
| `[53]` | 1 | 是否为 H 键受体 | 0/1 二值：N 或 O |
| `[54]` | 1 | 重原子邻居数占比 | 标量：`degree / 4.0` |

#### 聚合方式

对于含 N 个原子的分子：

```
atom_feats = np.array([get_atom_features(a) for a in mol.GetAtoms()])  # shape: (N, 55)

mean_feat = atom_feats.mean(axis=0)  # [55]
std_feat  = atom_feats.std(axis=0)   # [55]
min_feat  = atom_feats.min(axis=0)   # [55]
max_feat  = atom_feats.max(axis=0)   # [55]

atom_agg = concat([mean_feat, std_feat, min_feat, max_feat])  # [220]
```

#### 设计要点

- 使用 **四个统计量**（均值/标准差/最小值/最大值）而不是全局池化，保留了原子特征在全分子上的分布信息
- Gasteiger 电荷需要在分子层面预先计算（`AllChem.ComputeGasteigerCharges`），如果计算失败则跳过
- 环信息通过 `mol.GetRingInfo().AtomRings()` 逐原子检测，每个原子仅匹配第一个环

---

### 3.2 键级特征 (Bond Features) — 14 维

函数：`get_bond_features(bond: Chem.Bond) -> np.ndarray`

对分子中**每个化学键**提取 14 维特征，然后聚合为 4 个统计量（mean/std/min/max），得到 `14 × 4 = 56` 维。

#### 14 维逐位编码表

| 索引范围 | 维度 | 内容 | 编码方式 |
|---------|------|------|---------|
| `[0:4)` | 4 | 键类型 | one-hot：SINGLE(0), DOUBLE(1), TRIPLE(2), AROMATIC(3)，未知映射到 0 |
| `[4]` | 1 | 是否共轭 | 0/1 二值 (`GetIsConjugated`) |
| `[5]` | 1 | 是否在环中 | 0/1 二值 (`IsInRing`) |
| `[6:12)` | 6 | 立体构型 | one-hot：NONE(0), ANY(1), Z(2), E(3), CIS(4), TRANS(5) |
| `[12]` | 1 | 是否为芳香键 | 0/1 二值（与 `[0:4]` 中 AROMATIC 冗余，但单独保留便于聚合后区分） |
| `[13]` | 1 | 是否在环中 | 0/1 二值（与 `[5]` 冗余） |

#### 设计要点

- 索引 5 和 13 均表示环成员信息，这是设计上的冗余，聚合后仍会保留一定区分性
- 对于不含化学键的单原子分子，返回全零向量 (56,)

---

### 3.3 药效团特征 (Pharmacophore Features) — 194 维

函数：`get_pharmacophore_features(mol: Chem.Mol) -> np.ndarray`

- 使用 RDKit 的 `MACCSkeys.GenMACCSKeys(mol)` 生成 **166 位 MACCS 结构键**
- 由于 MACCS 标准输出为 167 位（含第 0 位全 0），代码将其填充至 194 维
- 剩余未使用维度保持为 0

**注意**：此处的 194 维并非 MACCS 原始标准的 166 维，而是预留了额外空间，可能是为了与 PharmHGT 论文中的药效团维度对齐，或为未来扩展保留。

---

### 3.4 反应性特征 (Reaction / BRICS Features) — 34 维

函数：`get_reaction_features(mol: Chem.Mol) -> np.ndarray`

#### 原理

利用 **BRICS**（Breaking of Retrosynthetically Interesting Chemical Substructures）断裂规则将分子分解为碎片，保留碎片包含的官能团信息和碎片间的反应连接信息。

#### 计算步骤

1. **过滤**：如果可旋转键数 (`NumRotatableBonds`) < 1，直接返回全零向量（刚性小分子无碎片意义）
2. **BRICS 分解**：调用 `BRICS.BRICSDecompose(mol, returnMols=False)` 生成碎片字符串列表
3. **安全限制**：最多处理 128 个碎片，防止异常分子导致死循环
4. **哈希分桶**：对每个碎片字符串计算 `zlib.crc32(frag.encode()) % 34`，将碎片映射到 34 个桶中，对应位置计数 +1
5. **归一化**：每个桶的计数除以碎片总数 `max(len(frags), 1)`，得到频率直方图

#### 设计要点

- CRC32 哈希将任意碎片字符串无冲突地映射到固定 34 维空间，无需预设碎片类别
- 归一化使特征不受分子大小影响，反映碎片类型分布而非绝对数量
- 采用 try-except 兜底，BRICS 分解失败时返回全零向量，保证鲁棒性

---

### 3.5 表面活性剂检测与特征 — 6 维

函数：`detect_surfactant(smiles: str) -> (head_mask, tail_mask, surf_type)`

这是本项目**特有的领域知识模块**，专门为表面活性剂分子设计。

#### 3.5.1 反离子排除

首先识别并排除常见的反离子，防止它们被误判为头基或尾链：

| 反离子 | SMARTS 模式 |
|--------|------------|
| Na⁺ | `[Na+]` |
| Li⁺ | `[Li+]` |
| K⁺ | `[K+]` |
| Cl⁻ | `[Cl-]` |
| Br⁻ | `[Br-]` |
| I⁻ | `[I-]` |

匹配到的原子在后续处理中从头基/尾链掩码中排除。

#### 3.5.2 表面活性剂类型判定

通过检测分子中是否含有带负电和/或带正电的原子，按优先级确定类型：

| 判定条件 | 类型 |
|---------|------|
| 同时含有 `[O-]` / `[S-]` 和 `[N+]` / `[n+]` | 两性离子 (Zwitterionic) |
| 仅含 `[O-]` / `[S-]` | 阴离子 (Anionic) |
| 仅含 `[N+]` / `[n+]` | 阳离子 (Cationic) |
| 两者均不含 | 非离子 (Nonionic) |

#### 3.5.3 头基（亲水基团）检测

根据判定的表面活性剂类型，使用对应的 SMARTS 子结构模式进行匹配：

**阴离子基团**：

| 基团名 | SMARTS |
|--------|--------|
| 磺酸盐 (sulfonate) | `S(=O)(=O)[O-]` |
| 硫酸酯 (sulfate) | `OS(=O)(=O)[O-]` |
| 羧酸盐 (carboxylate) | `C(=O)[O-]` |
| 磷酸酯 (phosphate) | `OP(=O)([O-])[O-]` |

**阳离子基团**：

| 基团名 | SMARTS |
|--------|--------|
| 季铵 (quat_ammonium) | `[N+](C)(C)C` |
| 铵 (ammonium) | `[NH3+]` |
| 吡啶 (pyridinium) | `[n+]1ccccc1` |
| 咪唑 (imidazolium) | `[n+]1cncc1` |

**非离子基团**：

| 基团名 | SMARTS |
|--------|--------|
| 羟基 (hydroxyl) | `[OH]` |
| 醚 (ether) | `COC` |
| 聚氧乙烯 (polyoxyethylene) | `CCOCCO` |
| 酰胺 (amide) | `NC(=O)` |
| 酯 (ester) | `C(=O)OC` |

匹配到的原子（排除反离子）在 `head_mask` 中标记为 `True`。

#### 3.5.4 尾链（疏水碳链）检测

使用 **DFS（深度优先搜索）** 寻找分子中最长的连续碳链：

1. 遍历所有碳原子（排除反离子）
2. 对每个碳原子启动 DFS，只沿碳原子（`atomic_num == 6`）扩展
3. 记录最长路径
4. 如果最长路径 ≥ 4 个碳原子，则认为该路径为疏水尾链
5. 否则将所有未被标记为头基的碳原子标记为尾链（兜底策略）

**DFS 算法实现**：
```
def dfs_longest(start, visited, chain):
    best = chain
    for neighbor in adjacency[start]:
        if neighbor in visited:
            continue
        if atom[neighbor] is carbon and not counterion:
            visited.add(neighbor)
            result = dfs_longest(neighbor, visited, chain + [neighbor])
            if len(result) > len(best):
                best = result
            visited.remove(neighbor)
    return best
```

#### 3.5.5 最终特征向量 (6 维)

| 索引 | 内容 | 编码 |
|------|------|------|
| 0 | 阴离子 | one-hot |
| 1 | 阳离子 | one-hot |
| 2 | 非离子 | one-hot |
| 3 | 两性离子 | one-hot |
| 4 | 头基原子占比 | `head_mask.sum() / n_atoms` |
| 5 | 尾链原子占比 | `tail_mask.sum() / n_atoms` |

---

### 3.6 分子描述符 (Molecular Descriptors) — 12 维

函数：`build_feature_vector` 直接计算，所有描述符经过归一化处理以适配树模型。

| 序号 | 名称 | 计算方式 | 归一化分母 |
|------|------|---------|-----------|
| 0 | MolWt | `Descriptors.MolWt` | `/ 500.0` |
| 1 | LogP | `Descriptors.MolLogP` | `/ 10.0` |
| 2 | TPSA | `Descriptors.TPSA` | `/ 200.0` |
| 3 | RotBonds | `Descriptors.NumRotatableBonds` | `/ max(n_atoms, 1)` |
| 4 | HBA | `Descriptors.NumHAcceptors` | `/ max(n_atoms, 1)` |
| 5 | HBD | `Descriptors.NumHDonors` | `/ max(n_atoms, 1)` |
| 6 | NumRings | `rdMolDescriptors.CalcNumRings` | `/ 20.0` |
| 7 | AroRings | `rdMolDescriptors.CalcNumAromaticRings` | `/ 10.0` |
| 8 | AliRings | `rdMolDescriptors.CalcNumAliphaticRings` | `/ 10.0` |
| 9 | FracSP3 | `rdMolDescriptors.CalcFractionCSP3` | 原生 [0,1] |
| 10 | HeavyAtoms | `mol.GetNumHeavyAtoms` | `/ 100.0` |
| 11 | NAtoms | `mol.GetNumAtoms` | `/ 200.0` |

**归一化设计思路**：将描述符缩放到大致 [0, 1] 或 [-1, 1] 区间，消除量纲差异对树模型分裂点选择的影响。

---

## 4. 缓存机制

### 4.1 缓存目录结构

```
data/features/pharmhgt/
├── X_train.npy       # 训练集特征 (n_train_valid × 522)
├── y_train.npy       # 训练集目标值 (n_train_valid,)
├── X_test.npy        # 测试集特征 (n_test_valid × 522)
├── y_test.npy        # 测试集目标值 (n_test_valid,)
└── metadata.json     # 元数据（用于缓存有效性校验）
```

### 4.2 数据变更检测

```python
def _smiles_hash(df, smiles_col='SMILES'):
    combined = ''.join(df[smiles_col].dropna().values)
    return hashlib.md5(combined.encode()).hexdigest()
```

- 将所有 SMILES 字符串拼接后计算 **MD5 哈希**
- 训练集和测试集分别计算哈希值
- `metadata.json` 中同时存储 `train_smiles_hash` 和 `test_smiles_hash`

### 4.3 缓存命中判定流程

```
调用 load_or_compute_features()
        │
        ├─ 读取 metadata.json ──→ 不存在 → Cache MISS
        │
        ├─ 对比 train_smiles_hash ──→ 不匹配 → Cache MISS
        │
        ├─ 对比 test_smiles_hash ──→ 不匹配 → Cache MISS
        │
        └─ 全部匹配 → Cache HIT → 直接加载 4 个 .npy 文件
```

### 4.4 缓存未命中时的处理

1. 对训练集和测试集分别执行 `_featurize_dataframe`
2. 对计算结果做 NaN/Inf 安全检查，异常值替换为 0
3. 保存 X_train.npy, y_train.npy, X_test.npy, y_test.npy
4. 写入 metadata.json（含 SMILES 哈希值、文件名、目标列、特征维度）

### 4.5 强制重计算

通过 `force_recompute=True` 参数可跳过缓存检查，强制重新计算。

---

## 5. 特征命名规范

文件末尾定义了完整的 `FEATURE_NAMES` 列表（长度 522），用于特征重要性分析时的可解释性。

| 范围 | 数量 | 命名模式 | 示例 |
|------|------|---------|------|
| `[0:220)` | 220 | `atom_mean/atom_std/atom_min/atom_max_{dim}` | `atom_mean_0`, `atom_std_14` |
| `[220:276)` | 56 | `bond_mean/bond_std/bond_min/bond_max_{dim}` | `bond_mean_3`, `bond_max_10` |
| `[276:470)` | 194 | `maccs_{i}` | `maccs_0`, `maccs_166` |
| `[470:504)` | 34 | `brics_{i}` | `brics_0`, `brics_33` |
| `[504:508)` | 4 | `surf_anionic`, `surf_cationic`, `surf_nonionic`, `surf_zwitterionic` | |
| `[508:510)` | 2 | `head_ratio`, `tail_ratio` | |
| `[510:522)` | 12 | 描述符名称 | `MolWt`, `LogP`, `TPSA`, `RotBonds`, `HBA`, `HBD`, `NumRings`, `AroRings`, `AliRings`, `FracSP3`, `HeavyAtoms`, `NAtoms` |

---

## 6. API 参考

### 6.1 单分子 API

```python
from smiles_to_features_pharmhgt import smiles_to_features_pharmhgt

vec = smiles_to_features_pharmhgt("CCO")
print(vec.shape)  # (522,)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `smiles` | str | SMILES 字符串 |
| **返回** | `np.ndarray` | 522 维特征向量，SMILES 无效时返回 `None` |

### 6.2 批量 API

```python
from smiles_to_features_pharmhgt import load_or_compute_features

X_train, y_train, X_test, y_test = load_or_compute_features(
    train_csv='./data/surfpro_imputed.csv',
    test_csv='./data/surfpro_test.csv',
    target_col='pCMC',
    smiles_col='SMILES',
    cache_dir=None,
    force_recompute=False,
    verbose=True
)
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `train_csv` | `./data/surfpro_imputed.csv` | 训练集 CSV 路径 |
| `test_csv` | `./data/surfpro_test.csv` | 测试集 CSV 路径 |
| `target_col` | `'pCMC'` | 目标变量列名 |
| `smiles_col` | `'SMILES'` | SMILES 列名 |
| `cache_dir` | `'data/features/pharmhgt/'` | 缓存目录 |
| `force_recompute` | `False` | 是否强制重计算 |
| `verbose` | `True` | 是否输出进度信息 |

### 6.3 训练脚本集成示例

所有训练脚本（`train_catboost_use_pharmhgt_features.py` 等）均遵循如下模式：

```python
from smiles_to_features_pharmhgt import load_or_compute_features

X_train, y_train, X_test, y_test = load_or_compute_features()

# 训练树模型
model = SomeBoostingModel()
model.fit(X_train, y_train)
```

---

## 7. 与原始 PharmHGT 论文的关系

### 7.1 借鉴点

原始 PharmHGT（Jiang et al., 2023, *Communications Chemistry*）的核心贡献是提出了一种**异构分子图**表示方法，包含原子级视图、药效团级视图和连接级视图三个层次，并用 Graph Transformer 进行消息传递。本文件中借鉴了以下设计思路：

| PharmHGT 论文理念 | 本文件对应实现 |
|-----------------|--------------|
| 原子级视图 (Atom-level view) | `get_atom_features` — 55 维原子编码 |
| 药效团级视图 (Pharm-level view) | `get_pharmacophore_features` — MACCS 194 维 |
| 反应信息 (BRICS reaction types) | `get_reaction_features` — BRICS 碎片哈希分桶 |
| 多层次信息融合 | `build_feature_vector` — 6 个模块拼接融合 |

### 7.2 关键差异

| 维度 | 原始 PharmHGT (GNN) | 本文件 (手工特征) |
|------|--------------------|------------------|
| 模型类型 | Heterogeneous Graph Transformer | 树模型 (CatBoost/LightGBM/XGBoost) |
| 特征表示 | 图结构 + 节点/边嵌入经消息传递学习 | 固定 522 维向量，原子级聚合代替消息传递 |
| 表面活性剂 | 无领域知识 | 显式检测头基/尾链 |
| 碎片特征 | BRICS 碎片作为独立节点，保留图拓扑 | BRICS 碎片哈希为固定长度直方图，丢失拓扑信息 |
| 计算效率 | 低（需 GPU + 消息传递迭代） | 高（单分子毫秒级，CPU 即可） |

### 7.3 命名由来

文件名中的 "pharmhgt" 表示这是一个受 PharmHGT 启发的特征工程方案，但本质上是对原始 GNN 方法的**降维替代方案**——将端到端学习的图表示转化为可解释的手工特征，使其适用于工业界更常用的树模型。

---

## 8. 注意事项

1. **NaN/Inf 安全**：在批量处理和单分子 API 中均未对输出做 NaN/Inf 检查，但在缓存层（`load_or_compute_features`）会自动修复。直接调用 `smiles_to_features_pharmhgt` 时建议自行检查。
2. **Gasteiger 电荷依赖**：原子特征中部分信息依赖于 `ComputeGasteigerCharges`，该计算可能对大分子失败，此时对应特征位保持为 0。
3. **BRICS 性能**：`BRICSDecompose` 对某些复杂分子可能耗时较长，代码设置了 128 碎片上限防止死循环。
4. **尾链检测局限性**：DFS 找到的是最长的连续碳链，对于含支链的表面活性剂（如双尾链），只能捕获其中一条，可能遗漏第二条尾链。
5. **特征冗余**：MACCS 维度无明确上限（194 维中大量为 0），且原子/键特征中存在少量冗余（如芳香性在原子和键级别均有编码），这是设计上的有意取舍。
