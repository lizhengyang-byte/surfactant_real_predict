# PharmaHGT 522 维特征参考表

> 基于 `smiles_to_features_pharmhgt.py` 自动整理  
> 用途: SHAP 分析时对照特征索引与含义

---

## 1\. 特征总览

| 模块 | 维度 | 索引范围 | 说明 |
| --- | --- | --- | --- |
| 原子聚合特征 | 220 | 000–219 | 55 维原子特征 × 4 统计量 (mean/std/min/max) |
| 键聚合特征 | 56 | 220–275 | 14 维键特征 × 4 统计量 |
| MACCS 指纹 | 194 | 276–469 | 药效团特征（填充至 194） |
| BRICS 碎片 | 34 | 470–503 | 反应性碎片 CRC32 分桶直方图 |
| 表面活性剂类型 | 4 | 504–507 | 阴/阳/非/两性 one-hot |
| 头基/尾链比例 | 2 | 508–509 | head_ratio, tail_ratio |
| 分子描述符 | 12 | 510–521 | RDKit 全局描述符（归一化） |
| **总计** | **522** | **000–521** |  |

---

## 2\. 原子特征 (55 维, 索引 0–54)

`get_atom_features()` 返回的原始原子特征向量。聚合后每组 55 维 × 4 统计量 = 220 维。

| 索引 | 名称 | 维度 | 说明 |
| --- | --- | --- | --- |
| 0–15 | atom_type_onehot | 16 | 原子序数 one-hot: 1(H), 3(Li), 5(B), 6(C), 7(N), 8(O), 9(F), 11(Na), 14(Si), 15(P), 16(S), 17(Cl), 19(K), 35(Br), 53(I), 79(Au) |
| 16–21 | degree_onehot | 6 | 度数 one-hot: 0, 1, 2, 3, 4, ≥5 |
| 22 | formal_charge | 1 | 形式电荷 (clip[-2,2]/2 → [-1,1]) |
| 23–27 | implicit_h_onehot | 5 | 隐氢数 one-hot: 0, 1, 2, 3, ≥4 |
| 28–32 | hybridization_onehot | 5 | 杂化方式 one-hot: SP, SP2, SP3, SP3D, SP3D2 |
| 33 | is_aromatic | 1 | 是否芳香原子 |
| 34 | in_ring | 1 | 是否在环中 |
| 35 | mass_div_100 | 1 | 原子质量 / 100 |
| 36 | is_chiral | 1 | 是否手性中心 |
| 37 | radical_electrons | 1 | 自由基电子数 / 2 |
| 38–41 | valence_onehot | 4 | 显价 one-hot: 1, 2, 3, ≥4 |
| 42–45 | ring_size_onehot | 4 | 环大小 one-hot: 3, 4, 5, 6 元环 |
| 46–49 | gasteiger_bins | 4 | Gasteiger 电荷分桶: [-1,-0.5), [-0.5,0), [0,0.5), [0.5,1] |
| 50 | ring_ge7 | 1 | 是否在 ≥7 元环中 |
| 51 | is_N_or_O | 1 | 是否为 N 或 O 原子 |
| 52 | is_h_donor | 1 | 是否为 H 键供体 (N/O + H) |
| 53 | is_h_acceptor | 1 | 是否为 H 键受体 (N/O) |
| 54 | heavy_neighbors_div4 | 1 | 重原子邻居数 / 4 |

### 聚合统计量 (4 种)

每种统计量生成 55 维，拼接为 220 维：

| 索引范围 | 名称 | 统计量 |
| --- | --- | --- |
| 000–054 | `atom_mean_0` – `atom_mean_54` | 各原子特征的均值 |
| 055–109 | `atom_std_0` – `atom_std_54` | 各原子特征的标准差 |
| 110–164 | `atom_min_0` – `atom_min_54` | 各原子特征的最小值 |
| 165–219 | `atom_max_0` – `atom_max_54` | 各原子特征的最大值 |

> 例: `atom_mean_46` = 分子中所有原子 Gasteiger 电荷分桶 0 的均值  
> 例: `atom_std_16` = 分子中所有原子度数为 0 的占比标准差

---

## 3\. 键特征 (14 维, 索引 0–13)

`get_bond_features()` 返回的原始键特征向量。聚合后每组 14 维 × 4 统计量 = 56 维。

| 索引 | 名称 | 维度 | 说明 |
| --- | --- | --- | --- |
| 0–3 | bond_type_onehot | 4 | 键型 one-hot: SINGLE, DOUBLE, TRIPLE, AROMATIC |
| 4 | is_conjugated | 1 | 是否共轭 |
| 5 | in_ring | 1 | 是否在环中 |
| 6–11 | stereo_onehot | 6 | 立体化学 one-hot: NONE, ANY, Z, E, CIS, TRANS |
| 12 | is_aromatic | 1 | 是否芳香键 |
| 13 | in_ring_dup | 1 | 是否在环中（重复，与索引 5 冗余） |

### 聚合统计量 (4 种)

| 索引范围 | 名称 | 统计量 |
| --- | --- | --- |
| 220–233 | `bond_mean_0` – `bond_mean_13` | 各键特征的均值 |
| 234–247 | `bond_std_0` – `bond_std_13` | 各键特征的标准差 |
| 248–261 | `bond_min_0` – `bond_min_13` | 各键特征的最小值 |
| 262–275 | `bond_max_0` – `bond_max_13` | 各键特征的最大值 |

---

## 4\. MACCS 指纹 (194 维, 索引 276–469)

| 索引范围 | 名称 | 说明 |
| --- | --- | --- |
| 276–469 | `maccs_0` – `maccs_193` | MACCS 结构键 (166 bits, 填充至 194) |

> MACCS 键是预定义的 166 个分子子结构模式，`maccs_105` 等已被 SHAP 分析识别为重要特征。

---

## 5\. BRICS 碎片 (34 维, 索引 470–503)

| 索引范围 | 名称 | 说明 |
| --- | --- | --- |
| 470–503 | `brics_0` – `brics_33` | BRICS 分解碎片 CRC32 哈希分桶直方图（归一化） |

> 将分子按 BRICS 规则分解为片段，每个片段字符串的 CRC32 值取模 34 落入对应桶。

---

## 6\. 表面活性剂特征 (6 维, 索引 504–509)

| 索引 | 名称 | 说明 |
| --- | --- | --- |
| 504 | `surf_anionic` | 阴离子型 (含 [O-] / [S-]) |
| 505 | `surf_cationic` | 阳离子型 (含 [N+] / [n+]) |
| 506 | `surf_nonionic` | 非离子型（默认，无离子基团） |
| 507 | `surf_zwitterionic` | 两性型（同时含阴/阳离子基团） |
| 508 | `head_ratio` | 头基原子数 / 总原子数（SMARTS 匹配 + 反离子排除） |
| 509 | `tail_ratio` | 尾链碳原子数 / 总原子数（DFS 最长碳链检测 ≥4） |

---

## 7\. 分子描述符 (12 维, 索引 510–521)

| 索引 | 名称 | 归一化方法 | 说明 |
| --- | --- | --- | --- |
| 510 | `MolWt` | / 500 | 分子量 |
| 511 | `LogP` | / 10 | 脂水分配系数 (Wildman-Crippen) |
| 512 | `TPSA` | / 200 | 极性表面积 |
| 513 | `RotBonds` | / n_atoms | 可旋转键数 |
| 514 | `HBA` | / n_atoms | H 键受体数 |
| 515 | `HBD` | / n_atoms | H 键供体数 |
| 516 | `NumRings` | / 20 | 环总数 |
| 517 | `AroRings` | / 10 | 芳香环数 |
| 518 | `AliRings` | / 10 | 脂肪环数 |
| 519 | `FracSP3` | 原生 [0,1] | SP3 碳比例 |
| 520 | `HeavyAtoms` | / 100 | 重原子数 |
| 521 | `NAtoms` | / 200 | 总原子数 |

---

## 8\. SHAP 分析快速对照

### 所有模型一致认可的关键特征

| SHAP 排名 | 特征索引 | 名称 | 物理意义 |
| --- | --- | --- | --- |
| 1 | 511 | `LogP` | 亲水-疏水平衡，CMC 首要决定因素 |
| 2 | 520 | `HeavyAtoms` | 分子大小 |
| 3 | 510 | `MolWt` | 分子量 |
| 4 | 509 | `tail_ratio` | 疏水尾链占比 |

### 常见重要特征索引速查

| 索引 | 名称 | 所属模块 |
| --- | --- | --- |
| 511 | LogP | 分子描述符 |
| 520 | HeavyAtoms | 分子描述符 |
| 510 | MolWt | 分子描述符 |
| 521 | NAtoms | 分子描述符 |
| 509 | tail_ratio | 表面活性剂特征 |
| 508 | head_ratio | 表面活性剂特征 |
| 513 | RotBonds | 分子描述符 |
| 512 | TPSA | 分子描述符 |
| 276 | maccs_0 | MACCS 指纹 |
| 470 | brics_0 | BRICS 碎片 |
| 504–507 | surf_* | 表面活性剂类型 |
| 35 | atom_mean_35 / atom_std_35 | 原子聚合 (原子质量) |
| 46 | atom_mean_46 / atom_std_46 | 原子聚合 (Gasteiger 电荷) |
| 54 | atom_mean_54 / atom_std_54 | 原子聚合 (重原子邻居数) |
| 16 | atom_mean_16 / atom_std_16 | 原子聚合 (度数=0) |
| 22 | atom_mean_22 / atom_std_22 | 原子聚合 (形式电荷) |

---

## 9\. 全部 522 维特征完整明细

| 全局索引 | 特征名称 | 所属模块 | 详细说明 |
| --- | --- | --- | --- |
| 0 | `atom_mean_0` | 原子聚合 | 原子类型=H (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 1 | `atom_mean_1` | 原子聚合 | 原子类型=Li (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 2 | `atom_mean_2` | 原子聚合 | 原子类型=B (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 3 | `atom_mean_3` | 原子聚合 | 原子类型=C (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 4 | `atom_mean_4` | 原子聚合 | 原子类型=N (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 5 | `atom_mean_5` | 原子聚合 | 原子类型=O (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 6 | `atom_mean_6` | 原子聚合 | 原子类型=F (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 7 | `atom_mean_7` | 原子聚合 | 原子类型=Na (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 8 | `atom_mean_8` | 原子聚合 | 原子类型=Si (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 9 | `atom_mean_9` | 原子聚合 | 原子类型=P (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 10 | `atom_mean_10` | 原子聚合 | 原子类型=S (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 11 | `atom_mean_11` | 原子聚合 | 原子类型=Cl (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 12 | `atom_mean_12` | 原子聚合 | 原子类型=K (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 13 | `atom_mean_13` | 原子聚合 | 原子类型=Br (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 14 | `atom_mean_14` | 原子聚合 | 原子类型=I (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 15 | `atom_mean_15` | 原子聚合 | 原子类型=Au (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 16 | `atom_mean_16` | 原子聚合 | 度数=0 (度数 one-hot: 0, 1, 2, 3, 4, ≥5) |
| 17 | `atom_mean_17` | 原子聚合 | 度数=1 (度数 one-hot: 0, 1, 2, 3, 4, ≥5) |
| 18 | `atom_mean_18` | 原子聚合 | 度数=2 (度数 one-hot: 0, 1, 2, 3, 4, ≥5) |
| 19 | `atom_mean_19` | 原子聚合 | 度数=3 (度数 one-hot: 0, 1, 2, 3, 4, ≥5) |
| 20 | `atom_mean_20` | 原子聚合 | 度数=4 (度数 one-hot: 0, 1, 2, 3, 4, ≥5) |
| 21 | `atom_mean_21` | 原子聚合 | 度数=5+ (度数 one-hot: 0, 1, 2, 3, 4, ≥5) |
| 22 | `atom_mean_22` | 原子聚合 | formal_charge: 形式电荷 clip[-2,2]/2 → [-1,1] |
| 23 | `atom_mean_23` | 原子聚合 | 隐氢数=0 (隐氢数 one-hot: 0, 1, 2, 3, ≥4) |
| 24 | `atom_mean_24` | 原子聚合 | 隐氢数=1 (隐氢数 one-hot: 0, 1, 2, 3, ≥4) |
| 25 | `atom_mean_25` | 原子聚合 | 隐氢数=2 (隐氢数 one-hot: 0, 1, 2, 3, ≥4) |
| 26 | `atom_mean_26` | 原子聚合 | 隐氢数=3 (隐氢数 one-hot: 0, 1, 2, 3, ≥4) |
| 27 | `atom_mean_27` | 原子聚合 | 隐氢数=4+ (隐氢数 one-hot: 0, 1, 2, 3, ≥4) |
| 28 | `atom_mean_28` | 原子聚合 | 杂化=SP (杂化方式 one-hot: SP, SP2, SP3, SP3D, SP3D2) |
| 29 | `atom_mean_29` | 原子聚合 | 杂化=SP2 (杂化方式 one-hot: SP, SP2, SP3, SP3D, SP3D2) |
| 30 | `atom_mean_30` | 原子聚合 | 杂化=SP3 (杂化方式 one-hot: SP, SP2, SP3, SP3D, SP3D2) |
| 31 | `atom_mean_31` | 原子聚合 | 杂化=SP3D (杂化方式 one-hot: SP, SP2, SP3, SP3D, SP3D2) |
| 32 | `atom_mean_32` | 原子聚合 | 杂化=SP3D2 (杂化方式 one-hot: SP, SP2, SP3, SP3D, SP3D2) |
| 33 | `atom_mean_33` | 原子聚合 | is_aromatic: 是否芳香原子 |
| 34 | `atom_mean_34` | 原子聚合 | in_ring: 是否在环中 |
| 35 | `atom_mean_35` | 原子聚合 | mass_div_100: 原子质量 / 100 |
| 36 | `atom_mean_36` | 原子聚合 | is_chiral: 是否手性中心 |
| 37 | `atom_mean_37` | 原子聚合 | radical_electrons: 自由基电子数 / 2 |
| 38 | `atom_mean_38` | 原子聚合 | 显价=1 (显价 one-hot: 1, 2, 3, ≥4) |
| 39 | `atom_mean_39` | 原子聚合 | 显价=2 (显价 one-hot: 1, 2, 3, ≥4) |
| 40 | `atom_mean_40` | 原子聚合 | 显价=3 (显价 one-hot: 1, 2, 3, ≥4) |
| 41 | `atom_mean_41` | 原子聚合 | 显价=4+ (显价 one-hot: 1, 2, 3, ≥4) |
| 42 | `atom_mean_42` | 原子聚合 | 3 元环 (环大小 one-hot: 3, 4, 5, 6 元环) |
| 43 | `atom_mean_43` | 原子聚合 | 4 元环 (环大小 one-hot: 3, 4, 5, 6 元环) |
| 44 | `atom_mean_44` | 原子聚合 | 5 元环 (环大小 one-hot: 3, 4, 5, 6 元环) |
| 45 | `atom_mean_45` | 原子聚合 | 6 元环 (环大小 one-hot: 3, 4, 5, 6 元环) |
| 46 | `atom_mean_46` | 原子聚合 | Gasteiger 电荷 [-1,-0.5) (Gasteiger 电荷分桶: [-1,-0.5), [-0.5,0), [0,0.5), [0.5,1]) |
| 47 | `atom_mean_47` | 原子聚合 | Gasteiger 电荷 [-0.5,0) (Gasteiger 电荷分桶: [-1,-0.5), [-0.5,0), [0,0.5), [0.5,1]) |
| 48 | `atom_mean_48` | 原子聚合 | Gasteiger 电荷 [0,0.5) (Gasteiger 电荷分桶: [-1,-0.5), [-0.5,0), [0,0.5), [0.5,1]) |
| 49 | `atom_mean_49` | 原子聚合 | Gasteiger 电荷 [0.5,1] (Gasteiger 电荷分桶: [-1,-0.5), [-0.5,0), [0,0.5), [0.5,1]) |
| 50 | `atom_mean_50` | 原子聚合 | ring_ge7: 是否在 ≥7 元环中 |
| 51 | `atom_mean_51` | 原子聚合 | is_N_or_O: 是否为 N 或 O 原子 |
| 52 | `atom_mean_52` | 原子聚合 | is_h_donor: 是否为 H 键供体 (N/O + H) |
| 53 | `atom_mean_53` | 原子聚合 | is_h_acceptor: 是否为 H 键受体 (N/O) |
| 54 | `atom_mean_54` | 原子聚合 | heavy_neighbors_div4: 重原子邻居数 / 4 |
| 55 | `atom_std_0` | 原子聚合 | 原子类型=H (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 56 | `atom_std_1` | 原子聚合 | 原子类型=Li (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 57 | `atom_std_2` | 原子聚合 | 原子类型=B (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 58 | `atom_std_3` | 原子聚合 | 原子类型=C (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 59 | `atom_std_4` | 原子聚合 | 原子类型=N (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 60 | `atom_std_5` | 原子聚合 | 原子类型=O (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 61 | `atom_std_6` | 原子聚合 | 原子类型=F (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 62 | `atom_std_7` | 原子聚合 | 原子类型=Na (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 63 | `atom_std_8` | 原子聚合 | 原子类型=Si (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 64 | `atom_std_9` | 原子聚合 | 原子类型=P (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 65 | `atom_std_10` | 原子聚合 | 原子类型=S (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 66 | `atom_std_11` | 原子聚合 | 原子类型=Cl (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 67 | `atom_std_12` | 原子聚合 | 原子类型=K (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 68 | `atom_std_13` | 原子聚合 | 原子类型=Br (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 69 | `atom_std_14` | 原子聚合 | 原子类型=I (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 70 | `atom_std_15` | 原子聚合 | 原子类型=Au (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 71 | `atom_std_16` | 原子聚合 | 度数=0 (度数 one-hot: 0, 1, 2, 3, 4, ≥5) |
| 72 | `atom_std_17` | 原子聚合 | 度数=1 (度数 one-hot: 0, 1, 2, 3, 4, ≥5) |
| 73 | `atom_std_18` | 原子聚合 | 度数=2 (度数 one-hot: 0, 1, 2, 3, 4, ≥5) |
| 74 | `atom_std_19` | 原子聚合 | 度数=3 (度数 one-hot: 0, 1, 2, 3, 4, ≥5) |
| 75 | `atom_std_20` | 原子聚合 | 度数=4 (度数 one-hot: 0, 1, 2, 3, 4, ≥5) |
| 76 | `atom_std_21` | 原子聚合 | 度数=5+ (度数 one-hot: 0, 1, 2, 3, 4, ≥5) |
| 77 | `atom_std_22` | 原子聚合 | formal_charge: 形式电荷 clip[-2,2]/2 → [-1,1] |
| 78 | `atom_std_23` | 原子聚合 | 隐氢数=0 (隐氢数 one-hot: 0, 1, 2, 3, ≥4) |
| 79 | `atom_std_24` | 原子聚合 | 隐氢数=1 (隐氢数 one-hot: 0, 1, 2, 3, ≥4) |
| 80 | `atom_std_25` | 原子聚合 | 隐氢数=2 (隐氢数 one-hot: 0, 1, 2, 3, ≥4) |
| 81 | `atom_std_26` | 原子聚合 | 隐氢数=3 (隐氢数 one-hot: 0, 1, 2, 3, ≥4) |
| 82 | `atom_std_27` | 原子聚合 | 隐氢数=4+ (隐氢数 one-hot: 0, 1, 2, 3, ≥4) |
| 83 | `atom_std_28` | 原子聚合 | 杂化=SP (杂化方式 one-hot: SP, SP2, SP3, SP3D, SP3D2) |
| 84 | `atom_std_29` | 原子聚合 | 杂化=SP2 (杂化方式 one-hot: SP, SP2, SP3, SP3D, SP3D2) |
| 85 | `atom_std_30` | 原子聚合 | 杂化=SP3 (杂化方式 one-hot: SP, SP2, SP3, SP3D, SP3D2) |
| 86 | `atom_std_31` | 原子聚合 | 杂化=SP3D (杂化方式 one-hot: SP, SP2, SP3, SP3D, SP3D2) |
| 87 | `atom_std_32` | 原子聚合 | 杂化=SP3D2 (杂化方式 one-hot: SP, SP2, SP3, SP3D, SP3D2) |
| 88 | `atom_std_33` | 原子聚合 | is_aromatic: 是否芳香原子 |
| 89 | `atom_std_34` | 原子聚合 | in_ring: 是否在环中 |
| 90 | `atom_std_35` | 原子聚合 | mass_div_100: 原子质量 / 100 |
| 91 | `atom_std_36` | 原子聚合 | is_chiral: 是否手性中心 |
| 92 | `atom_std_37` | 原子聚合 | radical_electrons: 自由基电子数 / 2 |
| 93 | `atom_std_38` | 原子聚合 | 显价=1 (显价 one-hot: 1, 2, 3, ≥4) |
| 94 | `atom_std_39` | 原子聚合 | 显价=2 (显价 one-hot: 1, 2, 3, ≥4) |
| 95 | `atom_std_40` | 原子聚合 | 显价=3 (显价 one-hot: 1, 2, 3, ≥4) |
| 96 | `atom_std_41` | 原子聚合 | 显价=4+ (显价 one-hot: 1, 2, 3, ≥4) |
| 97 | `atom_std_42` | 原子聚合 | 3 元环 (环大小 one-hot: 3, 4, 5, 6 元环) |
| 98 | `atom_std_43` | 原子聚合 | 4 元环 (环大小 one-hot: 3, 4, 5, 6 元环) |
| 99 | `atom_std_44` | 原子聚合 | 5 元环 (环大小 one-hot: 3, 4, 5, 6 元环) |
| 100 | `atom_std_45` | 原子聚合 | 6 元环 (环大小 one-hot: 3, 4, 5, 6 元环) |
| 101 | `atom_std_46` | 原子聚合 | Gasteiger 电荷 [-1,-0.5) (Gasteiger 电荷分桶: [-1,-0.5), [-0.5,0), [0,0.5), [0.5,1]) |
| 102 | `atom_std_47` | 原子聚合 | Gasteiger 电荷 [-0.5,0) (Gasteiger 电荷分桶: [-1,-0.5), [-0.5,0), [0,0.5), [0.5,1]) |
| 103 | `atom_std_48` | 原子聚合 | Gasteiger 电荷 [0,0.5) (Gasteiger 电荷分桶: [-1,-0.5), [-0.5,0), [0,0.5), [0.5,1]) |
| 104 | `atom_std_49` | 原子聚合 | Gasteiger 电荷 [0.5,1] (Gasteiger 电荷分桶: [-1,-0.5), [-0.5,0), [0,0.5), [0.5,1]) |
| 105 | `atom_std_50` | 原子聚合 | ring_ge7: 是否在 ≥7 元环中 |
| 106 | `atom_std_51` | 原子聚合 | is_N_or_O: 是否为 N 或 O 原子 |
| 107 | `atom_std_52` | 原子聚合 | is_h_donor: 是否为 H 键供体 (N/O + H) |
| 108 | `atom_std_53` | 原子聚合 | is_h_acceptor: 是否为 H 键受体 (N/O) |
| 109 | `atom_std_54` | 原子聚合 | heavy_neighbors_div4: 重原子邻居数 / 4 |
| 110 | `atom_min_0` | 原子聚合 | 原子类型=H (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 111 | `atom_min_1` | 原子聚合 | 原子类型=Li (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 112 | `atom_min_2` | 原子聚合 | 原子类型=B (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 113 | `atom_min_3` | 原子聚合 | 原子类型=C (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 114 | `atom_min_4` | 原子聚合 | 原子类型=N (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 115 | `atom_min_5` | 原子聚合 | 原子类型=O (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 116 | `atom_min_6` | 原子聚合 | 原子类型=F (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 117 | `atom_min_7` | 原子聚合 | 原子类型=Na (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 118 | `atom_min_8` | 原子聚合 | 原子类型=Si (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 119 | `atom_min_9` | 原子聚合 | 原子类型=P (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 120 | `atom_min_10` | 原子聚合 | 原子类型=S (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 121 | `atom_min_11` | 原子聚合 | 原子类型=Cl (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 122 | `atom_min_12` | 原子聚合 | 原子类型=K (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 123 | `atom_min_13` | 原子聚合 | 原子类型=Br (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 124 | `atom_min_14` | 原子聚合 | 原子类型=I (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 125 | `atom_min_15` | 原子聚合 | 原子类型=Au (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 126 | `atom_min_16` | 原子聚合 | 度数=0 (度数 one-hot: 0, 1, 2, 3, 4, ≥5) |
| 127 | `atom_min_17` | 原子聚合 | 度数=1 (度数 one-hot: 0, 1, 2, 3, 4, ≥5) |
| 128 | `atom_min_18` | 原子聚合 | 度数=2 (度数 one-hot: 0, 1, 2, 3, 4, ≥5) |
| 129 | `atom_min_19` | 原子聚合 | 度数=3 (度数 one-hot: 0, 1, 2, 3, 4, ≥5) |
| 130 | `atom_min_20` | 原子聚合 | 度数=4 (度数 one-hot: 0, 1, 2, 3, 4, ≥5) |
| 131 | `atom_min_21` | 原子聚合 | 度数=5+ (度数 one-hot: 0, 1, 2, 3, 4, ≥5) |
| 132 | `atom_min_22` | 原子聚合 | formal_charge: 形式电荷 clip[-2,2]/2 → [-1,1] |
| 133 | `atom_min_23` | 原子聚合 | 隐氢数=0 (隐氢数 one-hot: 0, 1, 2, 3, ≥4) |
| 134 | `atom_min_24` | 原子聚合 | 隐氢数=1 (隐氢数 one-hot: 0, 1, 2, 3, ≥4) |
| 135 | `atom_min_25` | 原子聚合 | 隐氢数=2 (隐氢数 one-hot: 0, 1, 2, 3, ≥4) |
| 136 | `atom_min_26` | 原子聚合 | 隐氢数=3 (隐氢数 one-hot: 0, 1, 2, 3, ≥4) |
| 137 | `atom_min_27` | 原子聚合 | 隐氢数=4+ (隐氢数 one-hot: 0, 1, 2, 3, ≥4) |
| 138 | `atom_min_28` | 原子聚合 | 杂化=SP (杂化方式 one-hot: SP, SP2, SP3, SP3D, SP3D2) |
| 139 | `atom_min_29` | 原子聚合 | 杂化=SP2 (杂化方式 one-hot: SP, SP2, SP3, SP3D, SP3D2) |
| 140 | `atom_min_30` | 原子聚合 | 杂化=SP3 (杂化方式 one-hot: SP, SP2, SP3, SP3D, SP3D2) |
| 141 | `atom_min_31` | 原子聚合 | 杂化=SP3D (杂化方式 one-hot: SP, SP2, SP3, SP3D, SP3D2) |
| 142 | `atom_min_32` | 原子聚合 | 杂化=SP3D2 (杂化方式 one-hot: SP, SP2, SP3, SP3D, SP3D2) |
| 143 | `atom_min_33` | 原子聚合 | is_aromatic: 是否芳香原子 |
| 144 | `atom_min_34` | 原子聚合 | in_ring: 是否在环中 |
| 145 | `atom_min_35` | 原子聚合 | mass_div_100: 原子质量 / 100 |
| 146 | `atom_min_36` | 原子聚合 | is_chiral: 是否手性中心 |
| 147 | `atom_min_37` | 原子聚合 | radical_electrons: 自由基电子数 / 2 |
| 148 | `atom_min_38` | 原子聚合 | 显价=1 (显价 one-hot: 1, 2, 3, ≥4) |
| 149 | `atom_min_39` | 原子聚合 | 显价=2 (显价 one-hot: 1, 2, 3, ≥4) |
| 150 | `atom_min_40` | 原子聚合 | 显价=3 (显价 one-hot: 1, 2, 3, ≥4) |
| 151 | `atom_min_41` | 原子聚合 | 显价=4+ (显价 one-hot: 1, 2, 3, ≥4) |
| 152 | `atom_min_42` | 原子聚合 | 3 元环 (环大小 one-hot: 3, 4, 5, 6 元环) |
| 153 | `atom_min_43` | 原子聚合 | 4 元环 (环大小 one-hot: 3, 4, 5, 6 元环) |
| 154 | `atom_min_44` | 原子聚合 | 5 元环 (环大小 one-hot: 3, 4, 5, 6 元环) |
| 155 | `atom_min_45` | 原子聚合 | 6 元环 (环大小 one-hot: 3, 4, 5, 6 元环) |
| 156 | `atom_min_46` | 原子聚合 | Gasteiger 电荷 [-1,-0.5) (Gasteiger 电荷分桶: [-1,-0.5), [-0.5,0), [0,0.5), [0.5,1]) |
| 157 | `atom_min_47` | 原子聚合 | Gasteiger 电荷 [-0.5,0) (Gasteiger 电荷分桶: [-1,-0.5), [-0.5,0), [0,0.5), [0.5,1]) |
| 158 | `atom_min_48` | 原子聚合 | Gasteiger 电荷 [0,0.5) (Gasteiger 电荷分桶: [-1,-0.5), [-0.5,0), [0,0.5), [0.5,1]) |
| 159 | `atom_min_49` | 原子聚合 | Gasteiger 电荷 [0.5,1] (Gasteiger 电荷分桶: [-1,-0.5), [-0.5,0), [0,0.5), [0.5,1]) |
| 160 | `atom_min_50` | 原子聚合 | ring_ge7: 是否在 ≥7 元环中 |
| 161 | `atom_min_51` | 原子聚合 | is_N_or_O: 是否为 N 或 O 原子 |
| 162 | `atom_min_52` | 原子聚合 | is_h_donor: 是否为 H 键供体 (N/O + H) |
| 163 | `atom_min_53` | 原子聚合 | is_h_acceptor: 是否为 H 键受体 (N/O) |
| 164 | `atom_min_54` | 原子聚合 | heavy_neighbors_div4: 重原子邻居数 / 4 |
| 165 | `atom_max_0` | 原子聚合 | 原子类型=H (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 166 | `atom_max_1` | 原子聚合 | 原子类型=Li (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 167 | `atom_max_2` | 原子聚合 | 原子类型=B (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 168 | `atom_max_3` | 原子聚合 | 原子类型=C (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 169 | `atom_max_4` | 原子聚合 | 原子类型=N (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 170 | `atom_max_5` | 原子聚合 | 原子类型=O (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 171 | `atom_max_6` | 原子聚合 | 原子类型=F (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 172 | `atom_max_7` | 原子聚合 | 原子类型=Na (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 173 | `atom_max_8` | 原子聚合 | 原子类型=Si (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 174 | `atom_max_9` | 原子聚合 | 原子类型=P (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 175 | `atom_max_10` | 原子聚合 | 原子类型=S (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 176 | `atom_max_11` | 原子聚合 | 原子类型=Cl (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 177 | `atom_max_12` | 原子聚合 | 原子类型=K (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 178 | `atom_max_13` | 原子聚合 | 原子类型=Br (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 179 | `atom_max_14` | 原子聚合 | 原子类型=I (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 180 | `atom_max_15` | 原子聚合 | 原子类型=Au (原子序数 one-hot: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Na(11), Si(14), P(15), S(16), Cl(17), K(19), Br(35), I(53), Au(79)) |
| 181 | `atom_max_16` | 原子聚合 | 度数=0 (度数 one-hot: 0, 1, 2, 3, 4, ≥5) |
| 182 | `atom_max_17` | 原子聚合 | 度数=1 (度数 one-hot: 0, 1, 2, 3, 4, ≥5) |
| 183 | `atom_max_18` | 原子聚合 | 度数=2 (度数 one-hot: 0, 1, 2, 3, 4, ≥5) |
| 184 | `atom_max_19` | 原子聚合 | 度数=3 (度数 one-hot: 0, 1, 2, 3, 4, ≥5) |
| 185 | `atom_max_20` | 原子聚合 | 度数=4 (度数 one-hot: 0, 1, 2, 3, 4, ≥5) |
| 186 | `atom_max_21` | 原子聚合 | 度数=5+ (度数 one-hot: 0, 1, 2, 3, 4, ≥5) |
| 187 | `atom_max_22` | 原子聚合 | formal_charge: 形式电荷 clip[-2,2]/2 → [-1,1] |
| 188 | `atom_max_23` | 原子聚合 | 隐氢数=0 (隐氢数 one-hot: 0, 1, 2, 3, ≥4) |
| 189 | `atom_max_24` | 原子聚合 | 隐氢数=1 (隐氢数 one-hot: 0, 1, 2, 3, ≥4) |
| 190 | `atom_max_25` | 原子聚合 | 隐氢数=2 (隐氢数 one-hot: 0, 1, 2, 3, ≥4) |
| 191 | `atom_max_26` | 原子聚合 | 隐氢数=3 (隐氢数 one-hot: 0, 1, 2, 3, ≥4) |
| 192 | `atom_max_27` | 原子聚合 | 隐氢数=4+ (隐氢数 one-hot: 0, 1, 2, 3, ≥4) |
| 193 | `atom_max_28` | 原子聚合 | 杂化=SP (杂化方式 one-hot: SP, SP2, SP3, SP3D, SP3D2) |
| 194 | `atom_max_29` | 原子聚合 | 杂化=SP2 (杂化方式 one-hot: SP, SP2, SP3, SP3D, SP3D2) |
| 195 | `atom_max_30` | 原子聚合 | 杂化=SP3 (杂化方式 one-hot: SP, SP2, SP3, SP3D, SP3D2) |
| 196 | `atom_max_31` | 原子聚合 | 杂化=SP3D (杂化方式 one-hot: SP, SP2, SP3, SP3D, SP3D2) |
| 197 | `atom_max_32` | 原子聚合 | 杂化=SP3D2 (杂化方式 one-hot: SP, SP2, SP3, SP3D, SP3D2) |
| 198 | `atom_max_33` | 原子聚合 | is_aromatic: 是否芳香原子 |
| 199 | `atom_max_34` | 原子聚合 | in_ring: 是否在环中 |
| 200 | `atom_max_35` | 原子聚合 | mass_div_100: 原子质量 / 100 |
| 201 | `atom_max_36` | 原子聚合 | is_chiral: 是否手性中心 |
| 202 | `atom_max_37` | 原子聚合 | radical_electrons: 自由基电子数 / 2 |
| 203 | `atom_max_38` | 原子聚合 | 显价=1 (显价 one-hot: 1, 2, 3, ≥4) |
| 204 | `atom_max_39` | 原子聚合 | 显价=2 (显价 one-hot: 1, 2, 3, ≥4) |
| 205 | `atom_max_40` | 原子聚合 | 显价=3 (显价 one-hot: 1, 2, 3, ≥4) |
| 206 | `atom_max_41` | 原子聚合 | 显价=4+ (显价 one-hot: 1, 2, 3, ≥4) |
| 207 | `atom_max_42` | 原子聚合 | 3 元环 (环大小 one-hot: 3, 4, 5, 6 元环) |
| 208 | `atom_max_43` | 原子聚合 | 4 元环 (环大小 one-hot: 3, 4, 5, 6 元环) |
| 209 | `atom_max_44` | 原子聚合 | 5 元环 (环大小 one-hot: 3, 4, 5, 6 元环) |
| 210 | `atom_max_45` | 原子聚合 | 6 元环 (环大小 one-hot: 3, 4, 5, 6 元环) |
| 211 | `atom_max_46` | 原子聚合 | Gasteiger 电荷 [-1,-0.5) (Gasteiger 电荷分桶: [-1,-0.5), [-0.5,0), [0,0.5), [0.5,1]) |
| 212 | `atom_max_47` | 原子聚合 | Gasteiger 电荷 [-0.5,0) (Gasteiger 电荷分桶: [-1,-0.5), [-0.5,0), [0,0.5), [0.5,1]) |
| 213 | `atom_max_48` | 原子聚合 | Gasteiger 电荷 [0,0.5) (Gasteiger 电荷分桶: [-1,-0.5), [-0.5,0), [0,0.5), [0.5,1]) |
| 214 | `atom_max_49` | 原子聚合 | Gasteiger 电荷 [0.5,1] (Gasteiger 电荷分桶: [-1,-0.5), [-0.5,0), [0,0.5), [0.5,1]) |
| 215 | `atom_max_50` | 原子聚合 | ring_ge7: 是否在 ≥7 元环中 |
| 216 | `atom_max_51` | 原子聚合 | is_N_or_O: 是否为 N 或 O 原子 |
| 217 | `atom_max_52` | 原子聚合 | is_h_donor: 是否为 H 键供体 (N/O + H) |
| 218 | `atom_max_53` | 原子聚合 | is_h_acceptor: 是否为 H 键受体 (N/O) |
| 219 | `atom_max_54` | 原子聚合 | heavy_neighbors_div4: 重原子邻居数 / 4 |
| 220 | `bond_mean_0` | 键聚合 | 键型=SINGLE (键型 one-hot: SINGLE, DOUBLE, TRIPLE, AROMATIC) |
| 221 | `bond_mean_1` | 键聚合 | 键型=DOUBLE (键型 one-hot: SINGLE, DOUBLE, TRIPLE, AROMATIC) |
| 222 | `bond_mean_2` | 键聚合 | 键型=TRIPLE (键型 one-hot: SINGLE, DOUBLE, TRIPLE, AROMATIC) |
| 223 | `bond_mean_3` | 键聚合 | 键型=AROMATIC (键型 one-hot: SINGLE, DOUBLE, TRIPLE, AROMATIC) |
| 224 | `bond_mean_4` | 键聚合 | is_conjugated: 是否共轭 |
| 225 | `bond_mean_5` | 键聚合 | in_ring: 是否在环中 |
| 226 | `bond_mean_6` | 键聚合 | 立体=NONE (立体化学 one-hot: NONE, ANY, Z, E, CIS, TRANS) |
| 227 | `bond_mean_7` | 键聚合 | 立体=ANY (立体化学 one-hot: NONE, ANY, Z, E, CIS, TRANS) |
| 228 | `bond_mean_8` | 键聚合 | 立体=Z (立体化学 one-hot: NONE, ANY, Z, E, CIS, TRANS) |
| 229 | `bond_mean_9` | 键聚合 | 立体=E (立体化学 one-hot: NONE, ANY, Z, E, CIS, TRANS) |
| 230 | `bond_mean_10` | 键聚合 | 立体=CIS (立体化学 one-hot: NONE, ANY, Z, E, CIS, TRANS) |
| 231 | `bond_mean_11` | 键聚合 | 立体=TRANS (立体化学 one-hot: NONE, ANY, Z, E, CIS, TRANS) |
| 232 | `bond_mean_12` | 键聚合 | is_aromatic: 是否芳香键 |
| 233 | `bond_mean_13` | 键聚合 | in_ring_dup: 是否在环中（冗余） |
| 234 | `bond_std_0` | 键聚合 | 键型=SINGLE (键型 one-hot: SINGLE, DOUBLE, TRIPLE, AROMATIC) |
| 235 | `bond_std_1` | 键聚合 | 键型=DOUBLE (键型 one-hot: SINGLE, DOUBLE, TRIPLE, AROMATIC) |
| 236 | `bond_std_2` | 键聚合 | 键型=TRIPLE (键型 one-hot: SINGLE, DOUBLE, TRIPLE, AROMATIC) |
| 237 | `bond_std_3` | 键聚合 | 键型=AROMATIC (键型 one-hot: SINGLE, DOUBLE, TRIPLE, AROMATIC) |
| 238 | `bond_std_4` | 键聚合 | is_conjugated: 是否共轭 |
| 239 | `bond_std_5` | 键聚合 | in_ring: 是否在环中 |
| 240 | `bond_std_6` | 键聚合 | 立体=NONE (立体化学 one-hot: NONE, ANY, Z, E, CIS, TRANS) |
| 241 | `bond_std_7` | 键聚合 | 立体=ANY (立体化学 one-hot: NONE, ANY, Z, E, CIS, TRANS) |
| 242 | `bond_std_8` | 键聚合 | 立体=Z (立体化学 one-hot: NONE, ANY, Z, E, CIS, TRANS) |
| 243 | `bond_std_9` | 键聚合 | 立体=E (立体化学 one-hot: NONE, ANY, Z, E, CIS, TRANS) |
| 244 | `bond_std_10` | 键聚合 | 立体=CIS (立体化学 one-hot: NONE, ANY, Z, E, CIS, TRANS) |
| 245 | `bond_std_11` | 键聚合 | 立体=TRANS (立体化学 one-hot: NONE, ANY, Z, E, CIS, TRANS) |
| 246 | `bond_std_12` | 键聚合 | is_aromatic: 是否芳香键 |
| 247 | `bond_std_13` | 键聚合 | in_ring_dup: 是否在环中（冗余） |
| 248 | `bond_min_0` | 键聚合 | 键型=SINGLE (键型 one-hot: SINGLE, DOUBLE, TRIPLE, AROMATIC) |
| 249 | `bond_min_1` | 键聚合 | 键型=DOUBLE (键型 one-hot: SINGLE, DOUBLE, TRIPLE, AROMATIC) |
| 250 | `bond_min_2` | 键聚合 | 键型=TRIPLE (键型 one-hot: SINGLE, DOUBLE, TRIPLE, AROMATIC) |
| 251 | `bond_min_3` | 键聚合 | 键型=AROMATIC (键型 one-hot: SINGLE, DOUBLE, TRIPLE, AROMATIC) |
| 252 | `bond_min_4` | 键聚合 | is_conjugated: 是否共轭 |
| 253 | `bond_min_5` | 键聚合 | in_ring: 是否在环中 |
| 254 | `bond_min_6` | 键聚合 | 立体=NONE (立体化学 one-hot: NONE, ANY, Z, E, CIS, TRANS) |
| 255 | `bond_min_7` | 键聚合 | 立体=ANY (立体化学 one-hot: NONE, ANY, Z, E, CIS, TRANS) |
| 256 | `bond_min_8` | 键聚合 | 立体=Z (立体化学 one-hot: NONE, ANY, Z, E, CIS, TRANS) |
| 257 | `bond_min_9` | 键聚合 | 立体=E (立体化学 one-hot: NONE, ANY, Z, E, CIS, TRANS) |
| 258 | `bond_min_10` | 键聚合 | 立体=CIS (立体化学 one-hot: NONE, ANY, Z, E, CIS, TRANS) |
| 259 | `bond_min_11` | 键聚合 | 立体=TRANS (立体化学 one-hot: NONE, ANY, Z, E, CIS, TRANS) |
| 260 | `bond_min_12` | 键聚合 | is_aromatic: 是否芳香键 |
| 261 | `bond_min_13` | 键聚合 | in_ring_dup: 是否在环中（冗余） |
| 262 | `bond_max_0` | 键聚合 | 键型=SINGLE (键型 one-hot: SINGLE, DOUBLE, TRIPLE, AROMATIC) |
| 263 | `bond_max_1` | 键聚合 | 键型=DOUBLE (键型 one-hot: SINGLE, DOUBLE, TRIPLE, AROMATIC) |
| 264 | `bond_max_2` | 键聚合 | 键型=TRIPLE (键型 one-hot: SINGLE, DOUBLE, TRIPLE, AROMATIC) |
| 265 | `bond_max_3` | 键聚合 | 键型=AROMATIC (键型 one-hot: SINGLE, DOUBLE, TRIPLE, AROMATIC) |
| 266 | `bond_max_4` | 键聚合 | is_conjugated: 是否共轭 |
| 267 | `bond_max_5` | 键聚合 | in_ring: 是否在环中 |
| 268 | `bond_max_6` | 键聚合 | 立体=NONE (立体化学 one-hot: NONE, ANY, Z, E, CIS, TRANS) |
| 269 | `bond_max_7` | 键聚合 | 立体=ANY (立体化学 one-hot: NONE, ANY, Z, E, CIS, TRANS) |
| 270 | `bond_max_8` | 键聚合 | 立体=Z (立体化学 one-hot: NONE, ANY, Z, E, CIS, TRANS) |
| 271 | `bond_max_9` | 键聚合 | 立体=E (立体化学 one-hot: NONE, ANY, Z, E, CIS, TRANS) |
| 272 | `bond_max_10` | 键聚合 | 立体=CIS (立体化学 one-hot: NONE, ANY, Z, E, CIS, TRANS) |
| 273 | `bond_max_11` | 键聚合 | 立体=TRANS (立体化学 one-hot: NONE, ANY, Z, E, CIS, TRANS) |
| 274 | `bond_max_12` | 键聚合 | is_aromatic: 是否芳香键 |
| 275 | `bond_max_13` | 键聚合 | in_ring_dup: 是否在环中（冗余） |
| 276 | `maccs_0` | MACCS 指纹 | MACCS 结构键 bit 0 |
| 277 | `maccs_1` | MACCS 指纹 | MACCS 结构键 bit 1 |
| 278 | `maccs_2` | MACCS 指纹 | MACCS 结构键 bit 2 |
| 279 | `maccs_3` | MACCS 指纹 | MACCS 结构键 bit 3 |
| 280 | `maccs_4` | MACCS 指纹 | MACCS 结构键 bit 4 |
| 281 | `maccs_5` | MACCS 指纹 | MACCS 结构键 bit 5 |
| 282 | `maccs_6` | MACCS 指纹 | MACCS 结构键 bit 6 |
| 283 | `maccs_7` | MACCS 指纹 | MACCS 结构键 bit 7 |
| 284 | `maccs_8` | MACCS 指纹 | MACCS 结构键 bit 8 |
| 285 | `maccs_9` | MACCS 指纹 | MACCS 结构键 bit 9 |
| 286 | `maccs_10` | MACCS 指纹 | MACCS 结构键 bit 10 |
| 287 | `maccs_11` | MACCS 指纹 | MACCS 结构键 bit 11 |
| 288 | `maccs_12` | MACCS 指纹 | MACCS 结构键 bit 12 |
| 289 | `maccs_13` | MACCS 指纹 | MACCS 结构键 bit 13 |
| 290 | `maccs_14` | MACCS 指纹 | MACCS 结构键 bit 14 |
| 291 | `maccs_15` | MACCS 指纹 | MACCS 结构键 bit 15 |
| 292 | `maccs_16` | MACCS 指纹 | MACCS 结构键 bit 16 |
| 293 | `maccs_17` | MACCS 指纹 | MACCS 结构键 bit 17 |
| 294 | `maccs_18` | MACCS 指纹 | MACCS 结构键 bit 18 |
| 295 | `maccs_19` | MACCS 指纹 | MACCS 结构键 bit 19 |
| 296 | `maccs_20` | MACCS 指纹 | MACCS 结构键 bit 20 |
| 297 | `maccs_21` | MACCS 指纹 | MACCS 结构键 bit 21 |
| 298 | `maccs_22` | MACCS 指纹 | MACCS 结构键 bit 22 |
| 299 | `maccs_23` | MACCS 指纹 | MACCS 结构键 bit 23 |
| 300 | `maccs_24` | MACCS 指纹 | MACCS 结构键 bit 24 |
| 301 | `maccs_25` | MACCS 指纹 | MACCS 结构键 bit 25 |
| 302 | `maccs_26` | MACCS 指纹 | MACCS 结构键 bit 26 |
| 303 | `maccs_27` | MACCS 指纹 | MACCS 结构键 bit 27 |
| 304 | `maccs_28` | MACCS 指纹 | MACCS 结构键 bit 28 |
| 305 | `maccs_29` | MACCS 指纹 | MACCS 结构键 bit 29 |
| 306 | `maccs_30` | MACCS 指纹 | MACCS 结构键 bit 30 |
| 307 | `maccs_31` | MACCS 指纹 | MACCS 结构键 bit 31 |
| 308 | `maccs_32` | MACCS 指纹 | MACCS 结构键 bit 32 |
| 309 | `maccs_33` | MACCS 指纹 | MACCS 结构键 bit 33 |
| 310 | `maccs_34` | MACCS 指纹 | MACCS 结构键 bit 34 |
| 311 | `maccs_35` | MACCS 指纹 | MACCS 结构键 bit 35 |
| 312 | `maccs_36` | MACCS 指纹 | MACCS 结构键 bit 36 |
| 313 | `maccs_37` | MACCS 指纹 | MACCS 结构键 bit 37 |
| 314 | `maccs_38` | MACCS 指纹 | MACCS 结构键 bit 38 |
| 315 | `maccs_39` | MACCS 指纹 | MACCS 结构键 bit 39 |
| 316 | `maccs_40` | MACCS 指纹 | MACCS 结构键 bit 40 |
| 317 | `maccs_41` | MACCS 指纹 | MACCS 结构键 bit 41 |
| 318 | `maccs_42` | MACCS 指纹 | MACCS 结构键 bit 42 |
| 319 | `maccs_43` | MACCS 指纹 | MACCS 结构键 bit 43 |
| 320 | `maccs_44` | MACCS 指纹 | MACCS 结构键 bit 44 |
| 321 | `maccs_45` | MACCS 指纹 | MACCS 结构键 bit 45 |
| 322 | `maccs_46` | MACCS 指纹 | MACCS 结构键 bit 46 |
| 323 | `maccs_47` | MACCS 指纹 | MACCS 结构键 bit 47 |
| 324 | `maccs_48` | MACCS 指纹 | MACCS 结构键 bit 48 |
| 325 | `maccs_49` | MACCS 指纹 | MACCS 结构键 bit 49 |
| 326 | `maccs_50` | MACCS 指纹 | MACCS 结构键 bit 50 |
| 327 | `maccs_51` | MACCS 指纹 | MACCS 结构键 bit 51 |
| 328 | `maccs_52` | MACCS 指纹 | MACCS 结构键 bit 52 |
| 329 | `maccs_53` | MACCS 指纹 | MACCS 结构键 bit 53 |
| 330 | `maccs_54` | MACCS 指纹 | MACCS 结构键 bit 54 |
| 331 | `maccs_55` | MACCS 指纹 | MACCS 结构键 bit 55 |
| 332 | `maccs_56` | MACCS 指纹 | MACCS 结构键 bit 56 |
| 333 | `maccs_57` | MACCS 指纹 | MACCS 结构键 bit 57 |
| 334 | `maccs_58` | MACCS 指纹 | MACCS 结构键 bit 58 |
| 335 | `maccs_59` | MACCS 指纹 | MACCS 结构键 bit 59 |
| 336 | `maccs_60` | MACCS 指纹 | MACCS 结构键 bit 60 |
| 337 | `maccs_61` | MACCS 指纹 | MACCS 结构键 bit 61 |
| 338 | `maccs_62` | MACCS 指纹 | MACCS 结构键 bit 62 |
| 339 | `maccs_63` | MACCS 指纹 | MACCS 结构键 bit 63 |
| 340 | `maccs_64` | MACCS 指纹 | MACCS 结构键 bit 64 |
| 341 | `maccs_65` | MACCS 指纹 | MACCS 结构键 bit 65 |
| 342 | `maccs_66` | MACCS 指纹 | MACCS 结构键 bit 66 |
| 343 | `maccs_67` | MACCS 指纹 | MACCS 结构键 bit 67 |
| 344 | `maccs_68` | MACCS 指纹 | MACCS 结构键 bit 68 |
| 345 | `maccs_69` | MACCS 指纹 | MACCS 结构键 bit 69 |
| 346 | `maccs_70` | MACCS 指纹 | MACCS 结构键 bit 70 |
| 347 | `maccs_71` | MACCS 指纹 | MACCS 结构键 bit 71 |
| 348 | `maccs_72` | MACCS 指纹 | MACCS 结构键 bit 72 |
| 349 | `maccs_73` | MACCS 指纹 | MACCS 结构键 bit 73 |
| 350 | `maccs_74` | MACCS 指纹 | MACCS 结构键 bit 74 |
| 351 | `maccs_75` | MACCS 指纹 | MACCS 结构键 bit 75 |
| 352 | `maccs_76` | MACCS 指纹 | MACCS 结构键 bit 76 |
| 353 | `maccs_77` | MACCS 指纹 | MACCS 结构键 bit 77 |
| 354 | `maccs_78` | MACCS 指纹 | MACCS 结构键 bit 78 |
| 355 | `maccs_79` | MACCS 指纹 | MACCS 结构键 bit 79 |
| 356 | `maccs_80` | MACCS 指纹 | MACCS 结构键 bit 80 |
| 357 | `maccs_81` | MACCS 指纹 | MACCS 结构键 bit 81 |
| 358 | `maccs_82` | MACCS 指纹 | MACCS 结构键 bit 82 |
| 359 | `maccs_83` | MACCS 指纹 | MACCS 结构键 bit 83 |
| 360 | `maccs_84` | MACCS 指纹 | MACCS 结构键 bit 84 |
| 361 | `maccs_85` | MACCS 指纹 | MACCS 结构键 bit 85 |
| 362 | `maccs_86` | MACCS 指纹 | MACCS 结构键 bit 86 |
| 363 | `maccs_87` | MACCS 指纹 | MACCS 结构键 bit 87 |
| 364 | `maccs_88` | MACCS 指纹 | MACCS 结构键 bit 88 |
| 365 | `maccs_89` | MACCS 指纹 | MACCS 结构键 bit 89 |
| 366 | `maccs_90` | MACCS 指纹 | MACCS 结构键 bit 90 |
| 367 | `maccs_91` | MACCS 指纹 | MACCS 结构键 bit 91 |
| 368 | `maccs_92` | MACCS 指纹 | MACCS 结构键 bit 92 |
| 369 | `maccs_93` | MACCS 指纹 | MACCS 结构键 bit 93 |
| 370 | `maccs_94` | MACCS 指纹 | MACCS 结构键 bit 94 |
| 371 | `maccs_95` | MACCS 指纹 | MACCS 结构键 bit 95 |
| 372 | `maccs_96` | MACCS 指纹 | MACCS 结构键 bit 96 |
| 373 | `maccs_97` | MACCS 指纹 | MACCS 结构键 bit 97 |
| 374 | `maccs_98` | MACCS 指纹 | MACCS 结构键 bit 98 |
| 375 | `maccs_99` | MACCS 指纹 | MACCS 结构键 bit 99 |
| 376 | `maccs_100` | MACCS 指纹 | MACCS 结构键 bit 100 |
| 377 | `maccs_101` | MACCS 指纹 | MACCS 结构键 bit 101 |
| 378 | `maccs_102` | MACCS 指纹 | MACCS 结构键 bit 102 |
| 379 | `maccs_103` | MACCS 指纹 | MACCS 结构键 bit 103 |
| 380 | `maccs_104` | MACCS 指纹 | MACCS 结构键 bit 104 |
| 381 | `maccs_105` | MACCS 指纹 | MACCS 结构键 bit 105 |
| 382 | `maccs_106` | MACCS 指纹 | MACCS 结构键 bit 106 |
| 383 | `maccs_107` | MACCS 指纹 | MACCS 结构键 bit 107 |
| 384 | `maccs_108` | MACCS 指纹 | MACCS 结构键 bit 108 |
| 385 | `maccs_109` | MACCS 指纹 | MACCS 结构键 bit 109 |
| 386 | `maccs_110` | MACCS 指纹 | MACCS 结构键 bit 110 |
| 387 | `maccs_111` | MACCS 指纹 | MACCS 结构键 bit 111 |
| 388 | `maccs_112` | MACCS 指纹 | MACCS 结构键 bit 112 |
| 389 | `maccs_113` | MACCS 指纹 | MACCS 结构键 bit 113 |
| 390 | `maccs_114` | MACCS 指纹 | MACCS 结构键 bit 114 |
| 391 | `maccs_115` | MACCS 指纹 | MACCS 结构键 bit 115 |
| 392 | `maccs_116` | MACCS 指纹 | MACCS 结构键 bit 116 |
| 393 | `maccs_117` | MACCS 指纹 | MACCS 结构键 bit 117 |
| 394 | `maccs_118` | MACCS 指纹 | MACCS 结构键 bit 118 |
| 395 | `maccs_119` | MACCS 指纹 | MACCS 结构键 bit 119 |
| 396 | `maccs_120` | MACCS 指纹 | MACCS 结构键 bit 120 |
| 397 | `maccs_121` | MACCS 指纹 | MACCS 结构键 bit 121 |
| 398 | `maccs_122` | MACCS 指纹 | MACCS 结构键 bit 122 |
| 399 | `maccs_123` | MACCS 指纹 | MACCS 结构键 bit 123 |
| 400 | `maccs_124` | MACCS 指纹 | MACCS 结构键 bit 124 |
| 401 | `maccs_125` | MACCS 指纹 | MACCS 结构键 bit 125 |
| 402 | `maccs_126` | MACCS 指纹 | MACCS 结构键 bit 126 |
| 403 | `maccs_127` | MACCS 指纹 | MACCS 结构键 bit 127 |
| 404 | `maccs_128` | MACCS 指纹 | MACCS 结构键 bit 128 |
| 405 | `maccs_129` | MACCS 指纹 | MACCS 结构键 bit 129 |
| 406 | `maccs_130` | MACCS 指纹 | MACCS 结构键 bit 130 |
| 407 | `maccs_131` | MACCS 指纹 | MACCS 结构键 bit 131 |
| 408 | `maccs_132` | MACCS 指纹 | MACCS 结构键 bit 132 |
| 409 | `maccs_133` | MACCS 指纹 | MACCS 结构键 bit 133 |
| 410 | `maccs_134` | MACCS 指纹 | MACCS 结构键 bit 134 |
| 411 | `maccs_135` | MACCS 指纹 | MACCS 结构键 bit 135 |
| 412 | `maccs_136` | MACCS 指纹 | MACCS 结构键 bit 136 |
| 413 | `maccs_137` | MACCS 指纹 | MACCS 结构键 bit 137 |
| 414 | `maccs_138` | MACCS 指纹 | MACCS 结构键 bit 138 |
| 415 | `maccs_139` | MACCS 指纹 | MACCS 结构键 bit 139 |
| 416 | `maccs_140` | MACCS 指纹 | MACCS 结构键 bit 140 |
| 417 | `maccs_141` | MACCS 指纹 | MACCS 结构键 bit 141 |
| 418 | `maccs_142` | MACCS 指纹 | MACCS 结构键 bit 142 |
| 419 | `maccs_143` | MACCS 指纹 | MACCS 结构键 bit 143 |
| 420 | `maccs_144` | MACCS 指纹 | MACCS 结构键 bit 144 |
| 421 | `maccs_145` | MACCS 指纹 | MACCS 结构键 bit 145 |
| 422 | `maccs_146` | MACCS 指纹 | MACCS 结构键 bit 146 |
| 423 | `maccs_147` | MACCS 指纹 | MACCS 结构键 bit 147 |
| 424 | `maccs_148` | MACCS 指纹 | MACCS 结构键 bit 148 |
| 425 | `maccs_149` | MACCS 指纹 | MACCS 结构键 bit 149 |
| 426 | `maccs_150` | MACCS 指纹 | MACCS 结构键 bit 150 |
| 427 | `maccs_151` | MACCS 指纹 | MACCS 结构键 bit 151 |
| 428 | `maccs_152` | MACCS 指纹 | MACCS 结构键 bit 152 |
| 429 | `maccs_153` | MACCS 指纹 | MACCS 结构键 bit 153 |
| 430 | `maccs_154` | MACCS 指纹 | MACCS 结构键 bit 154 |
| 431 | `maccs_155` | MACCS 指纹 | MACCS 结构键 bit 155 |
| 432 | `maccs_156` | MACCS 指纹 | MACCS 结构键 bit 156 |
| 433 | `maccs_157` | MACCS 指纹 | MACCS 结构键 bit 157 |
| 434 | `maccs_158` | MACCS 指纹 | MACCS 结构键 bit 158 |
| 435 | `maccs_159` | MACCS 指纹 | MACCS 结构键 bit 159 |
| 436 | `maccs_160` | MACCS 指纹 | MACCS 结构键 bit 160 |
| 437 | `maccs_161` | MACCS 指纹 | MACCS 结构键 bit 161 |
| 438 | `maccs_162` | MACCS 指纹 | MACCS 结构键 bit 162 |
| 439 | `maccs_163` | MACCS 指纹 | MACCS 结构键 bit 163 |
| 440 | `maccs_164` | MACCS 指纹 | MACCS 结构键 bit 164 |
| 441 | `maccs_165` | MACCS 指纹 | MACCS 结构键 bit 165 |
| 442 | `maccs_166` | MACCS 指纹 | MACCS 结构键 bit 166 |
| 443 | `maccs_167` | MACCS 指纹 | MACCS 结构键 bit 167 |
| 444 | `maccs_168` | MACCS 指纹 | MACCS 结构键 bit 168 |
| 445 | `maccs_169` | MACCS 指纹 | MACCS 结构键 bit 169 |
| 446 | `maccs_170` | MACCS 指纹 | MACCS 结构键 bit 170 |
| 447 | `maccs_171` | MACCS 指纹 | MACCS 结构键 bit 171 |
| 448 | `maccs_172` | MACCS 指纹 | MACCS 结构键 bit 172 |
| 449 | `maccs_173` | MACCS 指纹 | MACCS 结构键 bit 173 |
| 450 | `maccs_174` | MACCS 指纹 | MACCS 结构键 bit 174 |
| 451 | `maccs_175` | MACCS 指纹 | MACCS 结构键 bit 175 |
| 452 | `maccs_176` | MACCS 指纹 | MACCS 结构键 bit 176 |
| 453 | `maccs_177` | MACCS 指纹 | MACCS 结构键 bit 177 |
| 454 | `maccs_178` | MACCS 指纹 | MACCS 结构键 bit 178 |
| 455 | `maccs_179` | MACCS 指纹 | MACCS 结构键 bit 179 |
| 456 | `maccs_180` | MACCS 指纹 | MACCS 结构键 bit 180 |
| 457 | `maccs_181` | MACCS 指纹 | MACCS 结构键 bit 181 |
| 458 | `maccs_182` | MACCS 指纹 | MACCS 结构键 bit 182 |
| 459 | `maccs_183` | MACCS 指纹 | MACCS 结构键 bit 183 |
| 460 | `maccs_184` | MACCS 指纹 | MACCS 结构键 bit 184 |
| 461 | `maccs_185` | MACCS 指纹 | MACCS 结构键 bit 185 |
| 462 | `maccs_186` | MACCS 指纹 | MACCS 结构键 bit 186 |
| 463 | `maccs_187` | MACCS 指纹 | MACCS 结构键 bit 187 |
| 464 | `maccs_188` | MACCS 指纹 | MACCS 结构键 bit 188 |
| 465 | `maccs_189` | MACCS 指纹 | MACCS 结构键 bit 189 |
| 466 | `maccs_190` | MACCS 指纹 | MACCS 结构键 bit 190 |
| 467 | `maccs_191` | MACCS 指纹 | MACCS 结构键 bit 191 |
| 468 | `maccs_192` | MACCS 指纹 | MACCS 结构键 bit 192 |
| 469 | `maccs_193` | MACCS 指纹 | MACCS 结构键 bit 193 |
| 470 | `brics_0` | BRICS 碎片 | BRICS 碎片 CRC32 hash → bucket 0（归一化频次） |
| 471 | `brics_1` | BRICS 碎片 | BRICS 碎片 CRC32 hash → bucket 1（归一化频次） |
| 472 | `brics_2` | BRICS 碎片 | BRICS 碎片 CRC32 hash → bucket 2（归一化频次） |
| 473 | `brics_3` | BRICS 碎片 | BRICS 碎片 CRC32 hash → bucket 3（归一化频次） |
| 474 | `brics_4` | BRICS 碎片 | BRICS 碎片 CRC32 hash → bucket 4（归一化频次） |
| 475 | `brics_5` | BRICS 碎片 | BRICS 碎片 CRC32 hash → bucket 5（归一化频次） |
| 476 | `brics_6` | BRICS 碎片 | BRICS 碎片 CRC32 hash → bucket 6（归一化频次） |
| 477 | `brics_7` | BRICS 碎片 | BRICS 碎片 CRC32 hash → bucket 7（归一化频次） |
| 478 | `brics_8` | BRICS 碎片 | BRICS 碎片 CRC32 hash → bucket 8（归一化频次） |
| 479 | `brics_9` | BRICS 碎片 | BRICS 碎片 CRC32 hash → bucket 9（归一化频次） |
| 480 | `brics_10` | BRICS 碎片 | BRICS 碎片 CRC32 hash → bucket 10（归一化频次） |
| 481 | `brics_11` | BRICS 碎片 | BRICS 碎片 CRC32 hash → bucket 11（归一化频次） |
| 482 | `brics_12` | BRICS 碎片 | BRICS 碎片 CRC32 hash → bucket 12（归一化频次） |
| 483 | `brics_13` | BRICS 碎片 | BRICS 碎片 CRC32 hash → bucket 13（归一化频次） |
| 484 | `brics_14` | BRICS 碎片 | BRICS 碎片 CRC32 hash → bucket 14（归一化频次） |
| 485 | `brics_15` | BRICS 碎片 | BRICS 碎片 CRC32 hash → bucket 15（归一化频次） |
| 486 | `brics_16` | BRICS 碎片 | BRICS 碎片 CRC32 hash → bucket 16（归一化频次） |
| 487 | `brics_17` | BRICS 碎片 | BRICS 碎片 CRC32 hash → bucket 17（归一化频次） |
| 488 | `brics_18` | BRICS 碎片 | BRICS 碎片 CRC32 hash → bucket 18（归一化频次） |
| 489 | `brics_19` | BRICS 碎片 | BRICS 碎片 CRC32 hash → bucket 19（归一化频次） |
| 490 | `brics_20` | BRICS 碎片 | BRICS 碎片 CRC32 hash → bucket 20（归一化频次） |
| 491 | `brics_21` | BRICS 碎片 | BRICS 碎片 CRC32 hash → bucket 21（归一化频次） |
| 492 | `brics_22` | BRICS 碎片 | BRICS 碎片 CRC32 hash → bucket 22（归一化频次） |
| 493 | `brics_23` | BRICS 碎片 | BRICS 碎片 CRC32 hash → bucket 23（归一化频次） |
| 494 | `brics_24` | BRICS 碎片 | BRICS 碎片 CRC32 hash → bucket 24（归一化频次） |
| 495 | `brics_25` | BRICS 碎片 | BRICS 碎片 CRC32 hash → bucket 25（归一化频次） |
| 496 | `brics_26` | BRICS 碎片 | BRICS 碎片 CRC32 hash → bucket 26（归一化频次） |
| 497 | `brics_27` | BRICS 碎片 | BRICS 碎片 CRC32 hash → bucket 27（归一化频次） |
| 498 | `brics_28` | BRICS 碎片 | BRICS 碎片 CRC32 hash → bucket 28（归一化频次） |
| 499 | `brics_29` | BRICS 碎片 | BRICS 碎片 CRC32 hash → bucket 29（归一化频次） |
| 500 | `brics_30` | BRICS 碎片 | BRICS 碎片 CRC32 hash → bucket 30（归一化频次） |
| 501 | `brics_31` | BRICS 碎片 | BRICS 碎片 CRC32 hash → bucket 31（归一化频次） |
| 502 | `brics_32` | BRICS 碎片 | BRICS 碎片 CRC32 hash → bucket 32（归一化频次） |
| 503 | `brics_33` | BRICS 碎片 | BRICS 碎片 CRC32 hash → bucket 33（归一化频次） |
| 504 | `surf_anionic` | 表面活性剂类型 | 阴离子型（含 [O-]/[S-]） |
| 505 | `surf_cationic` | 表面活性剂类型 | 阳离子型（含 [N+]/[n+]） |
| 506 | `surf_nonionic` | 表面活性剂类型 | 非离子型（默认） |
| 507 | `surf_zwitterionic` | 表面活性剂类型 | 两性型（阴+阳） |
| 508 | `head_ratio` | 头基/尾链比例 | 头基原子数 / 总原子数（SMARTS 匹配 + 反离子排除） |
| 509 | `tail_ratio` | 头基/尾链比例 | 尾链碳原子数 / 总原子数（DFS 最长碳链 ≥4） |
| 510 | `MolWt` | 分子描述符 | 分子量 / 500 |
| 511 | `LogP` | 分子描述符 | 脂水分配系数 / 10 |
| 512 | `TPSA` | 分子描述符 | 极性表面积 / 200 |
| 513 | `RotBonds` | 分子描述符 | 可旋转键数 / n_atoms |
| 514 | `HBA` | 分子描述符 | H 键受体数 / n_atoms |
| 515 | `HBD` | 分子描述符 | H 键供体数 / n_atoms |
| 516 | `NumRings` | 分子描述符 | 环总数 / 20 |
| 517 | `AroRings` | 分子描述符 | 芳香环数 / 10 |
| 518 | `AliRings` | 分子描述符 | 脂肪环数 / 10 |
| 519 | `FracSP3` | 分子描述符 | SP3 碳比例 |
| 520 | `HeavyAtoms` | 分子描述符 | 重原子数 / 100 |
| 521 | `NAtoms` | 分子描述符 | 总原子数 / 200 |