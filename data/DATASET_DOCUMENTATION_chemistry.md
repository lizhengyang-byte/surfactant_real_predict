# 表面活性剂真实实验数据集说明

本文档面向化学、胶体与界面化学、表面活性剂配方和机器学习建模人员，说明当前工作区内真实实验表面活性剂数据集的来源、字段、单位、适用范围和注意事项。

## 1. 数据集总览

本项目当前保留两类数据表：

1. **EO/PO/extended 表面活性剂任务主表**
   - 文件：`data/extended_real/extended_eopo_real_properties.csv`
   - 行数：965
   - 唯一 SMILES/混合体系：483
   - 用途：面向 EO、PO、extended surfactant 的性质建模，尤其是 CMC。

2. **广义真实表面活性剂 CMC 大库**
   - 文件：`data/real_cmc_crawl/processed/all_real_surfactant_cmc_bulk_with_mixtures.csv`
   - 行数：5033
   - 唯一 SMILES/混合体系：1318
   - 用途：面向广义表面活性剂 CMC 的大样本预训练或基线模型训练。

所有表均采用统一字段：

```csv
SMILES,type,EO,PO,CMC,AW_ST_CMC,Gamma_max,Area_min,Pi_CMC,pC20,source
```

本数据集没有保留 InChI、CAS、CID、PubChem ID 等字段。

## 2. 数据来源

### 2.1 PSDI/NIST CMC 数据集

来源文件：

- `data/real_cmc_crawl/raw/PSDI_NIST_CMC_dataset.zip`
- `data/real_cmc_crawl/processed/psdi_nist_real_cmc_molar.csv`
- `data/real_cmc_crawl/processed/psdi_nist_eopo_cmc_subset.csv`

原始来源：

- PSDI-UK 数据仓库：https://github.com/PSDI-UK/psdi-datasets
- NIST 原始书籍 DOI：https://doi.org/10.6028/NBS.NSRDS.36
- 原书：*Critical Micelle Concentrations of Aqueous Surfactant Systems*

说明：

- NIST 数据是 1926-1966 年文献 CMC 汇编，包含多种表面活性剂、温度、添加剂和原始文献代码。
- PSDI 数据将原书表格数字化，并尝试为化合物补充 SMILES。
- 本项目只采用拆分字段中单位为 `M` 的 CMC 数值，直接作为 `mol/L`。
- 原数据中的 `D`、`W`、`P`、`N` 等非摩尔单位未转换，避免引入不确定性。
- NIST 行的温度、添加剂、作者和参考文献代码保存在 `source` 字段中。

注意：

- PSDI 的 SMILES 是根据名称和分子量推断得到，适合建模使用，但不是人工逐一校核的结构数据库。
- 若用于高精度结构-性质解释，应优先复核关键分子的结构。

### 2.2 P2SURF 单组分 CMC 数据

来源文件：

- `data/raw_sources/P2SURF/Single_Surfactant_CMC/total_smile_and_property.csv`
- `data/real_cmc_crawl/processed/p2surf_all_single_real_cmc.csv`
- `data/extended_real/p2surf_eopo_cmc_subset.csv`

原始来源：

- P2SURF GitHub：https://github.com/sandialabs/P2SURF

说明：

- P2SURF 提供单组分表面活性剂实验 CMC 数据。
- 原字段为 `logCMC`，notebook 图轴标注为 `logCMC (uM)`。
- 本项目转换公式：

```text
CMC (mol/L) = 10^(logCMC - 6)
```

### 2.3 P2SURF 二元混合体系 CMC 数据

来源文件：

- `data/raw_sources/P2SURF/Mixture_surfactants_CMC/.../updated_CMC-Data-for-mixed-micelle.csv`
- `data/real_cmc_crawl/processed/p2surf_mixture_real_cmc.csv`
- `data/real_cmc_crawl/processed/all_real_surfactant_cmc_bulk_with_mixtures.csv`

说明：

- 该数据包含二元表面活性剂混合体系的实验 CMC。
- 混合体系的 `SMILES` 表示为：

```text
SMILES_A.SMILES_B
```

- 两组分摩尔分数记录在 `source` 字段：

```text
mol_fraction_A=...; mol_fraction_B=...
```

注意：

- 混合体系不是单一分子，不能直接按普通单分子 QSAR 处理。
- 如果模型不支持混合物，应使用不含 mixture 的单组分表。

### 2.4 Molecules 2021 PO extended anionic surfactants

来源文件：

- `data/extended_real/pmma_extended_anionic_surfactants.pdf`

文献：

- Jiang et al., *Wettability of a Polymethylmethacrylate Surface by Extended Anionic Surfactants: Effect of Branched Chains*, Molecules 2021, 26, 863.
- DOI：https://doi.org/10.3390/molecules26040863

提取内容：

- L-C12PO4S
- G-C12PO4S
- G-C16PO4S
- 以及文中 Table 1 引用的 C12PnS 和 PPnS 文献值

性质：

- `CMC`
- `AW_ST_CMC`
- `Gamma_max`
- `Area_min`

单位处理：

- 表中 CMC 为 `10^-5 mol/L`，已换算为 `mol/L`。
- 表中 `10^10 Gamma_max (mol/cm2)` 已换算为 `mol/m2`。
- `Area_min` 保持 `nm2`。

### 2.5 Molecules 2024 OE3P3S 及混合体系

来源文件：

- `data/extended_real/molecules_2024_oe3p3s.pdf`

文献：

- Li/Ren et al., *Adsorption and Aggregation Behaviors of Oleyl Alcohol-Based Extended Surfactant and Its Mixtures*, Molecules 2024, 29, 2570.
- DOI：https://doi.org/10.3390/molecules29112570

提取内容：

- OE3P3S 单组分
- OE3P3S/CTAB、OE3P3S/DTAB、OE3P3S/TTAB 不同比例混合体系

性质：

- `CMC`
- `AW_ST_CMC`
- `Gamma_max`
- `Area_min`

单位处理：

- 表中 CMC 为 `mmol/L`，已换算为 `mol/L`。
- 表中 `Gamma_max` 为 `umol/m2`，已换算为 `mol/m2`。
- 表中 `Amin` 数值按物理量级作为 Angstrom2/molecule 处理，并换算为 `nm2`。

### 2.6 RSC Advances 2025 branched alcohol alkoxylate sulfates

来源：

- Yue et al., *Synthesis and properties of branched alcohol alkoxylate sulfates*, RSC Advances 2025.
- DOI：https://doi.org/10.1039/D5RA05419B

提取内容：

- C10E3P3S
- C10P3E3S
- C10E3S

性质：

- `CMC`
- `AW_ST_CMC`
- `Gamma_max`
- `Area_min`
- `pC20`

### 2.7 本地中文 PDF：磷酸酯型 extended 表面活性剂

来源文件：

- `D:\study\表面活性剂\延展性表面活性剂\磷酸酯型extended表面活性剂的性能研究.pdf`

提取内容：

- M100、M95、M90、M85、M80 磷酸酯型 C12P3P extended surfactant 样品
- SLE3S 参比样品

注意：

- M95-M80 为不同单/双酯组成的真实样品，属于混合组成，不是纯单一化合物。
- 组成信息写在 `source` 字段中。

## 3. 字段定义

### SMILES

分子或混合体系结构表示。

- 单组分：普通 SMILES。
- 二元混合体系：`SMILES_A.SMILES_B`。
- 盐形式保留反离子，例如 `[Na+]`、`[Br-]`。

### type

粗略化学类别。

常见值：

- `anionic`
- `cationic`
- `nonionic`
- `anionic-cationic mixture`
- `anionic mixture`
- `anionic aromatic`
- `anionic branched`
- `zwitterionic/ionic`

该字段用于建模分组或粗筛，不应视为严格 IUPAC 分类。

### EO

乙氧基单元数。

对于明确命名或结构的 EO surfactant，记录整数或平均值。例如：

- SLE3S：EO = 3
- OE3P3S：EO = 3.4

对于混合体系：

- P2SURF mixture 中使用摩尔分数加权平均 EO。
- 文献中明确标注混合比例的体系在 `source` 中保留比例信息。

### PO

丙氧基单元数。

对于 extended surfactant，PO 是核心结构字段。例如：

- C12PO4S：PO = 4
- OE3P3S：PO = 3

### CMC

临界胶束浓度。

统一单位：

```text
mol/L
```

### AW_ST_CMC

CMC 处空气-水表面张力。

统一单位：

```text
mN/m
```

对应文献中常见的：

- `gamma_CMC`
- `surface tension at CMC`
- `critical surface tension`

### Gamma_max

最大表面过剩吸附量。

统一单位：

```text
mol/m2
```

### Area_min

最小分子占据面积。

统一单位：

```text
nm2/molecule
```

### Pi_CMC

CMC 处表面压。

当前没有可靠批量数据，暂为空。

### pC20

降低纯水表面张力 20 mN/m 所需浓度的负对数，按原文报告值记录。

当前只有少量 RSC Advances 2025 数据。

### source

来源追踪字段，包含：

- 数据库或文献名称
- 表格编号或文件名
- DOI 或 GitHub 链接
- 温度、添加剂、摩尔分数、组成信息等无法单独建列但化学上重要的信息

## 4. 单位换算规则

### P2SURF

```text
CMC (mol/L) = 10^(logCMC - 6)
```

### PSDI/NIST

只接受拆分单位为 `M` 的 CMC：

```text
CMC = CMC Value where CMC Units == M
```

### mmol/L

```text
mol/L = mmol/L / 1000
```

### umol/m2

```text
mol/m2 = umol/m2 * 1e-6
```

### mol/cm2

```text
mol/m2 = mol/cm2 * 1e4
```

对于 Molecules 2021 中 `10^10 Gamma_max (mol/cm2)`：

```text
Gamma_max (mol/m2) = table_value * 1e-6
```

### Angstrom2/molecule

```text
nm2/molecule = Angstrom2/molecule * 0.01
```

## 5. 推荐建模方式

### 5.1 CMC 大样本模型

推荐文件：

```text
data/real_cmc_crawl/processed/all_real_surfactant_cmc_bulk.csv
```

或包含混合体系：

```text
data/real_cmc_crawl/processed/all_real_surfactant_cmc_bulk_with_mixtures.csv
```

适合：

- CMC 预训练
- 大样本 baseline
- 表面活性剂类别泛化模型

不适合：

- 直接训练 `AW_ST_CMC`
- 直接训练 `Gamma_max`
- 直接训练 `Area_min`

因为这些大库只有 CMC。

### 5.2 EO/PO/extended 专用模型

推荐文件：

```text
data/extended_real/extended_eopo_real_properties.csv
```

适合：

- EO/PO 参数影响 CMC 的模型
- extended surfactant CMC 模型
- PO/EO 顺序、支化、混合体系的粗粒度建模

### 5.3 多任务性质模型

建议按目标拆分：

- `CMC`：所有 CMC 非空行。
- `AW_ST_CMC`：只取该列非空行。
- `Gamma_max`：只取该列非空行。
- `Area_min`：只取该列非空行。
- `pC20`：当前样本很少，不建议单独训练复杂模型。

不要把空值当 0。

## 6. 化学注意事项

### 6.1 温度和添加剂影响

CMC 对温度、盐、添加剂、纯度和测定方法敏感。

当前统一表为了保持用户要求的列格式，没有单独建立温度和添加剂列，而是放在 `source` 中。

如果后续做高精度模型，建议扩展字段：

```text
temperature, additive, additive_concentration, method
```

### 6.2 混合体系不是单分子

混合体系行不能简单等同于一个真实分子。

当前做法：

- `SMILES = SMILES_A.SMILES_B`
- `type` 中标明 mixture
- 摩尔分数保存在 `source`

如果模型使用分子图输入，应单独设计 mixture featurizer。

### 6.3 EO/PO 平均数

商业醇醚和 extended surfactant 经常是分布样品，EO/PO 是平均加成数。

例如：

- OE3P3S 中 EO = 3.4，PO = 3.0

这类行适合配方和经验建模，但不应理解为严格单分散分子。

### 6.4 SMILES 可信度分层

建议按来源分层：

1. **高可信结构**
   - 论文结构式明确给出的 Molecules/RSC/本地 PDF 行。

2. **中等可信结构**
   - P2SURF 行，直接提供 SMILES 和实验 CMC。

3. **需要关键样本复核**
   - PSDI/NIST 行，CMC 来自经典文献汇编，SMILES 来自 PSDI 根据名称的结构补全。

## 7. 当前数据规模

截至当前版本：

### EO/PO/extended 主表

```text
文件：data/extended_real/extended_eopo_real_properties.csv
行数：965
唯一 SMILES/混合体系：483
CMC 非空：965
AW_ST_CMC 非空：29
Gamma_max 非空：29
Area_min 非空：29
pC20 非空：3
```

### 广义 CMC 大库，单组分

```text
文件：data/real_cmc_crawl/processed/all_real_surfactant_cmc_bulk.csv
行数：4054
唯一 SMILES：796
CMC 非空：4054
```

### 广义 CMC 大库，含混合体系

```text
文件：data/real_cmc_crawl/processed/all_real_surfactant_cmc_bulk_with_mixtures.csv
行数：5033
唯一 SMILES/混合体系：1318
CMC 非空：5033
```

## 8. 不包含内容

本数据集当前不包含：

- SurfPro 数据。
- 虚拟生成分子。
- 预测值冒充实验值。
- InChI、CAS、CID、PubChem ID。
- 未能确认单位的 CMC 数据。

## 9. 建议的下一步

1. 为 `AW_ST_CMC/Gamma_max/Area_min/pC20` 继续爬取论文 PDF 和 Supporting Information。
2. 增加温度、盐浓度和实验方法字段，提升 CMC 模型的化学可解释性。
3. 对 PSDI/NIST 中高频样本的 SMILES 进行人工或 RDKit 校验。
4. 为混合体系单独建立特征：组分 A、组分 B、摩尔分数、单组分性质。
