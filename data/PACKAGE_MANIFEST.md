# 压缩包清单

压缩包名称：surfactant_real_datasets_docs_20260721.zip

## 包含内容

- data/DATASET_DOCUMENTATION_chemistry.md：面向化学人员的完整数据集说明文档。
- data/文件夹和文件说明.md：逐文件夹、逐文件中文说明。
- data/extended_real/：EO/PO/extended 表面活性剂主表、论文 PDF 和说明。
- data/real_cmc_crawl/：真实 CMC 大库、处理后 CSV、原始 NIST/PSDI 文件和说明。
- data/raw_sources/P2SURF/：P2SURF 原始单组分和混合体系数据。

## 故意排除内容

- SurfPro 文件。
- 虚拟生成分子库。
- 预测值冒充实验值的数据。
- 模型缓存文件、特征缓存和无关工程文件。

## 最重要的数据文件

- data/extended_real/extended_eopo_real_properties.csv：EO/PO/extended 主表。
- data/real_cmc_crawl/processed/all_real_surfactant_cmc_bulk.csv：真实单组分 CMC 大库。
- data/real_cmc_crawl/processed/all_real_surfactant_cmc_bulk_with_mixtures.csv：包含混合体系的真实 CMC 大库。
- data/real_cmc_crawl/processed/p2surf_mixture_real_cmc.csv：P2SURF 二元混合体系 CMC。
- data/real_cmc_crawl/processed/psdi_nist_real_cmc_molar.csv：PSDI/NIST 中单位明确为 M 的 CMC。
- data/real_cmc_crawl/processed/p2surf_all_single_real_cmc.csv：P2SURF 单组分 CMC。

## 建议阅读顺序

1. 先读 data/文件夹和文件说明.md。
2. 再读 data/DATASET_DOCUMENTATION_chemistry.md。
3. 根据建模目标选择 CSV 文件。
