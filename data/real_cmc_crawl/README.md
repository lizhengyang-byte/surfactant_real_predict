# Real Surfactant CMC Crawl

This directory stores online real experimental surfactant CMC datasets crawled without SurfPro or virtual molecule generation.

Processed files:

- `processed/all_real_surfactant_cmc_bulk.csv`
  - Bulk real CMC table for broad CMC pretraining.
  - Sources: PSDI/NIST CMC collection and P2SURF single-surfactant dataset.
  - Columns match the project training schema: `SMILES,type,EO,PO,CMC,AW_ST_CMC,Gamma_max,Area_min,Pi_CMC,pC20,source`.

- `processed/all_real_surfactant_cmc_bulk_with_mixtures.csv`
  - Bulk real CMC table including single surfactants and P2SURF binary mixtures.
  - Use this when the model should learn mixture CMC behavior.
  - Mixture molar fractions are recorded inside `source`.

- `processed/psdi_nist_real_cmc_molar.csv`
  - Molar-unit rows extracted from PSDI/NIST Critical Micelle Concentrations of Aqueous Surfactant Systems.
  - Only split CMC values with unit `M` were used.
  - Older units such as `D`, `W`, `P`, and `N` were not converted.

- `processed/psdi_nist_eopo_cmc_subset.csv`
  - EO/PO subset from PSDI/NIST, filtered by explicit names such as `OXYETHYLENE`, `ETHOXY`, `OXYPROPYLENE`, `PROPOXY`, or `OXYPROP`.
  - This subset has also been appended to `../extended_real/extended_eopo_real_properties.csv`.

- `processed/p2surf_all_single_real_cmc.csv`
  - All single-surfactant P2SURF experimental CMC rows.
  - P2SURF `logCMC` is labeled in micro-molar units and converted with `CMC mol/L = 10^(logCMC - 6)`.

- `processed/p2surf_mixture_real_cmc.csv`
  - P2SURF binary mixed-micelle experimental CMC rows.
  - Mixture SMILES are represented as `SMILES_A.SMILES_B`.
  - Molar fractions are recorded inside `source`.

Raw files:

- `raw/PSDI_NIST_CMC_dataset.zip`
- `raw/PSDI_NIST_CMC_dataset/`

Use:

- Use `processed/all_real_surfactant_cmc_bulk.csv` for broad single-surfactant CMC model pretraining.
- Use `processed/all_real_surfactant_cmc_bulk_with_mixtures.csv` for broad CMC training when mixtures are acceptable.
- Use `../extended_real/extended_eopo_real_properties.csv` for the EO/PO/extended surfactant task table.
- Use rows with non-empty `AW_ST_CMC`, `Gamma_max`, `Area_min`, or `pC20` only for those property-specific targets.

Source links:

- PSDI/NIST dataset: https://github.com/PSDI-UK/psdi-datasets
- Original NIST book DOI: https://doi.org/10.6028/NBS.NSRDS.36
- P2SURF: https://github.com/sandialabs/P2SURF
