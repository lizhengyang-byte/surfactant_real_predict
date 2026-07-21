# Extended EO/PO Surfactant Real Data

Main file:

- `extended_eopo_real_properties.csv`

Columns:

- `SMILES`: molecule or explicitly marked mixture SMILES.
- `type`: surfactant class.
- `EO`, `PO`: ethylene oxide and propylene oxide unit counts where derivable from the reported structure/SMILES.
- `CMC`: critical micelle concentration in mol/L.
- `AW_ST_CMC`: air-water surface tension at CMC in mN/m.
- `Gamma_max`: maximum surface excess in mol/m2.
- `Area_min`: minimum molecular area in nm2.
- `Pi_CMC`, `pC20`: left blank when the source did not report exact values.
- `source`: paper/dataset/table used for traceability.

Source files in this folder:

- `extended_eopo_real_properties.csv`: merged training table.
- `p2surf_eopo_cmc_subset.csv`: EO/PO/polyether subset filtered from the P2SURF single-surfactant experimental CMC dataset.
- P2SURF binary mixture EO/PO rows are appended directly to `extended_eopo_real_properties.csv`; the full processed mixture table is stored at `../real_cmc_crawl/processed/p2surf_mixture_real_cmc.csv`.
- `molecules_2024_oe3p3s.pdf`: OE3P3S and OE3P3S/ATAB mixture surface properties.
- `pmma_extended_anionic_surfactants.pdf`: PO sulfate surface activity table from Molecules 2021.
- `oklahoma_extended_surfactant.pdf`: real extended-surfactant dissertation source kept for later extraction, but not merged where no unique SMILES was available.

Use by model target:

- CMC model: use all rows with non-empty `CMC`.
- AW_ST_CMC model: use only rows with non-empty `AW_ST_CMC`; P2SURF rows should be excluded for this target.
- Gamma_max and Area_min models: use only rows from paper tables with non-empty `Gamma_max` and `Area_min`.
- pC20 model: currently sparse; use only rows where `pC20` is present.

Notes:

- SurfPro and virtual/generated molecules are intentionally not used.
- No InChI, CAS, CID, or PubChem metadata columns are included.
- P2SURF reports `logCMC` in micro-molar units. It was converted as `CMC mol/L = 10^(logCMC - 6)`.
- Some literature rows are mixtures; those are explicitly marked in `type` and `source`.
- PSDI/NIST rows use only extracted CMC values whose split unit is `M`; older non-molar units were not converted.
