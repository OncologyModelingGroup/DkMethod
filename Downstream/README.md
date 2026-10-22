# Downstream manuscript analyses

These are the original collaborator-authored analysis files, preserved unchanged. They analyze clinical outcomes, gene expression, pathways, and multi-omics associations with estimated glioblastoma D/k. They are **not** part of the MRI-to-D/k runner in `Pipeline/run.py`.

| Analysis | Original file | Required input files |
| --- | --- | --- |
| Clinical comparisons and survival | `clinicalplots.ipynb` | `TableS1.PatientData.20151020.v3.xlsx`, `Dks_7_30_25.csv` |
| Predictive-gene heatmap | `Predictivegenes_heatmap.ipynb` | `FullGenes.csv`, TCGA-GBM expression TSV, `Dks_7_30_25.csv` |
| Reactome / GSVA | `reactome_gsva.ipynb` | TCGA-GBM expression TSV, `Dks_7_30_25.csv` |
| Multi-omics factor analysis | `mofa_analysis.R` | TCGA-GBM expression TSV, `TCGA_GBM_radiomicFeatures.csv`, `Dks_7_30_25.csv` |

The clinical notebook identifies its source spreadsheet as Table S1 from Ceccarelli et al., *Cell* (2016), and records the original [supplementary spreadsheet download](https://www.cell.com/cms/10.1016/j.cell.2015.12.028/attachment/92d4425d-44e2-4c91-9ea8-98b58a172a43/mmc2.xlsx). This link comes from the original notebook and has not been download-verified in this audit. The notebook expects the local filename `TableS1.PatientData.20151020.v3.xlsx` and uses `skiprows=1`.

The gene-heatmap notebook expects `FullGenes.csv` to have at least `gene` and `coef` columns. Its code **reads** that table; it does not generate it. Do not reconstruct it from the heatmap image or invent coefficients. The downstream notebooks likewise read `Dks_7_30_25.csv` rather than creating it.

The expression filename expected by the scripts is `GBM.transcriptome__ht_hg_u133a__broad_mit_edu__Level_3__gene_rma__data.data.txt`.

## Preparing inputs

The publication repository is intended to contain **derived analysis tables only**. Raw BraTS MRI, TCGA expression, clinical source spreadsheets, and radiomics source data should be obtained from their original providers where redistribution is unnecessary. Any derived table added to the repository should have documented provenance and a checked redistribution status.

**Repository inventory finding:** The archived derived tables were found and copied byte-for-byte to [`Data/Dks_7_30_25.csv`](../Data/Dks_7_30_25.csv) (102 patient rows) and [`Data/FullGenes.csv`](../Data/FullGenes.csv) (98 gene rows). Seven historical D/k table copies and three gene table copies have matching Git blob hashes within each group. The radiomics input `Legacy/TCGA_GBM_radiomicFeatures.csv` also exists. See [Data/README.md](../Data/README.md) for exact provenance.

**The tables are now located but have not been fully validated against the final manuscript results.** `Pipeline/run.py` produces a different `dkDf.csv`, and `Legacy/Data1D/BratsPatientDks_5_1_25.csv` is not interchangeable with the recovered `Dks_7_30_25.csv`. Preserve the original inputs and use working copies or symlinks to satisfy the original scripts' historical relative paths.

The original scripts use different relative paths. The clinical notebook reads its spreadsheet and D/k table from its current working directory, while the other scripts refer to files one directory above `Downstream/`. Stage files accordingly **without editing the original scripts**; keep a record of exact filenames, source versions, and sample-ID matching. No single execution command is yet verified to reproduce all manuscript figures.

## Software environments

The root `requirements.txt` supports the MRI/D/k pipeline, **not the entire downstream environment**. The notebooks additionally use packages including `openpyxl` (Excel loading), `scikit-learn`, `gseapy`, `mygene`, `statsmodels`, `rpy2`, and Jupyter; the clinical notebook also calls R survival packages. `mofa_analysis.R` uses R packages including `MOFA2`, `reticulate`, `readr`, `dplyr`, `ggplot2`, `ggpubr`, `matrixStats`, `tibble`, and `writexl`, and selects a Conda environment named `mofa_env`. These are observed dependencies, not a tested lockfile.

Run analyses with the matching source data and record the package versions used. The original files may contain saved notebook outputs; those outputs alone do not establish reproducibility.

## Scope and verification

The MRI pipeline and standalone numerical method have synthetic validation, but the complete downstream analyses have **not** been rerun here on the original manuscript input tables. The repository should not claim end-to-end manuscript figure reproduction until those inputs are matched and the outputs checked.

For additional provenance, survival-analysis caveats, and the archived 102-patient statistical summary, see [Pipeline/DOWNSTREAM.md](../Pipeline/DOWNSTREAM.md).
