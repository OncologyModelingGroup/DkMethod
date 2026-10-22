# Downstream analysis inputs and provenance

The original collaborator-authored scripts remain in `Downstream/` without modification. Their input paths are **not** automatically satisfied by `Pipeline/run.py`. Run each analysis only after supplying the matching files and documenting the source of every dataset.

| Analysis | Source file | Expected external inputs (as written) |
|---|---|---|
| Clinical comparisons and survival | `Downstream/clinicalplots.ipynb` | `TableS1.PatientData.20151020.v3.xlsx`, `Dks_7_30_25.csv`; R/rpy2 and survival packages |
| Predictive gene heatmap | `Downstream/Predictivegenes_heatmap.ipynb` | `FullGenes.csv`, `GBM.transcriptome__ht_hg_u133a__broad_mit_edu__Level_3__gene_rma__data.data.txt`, `Dks_7_30_25.csv` |
| Reactome / GSVA | `Downstream/reactome_gsva.ipynb` | Transcriptome TSV above, `Dks_7_30_25.csv`; gseapy, mygene, statsmodels |
| Multi-omics factor analysis | `Downstream/mofa_analysis.R` | Transcriptome TSV above, `TCGA_GBM_radiomicFeatures.csv`, `Dks_7_30_25.csv`; MOFA2 and R packages, conda `mofa_env` |

**Important:** The original `Dks_7_30_25.csv` is a derived analysis table, not simply the output of the new pipeline. The pipeline exports `dkDf.csv` with `id`, `D`, `k`, `t`, and `D/k`; the clinical notebook also accesses `Swanson D/k`. Do not substitute `dkDf.csv` for `Dks_7_30_25.csv` without establishing the required joins, provenance, and any additional columns.

The relative paths in the original notebooks/R script depend on the working directory and are not standardized. Existing files in `Legacy/` may be useful provenance sources but should not be deleted or silently edited. A reproducible publication release still requires explicit input manifests (checksums and origins), harmonized sample IDs, and confirmation that the generated D/k estimates match the published analysis.

## Explicit D/k comparison bridge

After obtaining the original, validated Swanson comparison data, create a new table without modifying either source:

```bash
python Pipeline/prepare_downstream.py --estimated output/dkDf.csv --comparison /path/to/validated_comparison.csv --output output/dk_comparison.csv
```

The comparison file must contain unique `id` values and a `Swanson D/k` column. The script joins by patient ID, retains all estimated patients (unmatched comparison values remain missing), reports match counts, and refuses to overwrite an input. **This does not reconstruct all columns of the original `Dks_7_30_25.csv` and does not establish clinical or transcriptomic alignment.** Test with `python -m unittest discover -s Tests -p 'test_downstream_bridge.py' -v`.

## Private-archive-only historical analysis

The historical-cohort analyzer below requires a file retained only in the private research archive. It is **excluded from the clean publication snapshot** together with its regression test. The commands below apply to the private research checkout only; do not expect them to work from the public snapshot.

## Archived 102-patient cohort: reproducible statistics

The preserved `Legacy/Data1D/BratsPatientDks_5_1_25.csv` contains 102 patient-level D/k estimates and Swanson comparators, of which 101 have observed survival information (89 deaths, 12 censored; `TCGA-12-1601` lacks both survival fields). Generate a machine-readable statistical summary without modifying the source:

```bash
python Pipeline/analyze_historical_cohort.py --output output/historical-cohort-summary.json
```

This calculates log10 Pearson and Spearman D/k-comparator correlations on all 102 patients, and an **uncut continuous log10(D/k) Cox proportional hazards model** on 101 survival records. Its hazard ratio describes a tenfold increase in D/k; it is not a per-unit raw D/k hazard ratio. The Cox result is an unadjusted association, not a causal estimate, and a rank-transformed Schoenfeld-residual proportional-hazards test gave p = 0.594 on the archived cohort (no detected violation, not proof that the assumption holds). The unadjusted Cox result was HR = 1.823 per tenfold D/k increase (95% CI 0.679–4.899; p = 0.234). The GitHub Actions `historical-cohort-summary` artifact records the generated JSON for each validated workflow run.

**Do not conflate the analyses:** `Downstream/clinicalplots.ipynb` uses R `survminer::surv_cutpoint` (maxstat) and has a saved optimized D/k cutoff of 2.081. A log-rank p-value after selecting that cutoff on the same dataset is not equivalent to a p-value for a prespecified threshold; the threshold-search procedure and inference limitations should be disclosed. The historical CSV and this analysis are not proof that the original MRI-to-D/k model or manuscript validation statistics were reproduced. Original downstream collaborator files remain unchanged.
