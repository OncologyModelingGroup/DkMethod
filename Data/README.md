# Derived data for manuscript analyses

This directory is reserved for **derived analysis tables only**. The two derived analysis tables are included to support reproducibility and reuse. They were copied **byte-for-byte** from archived analysis files; no numerical analyses were rerun as part of this copy. Their precise relationship to every final manuscript result has not been independently revalidated.

## Required derived tables

| Expected filename | Used by | Minimum known fields | Current status |
| --- | --- | --- | --- |
| `Dks_7_30_25.csv` | All four downstream analyses | `id`, `D/k`; clinical notebook also uses `Swanson D/k` | Included for public release (102 rows; unique TCGA-style patient IDs) |
| `FullGenes.csv` | Predictive-gene heatmap | `gene`, `coef` | Included for public release (98 rows) |

Provenance: `Data/Dks_7_30_25.csv` comes from `Legacy/BaharResults/Dks_7_30_25.csv` (Git blob `d62df4880b61819d87da7607ecb85d4b435e9fcb`); seven archived copies have the same blob. `Data/FullGenes.csv` comes from `Legacy/BaharResultsGenes/ElasticNetGenesHeatmap/FullGenes.csv` (Git blob `2f55dd9e76dd7f94d8f476d8f4351383a2908083`); three archived copies have the same blob. The archived `Legacy/Data1D/BratsPatientDks_5_1_25.csv` is a **different** 102-patient historical table and is not a verified replacement for `Dks_7_30_25.csv`.

## Patient-level D/k estimates and reuse

`Dks_7_30_25.csv` provides one record for each of 102 patients. The `id` column contains TCGA-style patient identifiers (for example, `TCGA-02-0006`), intended to support joins with publicly available TCGA-GBM molecular and clinical datasets. These are **not** patient names or direct personal identifiers. The exact availability and access conditions of any external dataset must be checked with its distributor.

| Column | Description |
| --- | --- |
| Unnamed first column | Exported row index; not a patient identifier |
| `id` | TCGA-style patient barcode |
| `D/k` | Estimated migration-to-proliferation ratio from the radial-density method described in the accompanying manuscript |
| `Swanson D/k` | Comparator D/k estimate retained from the original analysis table |

The table preserves the original values and column labels. Both D/k columns are model-derived estimates, **not direct clinical measurements**. Consult the manuscript and original analysis methods for the definitions, assumptions, and units; this README does not independently establish units or validate matching of all 102 identifiers against external TCGA records.

To join with TCGA molecular data, match on the appropriate patient-level barcode after checking whether the external table uses patient, sample, or aliquot identifiers. Multiple molecular samples may correspond to one patient. Do not assume that every patient has publicly available matched RNA measurements; the manuscript reports a smaller matched molecular subset.

The estimates are shared to enable independent biological association studies, comparisons with alternative methods, and reproducibility. Cite the accompanying manuscript when using them.

## Source datasets (not to duplicate here)

- BraTS 2017 MRI and segmentations: obtain from the [BraTS 2017 distribution](https://www.med.upenn.edu/sbia/brats2017/registration.html).
- TCGA-GBM gene expression: original downstream scripts refer to `GBM.transcriptome__ht_hg_u133a__broad_mit_edu__Level_3__gene_rma__data.data.txt`; see [FireBrowse](https://firebrowse.org/) and [GDAC](https://gdac.broadinstitute.org/).
- TCGA-GBM radiomic features: an archived table exists at `Legacy/TCGA_GBM_radiomicFeatures.csv`. Its version and suitability for release have not yet been checked.
- Clinical Table S1: the original clinical notebook cites [Ceccarelli et al. (2016) supplementary spreadsheet](https://www.cell.com/cms/10.1016/j.cell.2015.12.028/attachment/92d4425d-44e2-4c91-9ea8-98b58a172a43/mmc2.xlsx). The link has not been download-verified.

## Provenance and limitations

The archived tables are preserved unchanged. Their row counts and column names have been checked; independent confirmation of exact final-figure provenance and external dataset matching remains outstanding. No restricted original MRI or controlled-access molecular source files are included in this directory.

The collaborator-authored scripts in `Downstream/` use historical relative paths and are **not** automatically configured to read `Data/`. Their working-directory requirements are documented in [Downstream/README.md](../Downstream/README.md). The original scripts must remain unchanged.

Publication of these derived tables is intended; review the remaining repository contents and redistribution permissions before changing visibility.

## Read-only verification (2026-10-20)

- `Dks_7_30_25.csv`: 102 data rows, 102 unique nonempty `id` values, header `,id,Swanson D/k,D/k`; Git blob `d62df4880b61819d87da7607ecb85d4b435e9fcb`. Seven copies in `Legacy/` are blob-identical.
- `FullGenes.csv`: 98 data rows, header `gene,coef`; Git blob `2f55dd9e76dd7f94d8f476d8f4351383a2908083`. Three copies in `Legacy/` are blob-identical.
- The archived `Legacy/BaharResultsGenes/MOFAfinal/Factor6_vs_Dk_ScatterData.csv` contains 91 sample rows, consistent in count with the manuscript's reported matched transcriptomics cohort; **this is not proof that the final figure or analysis is reproduced**.
- The repository's current tracked-file tree contains no files named like credential files or raw NIfTI/DICOM images, based on a filename scan. This does **not** establish that historical Git commits, notebooks, or binary archives contain no sensitive content.

**Release caution:** Review the publication snapshot, notebook outputs, and redistribution permissions before changing visibility. The new repository must use fresh Git history; the original private history must not be transferred.
