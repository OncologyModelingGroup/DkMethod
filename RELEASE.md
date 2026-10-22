# Publication release boundary

**Decision pending author approval.** The existing `main` repository is a research archive, not a minimal publication package. Keep it private until its contents and history are reviewed.

## Inventory (2026-10-20)

The current tree contains 1,868 tracked files totaling about 107 MB. `Legacy/` accounts for 1,811 files and about 106 MB (over 99% of tracked bytes), including archived results, figures, notebooks, pickles, spreadsheets, a MOFA HDF5 model, and TCGA-derived tables. A filename scan found no obvious credential files or raw NIfTI/DICOM images; **this is not a content or history clearance**. The original repository has approximately 170 commits.

The publication's primary reusable material is in the original scientific Python modules, `Standalone/`, `Pipeline/`, `Downstream/`, `Tests/`, and `Data/`. The exact derived inputs `Data/Dks_7_30_25.csv` and `Data/FullGenes.csv` were copied unchanged from archived files, with provenance in `Data/README.md`.

## Recommended public distribution

**Publish a clean, separately created publication repository or history-free snapshot**, rather than making the complete research archive public. The clean publication tree should include:

- Root scientific modules: `Example.py`, `FitDkForEachPatient.py`, `LMclassNew.py`, `Model3Dand1D.py`, `PDEgrid.py`, `ProcessPatientData.py`.
- `Standalone/`, `Pipeline/`, `Downstream/`, `Tests/`, `Data/`, `requirements.txt`, `readme.md`, `LICENSE`, and selected CI workflows.
- The **minimal** archived file(s) genuinely required by the selected tests or reproducible analyses, only after inspecting them and confirming redistribution rights.

Do **not** copy `Legacy/` wholesale. It is neither necessary for a minimal method release nor fully audited. Do not delete or rewrite anything in the existing private research archive. Simply removing `Legacy/` in a new commit on the existing repository is **not sufficient**: previous Git commits would still expose the old files after making that repository public. `.gitignore` does not remove tracked files or history.

Before publishing, review the exact clean snapshot's contents, verify the selected tests against it, and check that the original collaborator scripts and their licenses/attribution are appropriately represented. Do not copy raw BraTS MRI or original TCGA expression data. Confirm that the two derived patient-level tables are approved for public distribution.

## Release workflow (not yet executed)

1. Freeze the intended file manifest and review it with the author.
2. Build the clean snapshot from approved files only, with no inherited Git history.
3. Run the targeted scientific and packaging checks on that exact snapshot.
4. Confirm manuscript Data Availability wording, repository URL, citation, and optional Zenodo DOI match the **actual public** contents.
5. Obtain explicit approval before creating a public repository, changing visibility, tagging a release, or publishing patient-derived data.

The current private repository remains the preserved research archive unless the author chooses a different approach.

## Exact snapshot manifest and test dependency (2026-10-20)

The current tree contains **57 non-`Legacy/` tracked files**. A minimal snapshot can select those files by path, but the existing CI workflow is **not self-contained** without one archived data file:

- `Tests/test_historical_cohort.py` imports `Pipeline/analyze_historical_cohort.py`, which defaults to `Legacy/Data1D/BratsPatientDks_5_1_25.csv`. The numerical-tests workflow runs both this test and the cohort summary command. This file contains 102 patient-level estimates with survival fields; its public distribution needs separate approval.
- `Tests/compare_private_patients.py` accepts an external private BraTS path and does not require bundled patient images. Its presence does not imply that four real patients are distributed.
- The other selected `Standalone/`, `Pipeline/`, `Downstream/`, `Tests/`, root scientific modules, `Data/`, and workflow files form the candidate snapshot. Preserve file contents; no collaborator code changes are necessary.

**Practical resolution:** include the archived 102-patient CSV in a clean release *only if approved*; otherwise remove the historical-cohort execution steps from the publication workflow and exclude both the archived-cohort analyzer and its test from the public snapshot and release verification. Do not silently make the test pass with a different dataset. A future public snapshot should be checked with a clean checkout and its own CI, not with the private archive's working tree.

A dedicated private GitHub Actions workflow (`.github/workflows/publication-snapshot.yml`) now builds and tests the isolated candidate without publishing artifacts. It must pass before release approval; a passing run has not yet been observed.

**Current status:** no publication snapshot or public repository has been created. The private archive remains unchanged apart from release documentation.

## Manuscript Data Availability correction (2026-10-20)

The draft statement that **individual patient-level D/k estimates are not included in the repository** is inaccurate for the proposed public snapshot: `Data/Dks_7_30_25.csv` contains 102 coded TCGA sample identifiers and patient-level D/k estimates. The snapshot also includes 98 gene coefficients in `Data/FullGenes.csv`. Both tables are derived data, not raw MRI or raw transcriptomics.

**Candidate replacement wording, contingent on approval to distribute the two tables:**

> The analysis code, standalone D/k estimation pipeline, and derived tables used in selected downstream analyses are available in the accompanying code repository [repository URL to be added after publication]. The derived tables include patient-level D/k estimates for 102 coded BraTS/TCGA cases and coefficients for 98 selected genes. Raw MRI and transcriptomic datasets are not redistributed; access to the source datasets is subject to their respective repositories' terms and conditions. Other clinical, molecular, and radiomic source data must be obtained from their original providers. The repository does not contain every intermediate result required to reproduce all downstream analyses from raw data.

This is a **draft**, not an approved manuscript change. Check the precise source provenance, original data licenses, and whether the 102-row derived table is authorized for open redistribution before using it. If redistribution is not approved, exclude that table from the public snapshot and rewrite the statement accordingly.

## Saved notebook output review

Read-only inspection of notebook JSON found 10 embedded PNG outputs in the clinical notebook, one in the predictive-gene heatmap notebook, and none in the Reactome notebook. The Reactome notebook retains stream output containing pathway-level correlations and mapping diagnostics. The inspected text outputs did not show obvious names or addresses, but embedded PNGs and all source cells still require content-level clearance. The originals remain untouched.

## Deferred history removal — explicit author instruction (2026-10-20)

The current `main` tree was cleaned in commit `97cc84e01d407382a2c345e88753ebcb1038f66e`: `Legacy/` (1,811 files), the archive-dependent historical cohort analyzer and test, the corresponding numerical-tests workflow, and an empty-directory placeholder were removed **from the current tree only**. They remain retrievable from earlier commits. Do **not** rewrite or purge Git history yet: the author wants to preserve it temporarily in case important research material is needed. **Before any public release, revisit and remove/sanitize the historical contents**, after reviewing and safeguarding anything worth retaining. Deleting from the current tree is not sufficient to prevent exposure if the repository is made public. The repository must remain private until that decision and the coauthor data-hosting discussion are resolved.
