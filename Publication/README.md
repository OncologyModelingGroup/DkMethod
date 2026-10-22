# Candidate publication snapshot

The private repository remains the full research archive. These scripts create and inspect a **separate, history-free local directory** for author review. They do not create a GitHub repository, delete files, or publish anything.

From a local checkout of the private repository:

```bash
python Publication/build_snapshot.py ../DkPublicationCandidate
python Publication/verify_snapshot.py ../DkPublicationCandidate
```

The builder requires a Git checkout and uses the tracked-file index as a strict source list. Untracked local files, caches, and symlinks are excluded or rejected; this prevents accidentally packaging unrelated files from a working directory. The CI workflow deliberately creates an untracked sentinel file and checks that it is not copied. The candidate directory does not inherit Git history.

The builder copies the root scientific modules, standalone method, pipeline, collaborator notebooks, derived tables, tests, and one parity workflow. It omits all of `Legacy/`, the historical-cohort analyzer and regression test that requires `Legacy/Data1D/BratsPatientDks_5_1_25.csv`, and the original numerical CI workflow that invokes that test. The full publication parity workflow remains; it must be verified independently on the candidate snapshot.

A private CI workflow at `.github/workflows/publication-snapshot.yml` builds the snapshot in an isolated runner directory, verifies it, and runs isolated synthetic, bridge, orchestration, numerical, and end-to-end integration tests. It does not publish or upload the snapshot. The isolated workflow passed on commit `a9768b34` ([GitHub Actions run #25](https://github.com/rafaelrbravo/DkMethod/actions/runs/39541321384)).

The verifier checks exclusions, derived-table row counts, required columns, unique patient IDs, and exact Git blob hashes pinned to the two recovered archived analysis tables. The builder and verifier additionally reject any unexpected file in `Data/`, preventing newly tracked patient-derived tables from silently entering a candidate snapshot. A changed table fails verification until its provenance and intended release contents are deliberately reviewed and the expected hash updated. It **does not** reproduce the paper's analyses, establish sharing permissions, or audit notebook outputs. Those remain release gates described in [RELEASE.md](../RELEASE.md).

No public release should be created without explicit approval.

## Notebook output inventory

The candidate includes three collaborator notebooks unchanged. A read-only structural inspection found:

| Notebook | Cells | Saved outputs | Embedded PNG outputs |
| --- | ---: | ---: | ---: |
| `Downstream/clinicalplots.ipynb` | 13 | 17 | 10 |
| `Downstream/Predictivegenes_heatmap.ipynb` | 3 | 1 | 1 |
| `Downstream/reactome_gsva.ipynb` | 26 | 5 | 0 |

These notebooks contain saved analysis outputs, not just code. Before release, inspect the images and text outputs for unintended patient-specific information, private paths, and redistribution constraints. Preserve collaborator originals unchanged in the private archive. If outputs must be removed from the public copies, obtain author approval for that transformation and document it; do not silently change scientific code.

## Isolated CI coverage

The private repository's `Verify clean publication snapshot` workflow builds a fresh candidate, checks the excluded archive and untracked-file sentinel, installs dependencies from the snapshot, and executes the preprocessed-array example, downstream bridge, pipeline orchestration, numerical smoke, real grid/fitting integration, and end-to-end synthetic NIfTI tests **from within that candidate**. Push triggers cover scientific code, pipeline, standalone implementation, tests, derived data, publication tooling, and dependency changes. The full 1000-coefficient numerical parity workflow is separate and manually runnable.

**Verified CI:** the isolated publication-snapshot workflow succeeded on commit `a9768b34` ([run #25](https://github.com/rafaelrbravo/DkMethod/actions/runs/39541321384)). This verifies the configured snapshot checks, not full patient-cohort or manuscript-figure reproduction.

## Completed visual review of saved figures (2026-10-20)

All 11 embedded PNG outputs were visually inspected: ten from `clinicalplots.ipynb` and one from `Predictivegenes_heatmap.ipynb`. The clinical figures comprise five scatter/regression plots (D/k or comparator against clinical measurements), three molecular-subtype boxplots, and two Kaplan–Meier survival plots with aggregate at-risk, censoring and event counts. The gene notebook contains a gene-expression heatmap with gene labels and D/k annotations. No patient names, coded sample IDs, private local paths, or other obvious direct identifiers were visible **in the rendered images**. Saved stream and plain-text outputs were also inspected for obvious identifiers; the Reactome notebook includes pathway correlations and mapping diagnostics.

**Scope:** This is a visual/output-content audit, not a formal privacy, re-identification, intellectual-property, or source-data licensing clearance. The notebooks' source code and underlying datasets may have different redistribution constraints. Do not claim approval to publish until the authors and data custodians confirm permissions.
