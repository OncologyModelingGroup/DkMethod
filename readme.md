# Glioblastoma D/k estimation from MRI

Research code accompanying **Bravo RR et al.**, *Estimating glioblastoma migration-to-proliferation ratio from single time-point MRI data via radial density distribution* (manuscript in preparation/submission; citation details to be updated).

The method estimates the migration-to-proliferation ratio (D/k) from radial tumor-density distributions derived from a single pretreatment MRI segmentation and a reaction–diffusion model.

## Installation

Requires Python 3.11 (the version tested in CI). From the repository root:

```bash
python -m pip install -r requirements.txt
```

## Data

Obtain the BraTS 2017 imaging data separately; patient MRI is **not** redistributed in this repository. The original input loader expects subject directories with segmentation files ending in `_ManuallyCorrected.nii.gz` or `_GlistrBoost.nii.gz`, paired with `_flair.nii.gz`. Generic `_seg.nii.gz` files are not recognized by the unchanged loader. Confirm your local dataset layout before running.

The [`Data/`](Data/README.md) directory includes 102 patient-level D/k estimates with TCGA-style barcodes, enabling linkage to compatible public TCGA-GBM molecular data, plus a derived gene-coefficient table. See the data README for column definitions, provenance, and limitations. Historical research outputs were removed from the current tree but remain recoverable from the private Git history; they are **not** a substitute for access to the original images.

## Run the reproducible pipeline

```bash
python Pipeline/run.py --help
python Pipeline/run.py --brats /path/to/BraTS2017 --output /path/to/results
```

The runner writes `paths.p`, `pts.p`, `grid.p`, `dks.p`, and `dkDf.csv`; it reuses existing checkpoints unless `--overwrite` is specified. Use a separate output directory for each model configuration. Intermediate pickle files must only be loaded from trusted sources.

**The default scientific grid is computationally expensive.** Read [Pipeline/README.md](Pipeline/README.md) before execution for parameter settings, reduced-size tests, resume behavior, and data assumptions. The legacy `Example.py` remains available but is not the recommended entry point for new runs.

## Validation and scope

The [full-grid parity workflow](.github/workflows/full-publication-parity.yml) compares original and standalone numerical methods on synthetic inputs. Historical-cohort tests and their archived inputs remain recoverable from earlier private commits but are not included in the current tree or clean publication snapshot. Synthetic tests **do not reproduce** the published BraTS 2017 patient estimates or downstream manuscript figures. Full clinical cohort reproduction remains unverified.

For publication packaging, validation boundaries, and release approval requirements, see [RELEASE.md](RELEASE.md) and [Publication/README.md](Publication/README.md). The publication snapshot excludes historical `Legacy/` files and Git history. Keep the original repository private and initialize a new history for the public repository.\n\nFor the original clinical, gene-expression, pathway, and multi-omics analyses, see [Downstream/README.md](Downstream/README.md) and the detailed [downstream provenance notes](Pipeline/DOWNSTREAM.md). The collaborator-authored analysis files are preserved unchanged; full downstream reproduction still requires the matching source and derived input tables.

## License

See [LICENSE](LICENSE). The manuscript citation should be updated with its final publication details when available.
