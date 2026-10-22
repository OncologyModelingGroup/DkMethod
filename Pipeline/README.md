# Publication pipeline

This directory provides a stage-by-stage entry point to the original D/k analysis. It **imports the original scientific code without modifying it**. The source data, collaborator analyses, and `Legacy/` are preserved.

## Setup

From the repository root, use Python 3.11 and install the tested scientific dependencies with `python -m pip install -r requirements.txt`. Obtain the BraTS 2017 data separately; the repository does not redistribute patient MRI.

Inspect the command-line options without importing the scientific dependencies:

```bash
python Pipeline/run.py --help
```

## Reproduce patient D/k estimates

```bash
python Pipeline/run.py --brats /path/to/BraTS2017 --output /path/to/results
```

The five sequential outputs are `paths.p`, `pts.p`, `grid.p`, `dks.p`, and `dkDf.csv`. Existing checkpoints are reused unless `--overwrite` is given. Intermediate files are Python pickles: **only load pickles you trust**.

The default model grid matches the root `Example.py` (1,000 diffusion values, one million steps, 201 spatial cells), which can be computationally expensive. For a quick grid-stage experiment using already prepared inputs:

```bash
python Pipeline/run.py --output /path/to/test-results --from-stage grid --to-stage grid --n-d 2 --steps 100 --side-len 21 --save-freq 10
```

This command is for development only, not reproduction of manuscript results. To regenerate only the results table from an existing `dks.p` checkpoint:

```bash
python Pipeline/run.py --output /path/to/results --from-stage table
```

Avoid mixing checkpoints generated with different model settings in one output directory. For a changed grid configuration, use a new output directory or regenerate all affected downstream stages with `--overwrite`.

## Status

The original MRI discovery, radial preprocessing, PDE grid generation, patient fitting and output-table stages have been exercised individually on synthetic inputs. A combined five-stage synthetic NIfTI integration test passed in [run 39506621926](https://github.com/rafaelrbravo/DkMethod/actions/runs/39506621926); later patient-ID changes are being validated separately. **The full pipeline has not yet been executed against the real BraTS 2017 cohort** and manuscript results are not yet reproduced. Downstream clinical, radiomic, and molecular notebooks remain separate and are not integrated into this runner.

## Continuous integration status

The GitHub Actions workflow at `.github/workflows/numerical-tests.yml` runs real PDE numerical tests, synthetic NIfTI preprocessing, real model-grid and fitting integration, the combined end-to-end synthetic NIfTI test, pipeline/bridge regression tests, and archived cohort statistics on Python 3.11. The real fitting integration passed in [run 39350206549](https://github.com/rafaelrbravo/DkMethod/actions/runs/39350206549). The combined end-to-end test passed in [run 39506621926](https://github.com/rafaelrbravo/DkMethod/actions/runs/39506621926), before the later full-subject-ID update. These tests validate synthetic scientific calculations and software integration, **not** original BraTS patient inference or manuscript figures.

## Automated orchestration tests

From the repository root, run:

```bash
python -m unittest discover -s Tests -p 'test_pipeline.py' -v
```

These tests use mocked scientific modules and verify stage ordering, checkpoints, restart behavior, and argument validation. They do not test the numerical PDE implementation or reproduce the manuscript figures.

## Small real-model smoke tests

After installing the full scientific dependencies, run from the repository root:

```bash
python -m unittest discover -s Tests -p 'test_numerical_smoke.py' -v
```

These exercise the actual 1D model, test spherical finite-volume mass conservation across grid spacings, logistic growth, zero-growth/zero-diffusion invariance, shell volumes, and recovery of known synthetic diffusion/time parameters. They are not a substitute for reproducing the full patient results. If a scientific dependency is missing, these tests skip rather than claim a numerical pass.

## Real integration tests

Run the actual scientific implementations on synthetic data, without mocking numerical algorithms:

```bash
python -m unittest discover -s Tests -p 'test_real_grid_pipeline.py' -v
python -m unittest discover -s Tests -p 'test_real_fitting_pipeline.py' -v
python -m unittest discover -s Tests -p 'test_end_to_end_synthetic_nifti.py' -v
```

The full integration test generates a small BraTS-style NIfTI segmentation and FLAIR volume and runs all five stages with a deliberately tiny diffusion grid. These test parameters are **not suitable for estimating clinical D/k**. Use the published settings and real cohort data for manuscript reproduction. The archived clinical survival analysis, including its continuous Cox and proportional-hazards diagnostic, is described in `DOWNSTREAM.md`.

## Dataset requirements and reproducibility boundary

The image-discovery stage expects BraTS 2017 subject directories under the supplied `--brats` root. The unchanged original discovery code recognizes segmentation files ending in `_ManuallyCorrected.nii.gz` or `_GlistrBoost.nii.gz`, plus an MRI mask image (the original selection order prefers `_flair`, then `_t2`, `_t1ce`, and `_t1`). A generic `_seg.nii.gz` filename is **not** recognized by the historical loader. The publication runner constructs input records using complete BraTS subject directory names as patient IDs. This avoids the original legacy loader's four-character filename slice (`[8:12]`), which could merge distinct patients. Subject directory names must be unique, and a corresponding mask image must be present. Check the discovered IDs against your expected cohort; if the available dataset has a different layout, prepare a separate, traceable input copy or manifest rather than modifying the original images or collaborator code.

Use a new output directory for each distinct model parameter set, retain all five generated checkpoints and the exact command used, and keep the original BraTS source read-only. The small synthetic tests demonstrate software execution only. A publication-grade reproduction must additionally compare patient IDs and estimated D/k values with the archived historical cohort, reconcile exclusions, and regenerate the relevant downstream figures and statistics. Do **not** delete `Legacy/` on the strength of synthetic CI alone.
