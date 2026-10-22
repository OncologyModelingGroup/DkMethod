# Standalone D/k estimation

This directory contains an independently implemented, modifiable version of the
BraTS MRI-to-D/k pipeline. It does **not** import the original DkMethod scientific
modules. The original code remains available separately for scientific comparison.

## Installation

Use Python 3.11 or newer, from the repository root:

```bash
python -m pip install numpy scipy nibabel numba
```

## Input data

Create one directory per patient under an input root. Each directory must contain
one segmentation named with `_ManuallyCorrected.`, `_GlistrBoost.`, or `_seg.` and one MRI volume named with
`_flair`, `_t2`, `_t1ce`, or `_t1`, in `.nii` or `.nii.gz` format.
If multiple segmentation types are present, the original priority order is used: manually corrected, GlistrBoost, then standard `_seg`. The MRI is used as a nonzero-voxel mask; its intensities are **not** used as
tumor density. Segmentation labels 1 and 4 indicate enhancing tumor; label 2
indicates nonenhancing tumor. All arrays must be registered and have the same
shape. At least one enhancing voxel is required.

```text
input/
  Patient_A/
    Patient_A_seg.nii.gz
    Patient_A_flair.nii.gz
  Patient_B/
    Patient_B_seg.nii.gz
    Patient_B_flair.nii.gz
```

## Start with your own preprocessed arrays

For users who already have a registered 3D segmentation and MRI mask, the
self-contained [example_preprocessed.py](example_preprocessed.py) is the simplest
starting point. It shows the entire processing path in one editable file:
arrays → radial counts → simulation grid → LM fitting → CSV.

```bash
python -m Standalone.example_preprocessed
```

Replace the two explicitly marked synthetic arrays in `main()` with your
preprocessed arrays. The `estimate()` function can also be imported directly for one patient. For multiple patients, use `estimate_many([(patient_id, segmentation, mask), ...])`; it builds the expensive simulation grid **once** and reuses it across the cohort.
Its small defaults are for a fast demonstration, **not** publication analysis.
The example does not import the original scientific modules.

## Generate synthetic example patients

Create four reproducible, artificial BraTS-like NIfTI subjects (no private
patient data):

```bash
python -m Standalone.generate_synthetic --output synthetic-input
python -m Standalone.run --input synthetic-input --output synthetic-results --grid-size 3 --steps 100 --save-freq 10 --fit-max-steps 100
```

The generator accepts `--n`, `--shape`, and `--seed`. Generated segmentations
contain enhancing and nonenhancing regions; MRI files provide valid-voxel
masks. These are **software test fixtures**, not biologically realistic
clinical examples. The same generated subjects can be compared to the
original implementation:

```bash
python Tests/compare_private_patients.py synthetic-input --grid-size 3 --steps 100 --save-freq 10 --fit-max-steps 100
```

That comparison script expects exactly four subjects. Its default LM time bound is 100 for tractable synthetic tests; use `--fit-max-steps 1000000` (and a matching publication-length grid) to compare the original publication bounds. Both reference and independent optimizers use the same bound. The private-patient comparator uses parameter-specific strict tolerances (0.5% relative error with zero absolute tolerance for D, 1e-12 relative error for k, and 0.5% relative error plus one step for fitted time).

## Run

From the repository root:

```bash
python -m Standalone.run --input input --output results
```

The output is `results/dkDf.csv` with columns `id,D,k,t,D/k`.

For a quick *nonpublication* smoke test:

```bash
python -m Standalone.run --input input --output smoke --grid-size 3 --steps 100 --save-freq 10
```

Default settings match the original intended grid: 1,000 log-spaced diffusion
coefficients from 1e-5 to 1e-2, fixed proliferation rate 1e-4, 1,000,000
simulation steps, radial model side length 201, and grid snapshots every
1,000 steps. **These settings are computationally expensive.** The standalone PDE solver supports optional Numba acceleration, but publication-scale runtime must be measured on the target machine.

## Customize your data or method

- `preprocessing.py`: edit `load_patient` or supply segmentation and mask
  arrays directly to `radial_counts`. You can replace the MRI loading step
  with your own preprocessed 3D arrays.
- `model.py`: modify the spherical finite-volume PDE or grid settings.
- `fitting.py`: modify the weighted grid search or bounded LM optimizer.
- `run.py`: replace `discover` and `load_patient` with your own data
  ingestion; `fit_patient` accepts a `RadialPatient` object.

## Scientific validation status

The independent standalone implementation has passed original-versus-standalone
comparisons for preprocessing, PDE simulation, and patient fitting on generated
synthetic data. Automated CI includes four generated NIfTI patient fits using
publication LM time bounds, plus a one-patient fitting comparison with a real
**1,000,000-step simulation horizon**, **1,000 saved snapshots**, and three
diffusion coefficients. The compiled solver was separately compared with the
original at the million-step horizon across five diffusion coefficients.
The complete scientific numerical CI suite passed in
[run 39515208580](https://github.com/rafaelrbravo/DkMethod/actions/runs/39515208580).

**Full publication-grid synthetic equivalence has passed; real-patient equivalence remains unverified.**
A separate, optional [full-grid validation script](../Tests/compare_full_publication_grid.py)
compares one generated NIfTI patient's original and standalone fits with all
**1,000 diffusion coefficients** and **1,000,000 simulation steps**:

```bash
python -m Tests.compare_full_publication_grid
```

This comparison is computationally demanding. [GitHub Actions run 39520589934](https://github.com/rafaelrbravo/DkMethod/actions/runs/39520589934) completed successfully, reporting matching original and standalone fit parameters for the synthetic patient at printed precision (D = 1.39315720e-05, k = 1e-4, fitted steps = 2.23532533e+05). The run started before stricter per-parameter assertions were committed, so it establishes agreement at the precision reported in its logs rather than independently proving that the later stricter assertions passed. Real-patient equivalence remains unverified.
It can also be started manually from the
[Full publication-grid parity GitHub Actions workflow](https://github.com/rafaelrbravo/DkMethod/actions/workflows/full-publication-parity.yml).
It does not run on ordinary pushes. The four private real BraTS examples
must be validated separately and must **never be committed**.

For short synthetic smoke tests, set `--fit-max-steps` to the test horizon so
the optimizer cannot propose the original million-step maximum. The default
remains `1000000`, matching the original paper's fitting bounds; restricting
this parameter changes the fitted search space and is **not** appropriate
for asserting publication-setting equivalence.

## Performance

The standalone PDE solver optionally uses Numba to compile the entire time-stepping loop. Install `numba` (shown above) for publication-scale grids; without it, a pure-NumPy/Python fallback remains available but is much slower. The accelerated solver is independently compared against the Python solver and original model in `Tests/test_compiled_solver.py`. A passing test establishes agreement only for the tested configurations; it does not establish publication-scale runtime or full cohort equivalence.

## Reproducible performance measurement

From the repository root, measure steady-state compiled grid throughput and snapshot memory on your hardware:

```bash
python -m Tests.benchmark_standalone_grid --grid-size 10 --steps 100000
```

The benchmark excludes first-call JIT compilation, prints measured runtime and snapshot bytes, and separately labels a **linear extrapolation** for a 1,000-by-1,000,000 grid. Extrapolated time is not a measured full-grid runtime; patient fitting and MRI loading are excluded. The same benchmark runs in numerical CI.

### Measured CI performance (2026-10-20)

On an Ubuntu GitHub Actions runner, the compiled PDE grid benchmark measured **0.325 seconds** for 10 diffusion coefficients × 100,000 steps, or approximately **3.08 million time steps/second** (after JIT warmup). Linear extrapolation to 1,000 coefficients × 1,000,000 steps is **324.5 seconds (~5.4 minutes)** for grid construction alone; this was **not** measured end-to-end. Stored float32 snapshot arrays for that full grid occupy **404 MB**, excluding Python and working memory. Hardware, solver settings, compilation, and fitting can change total runtime. Source: GitHub Actions run `39514311490`, `Tests/benchmark_standalone_grid.py`.

## Reuse the publication simulation grid

For repeated cohort runs with the same numerical settings, pass `--grid-cache /path/to/grid.npz` to `python -m Standalone.run`. The first run computes and saves the grid; subsequent runs load it without rerunning the million-step PDE simulations. The cache records grid size, time horizon, snapshot frequency, spatial discretization, time step and growth rate. A settings mismatch **raises an error** rather than silently using a scientifically incompatible grid. Writes are atomic, and cache arrays use NumPy NPZ with pickle disabled. Delete the cache file to force regeneration. A full publication grid contains approximately 404 MB of snapshot arrays, plus final states and file overhead.
