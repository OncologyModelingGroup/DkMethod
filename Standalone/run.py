"""Run the self-contained D/k method on BraTS-style NIfTI segmentations.

Example (from repository root):
    python -m Standalone.run --input /path/to/BraTS --output results \
        --grid-size 1000 --steps 1000000 --save-freq 1000

For a fast smoke test use --grid-size 3 --steps 100 --save-freq 10.
Default settings reproduce the published model/grid configuration but are
computationally expensive. No original DkMethod modules are imported.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .preprocessing import load_patient
from .model import ModelParams, ModelResult, run_model
from .fitting import fit_patient, result_row


def discover(input_dir):
    """Find one segmentation and one MRI mask per subject directory."""
    root = Path(input_dir)
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")
    subjects = []
    for directory in sorted([root, *root.rglob("*")]):
        if not directory.is_dir():
            continue
        files = [p for p in directory.iterdir() if p.is_file() and
                 (p.name.endswith(".nii") or p.name.endswith(".nii.gz"))]
        # Original FindSegPath prefers manually corrected segmentations,
        # then GlistrBoost. Also accept standard BraTS *_seg files.
        segs = []
        for token in ("_manuallycorrected.", "_glistrboost.", "_seg."):
            segs = [p for p in files if token in p.name.lower()]
            if segs:
                break
        if not segs:
            continue
        if len(segs) != 1:
            raise ValueError(f"Expected exactly one segmentation in {directory}")
        masks = []
        for token in ("_flair", "_t2", "_t1ce", "_t1"):
            masks = [p for p in files if token in p.name.lower() and p != segs[0]]
            if masks:
                break
        if not masks:
            raise ValueError(f"No MRI mask image in {directory}")
        if len(masks) > 1:
            raise ValueError(f"Ambiguous MRI mask images in {directory}: {masks}")
        subjects.append((directory.name, segs[0], masks[0]))
    if not subjects:
        raise ValueError(f"No *_seg*.nii[.gz] files under {root}")
    ids = [x[0] for x in subjects]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate subject directory names; cannot assign unique IDs")
    return subjects



def build_grid(grid_size, steps, save_freq, side_len, dx, dt, k, cache_path=None):
    """Build or load a model grid; reject caches with mismatched settings."""
    diffusion = np.logspace(-5, -2, grid_size)
    metadata = dict(grid_size=grid_size, steps=float(steps), save_freq=save_freq,
                    side_len=side_len, dx=float(dx), dt=float(dt), k=float(k))
    cache = Path(cache_path) if cache_path is not None else None
    if cache is not None and cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            stored = json.loads(str(data["metadata"].item()))
            if stored != metadata:
                raise ValueError(f"Grid cache settings mismatch: {cache}")
            saved_diffusion = data["diffusion"]
            if not np.array_equal(saved_diffusion, diffusion):
                raise ValueError(f"Grid cache diffusion values mismatch: {cache}")
            snapshots = data["snapshots"]
            final = data["final"]
            expected_count = (int(steps) - 1) // save_freq + 1 if steps >= 1 else 0
            if snapshots.shape != (grid_size, expected_count, side_len // 2 + 1):
                raise ValueError(f"Grid cache snapshots shape mismatch: {cache}")
            if final.shape != (grid_size, side_len // 2 + 1):
                raise ValueError(f"Grid cache final shape mismatch: {cache}")
            return [
                ModelResult(ModelParams(float(d), k, steps, side_len, dx, dt),
                            final[i].copy(), snapshots[i].copy(), save_freq)
                for i, d in enumerate(diffusion)
            ]
    grid = [run_model(ModelParams(float(d), k, steps, side_len, dx, dt), save_freq)
            for d in diffusion]
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        # Save atomically so interrupted jobs do not leave a valid-looking cache.
        temporary = cache.with_name(cache.name + ".partial.npz")
        try:
            np.savez(temporary, metadata=json.dumps(metadata, sort_keys=True),
                     diffusion=diffusion,
                     snapshots=np.stack([g.intermediateStates for g in grid]),
                     final=np.stack([g.finalState for g in grid]))
            temporary.replace(cache)
        finally:
            temporary.unlink(missing_ok=True)
    return grid


def run(input_dir, output_dir, grid_size=1000, steps=1e6, save_freq=1000,
        side_len=201, dx=1.0, dt=1.0, k=1e-4, fit_max_steps=1e6, grid_cache=None):
    """Discover patients, construct the grid, fit each patient, write dkDf.csv."""
    if grid_size < 1 or save_freq < 1 or steps <= 0 or side_len != 201 or fit_max_steps <= 0:
        raise ValueError("Require positive grid_size, steps, save_freq and side_len=201")
    if not np.isfinite([steps, dx, dt, k, fit_max_steps]).all():
        raise ValueError("All model and fit parameters must be finite")
    if dx <= 0 or dt <= 0 or k <= 0 or steps > fit_max_steps:
        raise ValueError("Require positive dx, dt, k and steps <= fit_max_steps")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    patients = [load_patient(seg, mask, patient_id) for patient_id, seg, mask in discover(input_dir)]
    grid = build_grid(grid_size, steps, save_freq, side_len, dx, dt, k, grid_cache)
    rows = [result_row(fit_patient(patient, grid, bounds=((-5, -2), (0, fit_max_steps)))) for patient in patients]
    csv_path = output / "dkDf.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "D", "k", "t", "D/k"])
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Root directory of BraTS patient folders")
    parser.add_argument("--output", required=True, help="Directory for dkDf.csv")
    parser.add_argument("--grid-size", type=int, default=1000)
    parser.add_argument("--steps", type=float, default=1e6)
    parser.add_argument("--save-freq", type=int, default=1000)
    parser.add_argument("--side-len", type=int, default=201)
    parser.add_argument("--dx", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--k", type=float, default=1e-4)
    parser.add_argument("--fit-max-steps", type=float, default=1e6, help="LM time upper bound; original default 1e6")
    parser.add_argument("--grid-cache", default=None, help="Optional reusable NPZ simulation grid; rejects mismatched settings")
    args = parser.parse_args()
    print(run(args.input, args.output, args.grid_size, args.steps,
              args.save_freq, args.side_len, args.dx, args.dt, args.k, args.fit_max_steps, args.grid_cache))


if __name__ == "__main__":
    main()
