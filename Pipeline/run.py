"""Run the published D/k estimation stages with explicit input and output paths.

The existing scientific implementations are imported unchanged from the repository root.
Each stage writes a checkpoint so expensive computations can be resumed independently.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def run(args):
    import ProcessPatientData as patient
    import Model3Dand1D as model
    import FitDkForEachPatient as fitting
    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    paths = out / "paths.p"
    patients = out / "pts.p"
    grid = out / "grid.p"
    fits = out / "dks.p"
    table = out / "dkDf.csv"

    stages = [
        ("paths", paths),
        ("patients", patients),
        ("grid", grid),
        ("fits", fits),
        ("table", table),
    ]
    stage_names = [name for name, _ in stages]
    start = stage_names.index(args.from_stage)
    stop = stage_names.index(args.to_stage)
    if start > stop:
        raise ValueError("--from-stage must precede --to-stage")
    required = {'patients': [paths], 'fits': [patients, grid], 'table': [fits]}
    for index in range(start, stop + 1):
        name, destination = stages[index]
        if destination.is_file() and not args.overwrite:
            if name == "grid":
                saved = pd.read_pickle(destination)
                expected_d = np.logspace(args.d_min_log, args.d_max_log, args.n_d)
                if len(saved) != args.n_d or any(
                    not np.isclose(result.params.d, d, rtol=1e-12, atol=0)
                    or result.params.k != args.k
                    or result.params.steps != args.steps
                    or result.params.sideLen != args.side_len
                    or result.params.dx != args.dx
                    or result.params.dt != args.dt
                    or result.saveFreq != args.save_freq
                    for result, d in zip(saved, expected_d)
                ):
                    raise ValueError("Existing grid checkpoint has different model parameters; use --overwrite to regenerate")
            print(f"Using existing {name}: {destination}", flush=True)
            continue
        for prerequisite in required.get(name, []):
            if not prerequisite.is_file():
                raise FileNotFoundError(f'Stage {name} requires {prerequisite}; generate its upstream stage first')
        if name == "paths":
            if args.brats is None:
                raise ValueError("--brats is required to generate paths")
            source = Path(args.brats).expanduser().resolve()
            if not source.is_dir():
                raise NotADirectoryError(f"BraTS input directory not found: {source}")
            # Build the same input records as the original discovery routine,
            # but use complete subject folder names instead of its four-character IDs.
            import os
            discovered = {}
            for folder, _, files in os.walk(source):
                segmentation = patient.FindSegPath(files)
                if segmentation is None:
                    continue
                mask = patient.FindMaskPath(files)
                if mask is None:
                    raise ValueError(f"No matching MRI mask found for {segmentation} in {folder}")
                patient_id = Path(folder).name
                if patient_id in discovered:
                    raise ValueError(f"Duplicate BraTS subject directory name {patient_id!r}: {folder}")
                discovered[patient_id] = {
                    "seg": segmentation,
                    "mask": mask,
                    "pt": patient_id,
                    "root": str(Path(folder).resolve()) + os.sep,
                }
            pd.to_pickle(discovered, destination)
            found = pd.read_pickle(destination)
            if not found:
                destination.unlink(missing_ok=True)
                raise ValueError(f"No segmented BraTS patients found in {source}; check dataset layout and filenames")
        elif name == "patients":
            patient.ProcessPts(pd.read_pickle(paths), str(destination), patient.GetLargestCom)
            processed = pd.read_pickle(destination)
            if not processed:
                destination.unlink(missing_ok=True)
                raise ValueError("Patient processing produced no usable patients")
        elif name == "grid":
            params = [model.ModelParams(d, k=args.k, steps=args.steps, sideLen=args.side_len, dx=args.dx, dt=args.dt) for d in np.logspace(args.d_min_log, args.d_max_log, args.n_d)]
            model.GenGrid(params, str(destination), saveFreq=args.save_freq)
            generated = pd.read_pickle(destination)
            if not generated:
                destination.unlink(missing_ok=True)
                raise ValueError("Model grid generation produced no simulations")
        elif name == "fits":
            fitting.GenPatientDks(pd.read_pickle(patients), pd.read_pickle(grid), str(destination))
            fitted = pd.read_pickle(destination)
            if not fitted:
                destination.unlink(missing_ok=True)
                raise ValueError("Patient fitting produced no estimates")
        else:
            fitting.GenPatientDkResDf(pd.read_pickle(fits), outPath=str(destination))
        if name == "table" and destination.is_file():
            generated_table = pd.read_csv(destination)
            if generated_table.empty or "D/k" not in generated_table.columns:
                destination.unlink(missing_ok=True)
                raise ValueError("Generated results table is empty or missing D/k estimates")
        if not destination.is_file():
            raise RuntimeError(f"Stage {name} did not produce {destination}")
        print(f"Completed {name}: {destination}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brats", help="Directory containing the BraTS 2017 patient folders")
    parser.add_argument("--output", required=True, help="Directory for generated checkpoints and results")
    parser.add_argument("--from-stage", choices=["paths", "patients", "grid", "fits", "table"], default="paths")
    parser.add_argument("--to-stage", choices=["paths", "patients", "grid", "fits", "table"], default="table")
    parser.add_argument("--overwrite", action="store_true", help="Recompute selected stages even when output exists")
    parser.add_argument("--n-d", type=int, default=1000)
    parser.add_argument("--d-min-log", type=float, default=-5)
    parser.add_argument("--d-max-log", type=float, default=-2)
    parser.add_argument("--k", type=float, default=1e-4)
    parser.add_argument("--steps", type=float, default=1e6)
    parser.add_argument("--side-len", type=int, default=201)
    parser.add_argument("--dx", type=float, default=1)
    parser.add_argument("--dt", type=float, default=1)
    parser.add_argument("--save-freq", type=int, default=1000)
    args = parser.parse_args(argv)
    if args.n_d < 1 or args.steps < 1 or args.save_freq < 1 or args.side_len < 2 or args.k <= 0 or args.dx <= 0 or args.dt <= 0:
        parser.error("n-d, save-freq, k, dx, dt must be positive; steps >= 1 and side-len >= 2")
    if args.steps <= args.save_freq:
        parser.error("steps must exceed save-freq to generate at least two distinct grid snapshots")
    if not all(np.isfinite(v) for v in (args.d_min_log, args.d_max_log, args.k, args.steps, args.dx, args.dt)):
        parser.error("Nonfinite model parameter")
    if args.d_min_log >= args.d_max_log or args.n_d < 2:
        parser.error("Invalid diffusion grid")
    log_stability = np.log(6) + args.d_max_log * np.log(10) + np.log(args.dt) - 2 * np.log(args.dx)
    if not np.isfinite(log_stability) or log_stability > 0:
        parser.error("Explicit radial diffusion may be unstable: require 6 * max(D) * dt / dx**2 <= 1")
    return args


if __name__ == "__main__":
    run(parse_args())
