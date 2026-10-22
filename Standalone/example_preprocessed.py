"""Minimal editable standalone D/k pipeline using your preprocessed arrays.

Run from the repository root:
    python -m Standalone.example_preprocessed

Replace the two example arrays in main() with your own 3D segmentation and
registered nonzero MRI mask. No original paper modules are imported.
"""
import csv
from pathlib import Path
import numpy as np
from .preprocessing import radial_counts
from .model import ModelParams, run_model
from .fitting import fit_patient, result_row


def estimate_many(patients, grid_size=3, steps=100, save_freq=10,
                  fit_max_steps=100):
    """Fit iterable of (patient_id, segmentation, mask) with one shared grid.

    This is the preferred interface for cohorts: simulations are constructed
    only once, rather than repeated for every patient.
    """
    if steps <= 0 or fit_max_steps < steps or grid_size < 1 or save_freq < 1:
        raise ValueError("Invalid grid or fitting horizon")
    grid = [
        run_model(ModelParams(float(d), 1e-4, steps, sideLen=201), save_freq)
        for d in np.logspace(-5, -2, grid_size)
    ]
    results = []
    for patient_id, segmentation, mask in patients:
        patient = radial_counts(segmentation, mask, patient_id)
        results.append(result_row(fit_patient(
            patient, grid, bounds=((-5, -2), (0, fit_max_steps)))))
    return results


def estimate(segmentation, mask, patient_id, grid_size=3, steps=100,
             save_freq=10, fit_max_steps=100):
    """Return fitted D/k for one preprocessed 3D patient.

    Segmentation: 3D array with BraTS labels 0,1,2,4.
    Mask: aligned 3D array, voxels >0 indicate valid tissue.
    Use publication settings (1000, 1e6, 1000, 1e6) only after validation.
    """
    return estimate_many(
        [(patient_id, segmentation, mask)], grid_size, steps,
        save_freq, fit_max_steps)[0]


def main():
    # REPLACE these two arrays with your own registered, preprocessed data.
    xyz = np.indices((25, 25, 25))
    r2 = sum((xyz[axis] - 12) ** 2 for axis in range(3))
    segmentation = np.zeros((25, 25, 25), dtype=np.int16)
    segmentation[r2 <= 7 ** 2] = 2
    segmentation[r2 <= 4 ** 2] = 4
    mask = np.ones_like(segmentation)

    result = estimate(segmentation, mask, "example_patient")
    destination = Path("example_dkDf.csv")
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "D", "k", "t", "D/k"])
        writer.writeheader()
        writer.writerow(result)
    print(f"Wrote {destination}: {result}")


if __name__ == "__main__":
    main()
