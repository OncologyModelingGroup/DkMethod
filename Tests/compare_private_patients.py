"""Compare standalone and original D/k fitting on four local real BraTS patients.

No patient data is uploaded or committed. Default fit bound 100 is for fast parity tests, not publication settings. Provide an input root containing
patient folders with *_seg.nii[.gz] and *_flair (or other MRI) NIfTI files.

Usage:
    python Tests/compare_private_patients.py /private/BraTS --grid-size 3 --steps 100 --save-freq 10

Uses identical original and standalone grid configurations. Defaults are
deliberately small for validation; increase settings for publication runs.
"""
import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ProcessPatientData as original_pre
import Model3Dand1D as original_model
import FitDkForEachPatient as original_fit
from Standalone.run import discover
from Standalone.preprocessing import load_patient
from Standalone.model import ModelParams, run_model
from Standalone.fitting import fit_patient


def compare(root, grid_size, steps, save_freq, fit_max_steps=100):
    if fit_max_steps < steps:
        raise ValueError('fit_max_steps must be >= steps')
    subjects = discover(root)
    if len(subjects) != 4:
        raise ValueError(f"Expected exactly four private example patients; found {len(subjects)}")
    diffusion = np.logspace(-5, -2, grid_size)
    independent_grid = [
        run_model(ModelParams(float(d), 1e-4, steps, 201), save_freq)
        for d in diffusion
    ]
    reference_grid = [
        original_model.RunModel1D(
            original_model.ModelParams(float(d), 1e-4, steps, 201),
            saveFreq=save_freq
        ) for d in diffusion
    ]
    failures = []
    for patient_id, seg, mask in subjects:
        ours = load_patient(seg, mask, patient_id)
        paths = {patient_id: {
            "pt": patient_id, "root": str(seg.parent) + "/",
            "seg": seg.name, "mask": mask.name
        }}
        reference = original_pre.GenRadialCounts(
            original_pre.GenPt(paths[patient_id], original_pre.GetLargestCom)
        )
        for name in ("counts", "empty", "non", "enh"):
            np.testing.assert_array_equal(getattr(ours, name), getattr(reference, name),
                                          err_msg=f"{patient_id}: {name}")
        np.testing.assert_allclose(ours.center, reference.com, rtol=0, atol=1e-12)
        fitted = fit_patient(ours, independent_grid, bounds=((-5, -2), (0, fit_max_steps)))
        truth = original_fit.PtRadialToModelResults(
            original_fit.GenRadialPt(reference, 0.9, 0.45, 101), "radial"
        )
        expected = original_model.GridLmSearch(
            truth, reference_grid, ((-5, -2), (0, fit_max_steps))
        )
        got = np.array([fitted.params.d, fitted.params.k, fitted.params.steps])
        want = np.array([expected.params.d, expected.params.k, expected.params.steps])
        print(f"{patient_id}: standalone={got} original={want} abs_diff={abs(got-want)}")
        try:
            np.testing.assert_allclose(got[0], want[0], rtol=5e-3, atol=0, err_msg=f"{patient_id}: D mismatch")
            np.testing.assert_allclose(got[1], want[1], rtol=1e-12, atol=0, err_msg=f"{patient_id}: k mismatch")
            np.testing.assert_allclose(got[2], want[2], rtol=5e-3, atol=1.0, err_msg=f"{patient_id}: time mismatch")
        except AssertionError as error:
            failures.append((patient_id, str(error)))
    if failures:
        raise AssertionError(f"Original-versus-standalone fit differs: {failures}")
    print("PASS: all four patient preprocessing and fitting comparisons")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--grid-size", type=int, default=3)
    parser.add_argument("--steps", type=float, default=100)
    parser.add_argument("--save-freq", type=int, default=10)
    parser.add_argument('--fit-max-steps', type=float, default=100,
                        help='LM time upper bound; use 1000000 for original publication bounds')
    args = parser.parse_args()
    compare(args.root, args.grid_size, args.steps, args.save_freq, args.fit_max_steps)


if __name__ == "__main__":
    main()
