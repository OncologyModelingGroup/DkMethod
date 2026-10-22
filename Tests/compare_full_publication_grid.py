"""Optional full-publication-grid parity check; runs in a separate manual CI workflow.

Run from repository root:
    python -m Tests.compare_full_publication_grid

Generates one reproducible synthetic NIfTI patient, independently computes
both 1000-coefficient / million-step grids, and compares original and
standalone D/k fits. Expect substantial RAM and compute time. This is NOT
a substitute for validation on the four private BraTS subjects.
"""
import gc
import tempfile
from pathlib import Path

import numpy as np

import ProcessPatientData as original_pre
import Model3Dand1D as original_model
import FitDkForEachPatient as original_fit
from Standalone.generate_synthetic import generate
from Standalone.run import discover, build_grid
from Standalone.preprocessing import load_patient
from Standalone.fitting import fit_patient


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = generate(Path(directory) / "patients", n=1, seed=1729)
        patient_id, seg, mask = discover(root)[0]
        patient = load_patient(seg, mask, patient_id)
        record = {"pt": patient_id, "root": str(seg.parent) + "/",
                  "seg": seg.name, "mask": mask.name}
        reference = original_pre.GenRadialCounts(
            original_pre.GenPt(record, original_pre.GetLargestCom))
        np.testing.assert_array_equal(patient.counts, reference.counts)

        diffusion = np.logspace(-5, -2, 1000)
        independent = build_grid(1000, 1_000_000, 1000, 201, 1, 1, 1e-4)
        actual_fit = fit_patient(patient, independent,
                                 bounds=((-5, -2), (0, 1_000_000)))
        actual = np.array([actual_fit.params.d, actual_fit.params.k,
                           actual_fit.params.steps])
        del independent
        gc.collect()

        original_grid = [
            original_model.RunModel1D(
                original_model.ModelParams(float(d), 1e-4, 1_000_000, 201),
                saveFreq=1000)
            for d in diffusion
        ]
        target = original_fit.PtRadialToModelResults(
            original_fit.GenRadialPt(reference, 0.9, 0.45, 101), "radial")
        expected_fit = original_model.GridLmSearch(
            target, original_grid, ((-5, -2), (0, 1_000_000)))
        expected = np.array([expected_fit.params.d, expected_fit.params.k,
                             expected_fit.params.steps])
        print("standalone:", actual, "original:", expected, "difference:", actual - expected)
        np.testing.assert_allclose(actual[0], expected[0], rtol=5e-3, atol=0, err_msg="D mismatch")
        np.testing.assert_allclose(actual[1], expected[1], rtol=1e-12, atol=0, err_msg="k mismatch")
        np.testing.assert_allclose(actual[2], expected[2], rtol=5e-3, atol=1.0, err_msg="time mismatch")
        print("PASS: full 1000-coefficient publication-grid fit parity")


if __name__ == "__main__":
    main()
