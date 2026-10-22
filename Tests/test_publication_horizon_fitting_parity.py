"""Original versus standalone full million-step grid and fitting comparison.

One generated NIfTI patient, 3 diffusion coefficients, 1,000,000 simulation
steps, 1,000 snapshots, and publication LM bounds. This tests full horizon
and time resolution but NOT the publication's full 1,000-coefficient grid.
"""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import ProcessPatientData as original_pre
import Model3Dand1D as original_model
import FitDkForEachPatient as original_fit
from Standalone.generate_synthetic import generate
from Standalone.run import discover
from Standalone.preprocessing import load_patient
from Standalone.model import ModelParams, run_model
from Standalone.fitting import fit_patient


class PublicationHorizonFittingParity(unittest.TestCase):
    def test_one_generated_patient_million_step_fitting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = generate(Path(directory) / "patients", n=1, seed=1729)
            patient_id, seg, mask = discover(root)[0]
            patient = load_patient(seg, mask, patient_id)
            path = {"pt": patient_id, "root": str(seg.parent) + "/",
                    "seg": seg.name, "mask": mask.name}
            reference = original_pre.GenRadialCounts(
                original_pre.GenPt(path, original_pre.GetLargestCom))
            np.testing.assert_array_equal(patient.counts, reference.counts)

            diffusion = np.logspace(-5, -2, 3)
            independent_grid = [
                run_model(ModelParams(float(d), 1e-4, 1_000_000, 201), 1000)
                for d in diffusion]
            reference_grid = [
                original_model.RunModel1D(
                    original_model.ModelParams(float(d), 1e-4, 1_000_000, 201),
                    saveFreq=1000)
                for d in diffusion]
            fitted = fit_patient(patient, independent_grid,
                                 bounds=((-5, -2), (0, 1_000_000)))
            truth = original_fit.PtRadialToModelResults(
                original_fit.GenRadialPt(reference, 0.9, 0.45, 101), "radial")
            expected = original_model.GridLmSearch(
                truth, reference_grid, ((-5, -2), (0, 1_000_000)))
            actual = np.array([fitted.params.d, fitted.params.k, fitted.params.steps])
            wanted = np.array([expected.params.d, expected.params.k, expected.params.steps])
            print("million-step fit: standalone=", actual, "original=", wanted)
            np.testing.assert_allclose(actual[0], wanted[0], rtol=5e-3, atol=0, err_msg="D mismatch")
            np.testing.assert_allclose(actual[1], wanted[1], rtol=1e-12, atol=0, err_msg="k mismatch")
            np.testing.assert_allclose(actual[2], wanted[2], rtol=5e-3, atol=1.0, err_msg="time mismatch")


if __name__ == "__main__":
    unittest.main()
