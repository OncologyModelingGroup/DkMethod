"""Original-versus-standalone fitting tests using a small synthetic model grid.

Run: python -m unittest discover -s Tests -p test_standalone_fitting.py -v
"""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Standalone.model import ModelParams, run_model, shell_volumes
from Standalone.fitting import levenberg_marquardt, fit_patient, result_row
from Standalone.preprocessing import RadialPatient


class FittingEquivalence(unittest.TestCase):
    def test_lm_on_known_linear_problem(self):
        from LMclassNew import LevenbergMarquardt
        x0 = np.array([0.0, 0.0])
        y = np.array([0.3, 0.7, 1.0])
        def fn(x):
            return np.array([x[0], x[1], x[0] + x[1]])
        bounds = ((-5, 5), (-5, 5))
        reference = LevenbergMarquardt(fn, bounds, x0, y)
        reference.DefaultRun(0.0001)
        x, prediction, _, _ = levenberg_marquardt(fn, x0, y, bounds)
        np.testing.assert_allclose(x, reference.bestX, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(prediction, reference.bestY, atol=1e-6, rtol=1e-6)

    def test_patient_fit_matches_original(self):
        import Model3Dand1D as original
        import FitDkForEachPatient as fit_original
        import ProcessPatientData as ppd
        from dataclasses import replace

        grid_params = [ModelParams(d, 1e-4, 80, 201) for d in (1e-5, 1e-4, 1e-3)]
        independent_grid = [run_model(p, save_freq=10) for p in grid_params]
        original_grid = [
            original.RunModel1D(original.ModelParams(p.d, p.k, p.steps, p.sideLen, p.dx, p.dt), saveFreq=10)
            for p in grid_params
        ]
        # Synthetic target: an actual model checkpoint, converted into
        # pseudo-patient radial counts with known densities.
        target = original_grid[1].intermediateStates[4]
        counts = np.full(101, 1000.0)
        enh = np.minimum(counts, np.maximum(0, target / 0.9 * counts))
        non = np.zeros(101)
        patient = RadialPatient("synthetic", np.zeros(3), counts, np.zeros(101), non, enh)
        ours = fit_patient(patient, independent_grid)
        old_counts = ppd.PtRadialCounts(counts, np.zeros(101), non, enh, np.zeros(3), "synthetic")
        old_density = fit_original.GenRadialPt(old_counts, 0.9, 0.45, 101)
        old_truth = fit_original.PtRadialToModelResults(old_density, "radial")
        reference = original.GridLmSearch(old_truth, original_grid, ((-5, -2), (0, 1e6)))
        self.assertEqual(result_row(ours)["id"], reference.id)
        np.testing.assert_allclose(
            [ours.params.d, ours.params.k, ours.params.steps],
            [reference.params.d, reference.params.k, reference.params.steps],
            rtol=5e-3, atol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
