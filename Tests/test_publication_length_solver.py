"""Publication-length PDE parity and measurable runtime smoke test.

Run from repository root:
    python -m unittest discover -s Tests -p test_publication_length_solver.py -v
"""
import time
import unittest
import numpy as np
import Model3Dand1D as original
from Standalone.model import ModelParams, run_model, njit


@unittest.skipIf(njit is None, "Numba not installed")
class PublicationLengthSolver(unittest.TestCase):
    def test_million_step_single_grid_entry(self):
        # Test the extremes and interior of the publication's log-spaced grid.
        for d in (1e-5, 3.1622776601683795e-5, 1e-4, 1e-3, 1e-2):
            with self.subTest(d=d):
                params = ModelParams(d, 1e-4, 1_000_000, 201)
                start = time.perf_counter()
                ours = run_model(params, save_freq=1000)
                elapsed = time.perf_counter() - start
                reference = original.RunModel1D(
                    original.ModelParams(params.d, params.k, params.steps, params.sideLen),
                    saveFreq=1000)
                np.testing.assert_allclose(ours.finalState, reference.finalState,
                                           rtol=2e-5, atol=1e-9)
                np.testing.assert_allclose(ours.intermediateStates, reference.intermediateStates,
                                           rtol=2e-5, atol=1e-9)
                self.assertEqual(ours.intermediateStates.shape, (1000, 101))
                print(f"Compiled million-step d={d:g}: {elapsed:.2f} seconds")


if __name__ == "__main__":
    unittest.main()
