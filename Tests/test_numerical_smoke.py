"""Small real numerical smoke tests, separate from mocked orchestration tests.

Run after installing the repository's scientific dependencies. These do not
require BraTS data and do not attempt manuscript-scale fitting.
"""
import unittest

import numpy as np


class NumericalSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import Model3Dand1D as model
        except ImportError as exc:
            raise unittest.SkipTest(f"Scientific dependency unavailable: {exc}")
        cls.model = model

    def test_small_radial_simulation_is_finite(self):
        params = self.model.ModelParams(d=1e-4, k=1e-4, steps=4, sideLen=21, dx=1, dt=1)
        result = self.model.RunModel1D(params, saveFreq=1)
        self.assertEqual(result.finalState.shape, (11,))
        self.assertEqual(result.intermediateStates.shape, (4, 11))
        self.assertTrue(np.isfinite(result.finalState).all())
        self.assertTrue((result.finalState >= 0).all())
        self.assertTrue((result.finalState <= 1).all())
        self.assertGreater(result.finalState.sum(), 0)

    def test_zero_diffusion_zero_growth_preserves_initial_field(self):
        params = self.model.ModelParams(d=0, k=0, steps=4, sideLen=21, dx=1, dt=1)
        initial = self.model.RunModel1D(self.model.ModelParams(d=0, k=0, steps=0, sideLen=21, dx=1, dt=1))
        final = self.model.RunModel1D(params)
        np.testing.assert_allclose(final.finalState, initial.finalState, rtol=1e-12, atol=1e-12)

    def test_diffusion_conserves_spherical_mass(self):
        """The radial finite-volume scheme has no-flux outer boundaries."""
        params = self.model.ModelParams(d=0.01, k=0, steps=10, sideLen=21, dx=1, dt=1)
        initial = self.model.RunModel1D(self.model.ModelParams(d=0, k=0, steps=0, sideLen=21, dx=1, dt=1))
        final = self.model.RunModel1D(params)
        volumes = self.model.GetShellVolArr(len(final.finalState))
        initial_mass = np.dot(initial.finalState, volumes)
        final_mass = np.dot(final.finalState, volumes)
        self.assertGreater(initial_mass, 0)
        np.testing.assert_allclose(final_mass, initial_mass, rtol=1e-4, atol=1e-9)

    def test_grid_search_recovers_known_diffusion_and_time(self):
        """An exact synthetic grid snapshot must be recoverable by the fitter."""
        grid = []
        for d in (1e-4, 1e-2):
            params = self.model.ModelParams(d=d, k=0, steps=8, sideLen=21, dx=1, dt=1)
            grid.append(self.model.RunModel1D(params, saveFreq=2))
        truth = self.model.ModelResults(
            params=grid[1].params, finalState=grid[1].intermediateStates[2].copy(),
            dimensons=1
        )
        recovered = self.model.GridSearch(truth, grid)
        self.assertEqual(recovered.d, 1e-2)
        self.assertEqual(recovered.steps, 4)

    def test_growth_without_diffusion_increases_mass(self):
        """Positive logistic growth should increase total cell density."""
        start = self.model.ModelParams(d=0, k=0, steps=0, sideLen=21, dx=1, dt=1)
        grown = self.model.ModelParams(d=0, k=0.1, steps=10, sideLen=21, dx=1, dt=1)
        initial = self.model.RunModel1D(start).finalState
        final = self.model.RunModel1D(grown).finalState
        volumes = self.model.GetShellVolArr(len(initial))
        self.assertGreater(np.dot(final, volumes), np.dot(initial, volumes))
        self.assertTrue(np.isfinite(final).all())

    def test_nonunit_spacing_mass(self):
        for dx in (0.5, 2.0):
            a = self.model.RunModel1D(self.model.ModelParams(0, 0, 0, 21, dx)).finalState
            b = self.model.RunModel1D(self.model.ModelParams(0.01, 0, 10, 21, dx)).finalState
            v = self.model.GetShellVolArr(len(a)) * dx**3
            np.testing.assert_allclose(a @ v, b @ v, rtol=1e-4, atol=1e-12)

    def test_shell_volumes_are_positive(self):
        volumes = self.model.GetShellVolArr(11)
        self.assertEqual(volumes.shape, (11,))
        self.assertTrue(np.isfinite(volumes).all())
        self.assertTrue((volumes > 0).all())


if __name__ == "__main__":
    unittest.main()
