"""Verify optional compiled solver against independent Python and original PDE."""
import unittest
import numpy as np
import Model3Dand1D as original
from Standalone.model import ModelParams, run_model, njit


@unittest.skipIf(njit is None, "Numba optional acceleration not installed")
class CompiledSolverParity(unittest.TestCase):
    def test_compiled_python_original(self):
        for d, k, steps, dx, dt in [
            (1e-5, 1e-4, 31, 1., 1.),
            (1e-3, 1e-2, 45.5, 1., 0.5),
            (1e-2, 1e-3, 73, 2., 1.),
        ]:
            with self.subTest(d=d, steps=steps):
                params = ModelParams(d, k, steps, 201, dx, dt)
                compiled = run_model(params, save_freq=7, accelerated=True)
                python = run_model(params, save_freq=7, accelerated=False)
                reference = original.RunModel1D(
                    original.ModelParams(d, k, steps, 201, dx, dt), saveFreq=7)
                for attribute in ("finalState", "intermediateStates"):
                    a = getattr(compiled, attribute)
                    np.testing.assert_allclose(a, getattr(python, attribute), rtol=2e-5, atol=1e-10)
                    np.testing.assert_allclose(a, getattr(reference, attribute), rtol=2e-5, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
