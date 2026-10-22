"""Exercise real model-grid generation and checkpointing with small stable parameters."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from Pipeline.run import parse_args, run


class RealGridIntegrationTests(unittest.TestCase):
    def test_grid_generation_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = parse_args(["--output", tmp, "--from-stage", "grid", "--to-stage", "grid", "--n-d", "2", "--d-min-log", "-5", "--d-max-log", "-4", "--steps", "20", "--save-freq", "5", "--side-len", "21"])
            run(args)
            checkpoint = Path(tmp) / "grid.p"
            self.assertTrue(checkpoint.is_file())
            grid = pd.read_pickle(checkpoint)
            self.assertEqual(len(grid), 2)
            for result, diffusion in zip(grid, np.logspace(-5, -4, 2)):
                self.assertAlmostEqual(result.params.d, diffusion)
                self.assertEqual(result.saveFreq, 5)
                self.assertGreaterEqual(len(result.intermediateStates), 2)
                self.assertTrue(np.isfinite(result.finalState).all())
                self.assertTrue(np.isfinite(result.intermediateStates).all())
            run(args)


if __name__ == "__main__":
    unittest.main()
