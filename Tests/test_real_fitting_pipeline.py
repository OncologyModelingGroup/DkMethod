"""Run original grid-search/Levenberg-Marquardt patient fitting through pipeline stages.

Uses a synthetic 101-bin radial profile, not clinical MRI. The original fitting
functions and PDE solver are never mocked. Keep this tiny for CI.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from Pipeline.run import parse_args, run
from ProcessPatientData import PtRadialCounts


class RealFittingIntegrationTests(unittest.TestCase):
    def test_real_fit_and_results_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            counts = np.ones(101, dtype=np.float64)
            enhancing = np.zeros(101, dtype=np.float64)
            nonenhancing = np.zeros(101, dtype=np.float64)
            enhancing[0] = 1
            nonenhancing[1:4] = 1
            patient = PtRadialCounts(counts, np.zeros(101), nonenhancing, enhancing, np.zeros(3), "synthetic")
            pd.to_pickle({"synthetic": patient}, root / "pts.p")
            args = parse_args(["--output", tmp, "--from-stage", "grid", "--to-stage", "table", "--n-d", "2", "--d-min-log", "-5", "--d-max-log", "-4", "--steps", "20", "--save-freq", "5", "--side-len", "201"])
            run(args)
            table = pd.read_csv(root / "dkDf.csv")
            self.assertEqual(len(table), 1)
            self.assertEqual(table.loc[0, "id"], "synthetic")
            self.assertTrue(np.isfinite(table.loc[0, ["D", "k", "t", "D/k"]].to_numpy(dtype=float)).all())
            self.assertGreater(table.loc[0, "D/k"], 0)
            self.assertAlmostEqual(table.loc[0, "D/k"], table.loc[0, "D"] / table.loc[0, "k"])
            self.assertEqual(len(pd.read_pickle(root / "dks.p")), 1)


if __name__ == "__main__":
    unittest.main()
