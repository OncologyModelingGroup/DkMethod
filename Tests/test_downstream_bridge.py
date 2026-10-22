"""Tests for the explicit D/k downstream comparison bridge."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "Pipeline" / "prepare_downstream.py"
spec = importlib.util.spec_from_file_location("prepare_downstream", SCRIPT)
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


class DownstreamBridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.est = self.root / "est.csv"
        self.comp = self.root / "comp.csv"
        self.out = self.root / "merged.csv"
        pd.DataFrame({"id": ["A", "B"], "D": [1, 2], "k": [1, 1], "t": [2, 3], "D/k": [1, 2]}).to_csv(self.est, index=False)
        pd.DataFrame({"id": ["B", "A"], "Swanson D/k": [20, 10]}).to_csv(self.comp, index=False)

    def test_id_join_not_row_order(self):
        result = bridge.prepare(self.est, self.comp, self.out)
        self.assertEqual(result["Swanson D/k"].tolist(), [10, 20])
        self.assertTrue(self.out.exists())

    def test_no_overlap_rejected(self):
        pd.DataFrame({"id": ["Z"], "Swanson D/k": [5]}).to_csv(self.comp, index=False)
        with self.assertRaisesRegex(ValueError, "No matching"):
            bridge.prepare(self.est, self.comp, self.out)

    def test_duplicate_ids_rejected(self):
        pd.DataFrame({"id": ["A", "A"], "Swanson D/k": [5, 6]}).to_csv(self.comp, index=False)
        with self.assertRaisesRegex(ValueError, "unique"):
            bridge.prepare(self.est, self.comp, self.out)

    def test_missing_comparator_rejected(self):
        pd.DataFrame({"id": ["A"]}).to_csv(self.comp, index=False)
        with self.assertRaisesRegex(ValueError, "missing columns"):
            bridge.prepare(self.est, self.comp, self.out)

    def test_nonfinite_estimate_rejected(self):
        pd.DataFrame({"id": ["A"], "D": [float("inf")], "k": [1], "t": [2], "D/k": [1]}).to_csv(self.est, index=False)
        with self.assertRaisesRegex(ValueError, "finite numeric"):
            bridge.prepare(self.est, self.comp, self.out)

    def test_nonnumeric_comparison_rejected(self):
        pd.DataFrame({"id": ["A", "B"], "Swanson D/k": ["invalid", "10"]}).to_csv(self.comp, index=False)
        with self.assertRaisesRegex(ValueError, "numeric finite"):
            bridge.prepare(self.est, self.comp, self.out)

    def test_ratio_mismatch(self):
        pd.DataFrame({"id":["A"],"D":[2],"k":[1],"t":[2],"D/k":[9]}).to_csv(self.est,index=False)
        with self.assertRaisesRegex(ValueError,"agree with D divided by k"):
            bridge.prepare(self.est,self.comp,self.out)

    def test_zero_k(self):
        pd.DataFrame({"id":["A"],"D":[2],"k":[0],"t":[2],"D/k":[1]}).to_csv(self.est,index=False)
        with self.assertRaisesRegex(ValueError,"must be positive"):
            bridge.prepare(self.est,self.comp,self.out)

    def test_input_overwrite_rejected(self):
        with self.assertRaisesRegex(ValueError, "overwrite"):
            bridge.prepare(self.est, self.comp, self.est)


if __name__ == "__main__":
    unittest.main()
