"""Exercise editable preprocessed-array example without NIfTI file loading."""
import unittest
import numpy as np
from Standalone.example_preprocessed import estimate, estimate_many


class PreprocessedExample(unittest.TestCase):
    def test_direct_arrays_to_dk(self):
        xyz = np.indices((19, 19, 19))
        r2 = sum((xyz[i] - 9) ** 2 for i in range(3))
        segmentation = np.zeros((19, 19, 19), dtype=np.int16)
        segmentation[r2 <= 5 ** 2] = 2
        segmentation[r2 <= 3 ** 2] = 4
        result = estimate(segmentation, np.ones_like(segmentation),
                          "preprocessed", grid_size=2, steps=10,
                          save_freq=5, fit_max_steps=10)
        self.assertEqual(result["id"], "preprocessed")
        self.assertTrue(np.isfinite(result["D/k"]))
        self.assertAlmostEqual(result["D/k"], result["D"] / result["k"])

    def test_cohort_reuses_grid(self):
        xyz = np.indices((19, 19, 19))
        r2 = sum((xyz[i] - 9) ** 2 for i in range(3))
        segmentation = np.zeros((19, 19, 19), dtype=np.int16)
        segmentation[r2 <= 5 ** 2] = 2
        segmentation[r2 <= 3 ** 2] = 4
        mask = np.ones_like(segmentation)
        results = estimate_many(
            [("patient_A", segmentation, mask), ("patient_B", segmentation, mask)],
            grid_size=2, steps=10, save_freq=5, fit_max_steps=10)
        self.assertEqual([x["id"] for x in results], ["patient_A", "patient_B"])
        self.assertAlmostEqual(results[0]["D/k"], results[1]["D/k"])



if __name__ == "__main__":
    unittest.main()
