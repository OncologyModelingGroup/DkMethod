"""Exercise the complete independent NIfTI -> D/k CSV path on synthetic data.

No private patient data is used or committed. Run from repository root:
    python -m unittest discover -s Tests -p test_standalone_end_to_end.py -v
"""
import csv
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from Standalone.run import run, discover


class StandaloneEndToEnd(unittest.TestCase):
    def test_synthetic_nifti_to_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "patients"
            for patient_id, center in [("synthetic_A", (12, 12, 12)), ("synthetic_B", (13, 11, 12))]:
                folder = input_dir / patient_id
                folder.mkdir(parents=True)
                xyz = np.indices((25, 25, 25))
                d2 = sum((xyz[i] - center[i]) ** 2 for i in range(3))
                segmentation = np.zeros((25, 25, 25), dtype=np.int16)
                segmentation[d2 <= 7**2] = 2
                segmentation[d2 <= 4**2] = 4
                mask = np.ones_like(segmentation)
                nib.save(nib.Nifti1Image(segmentation, np.eye(4)), str(folder / f"{patient_id}_seg.nii.gz"))
                nib.save(nib.Nifti1Image(mask, np.eye(4)), str(folder / f"{patient_id}_flair.nii.gz"))
            self.assertEqual(len(discover(input_dir)), 2)
            result = run(input_dir, root / "out", grid_size=2, steps=20, save_freq=5, fit_max_steps=20)
            self.assertTrue(result.exists())
            with result.open(newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, ["id", "D", "k", "t", "D/k"])
                rows = list(reader)
            self.assertEqual([row["id"] for row in rows], ["synthetic_A", "synthetic_B"])
            for row in rows:
                d, k, t, ratio = (float(row[key]) for key in ("D", "k", "t", "D/k"))
                self.assertTrue(np.all(np.isfinite([d, k, t, ratio])))
                self.assertGreater(d, 0)
                self.assertGreater(k, 0)
                self.assertAlmostEqual(ratio, d / k, places=8)


if __name__ == "__main__":
    unittest.main()
