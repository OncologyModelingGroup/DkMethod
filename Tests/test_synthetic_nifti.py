"""Exercise original BraTS NIfTI discovery and radial preprocessing on synthetic scans.

No patient data are distributed. This verifies the non-mocked paths -> patients
stages, including NIfTI loading, segmentation remapping, and 100 radial bins.
"""
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

import ProcessPatientData as patient


class SyntheticNiftiTests(unittest.TestCase):
    def test_discovery_and_radial_processing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "HGG" / "BraTS17_2013_001_1"
            folder.mkdir(parents=True)
            prefix = "BraTS17_2013_001_1"
            seg = np.zeros((13, 13, 13), dtype=np.float32)
            seg[5:8, 5:8, 5:8] = 4
            seg[4:9, 4:9, 4:9] = np.where(seg[4:9, 4:9, 4:9] == 4, 4, 2)
            flair = np.ones_like(seg)
            nib.save(nib.Nifti1Image(seg, np.eye(4)), folder / (prefix + "_ManuallyCorrected.nii.gz"))
            nib.save(nib.Nifti1Image(flair, np.eye(4)), folder / (prefix + "_flair.nii.gz"))
            paths_file = root / "paths.p"
            patients_file = root / "pts.p"
            patient.GenBratsPaths(str(root), str(paths_file))
            paths = pd.read_pickle(paths_file)
            self.assertEqual(len(paths), 1)
            patient.ProcessPts(paths, str(patients_file), patient.GetLargestCom)
            results = pd.read_pickle(patients_file)
            self.assertEqual(set(results), set(paths))
            radial = next(iter(results.values()))
            self.assertEqual(len(radial.counts), 101)
            self.assertTrue(np.isfinite(radial.com).all())
            self.assertGreater(radial.enh.sum(), 0)
            self.assertGreater(radial.non.sum(), 0)
            self.assertTrue(np.isfinite(radial.counts).all())


if __name__ == "__main__":
    unittest.main()
