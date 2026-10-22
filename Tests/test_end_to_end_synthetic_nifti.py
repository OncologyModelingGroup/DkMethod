"""Full original BraTS-to-D/k pipeline integration using generated NIfTI images."""
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from Pipeline.run import parse_args, run


class EndToEndSyntheticNiftiTests(unittest.TestCase):
    def test_nifti_to_patient_dk_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scans = root / "brats"
            folder = scans / "HGG" / "BraTS17_2013_001_1"
            folder.mkdir(parents=True)
            prefix = "BraTS17_2013_001_1"
            seg = np.zeros((13, 13, 13), dtype=np.float32)
            seg[4:9, 4:9, 4:9] = 2
            seg[5:8, 5:8, 5:8] = 4
            flair = np.ones_like(seg)
            nib.save(nib.Nifti1Image(seg, np.eye(4)), folder / (prefix + "_ManuallyCorrected.nii.gz"))
            nib.save(nib.Nifti1Image(flair, np.eye(4)), folder / (prefix + "_flair.nii.gz"))
            output = root / "results"
            args = parse_args(["--brats", str(scans), "--output", str(output), "--n-d", "2", "--d-min-log", "-5", "--d-max-log", "-4", "--steps", "20", "--save-freq", "5", "--side-len", "201"])
            run(args)
            self.assertTrue(all((output / name).is_file() for name in ("paths.p", "pts.p", "grid.p", "dks.p", "dkDf.csv")))
            patients = pd.read_pickle(output / "pts.p")
            table = pd.read_csv(output / "dkDf.csv", dtype={"id": str})
            self.assertEqual(len(patients), 1)
            self.assertEqual(len(table), 1)
            self.assertEqual(next(iter(patients)), prefix)
            self.assertEqual(table.loc[0, "id"], prefix)
            self.assertTrue(np.isfinite(table.loc[0, ["D", "k", "t", "D/k"]].to_numpy(dtype=float)).all())
            self.assertGreater(table.loc[0, "D/k"], 0)


if __name__ == "__main__":
    unittest.main()
