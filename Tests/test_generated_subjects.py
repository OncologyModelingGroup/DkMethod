"""Check reproducible four-subject NIfTI generation and preprocessing parity."""
import tempfile
import unittest
from pathlib import Path
import numpy as np

import ProcessPatientData as original
from Standalone.generate_synthetic import generate
from Standalone.run import discover
from Standalone.preprocessing import load_patient


class GeneratedSubjects(unittest.TestCase):
    def test_four_generated_subjects_match_original_preprocessing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = generate(Path(temp) / "generated", n=4, seed=1729)
            subjects = discover(root)
            self.assertEqual(len(subjects), 4)
            signatures = []
            for patient_id, seg, mask in subjects:
                ours = load_patient(seg, mask, patient_id)
                paths = {"root": str(seg.parent) + "/", "seg": seg.name,
                         "mask": mask.name, "pt": patient_id}
                reference = original.GenRadialCounts(
                    original.GenPt(paths, original.GetLargestCom))
                np.testing.assert_allclose(ours.center, reference.com, rtol=0, atol=1e-12)
                for name in ("counts", "empty", "non", "enh"):
                    np.testing.assert_array_equal(getattr(ours, name), getattr(reference, name))
                signatures.append((tuple(ours.center), tuple(ours.enh[:15])))
            self.assertEqual(len(set(signatures)), 4)


if __name__ == "__main__":
    unittest.main()
