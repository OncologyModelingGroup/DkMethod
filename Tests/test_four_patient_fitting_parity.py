"""End-to-end original versus independent fits on four generated NIfTI patients."""
import tempfile
import unittest
from pathlib import Path
from Standalone.generate_synthetic import generate
from Tests.compare_private_patients import compare


class FourPatientFittingParity(unittest.TestCase):
    def test_generated_four_patient_fits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = generate(Path(temporary) / "patients", n=4, seed=1729)
            compare(root, grid_size=3, steps=80, save_freq=10, fit_max_steps=80)


if __name__ == "__main__":
    unittest.main()
