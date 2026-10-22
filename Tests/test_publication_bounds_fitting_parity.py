"""Original-versus-independent LM fitting with publication time bounds.

Uses short synthetic PDE grid to isolate the important change from synthetic
fit bounds (80) to original publication LM bounds (1,000,000). This does not
claim equivalence for a full million-step grid.
"""
import tempfile
import unittest
from pathlib import Path
from Standalone.generate_synthetic import generate
from Tests.compare_private_patients import compare


class PublicationBoundsFittingParity(unittest.TestCase):
    def test_four_generated_subjects_publication_lm_bounds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = generate(Path(temporary) / "patients", n=4, seed=1729)
            compare(root, grid_size=3, steps=80, save_freq=10,
                    fit_max_steps=1_000_000)


if __name__ == "__main__":
    unittest.main()
