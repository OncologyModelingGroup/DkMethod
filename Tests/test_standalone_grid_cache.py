"""Verify cached and freshly computed standalone grids are identical."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from Standalone.run import build_grid


class GridCacheTest(unittest.TestCase):
    def test_roundtrip_and_reject_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "grid.npz"
            args = (3, 40, 10, 201, 1., 1., 1e-4)
            original = build_grid(*args, cache_path=cache)
            self.assertTrue(cache.exists())
            restored = build_grid(*args, cache_path=cache)
            self.assertEqual(len(original), len(restored))
            for a, b in zip(original, restored):
                self.assertEqual(a.params, b.params)
                np.testing.assert_array_equal(a.finalState, b.finalState)
                np.testing.assert_array_equal(a.intermediateStates, b.intermediateStates)
            with self.assertRaisesRegex(ValueError, "settings mismatch"):
                build_grid(3, 50, 10, 201, 1., 1., 1e-4, cache_path=cache)


if __name__ == "__main__":
    unittest.main()
