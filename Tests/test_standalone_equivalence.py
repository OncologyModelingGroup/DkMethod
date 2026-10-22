"""Numerical parity tests for independently implemented standalone D/k stages.

Run from repository root:
    python -m unittest discover -s Tests -p 'test_standalone_equivalence.py' -v

Uses deterministic in-memory synthetic segmentations; no patient files committed.
Original modules are imported only by this comparison test, never by the
standalone implementations. Numerical tolerances allow floating-point drift.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Standalone.preprocessing import radial_counts
from Standalone.model import ModelParams, run_model, shell_volumes, grid_search


class PreprocessingEquivalence(unittest.TestCase):
    def test_original_radial_counts(self):
        import ProcessPatientData as original
        shape = (33, 33, 33)
        xyz = np.indices(shape)
        cases = [
            ((16, 16, 16), (4, 7), False),
            ((12, 19, 15), (3, 8), False),
            ((17, 15, 16), (5, 9), True),
        ]
        for center, (r_enh, r_non), clipped in cases:
            with self.subTest(center=center, clipped=clipped):
                d2 = sum((xyz[i] - center[i]) ** 2 for i in range(3))
                seg = np.zeros(shape, dtype=np.int16)
                seg[d2 <= r_non**2] = 2
                seg[d2 <= r_enh**2] = 4
                seg[center] = 1
                mask = np.ones(shape, dtype=np.int16)
                if clipped:
                    mask[:8] = 0
                    mask[:, :5] = 0
                ours = radial_counts(seg, mask, "synthetic")
                mapped = np.where((seg == 1) | (seg == 4), original.ENH, np.where(seg == 2, original.NON, seg))
                original_mask = np.where(mask > 0, 1, -1)
                mapped = np.where(original_mask > 0, mapped, 0)
                center_original = original.GetLargestCom(mapped, original_mask)
                reference = original.GetRadialDist100(mapped, original_mask, center_original)
                np.testing.assert_allclose(ours.center, center_original, atol=1e-12, rtol=0)
                for got, expected in zip((ours.counts, ours.empty, ours.non, ours.enh), reference):
                    np.testing.assert_array_equal(got, expected)

    def test_no_enhancing_region_is_rejected(self):
        seg = np.zeros((9, 9, 9), dtype=int)
        with self.assertRaisesRegex(ValueError, "no enhancing"):
            radial_counts(seg, np.ones_like(seg), "negative")


class ModelEquivalence(unittest.TestCase):
    def test_original_forward_model(self):
        import Model3Dand1D as original
        for d, k, steps, dx, dt in [
            (1e-4, 1e-3, 24, 1.0, 1.0),
            (1e-2, 1e-3, 41.5, 1.0, 0.5),
            (1e-3, 1e-2, 75, 2.0, 1.0),
        ]:
            with self.subTest(d=d, k=k, steps=steps):
                params = ModelParams(d, k, steps, 21, dx, dt)
                ours = run_model(params, save_freq=5)
                ref = original.RunModel1D(original.ModelParams(d, k, steps, 21, dx, dt), saveFreq=5)
                np.testing.assert_allclose(ours.finalState, ref.finalState, rtol=2e-5, atol=1e-10)
                np.testing.assert_allclose(ours.intermediateStates, ref.intermediateStates, rtol=2e-5, atol=1e-10)

    def test_original_grid_search(self):
        import Model3Dand1D as original
        params = [ModelParams(d, 0.01, 60, 21) for d in (1e-4, 1e-3, 1e-2)]
        ours = [run_model(p, save_freq=10) for p in params]
        refs = [original.RunModel1D(original.ModelParams(p.d, p.k, p.steps, p.sideLen, p.dx, p.dt), saveFreq=10) for p in params]
        target = refs[1].intermediateStates[3].astype(float) + 1e-6
        weights = shell_volumes(len(target))
        np.testing.assert_allclose(weights, original.GetShellVolArr(len(target)), rtol=1e-14)
        entry, index, _ = grid_search(target, ours, weights)
        truth = original.ModelResults(None, target, None, weights=weights)
        best = original.GridSearch(truth, refs)
        self.assertEqual(entry.params.d, best.d)
        self.assertEqual(index * entry.saveFreq, best.steps)


if __name__ == "__main__":
    unittest.main()
