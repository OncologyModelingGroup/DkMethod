"""Run scientific numerical tests; missing dependencies are a failure, not a pass."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Tests"))
suite = unittest.defaultTestLoader.loadTestsFromName("test_numerical_smoke")
result = unittest.TextTestRunner(verbosity=2).run(suite)
if not result.wasSuccessful() or result.skipped or result.testsRun == 0:
    if result.skipped:
        print("Scientific numerical validation INCOMPLETE: tests were skipped.", file=sys.stderr)
    raise SystemExit(1)
