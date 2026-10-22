"""Compare standalone-pipeline checkpoints with the original scientific routines.

Usage:
  python Tests/compare_original_pipeline.py --output /path/to/pipeline/output

The existing pipeline must already have generated paths.p, pts.p, grid.p,
dks.p and dkDf.csv. This tool independently regenerates the final results
table using the original function, then compares every patient and numeric
result. It does NOT establish independent algorithm equivalence: both
paths call the same scientific implementation. Full equivalence requires
an independent standalone implementation and matched-input tests.
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def compare(output, rtol=1e-10, atol=1e-12):
    from FitDkForEachPatient import GenPatientDkResDf
    output = Path(output)
    required = ["paths.p", "pts.p", "grid.p", "dks.p", "dkDf.csv"]
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing checkpoints: " + ", ".join(missing))

    paths = pd.read_pickle(output / "paths.p")
    pts = pd.read_pickle(output / "pts.p")
    grid = pd.read_pickle(output / "grid.p")
    fits = pd.read_pickle(output / "dks.p")
    actual = pd.read_csv(output / "dkDf.csv", dtype={"id": str})
    expected = GenPatientDkResDf(fits)
    actual["id"] = actual["id"].astype(str)
    expected["id"] = expected["id"].astype(str)

    if len(set(paths)) != len(paths) or len(set(pts)) != len(pts):
        raise AssertionError("Duplicate patient IDs in checkpoints")
    if set(pts) != set(actual["id"]) or set(pts) != set(expected["id"]):
        raise AssertionError("Patient IDs differ between preprocessing, fits and table")
    if actual["id"].duplicated().any() or expected["id"].duplicated().any():
        raise AssertionError("Duplicate patient IDs in results")
    if not grid:
        raise AssertionError("Empty simulation grid")

    actual = actual.set_index("id").sort_index()
    expected = expected.set_index("id").sort_index()
    if set(actual.columns) != set(expected.columns):
        raise AssertionError(f"Columns differ: {list(actual.columns)} vs {list(expected.columns)}")
    for col in expected:
        a = pd.to_numeric(actual[col], errors="raise").to_numpy()
        b = pd.to_numeric(expected[col], errors="raise").to_numpy()
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise AssertionError(f"Nonfinite {col}")
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=col)
    print(f"PASS: {len(actual)} patients, {len(grid)} grid entries; "
          "original table function matches pipeline CSV.")
    return len(actual)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--atol", type=float, default=1e-12)
    args = parser.parse_args()
    compare(args.output, args.rtol, args.atol)
