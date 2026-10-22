"""Verify a publication snapshot without relying on the private archive."""
from pathlib import Path
import csv
import hashlib
import sys

EXPECTED = {
    "Data/Dks_7_30_25.csv": (102, ["id", "Swanson D/k", "D/k"]),
    "Data/FullGenes.csv": (98, ["gene", "coef"]),
}
EXPECTED_GIT_BLOBS = {
    "Data/Dks_7_30_25.csv": "d62df4880b61819d87da7607ecb85d4b435e9fcb",
    "Data/FullGenes.csv": "2f55dd9e76dd7f94d8f476d8f4351383a2908083",
}
FORBIDDEN = ["Legacy", ".git", "Tests/test_historical_cohort.py", "Pipeline/analyze_historical_cohort.py", ".github/workflows/numerical-tests.yml"]


def check(root):
    root = Path(root).resolve()
    for relative in FORBIDDEN:
        if (root / relative).exists():
            raise AssertionError(f"Forbidden publication item: {relative}")
    allowed_data = {"README.md", "Dks_7_30_25.csv", "FullGenes.csv"}
    actual_data = {p.name for p in (root / "Data").iterdir()}
    if actual_data != allowed_data:
        raise AssertionError(f"Unexpected or missing Data/ files: {sorted(actual_data ^ allowed_data)}")
    for relative, (expected_rows, fields) in EXPECTED.items():
        path = root / relative
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if not set(fields).issubset(reader.fieldnames or []):
                raise AssertionError(f"Missing required columns in {relative}")
            rows = list(reader)
        if len(rows) != expected_rows:
            raise AssertionError(f"Expected {expected_rows} rows in {relative}, got {len(rows)}")
        if relative.endswith("Dks_7_30_25.csv"):
            ids = [row["id"] for row in rows]
            if any(not x for x in ids) or len(set(ids)) != len(ids):
                raise AssertionError("Missing or duplicate patient IDs")
        data = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if blob != EXPECTED_GIT_BLOBS[relative]:
            raise AssertionError(f"Derived table differs from archived approved candidate: {relative}")
        print(f"PASS {relative}: {len(rows)} rows, SHA256 {hashlib.sha256(data).hexdigest()}")
    print("PASS: snapshot exclusions and derived-table structure")
    print("NOTE: numerical reproduction and release permissions still require review")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python Publication/verify_snapshot.py SNAPSHOT_DIRECTORY")
    check(sys.argv[1])
