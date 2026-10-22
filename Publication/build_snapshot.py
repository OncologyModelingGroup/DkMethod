"""Create a history-free candidate publication directory from a local checkout.

Run from the repository root:
    python Publication/build_snapshot.py /path/to/empty/output

This copies only Git-tracked, allowlisted files. It reads the Git index but never modifies the source repository or copies its history.
The destination must not exist or must be an empty directory.
"""
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
INCLUDE_ROOT = {
    ".gitattributes", "Example.py", "FitDkForEachPatient.py", "LICENSE",
    "LMclassNew.py", "Model3Dand1D.py", "PDEgrid.py",
    "ProcessPatientData.py", "RELEASE.md", "readme.md", "requirements.txt",
}
INCLUDE_DIRS = {"Data", "Downstream", "Pipeline", "Standalone", "Tests", "Publication"}
EXCLUDE = {
    "Tests/test_historical_cohort.py",
    "Pipeline/analyze_historical_cohort.py",
    ".github/workflows/numerical-tests.yml",
}
# The historical-cohort script is preserved for reference, but it requires
# an archived CSV not included in the public snapshot.
INCLUDE_EXPLICIT = {".github/workflows/full-publication-parity.yml"}


def manifest():
    tracked = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z", "--cached"],
        check=True, stdout=subprocess.PIPE,
    ).stdout.decode("utf-8").split("\0")
    paths = set()
    for relative in filter(None, tracked):
        path = Path(relative)
        if (relative in INCLUDE_ROOT
                or relative in INCLUDE_EXPLICIT
                or (path.parts and path.parts[0] in INCLUDE_DIRS)):
            if relative not in EXCLUDE:
                source = ROOT / relative
                if source.is_symlink():
                    raise ValueError(f"Refusing symlink in publication snapshot: {relative}")
                if not source.is_file():
                    raise FileNotFoundError(source)
                paths.add(relative)
    missing = (INCLUDE_ROOT | INCLUDE_EXPLICIT) - paths
    if missing:
        raise ValueError(f"Required publication files missing: {sorted(missing)}")
    return sorted(paths)


def main(destination):
    destination = Path(destination).expanduser().resolve()
    if destination == ROOT or ROOT in destination.parents:
        raise ValueError("Destination must be outside the source repository")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Destination must be empty")
    if not destination.exists():
        destination.mkdir(parents=True)
    paths = manifest()
    # Fail closed if any future tracked file introduces a patient-data format
    # into a selected directory without an explicit release review.
    allowed_data = {"Data/README.md", "Data/Dks_7_30_25.csv", "Data/FullGenes.csv"}
    unexpected_data = sorted(p for p in paths if p.startswith("Data/") and p not in allowed_data)
    if unexpected_data:
        raise ValueError(f"Unreviewed Data/ files require explicit release approval: {unexpected_data}")
    for relative in paths:
        if Path(relative).suffix.lower() in {".nii", ".dcm", ".hdf5", ".pkl", ".pickle", ".p"} or relative.lower().endswith(".nii.gz"):
            raise ValueError(f"Unexpected binary research artifact: {relative}")
    for relative in paths:
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    print(f"Copied {len(paths)} files into {destination}")
    print("No Git history, Legacy directory, or raw patient images copied.")
    print("Publication requires author approval and a final content audit.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python Publication/build_snapshot.py OUTPUT_DIRECTORY")
    main(sys.argv[1])
