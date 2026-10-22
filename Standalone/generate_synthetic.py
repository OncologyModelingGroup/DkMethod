"""Generate reproducible BraTS-like synthetic NIfTI subjects for D/k testing.

From repository root:
    python -m Standalone.generate_synthetic --output /tmp/dk-synthetic

No clinical images or patient data are used.
"""
import argparse
from pathlib import Path
import numpy as np


def generate(output, n=4, shape=33, seed=1729):
    """Create deterministic non-identical segmentations and MRI masks."""
    import nibabel as nib
    if n < 1 or shape < 25:
        raise ValueError("Require n>=1 and shape>=25")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    axes = np.indices((shape,) * 3, dtype=float)
    for i in range(n):
        patient_id = f"synthetic_{i+1:02d}"
        folder = root / patient_id
        folder.mkdir(parents=True, exist_ok=True)
        center = np.array([shape // 2 + int(rng.integers(-2, 3)) for _ in range(3)])
        scale = np.array([1.0, 0.9 + 0.1 * i / max(1, n), 1.1])
        dist2 = sum(((axes[j] - center[j]) / scale[j]) ** 2 for j in range(3))
        segmentation = np.zeros((shape,) * 3, dtype=np.int16)
        segmentation[dist2 <= (7 + i % 3) ** 2] = 2
        segmentation[dist2 <= (4 + i % 2) ** 2] = 4
        segmentation[tuple(center)] = 1
        mask = np.ones_like(segmentation, dtype=np.float32)
        mask[:(i % 3) + 1] = 0
        affine = np.eye(4)
        nib.save(nib.Nifti1Image(segmentation, affine), str(folder / f"{patient_id}_seg.nii.gz"))
        nib.save(nib.Nifti1Image(mask, affine), str(folder / f"{patient_id}_flair.nii.gz"))
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--shape", type=int, default=33)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    print(generate(args.output, args.n, args.shape, args.seed))


if __name__ == "__main__":
    main()
