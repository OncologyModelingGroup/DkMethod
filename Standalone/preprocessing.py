"""Standalone MRI segmentation-to-radial-density preprocessing.

Requires numpy and nibabel. No imports from the original DkMethod modules.
Implements the published preprocessing conventions: BraTS labels 1/4 as
enhancing, label 2 as nonenhancing; largest 6-connected enhancing component
defines the center; voxel distances round to nearest integer radius; densities
are 0.90 (enhancing), 0.45 (nonenhancing), 0 (other).
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class RadialPatient:
    id: str
    center: np.ndarray
    counts: np.ndarray
    empty: np.ndarray
    non: np.ndarray
    enh: np.ndarray

    def density(self, length=101, enhancing=0.90, nonenhancing=0.45):
        denom = np.where(self.counts == 0, 1, self.counts)
        values = (self.enh * enhancing + self.non * nonenhancing) / denom
        if length is not None:
            if length < len(values):
                raise ValueError("Requested length shorter than radial counts")
            values = np.pad(values, (0, length - len(values)))
        return values


def radial_counts(segmentation, mask, patient_id, max_radius=99.5):
    """Return 100 radial bins, matching original GetRadialDist100.

    Input arrays must have identical shape and segmentation labels 0/1/2/4.
    The mask uses strictly positive voxels as valid (as in original GenPt).
    """
    from scipy.ndimage import label
    seg = np.asarray(segmentation)
    valid = np.asarray(mask) > 0
    if seg.shape != valid.shape or seg.ndim != 3:
        raise ValueError("Segmentation and mask must have equal 3D shapes")
    mapped = np.where((seg == 1) | (seg == 4), 6, np.where(seg == 2, 5, seg))
    # Original GenPt resets invalid voxels to 0; GetLargestCom operates
    # on that masked tumor field.
    mapped = np.where(valid, mapped, 0)
    structure = np.zeros((3, 3, 3), dtype=np.int8)
    structure[1, 1, 1] = 1
    for axis in range(3):
        for offset in (-1, 1):
            index = [1, 1, 1]
            index[axis] += offset
            structure[tuple(index)] = 1
    labels, n = label(mapped == 6, structure=structure)
    if n == 0:
        raise ValueError(f"Patient {patient_id}: no enhancing component; original center undefined")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    center = np.argwhere(labels == np.argmax(sizes)).mean(axis=0)
    # Original GetRadialDist100 uses int(maxRad + 1.5) bins.
    n_bins = int(max_radius + 1.5)
    xyz = np.indices(seg.shape, dtype=np.float64)
    radius = np.sqrt(sum((xyz[i] - center[i]) ** 2 for i in range(3)))
    bins = np.floor(radius + 0.5).astype(np.int64)
    selected = valid & (bins < n_bins)
    counts = np.bincount(bins[selected], minlength=n_bins).astype(float)
    empty = np.bincount(bins[selected & (mapped == 0)], minlength=n_bins).astype(float)
    non = np.bincount(bins[selected & (mapped == 5)], minlength=n_bins).astype(float)
    enh = np.bincount(bins[selected & (mapped == 6)], minlength=n_bins).astype(float)
    # Original special case: if no valid voxel falls in radius zero, insert
    # the nearest-center voxel, even when it lies outside the mask.
    if counts[0] == 0:
        pos = tuple(np.floor(center + 0.5).astype(int))
        value = mapped[pos]
        counts[0] = 1
        if value == 0:
            empty[0] += 1
        elif value == 5:
            non[0] += 1
        elif value == 6:
            enh[0] += 1
    return RadialPatient(str(patient_id), center, counts, empty, non, enh)


def load_patient(seg_path, mask_path, patient_id=None):
    import nibabel as nib
    seg_path = Path(seg_path)
    mask_path = Path(mask_path)
    segmentation = nib.load(str(seg_path)).get_fdata()
    mask = nib.load(str(mask_path)).get_fdata()
    return radial_counts(segmentation, mask, patient_id or seg_path.parent.name)
