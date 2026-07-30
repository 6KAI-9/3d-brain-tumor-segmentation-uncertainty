from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


def _write_synthetic_case(directory: Path, name: str, spatial_size: int) -> dict[str, str]:
    rng = np.random.default_rng(0)
    image = rng.normal(size=(spatial_size, spatial_size, spatial_size, 4)).astype(np.float32)
    label = np.zeros((spatial_size, spatial_size, spatial_size), dtype=np.uint8)
    mid = spatial_size // 2
    label[mid - 2 : mid + 2, mid - 2 : mid + 2, mid - 2 : mid + 2] = 2  # non-background class

    affine = np.eye(4)
    image_path = directory / f"{name}.nii.gz"
    label_path = directory / f"{name}_seg.nii.gz"
    nib.save(nib.Nifti1Image(image, affine), str(image_path))
    nib.save(nib.Nifti1Image(label, affine), str(label_path))
    return {"image": str(image_path), "label": str(label_path)}


@pytest.fixture
def synthetic_datalist(tmp_path: Path) -> list[dict[str, str]]:
    return [_write_synthetic_case(tmp_path, f"case_{i}", spatial_size=32) for i in range(3)]
