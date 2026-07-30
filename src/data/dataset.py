"""Build train/val/test MONAI datasets for MSD Task01_BrainTumour from a saved split."""

from __future__ import annotations

from monai.data import Dataset

from src.data.splits import apply_split, build_full_datalist, load_splits
from src.data.transforms import get_train_transforms, get_val_transforms


def build_datasets(
    root_dir: str,
    splits_path: str,
    patch_size: tuple[int, int, int],
    num_samples: int,
    pos: float,
    neg: float,
    download: bool = True,
) -> tuple[Dataset, Dataset, Dataset]:
    """Return (train_ds, val_ds, test_ds) built from the case list and a saved split."""
    full_datalist = build_full_datalist(root_dir, download=download)
    splits = load_splits(splits_path)

    train_ds = Dataset(
        data=apply_split(full_datalist, splits["train"]),
        transform=get_train_transforms(patch_size, num_samples, pos, neg),
    )
    val_ds = Dataset(
        data=apply_split(full_datalist, splits["val"]),
        transform=get_val_transforms(),
    )
    test_ds = Dataset(
        data=apply_split(full_datalist, splits["test"]),
        transform=get_val_transforms(),
    )
    return train_ds, val_ds, test_ds
