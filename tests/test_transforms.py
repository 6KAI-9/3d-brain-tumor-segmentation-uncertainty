import torch
from monai.data import DataLoader, Dataset, list_data_collate

from src.data.transforms import get_train_transforms, get_val_transforms


def test_train_transforms_produce_correctly_shaped_binary_label_patches(synthetic_datalist):
    patch_size = (16, 16, 16)
    transform = get_train_transforms(patch_size, num_samples=2, pos=1, neg=1)
    ds = Dataset(data=synthetic_datalist, transform=transform)
    loader = DataLoader(ds, batch_size=1, collate_fn=list_data_collate)
    batch = next(iter(loader))

    assert batch["image"].shape == (2, 4, *patch_size)
    assert batch["label"].shape == (2, 1, *patch_size)
    assert set(torch.unique(batch["label"]).tolist()) <= {0.0, 1.0}


def test_val_transforms_return_full_volume_with_binary_label(synthetic_datalist):
    ds = Dataset(data=synthetic_datalist, transform=get_val_transforms())
    sample = ds[0]

    assert sample["image"].shape[0] == 4
    assert sample["label"].shape[0] == 1
    assert sample["label"].shape[1:] == sample["image"].shape[1:]
    assert set(torch.unique(sample["label"]).tolist()) <= {0.0, 1.0}
