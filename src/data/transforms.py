"""MONAI transform pipelines for MSD Task01_BrainTumour.

Images are 4-channel (FLAIR, T1w, T1gd, T2w). Labels are collapsed from the
multi-class MSD annotation (background/edema/non-enhancing/enhancing) down to a
binary whole-tumour mask (background=0, anything else=1).
"""

from __future__ import annotations

import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
)

_KEYS = ("image", "label")


def _binarize_whole_tumor(label: torch.Tensor) -> torch.Tensor:
    return (label > 0).float()


def _base_transforms() -> list:
    return [
        LoadImaged(keys=_KEYS),
        EnsureChannelFirstd(keys=_KEYS),
        Orientationd(keys=_KEYS, axcodes="RAS"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        Lambdad(keys="label", func=_binarize_whole_tumor),
    ]


def get_train_transforms(
    patch_size: tuple[int, int, int],
    num_samples: int,
    pos: float,
    neg: float,
) -> Compose:
    return Compose(
        [
            *_base_transforms(),
            RandCropByPosNegLabeld(
                keys=_KEYS,
                label_key="label",
                spatial_size=patch_size,
                pos=pos,
                neg=neg,
                num_samples=num_samples,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=_KEYS, spatial_axis=0, prob=0.5),
            RandFlipd(keys=_KEYS, spatial_axis=1, prob=0.5),
            RandFlipd(keys=_KEYS, spatial_axis=2, prob=0.5),
            RandRotate90d(keys=_KEYS, prob=0.5, spatial_axes=(0, 1)),
            EnsureTyped(keys=_KEYS, dtype=torch.float32),
        ]
    )


def get_val_transforms() -> Compose:
    return Compose(
        [
            *_base_transforms(),
            EnsureTyped(keys=_KEYS, dtype=torch.float32),
        ]
    )
