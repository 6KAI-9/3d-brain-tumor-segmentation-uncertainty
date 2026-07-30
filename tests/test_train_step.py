import copy

import torch
from monai.data import DataLoader, Dataset, list_data_collate
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from torch.cuda.amp import GradScaler

from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.attention_unet import build_attention_unet
from src.training.train import train_one_epoch, validate


def _tiny_model() -> torch.nn.Module:
    return build_attention_unet(
        in_channels=4, out_channels=1, channels=(4, 8, 16), strides=(2, 2), dropout=0.0
    )


def test_train_one_epoch_updates_weights_with_grad_accumulation(synthetic_datalist):
    device = torch.device("cpu")
    model = _tiny_model().to(device)
    before = copy.deepcopy(list(model.parameters()))

    patch_size = (16, 16, 16)
    train_ds = Dataset(
        data=synthetic_datalist,
        transform=get_train_transforms(patch_size, num_samples=1, pos=1, neg=1),
    )
    loader = DataLoader(train_ds, batch_size=1, collate_fn=list_data_collate)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scaler = GradScaler(enabled=False)
    loss_fn = DiceLoss(sigmoid=True)

    avg_loss = train_one_epoch(
        model, loader, optimizer, scaler, loss_fn, device, accum_steps=2, amp_enabled=False
    )

    assert avg_loss == avg_loss  # not NaN
    after = list(model.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after, strict=True))


def test_validate_returns_dice_in_unit_interval(synthetic_datalist):
    device = torch.device("cpu")
    model = _tiny_model().to(device)

    val_ds = Dataset(data=synthetic_datalist, transform=get_val_transforms())
    loader = DataLoader(val_ds, batch_size=1)

    inferer = SlidingWindowInferer(roi_size=(16, 16, 16), sw_batch_size=1, overlap=0.25)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    dice = validate(model, loader, inferer, dice_metric, device)
    assert 0.0 <= dice <= 1.0
