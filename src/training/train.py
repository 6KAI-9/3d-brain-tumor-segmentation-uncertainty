"""Baseline training loop: Attention U-Net, AMP, gradient accumulation, Dice -> MLflow.

No MC Dropout, no ECE, no visualization dashboard here — those are later milestones.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import mlflow
import torch
import yaml
from monai.data import DataLoader, list_data_collate
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.utils import set_determinism
from torch.cuda.amp import GradScaler, autocast

from src.data.dataset import build_datasets
from src.models.attention_unet import build_attention_unet


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    loss_fn: torch.nn.Module,
    device: torch.device,
    accum_steps: int,
    amp_enabled: bool,
) -> float:
    model.train()
    optimizer.zero_grad()
    running_loss = 0.0
    step = 0
    for step, batch in enumerate(loader):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        with autocast(enabled=amp_enabled):
            outputs = model(images)
            loss = loss_fn(outputs, labels) / accum_steps

        scaler.scale(loss).backward()
        running_loss += loss.item() * accum_steps

        if (step + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    if (step + 1) % accum_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return running_loss / (step + 1)


def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    inferer: SlidingWindowInferer,
    dice_metric: DiceMetric,
    device: torch.device,
) -> float:
    model.eval()
    dice_metric.reset()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = inferer(images, model)
            assert isinstance(outputs, torch.Tensor)
            outputs = (torch.sigmoid(outputs) > 0.5).float()
            dice_metric(y_pred=outputs, y=labels)
    result = dice_metric.aggregate()
    assert isinstance(result, torch.Tensor)
    return float(result.item())


def _flatten(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in config.items():
        full_key = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, prefix=f"{full_key}."))
        else:
            flat[full_key] = value
    return flat


def main(config_path: str) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    set_determinism(seed=config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(config["training"]["amp"]) and device.type == "cuda"

    data_cfg = config["data"]
    patch_size = tuple(data_cfg["patch_size"])
    train_ds, val_ds, _test_ds = build_datasets(
        root_dir=data_cfg["root_dir"],
        splits_path=data_cfg["splits_path"],
        patch_size=patch_size,
        num_samples=data_cfg["num_samples"],
        pos=data_cfg["pos"],
        neg=data_cfg["neg"],
        download=data_cfg["download"],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        collate_fn=list_data_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=data_cfg["num_workers"],
    )

    model_cfg = config["model"]
    model = build_attention_unet(
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        channels=tuple(model_cfg["channels"]),
        strides=tuple(model_cfg["strides"]),
        dropout=model_cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
    scaler = GradScaler(enabled=amp_enabled)
    loss_fn = DiceLoss(sigmoid=True)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    inferer = SlidingWindowInferer(roi_size=patch_size, sw_batch_size=1, overlap=0.25)

    checkpoint_dir = Path(config["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_dice = -1.0

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name=config["mlflow"]["run_name"]):
        mlflow.log_params(_flatten(config))

        for epoch in range(config["training"]["epochs"]):
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                loss_fn,
                device,
                config["training"]["accum_steps"],
                amp_enabled,
            )
            mlflow.log_metric("train_loss", train_loss, step=epoch)

            if (epoch + 1) % config["training"]["val_interval"] == 0:
                val_dice = validate(model, val_loader, inferer, dice_metric, device)
                mlflow.log_metric("val_dice", val_dice, step=epoch)
                if val_dice > best_dice:
                    best_dice = val_dice
                    torch.save(model.state_dict(), checkpoint_dir / "best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
