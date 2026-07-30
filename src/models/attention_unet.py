"""Plain Attention U-Net baseline (no MC Dropout / uncertainty head yet)."""

from __future__ import annotations

from monai.networks.nets import AttentionUnet


def build_attention_unet(
    in_channels: int = 4,
    out_channels: int = 1,
    channels: tuple[int, ...] = (16, 32, 64, 128, 256),
    strides: tuple[int, ...] = (2, 2, 2, 2),
    dropout: float = 0.0,
) -> AttentionUnet:
    return AttentionUnet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        dropout=dropout,
    )
