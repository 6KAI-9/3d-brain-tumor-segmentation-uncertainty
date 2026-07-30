import torch
from torch import nn

from src.models.attention_unet import build_attention_unet


def test_forward_pass_produces_expected_output_shape():
    model = build_attention_unet(
        in_channels=4,
        out_channels=1,
        channels=(4, 8, 16),
        strides=(2, 2),
        dropout=0.0,
    )
    x = torch.randn(1, 4, 16, 16, 16)
    out = model(x)
    assert out.shape == (1, 1, 16, 16, 16)


def test_baseline_has_no_active_dropout():
    model = build_attention_unet(dropout=0.0)
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            assert module.p == 0.0
