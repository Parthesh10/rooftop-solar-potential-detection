"""Geometry and padding tests (F-07, F-15)."""

import pytest
import torch

from model.unet import UNet
from utils import pad_to_multiple, unpad


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return UNet(3, 1, False).eval()


@pytest.mark.parametrize("size", [(64, 64), (250, 250), (224, 224), (100, 137)])
def test_pad_unpad_roundtrip(size):
    x = torch.randn(2, 3, *size)
    padded, pad = pad_to_multiple(x, 32)
    assert padded.shape[-2] % 32 == 0 and padded.shape[-1] % 32 == 0
    torch.testing.assert_close(unpad(padded, pad), x)


def test_pad_is_noop_when_already_aligned():
    x = torch.randn(1, 3, 64, 64)
    padded, pad = pad_to_multiple(x, 32)
    assert pad == (0, 0, 0, 0)
    assert padded is x


def test_pad_rejects_non_4d():
    with pytest.raises(ValueError):
        pad_to_multiple(torch.randn(3, 64, 64))


@torch.no_grad()
def test_forward_preserves_spatial_dims(model):
    """A padded input must come back at the original size."""
    x = torch.randn(1, 3, 250, 250)
    padded, pad = pad_to_multiple(x, 32)
    out = unpad(model(padded), pad)
    assert out.shape == (1, 1, 250, 250)


@torch.no_grad()
def test_squeeze_named_dim_survives_batch_of_one(model):
    """F-15: bare squeeze() collapses the batch dim on a trailing batch of 1.

    The test split has 53 images at batch_size=2, so this fired on every
    evaluation run the project ever did.
    """
    x = torch.randn(1, 3, 64, 64)
    logits = model(x)
    assert logits.shape == (1, 1, 64, 64)
    assert logits.squeeze(1).shape == (1, 64, 64)   # correct
    assert logits.squeeze().shape == (64, 64)       # the old bug: batch dim gone


@torch.no_grad()
def test_loss_shapes_align_for_batch_of_one(model):
    """The consequence: labels (1, H, W) vs a squeezed (H, W) would broadcast."""
    import torch.nn.functional as F

    x = torch.randn(1, 3, 64, 64)
    labels = torch.zeros(1, 64, 64)
    logits = model(x).squeeze(1)
    assert logits.shape == labels.shape
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    assert torch.isfinite(loss)


def test_logit_threshold_matches_sigmoid_threshold():
    """logits > 0  <=>  sigmoid(logits) > 0.5, used on the training fast path."""
    logits = torch.randn(1000) * 5
    torch.testing.assert_close(
        (logits > 0).float(), (torch.sigmoid(logits) > 0.5).float()
    )
