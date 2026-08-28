"""Model registry: the pretrained-encoder path (plan.md §4.2).

The verbatim U-Net stays reachable and unchanged; smp architectures share its
logit contract so the loss and inference path need no changes.
"""

import pytest
import torch

from model.registry import (
    build_model,
    is_scratch,
    model_spec,
    recommended_stats_key,
)
from model.unet import UNet

smp = pytest.importorskip("segmentation_models_pytorch")


def test_scratch_unet_is_the_verbatim_2023_network():
    m = build_model("unet", "scratch")
    assert isinstance(m, UNet)
    # same param count as the control the whole project is benchmarked against
    assert sum(p.numel() for p in m.parameters()) == 14_788_929


@pytest.mark.parametrize(
    "arch, encoder",
    [
        ("unet", "resnet34"),
        ("unet++", "efficientnet-b0"),
        ("deeplabv3+", "efficientnet-b2"),
        ("manet", "resnet34"),
    ],
)
def test_smp_models_emit_logits_at_input_resolution(arch, encoder):
    torch.manual_seed(0)
    m = build_model(arch, encoder, encoder_weights=None).eval()
    with torch.no_grad():
        out = m(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 1, 224, 224)
    # raw logits: no activation baked into the head, so the combo loss and
    # infer.predict_probs (which apply their own sigmoid) stay correct.
    assert isinstance(m.segmentation_head[-1].activation, torch.nn.Identity)


def test_unknown_arch_is_rejected():
    with pytest.raises(ValueError, match="unknown arch"):
        build_model("transformer-9000", "resnet34")


def test_smp_arch_without_encoder_is_rejected():
    with pytest.raises(ValueError, match="named encoder"):
        build_model("unet++", "scratch")


def test_is_scratch_recognises_the_aliases():
    assert is_scratch(None) and is_scratch("scratch") and is_scratch("NONE")
    assert not is_scratch("resnet34")


def test_recommended_stats_key_tracks_pretraining():
    assert recommended_stats_key("resnet34", "imagenet") == "imagenet"
    assert recommended_stats_key("resnet34", None) == "all"
    assert recommended_stats_key("scratch", "imagenet") == "all"


def test_model_spec_round_trips_through_build_model():
    spec = model_spec("unet++", "efficientnet-b0", "imagenet")
    assert spec["stats_key"] == "imagenet"
    m = build_model(spec["arch"], spec["encoder"], encoder_weights=None)
    with torch.no_grad():
        out = m.eval()(torch.randn(1, 3, 96, 96))
    assert out.shape == (1, 1, 96, 96)


def test_scratch_spec_reports_swiss_stats():
    spec = model_spec("unet", "scratch")
    assert spec == {
        "arch": "unet",
        "encoder": "scratch",
        "encoder_weights": None,
        "in_channels": 3,
        "out_channels": 1,
        "bilinear": False,
        "stats_key": "all",
    }
