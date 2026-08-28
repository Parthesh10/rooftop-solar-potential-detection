"""Model factory: the verbatim 2023 U-Net plus pretrained-encoder architectures.

Why this exists
---------------
Training a 14.8 M-parameter U-Net *from scratch* on 420 tiles was the binding
constraint on the 2023 result — not the loss recipe (see ``results/RESULTS.md``).
An ImageNet-pretrained encoder is the single change expected to move test IoU,
and it needs ``segmentation_models_pytorch``.

Contract
--------
Every model returned here emits **raw logits** shaped ``(N, out_channels, H, W)``
— identical to ``model.unet.UNet.forward`` — so the training loop, the combo
loss and ``infer.predict_probs`` are all unchanged. ``smp`` models are built with
``activation=None`` for exactly this reason.

The verbatim U-Net stays reachable as ``build_model("unet", "scratch")`` and is
kept byte-for-byte in ``model/unet.py``: it is the control that proves any delta
from a pretrained encoder is real.

Normalisation
-------------
A pretrained encoder expects ImageNet channel statistics, not the Swiss-set
statistics. ``recommended_stats_key`` returns the right ``config.NORM_STATS`` key
for a given encoder; ``scripts/train_swiss.py`` uses it unless ``--stats-key`` is
passed explicitly.
"""

from __future__ import annotations

from model.unet import UNet

__all__ = [
    "build_model",
    "build_from_config",
    "model_spec",
    "recommended_stats_key",
    "is_scratch",
    "SMP_ARCHS",
]

# alias -> segmentation_models_pytorch class name
SMP_ARCHS: dict[str, str] = {
    "unet": "Unet",
    "unet++": "UnetPlusPlus",
    "unetplusplus": "UnetPlusPlus",
    "unetpp": "UnetPlusPlus",
    "deeplabv3+": "DeepLabV3Plus",
    "deeplabv3plus": "DeepLabV3Plus",
    "dlv3p": "DeepLabV3Plus",
    "deeplabv3": "DeepLabV3",
    "manet": "MAnet",
    "fpn": "FPN",
    "pspnet": "PSPNet",
    "linknet": "Linknet",
    "pan": "PAN",
    "segformer": "Segformer",
}

_SCRATCH = {None, "", "scratch", "none", "random"}


def is_scratch(encoder: str | None) -> bool:
    """True when no pretrained encoder was requested."""
    return (encoder.lower() if isinstance(encoder, str) else encoder) in _SCRATCH


def recommended_stats_key(encoder: str | None, encoder_weights: str | None) -> str:
    """Which ``config.NORM_STATS`` key to normalise with.

    ImageNet stats for a pretrained encoder, the Swiss-set stats ("all")
    otherwise. Getting this wrong quietly costs accuracy — the encoder's early
    filters were fit to the ImageNet distribution.
    """
    return "imagenet" if (encoder_weights and not is_scratch(encoder)) else "all"


def build_model(
    arch: str = "unet",
    encoder: str | None = "scratch",
    encoder_weights: str | None = None,
    in_channels: int = 3,
    classes: int = 1,
):
    """Construct a segmentation model that outputs raw logits.

    Args:
        arch: ``"unet"`` (default), ``"unet++"``, ``"deeplabv3+"``, ``"manet"``,
            ``"fpn"``, ``"pspnet"``, ``"linknet"``, ``"pan"``, ``"segformer"`` —
            see ``SMP_ARCHS`` for every accepted spelling.
        encoder: ``"scratch"`` (or ``None``) for a randomly-initialised model;
            otherwise an ``smp`` encoder name such as ``"resnet34"`` or
            ``"efficientnet-b0"``.
        encoder_weights: ``"imagenet"`` to pull pretrained encoder weights,
            ``None`` to skip the download (use ``None`` when the full state is
            about to be loaded from a checkpoint).
        in_channels: input channels (3 for RGB).
        classes: output channels (1 for binary rooftop segmentation).

    ``build_model("unet", "scratch")`` returns the verbatim 2023 ``UNet``.
    """
    key = arch.strip().lower()

    if key in ("unet", "u-net") and is_scratch(encoder):
        # The control. Kept identical to the 2023 network on purpose.
        return UNet(in_channels, classes, False)

    if key not in SMP_ARCHS:
        raise ValueError(
            f"unknown arch {arch!r}. Known: 'unet' (scratch or smp), "
            + ", ".join(sorted(set(SMP_ARCHS) - {"unet"}))
        )

    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            f"arch={arch!r} encoder={encoder!r} needs segmentation-models-pytorch. "
            "Install it with:  pip install segmentation-models-pytorch"
        ) from exc

    if is_scratch(encoder):
        raise ValueError(
            f"arch={arch!r} needs a named encoder (e.g. 'resnet34', "
            "'efficientnet-b0'); only the plain 'unet' has a scratch variant"
        )

    factory = getattr(smp, SMP_ARCHS[key])
    return factory(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None,  # raw logits — the loss and infer path expect them
    )


def model_spec(
    arch: str = "unet",
    encoder: str | None = "scratch",
    encoder_weights: str | None = None,
    in_channels: int = 3,
    classes: int = 1,
) -> dict:
    """The block a manifest / summary needs to rebuild this model for inference."""
    scratch = is_scratch(encoder) and arch.strip().lower() in ("unet", "u-net")
    return {
        "arch": "unet" if scratch else arch.strip().lower(),
        "encoder": "scratch" if scratch else encoder,
        "encoder_weights": None if scratch else encoder_weights,
        "in_channels": in_channels,
        "out_channels": classes,
        "bilinear": False,
        "stats_key": recommended_stats_key(encoder, encoder_weights),
    }


def build_from_config(cfg):
    """Build the model described by a :class:`config.TrainConfig`."""
    return build_model(
        arch=getattr(cfg, "arch", "unet"),
        encoder=getattr(cfg, "encoder", "scratch"),
        encoder_weights=getattr(cfg, "encoder_weights", None),
    )
