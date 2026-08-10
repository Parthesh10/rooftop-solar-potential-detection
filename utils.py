"""Device handling, seeding, and size-padding helpers.

`seed_torch` used to live in hyperparameters/select_param.py, which imported
train.train, which imported hyperparameters.select_param — a circular import
that only survived because both sides used star-imports (F-19). Moving the
shared helpers here breaks the cycle.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from config import SIZE_DIVISOR


def get_device(prefer: str | None = None) -> torch.device:
    """Return the best available device.

    F-17: the old code called ``.cuda()`` unconditionally in places and relied on
    ``torch.cuda.is_available()`` guards elsewhere, so tensors silently stayed on
    the CPU when CUDA was missing. Resolve the device once and pass it around.
    """
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_torch(seed: int = 0, deterministic: bool = True) -> None:
    """Seed every RNG we touch.

    ``deterministic=True`` disables cuDNN autotuning, which costs ~20-30% speed.
    Set it False once you trust the pipeline and want throughput.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    ps = model.parameters()
    if trainable_only:
        ps = (p for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in ps)


# --------------------------------------------------------------------------- #
# Padding
# --------------------------------------------------------------------------- #
def pad_to_multiple(
    x: torch.Tensor, divisor: int = SIZE_DIVISOR, mode: str = "reflect"
) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    """Pad an ``(N, C, H, W)`` tensor so H and W divide by ``divisor``.

    Returns the padded tensor and the pad amounts ``(left, right, top, bottom)``
    so :func:`unpad` can restore the original geometry.

    F-07: the U-Net downsamples 4x, so a 250x250 input produced fractional
    feature-map sizes and every skip connection was being realigned by an
    asymmetric ``F.pad`` inside the decoder. Padding once, up front, keeps
    encoder and decoder features registered to the same grid.
    """
    if x.dim() != 4:
        raise ValueError(f"expected (N, C, H, W), got shape {tuple(x.shape)}")
    h, w = x.shape[-2:]
    ph = (divisor - h % divisor) % divisor
    pw = (divisor - w % divisor) % divisor
    if ph == 0 and pw == 0:
        return x, (0, 0, 0, 0)
    left, right = pw // 2, pw - pw // 2
    top, bottom = ph // 2, ph - ph // 2
    # `reflect` needs pad < input dim; fall back to replicate for tiny inputs.
    if mode == "reflect" and (max(left, right) >= w or max(top, bottom) >= h):
        mode = "replicate"
    return F.pad(x, (left, right, top, bottom), mode=mode), (left, right, top, bottom)


def unpad(x: torch.Tensor, pad: tuple[int, int, int, int]) -> torch.Tensor:
    """Undo :func:`pad_to_multiple`."""
    left, right, top, bottom = pad
    if left == right == top == bottom == 0:
        return x
    h, w = x.shape[-2:]
    return x[..., top : h - bottom if bottom else h, left : w - right if right else w]


__all__ = ["get_device", "seed_torch", "count_parameters", "pad_to_multiple", "unpad"]
