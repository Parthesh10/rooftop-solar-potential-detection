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


def select_amp(model, device, requested: str | bool = "auto", probe_shape=(2, 3, 224, 224)):
    """Choose a safe autocast dtype, verifying it on the real model.

    Returns ``(enabled, dtype)`` where dtype is None when AMP is off.

    Why this exists — measured on a GTX 1650 (sm_75, cuDNN 9.10.2), batch 8 @ 224:

        fp32   497 ms/step   3.32 GB   correct
        bf16  1299 ms/step   1.48 GB   correct but 2.6x SLOWER (emulated)
        fp16  1851 ms/step   3.15 GB   **NaN on every step**

    The fp16 failure is a bad cuDNN kernel for 64->64 convolutions on this
    architecture; the whole forward pass comes back NaN from the first block.
    ``torch.cuda.is_bf16_supported()`` also returns True here even though the
    hardware has no native bf16, so neither the dtype flags nor the compute
    capability can be trusted on their own.

    So: on pre-Ampere cards AMP is off by default (it buys nothing when there
    are no tensor cores), and whatever is selected is **probed for NaN against
    the actual model** before training starts. Silent NaN is far more expensive
    than a startup check.
    """
    if device is None or torch.device(device).type != "cuda":
        return False, None

    if requested in (False, "off", "none"):
        return False, None

    major = torch.cuda.get_device_properties(torch.device(device).index or 0).major

    if requested in (True, "auto"):
        if major < 8:
            print(
                f"[amp] compute {major}.x has no tensor cores — AMP disabled "
                f"(fp32 is both faster and safer here). Force with amp='bf16'."
            )
            return False, None
        candidates = [torch.bfloat16, torch.float16]
    else:
        candidates = {
            "fp16": [torch.float16], "float16": [torch.float16],
            "bf16": [torch.bfloat16], "bfloat16": [torch.bfloat16],
        }.get(str(requested), [torch.float16])

    x = torch.randn(*probe_shape, device=device)
    was_training = model.training
    model.eval()
    try:
        for dt in candidates:
            try:
                with torch.no_grad(), torch.autocast("cuda", dtype=dt):
                    out = model(x)
                if torch.isnan(out).any() or torch.isinf(out).any():
                    print(f"[amp] {dt} produced NaN/Inf on this GPU — rejecting it")
                    continue
                return True, dt
            except Exception as exc:
                print(f"[amp] {dt} unusable ({exc}) — rejecting it")
    finally:
        if was_training:
            model.train()

    print("[amp] no usable autocast dtype — running in fp32")
    return False, None


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
