"""The single inference path.

Fixes **F-01**, the worst defect in the codebase. ``process_data/import_test.py``
previously did:

    test = torch.tensor(np.transpose(test))   # BUG 1
    test = test.float()                       # BUG 2

1. ``np.transpose`` with no ``axes`` argument reverses *all* axes. On an
   ``(H, W, C)`` array that yields ``(C, W, H)`` — the channels land correctly
   but the image is **spatially transposed**, mirrored about its diagonal. The
   display code then transposed the *output* back, hiding the bug in every
   figure while corrupting every prediction.
2. No ``/255`` and no normalisation, so the network saw values ~60x larger in
   magnitude than anything it was trained on.

Both are fixed here, and — more importantly — this module is now the *only*
place inference preprocessing is defined. ``tests/test_preprocess_parity.py``
asserts byte-level agreement with the training dataloader, so the two can never
drift apart again.

Normalisation constants come from a model manifest written next to the weights,
not from a literal in the source.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from config import DEFAULT_STATS_KEY, REPO_ROOT, SIZE_DIVISOR, norm_stats
from model.unet import UNet
from utils import get_device, pad_to_multiple, unpad

__all__ = [
    "load_manifest",
    "preprocess",
    "load_model",
    "predict_probs",
    "predict_mask",
    "predict_large",
]

MANIFEST_PATH = REPO_ROOT / "model" / "manifest.json"


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
def load_manifest(path: str | Path | None = None) -> dict:
    """Load the model manifest: architecture, normalisation, threshold, metrics.

    A checkpoint that does not describe its own preprocessing is a loaded gun
    (F-01, F-18). Every saved run writes one of these.
    """
    path = Path(path) if path is not None else MANIFEST_PATH
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _entry_for(ckpt: str | Path, manifest: dict) -> dict:
    name = Path(ckpt).name
    return manifest.get("models", {}).get(name, {})


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def preprocess(
    image,
    mean: list[float] | None = None,
    std: list[float] | None = None,
    stats_key: str = DEFAULT_STATS_KEY,
    size: int | None = None,
    bgr: bool = False,
) -> torch.Tensor:
    """Turn an image into a normalised ``(1, 3, H, W)`` float tensor.

    Args:
        image: a path, a PIL image, or an ``(H, W, 3)`` uint8 array.
        mean/std: normalisation constants. Default: looked up by ``stats_key``.
        stats_key: key into ``config.NORM_STATS``.
        size: optional square resize. ``None`` keeps native resolution, which is
            what you want when the source GSD already matches training (z=19
            web-mercator tiles ~= Inria's 0.3 m/px). Resizing changes the
            effective ground sample distance and will cost accuracy.
        bgr: set True when the array came from ``cv2.imread``, which returns BGR.

    The channel order is fixed with an explicit ``.permute(2, 0, 1)`` — never a
    bare ``np.transpose``.
    """
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")

    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    else:
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise ValueError(f"expected (H, W, 3), got shape {arr.shape}")
        arr = arr[:, :, :3]
        if bgr:
            arr = arr[:, :, ::-1]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

    if size is not None:
        # INTER_AREA-equivalent for downscaling; PIL's ANTIALIAS/LANCZOS is the
        # closest match and avoids the ringing INTER_CUBIC introduced.
        arr = np.asarray(
            Image.fromarray(arr).resize((size, size), Image.Resampling.LANCZOS)
        )

    if mean is None or std is None:
        mean, std = norm_stats(stats_key)

    arr = np.ascontiguousarray(arr)
    if not arr.flags.writeable:
        # PIL/tifffile can hand back read-only buffers; torch.from_numpy on one
        # warns and yields a tensor whose in-place ops are undefined behaviour.
        arr = arr.copy()
    x = torch.from_numpy(arr)                        # (H, W, 3) uint8
    x = x.permute(2, 0, 1).float().div_(255.0)       # (3, H, W) in [0, 1]
    x = TF.normalize(x, mean=mean, std=std)
    return x.unsqueeze(0)                             # (1, 3, H, W)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def load_model(
    ckpt: str | Path,
    device: torch.device | str | None = None,
    manifest: dict | None = None,
) -> tuple[torch.nn.Module, dict]:
    """Load a checkpoint in ``eval`` mode and return ``(model, entry)``.

    F-03: the old ``test_model`` never called ``model.eval()``, so every
    published metric was measured with BatchNorm using *batch* statistics at
    batch size 2. Eval mode is set here, at load time, so no caller can forget.
    """
    device = torch.device(device) if device is not None else get_device()
    manifest = manifest if manifest is not None else load_manifest()
    entry = _entry_for(ckpt, manifest)

    arch = entry.get("arch", "unet")
    if arch != "unet":
        raise NotImplementedError(f"manifest requests arch={arch!r}, only 'unet' is built in")

    model = UNet(
        n_channels=entry.get("in_channels", 3),
        n_classes=entry.get("out_channels", 1),
        bilinear=entry.get("bilinear", False),
    ).to(device)

    state = torch.load(ckpt, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model" in state and "state_dict" not in state:
        state = state["model"]
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()  # ← F-03
    return model, entry


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
_TTA_D4 = (
    (0, False), (1, False), (2, False), (3, False),
    (0, True), (1, True), (2, True), (3, True),
)


@torch.inference_mode()
def predict_probs(
    model: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device | str | None = None,
    tta: bool = False,
) -> torch.Tensor:
    """Run the model on a preprocessed ``(N, 3, H, W)`` batch -> ``(N, 1, H, W)`` probs.

    Pads to a multiple of 32 before the forward pass and crops the result back
    (F-07), so arbitrary input sizes are safe and the decoder's skip connections
    stay registered to the encoder's grid.

    ``tta=True`` averages over the 8 dihedral transforms — typically worth
    +1-2 IoU for 8x the compute.
    """
    device = torch.device(device) if device is not None else next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        x = x.to(device)
        padded, pad = pad_to_multiple(x, SIZE_DIVISOR)

        if not tta:
            logits = model(padded)
        else:
            acc = None
            for k, flip in _TTA_D4:
                v = torch.rot90(padded, k, dims=(-2, -1))
                if flip:
                    v = torch.flip(v, dims=(-1,))
                out = model(v)
                if flip:
                    out = torch.flip(out, dims=(-1,))
                out = torch.rot90(out, -k, dims=(-2, -1))
                acc = out if acc is None else acc + out
            logits = acc / len(_TTA_D4)

        probs = torch.sigmoid(logits)
        return unpad(probs, pad)
    finally:
        if was_training:
            model.train()


def predict_mask(
    model: torch.nn.Module,
    image,
    threshold: float = 0.5,
    stats_key: str = DEFAULT_STATS_KEY,
    mean: list[float] | None = None,
    std: list[float] | None = None,
    size: int | None = None,
    bgr: bool = False,
    tta: bool = False,
    device: torch.device | str | None = None,
    return_probs: bool = False,
):
    """End-to-end: raw image -> boolean rooftop mask of the same H, W.

    This is the function the web API should call. Nothing else.
    """
    x = preprocess(image, mean=mean, std=std, stats_key=stats_key, size=size, bgr=bgr)
    probs = predict_probs(model, x, device=device, tta=tta)[0, 0].cpu().numpy()
    mask = probs > threshold
    return (mask, probs) if return_probs else mask


@torch.inference_mode()
def predict_large(
    model: torch.nn.Module,
    image,
    window: int = 512,
    stride: int = 256,
    threshold: float = 0.5,
    stats_key: str = DEFAULT_STATS_KEY,
    tta: bool = False,
    device: torch.device | str | None = None,
    return_probs: bool = False,
):
    """Sliding-window inference for images too large for one forward pass.

    Overlapping windows are blended with a cosine (Hann) weight so tile seams do
    not show up as grid artefacts in the output mask — which they will if you
    average uniformly or just take the last write.
    """
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    arr = np.asarray(image.convert("RGB")) if isinstance(image, Image.Image) else np.asarray(image)
    h, w = arr.shape[:2]

    if h <= window and w <= window:
        return predict_mask(
            model, arr, threshold=threshold, stats_key=stats_key, tta=tta,
            device=device, return_probs=return_probs,
        )

    device = torch.device(device) if device is not None else next(model.parameters()).device

    ramp = torch.hann_window(window, periodic=False, dtype=torch.float32)
    weight = torch.outer(ramp, ramp).clamp_min(1e-3).to(device)

    acc = torch.zeros((h, w), dtype=torch.float32, device=device)
    wsum = torch.zeros((h, w), dtype=torch.float32, device=device)

    ys = list(range(0, max(h - window, 0) + 1, stride)) or [0]
    xs = list(range(0, max(w - window, 0) + 1, stride)) or [0]
    if ys[-1] + window < h:
        ys.append(h - window)
    if xs[-1] + window < w:
        xs.append(w - window)

    for y in ys:
        for x0 in xs:
            patch = arr[y : y + window, x0 : x0 + window]
            ph, pw = patch.shape[:2]
            t = preprocess(patch, stats_key=stats_key)
            probs = predict_probs(model, t, device=device, tta=tta)[0, 0]
            acc[y : y + ph, x0 : x0 + pw] += probs * weight[:ph, :pw]
            wsum[y : y + ph, x0 : x0 + pw] += weight[:ph, :pw]

    probs = (acc / wsum.clamp_min(1e-6)).cpu().numpy()
    mask = probs > threshold
    return (mask, probs) if return_probs else mask
