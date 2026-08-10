"""Visualise a model's prediction on an arbitrary image.

The old implementation of ``import_and_show`` is what F-01 was about: it fed the
network a diagonally-transposed, unnormalised 0-255 tensor and then transposed
the *output* back so the figure looked plausible. Every MANIT prediction in the
2023 report (Figures 21-23) came through it.

All preprocessing now delegates to :mod:`infer`, so this module cannot drift
away from the training pipeline again.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from config import DEFAULT_STATS_KEY
from infer import predict_large, predict_mask

__all__ = ["import_and_show"]


def import_and_show(
    model,
    name: str | Path,
    threshold: float = 0.5,
    stats_key: str = DEFAULT_STATS_KEY,
    size: int | None = None,
    tta: bool = False,
    device=None,
    alpha: float = 0.45,
):
    """Run ``model`` on the image at ``name`` and show input / mask / overlay.

    Args:
        size: optional square resize. Leave ``None`` to keep native resolution —
            the model was trained at ~0.25-0.3 m/px, so resizing changes the
            effective ground sample distance and costs accuracy. The old code
            unconditionally forced 250x250.
        tta: average over the 8 dihedral transforms.

    Returns the boolean mask, so callers can measure area rather than only look
    at a picture.
    """
    image = Image.open(name).convert("RGB")
    arr = np.asarray(image)

    predictor = predict_large if max(arr.shape[:2]) > 1024 else predict_mask
    kwargs = dict(threshold=threshold, stats_key=stats_key, tta=tta, device=device)
    if predictor is predict_mask:
        kwargs["size"] = size
    mask = predictor(model, arr if size is None else image, **kwargs)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    axes[0].imshow(arr)
    axes[0].set_title("Input image")

    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Predicted mask (t={threshold})")

    overlay = arr.astype(np.float32) / 255.0
    if overlay.shape[:2] == mask.shape:
        tint = np.zeros_like(overlay)
        tint[..., 0] = 1.0  # red
        m = mask[..., None].astype(np.float32) * alpha
        overlay = overlay * (1 - m) + tint * m
    axes[2].imshow(np.clip(overlay, 0, 1))
    axes[2].set_title("Overlay")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    plt.show()

    coverage = float(mask.mean())
    print(f"predicted rooftop pixels: {int(mask.sum()):,} ({coverage:.1%} of image)")
    return mask
