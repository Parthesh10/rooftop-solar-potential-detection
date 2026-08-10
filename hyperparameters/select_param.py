"""Deprecated compatibility shim.

This module used to hold ``seed_torch`` and ``test_model`` and to import
``train.train``, which imported this module right back — a circular import that
only survived because both sides used star-imports (F-19).

The real implementations now live in:

* ``utils.seed_torch`` / ``utils.get_device``
* ``evaluate.evaluate`` — which, unlike the old ``test_model``, actually calls
  ``model.eval()`` (F-03) and runs under ``no_grad`` (F-06).

Existing notebook cells that do ``from hyperparameters.select_param import *``
keep working; they just get the corrected implementations.
"""

from __future__ import annotations

import warnings

from evaluate import evaluate
from utils import get_device, seed_torch

__all__ = ["seed_torch", "get_device", "evaluate", "test_model"]


def test_model(test_loader, model, device=None, threshold: float = 0.5):
    """Deprecated. Use :func:`evaluate.evaluate`.

    Returns ``[iou, accuracy, recall, precision]`` for continuity with the 2023
    notebook, but the numbers are **not** comparable to the ones recorded there:
    that harness left BatchNorm in training mode (F-03) and scored empty tiles
    as a perfect 1.0 (F-08).
    """
    warnings.warn(
        "test_model() is deprecated; use evaluate.evaluate(), which returns a "
        "dict including undefined-tile counts and the positive-pixel rate. "
        "Note these values will differ from the 2023 report — see F-03 and F-08.",
        DeprecationWarning,
        stacklevel=2,
    )
    res = evaluate(test_loader, model, device=device, threshold=threshold)
    return [res["iou"], res["accuracy"], res["recall"], res["precision"]]
