"""Segmentation metrics.

Rewritten to fix three defects that made every previously reported number
unverifiable — see potential-fixes.md:

* **F-08** — the old ``iou`` inverted both mask and prediction when the target
  was all-background, awarding IoU = 1.0 to every empty tile the model got
  right. On a dataset with many empty crops that silently inflated the mean by
  an unknown amount. Undefined tiles now return ``nan`` and must be aggregated
  with :func:`aggregate` (nan-aware), which also reports how many were skipped.
* **F-11** — ``recall`` divided by zero on empty tiles, producing ``nan`` that
  poisoned ``np.mean``. Both ``recall`` and ``precision`` now use the same
  explicit undefined-is-nan convention.
* **F-12** — ``sigmoid`` overflowed for inputs below about -709.

All functions accept float or boolean arrays of any matching shape. Predictions
are expected to be already thresholded (0/1); use :func:`binarize` otherwise.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "sigmoid",
    "binarize",
    "confusion",
    "iou",
    "accuracy",
    "recall",
    "precision",
    "f1",
    "aggregate",
    "summarize",
]


def sigmoid(x):
    """Numerically stable logistic function (F-12).

    ``1 / (1 + exp(-x))`` overflows to ``inf`` for very negative ``x`` and spams
    ``RuntimeWarning: overflow encountered in exp``. This branchy form is exact
    over the whole float64 range.
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def binarize(x, threshold: float = 0.5) -> np.ndarray:
    """Threshold probabilities (or logits, with ``threshold=0``) to a bool mask."""
    return np.asarray(x) > threshold


def _as_bool(a) -> np.ndarray:
    a = np.asarray(a)
    return a if a.dtype == bool else a > 0.5


def confusion(pred, target) -> tuple[int, int, int, int]:
    """Return ``(tp, fp, fn, tn)`` pixel counts."""
    p, t = _as_bool(pred), _as_bool(target)
    if p.shape != t.shape:
        raise ValueError(f"shape mismatch: pred {p.shape} vs target {t.shape}")
    tp = int(np.count_nonzero(p & t))
    fp = int(np.count_nonzero(p & ~t))
    fn = int(np.count_nonzero(~p & t))
    tn = int(np.count_nonzero(~p & ~t))
    return tp, fp, fn, tn


def iou(pred, target) -> float:
    """Positive-class Jaccard index.

    Returns ``nan`` when the union is empty — i.e. the target has no positive
    pixels *and* the model predicted none. That case is genuinely undefined, not
    a perfect score (F-08). Aggregate with :func:`aggregate`.
    """
    tp, fp, fn, _ = confusion(pred, target)
    union = tp + fp + fn
    return float(tp / union) if union > 0 else float("nan")


def accuracy(pred, target) -> float:
    """(TP + TN) / total.

    Near-meaningless on this task — the positive-pixel rate is roughly 10%, so
    predicting all-background already scores ~0.90. Reported only for continuity
    with the 2023 results tables.
    """
    tp, fp, fn, tn = confusion(pred, target)
    total = tp + fp + fn + tn
    return float((tp + tn) / total) if total > 0 else float("nan")


def recall(pred, target) -> float:
    """TP / (TP + FN); ``nan`` when the target has no positive pixels (F-11)."""
    tp, _, fn, _ = confusion(pred, target)
    denom = tp + fn
    return float(tp / denom) if denom > 0 else float("nan")


def precision(pred, target) -> float:
    """TP / (TP + FP); ``nan`` when nothing was predicted positive."""
    tp, fp, _, _ = confusion(pred, target)
    denom = tp + fp
    return float(tp / denom) if denom > 0 else float("nan")


def f1(pred, target) -> float:
    """Dice / F1 over the positive class; ``nan`` when both sides are empty."""
    tp, fp, fn, _ = confusion(pred, target)
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom > 0 else float("nan")


def aggregate(values) -> tuple[float, int, int]:
    """Nan-aware mean.

    Returns ``(mean, n_used, n_undefined)`` so a caller can report *how many*
    tiles were undefined rather than hiding them — the information F-08 lost.
    """
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    n_used = int(finite.sum())
    n_undef = int(arr.size - n_used)
    mean = float(arr[finite].mean()) if n_used else float("nan")
    return mean, n_used, n_undef


def summarize(per_tile: dict[str, list[float]]) -> dict[str, float]:
    """Collapse per-tile metric lists into a flat summary dict.

    Emits ``<name>`` plus ``<name>_undefined`` for each metric.
    """
    out: dict[str, float] = {}
    for name, vals in per_tile.items():
        mean, n_used, n_undef = aggregate(vals)
        out[name] = mean
        out[f"{name}_undefined"] = n_undef
        out[f"{name}_n"] = n_used
    return out
