"""Training objectives.

Fixes F-02: the original notebook used

    torch.nn.BCEWithLogitsLoss(weight=torch.FloatTensor([4]).cuda())

``weight`` is a *per-element rescaling factor* broadcast over the batch — a
scalar 4 multiplies every pixel's loss, positives and negatives alike, which is
mathematically identical to leaving the loss alone and quadrupling the learning
rate. It does nothing about class imbalance. The argument that up-weights
false negatives is ``pos_weight``.

Every run in ``plots/`` labelled ``loss4`` / ``loss5`` / ``loss6`` / ``loss9``
was therefore sweeping a learning-rate multiplier, not a class weight (F-24).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DiceLoss", "TverskyLoss", "ComboLoss", "compute_pos_weight", "build_loss"]


class DiceLoss(nn.Module):
    """Soft Dice on the positive class. Directly aligned with IoU."""

    def __init__(self, smooth: float = 1.0, from_logits: bool = True):
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(inputs) if self.from_logits else inputs
        probs = probs.reshape(probs.shape[0], -1)
        targets = targets.reshape(targets.shape[0], -1).to(probs.dtype)
        inter = (probs * targets).sum(dim=1)
        denom = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class TverskyLoss(nn.Module):
    """Generalised Dice with asymmetric FP/FN penalties.

    ``beta > alpha`` penalises false negatives harder — use it when recall on
    rooftops matters more than precision (e.g. an "upper bound on potential"
    framing for the web app).
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0,
                 from_logits: bool = True):
        super().__init__()
        self.alpha, self.beta = alpha, beta
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(inputs) if self.from_logits else inputs
        probs = probs.reshape(probs.shape[0], -1)
        targets = targets.reshape(targets.shape[0], -1).to(probs.dtype)
        tp = (probs * targets).sum(dim=1)
        fp = (probs * (1 - targets)).sum(dim=1)
        fn = ((1 - probs) * targets).sum(dim=1)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky.mean()


class ComboLoss(nn.Module):
    """``(1 - w) * BCEWithLogits(pos_weight) + w * Dice``.

    The default recipe: BCE gives stable per-pixel gradients, Dice optimises the
    metric we actually report.
    """

    def __init__(self, pos_weight: torch.Tensor | float | None = None,
                 dice_weight: float = 0.5, region_loss: nn.Module | None = None):
        super().__init__()
        if pos_weight is not None and not isinstance(pos_weight, torch.Tensor):
            pos_weight = torch.tensor([float(pos_weight)])
        # registered as a buffer so `.to(device)` moves it with the module —
        # the old code hardcoded `.cuda()`, which crashed on CPU-only machines.
        self.register_buffer("pos_weight", pos_weight)
        self.dice_weight = float(dice_weight)
        self.region = region_loss if region_loss is not None else DiceLoss()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.to(inputs.dtype)
        bce = F.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=self.pos_weight
        )
        if self.dice_weight <= 0:
            return bce
        return (1.0 - self.dice_weight) * bce + self.dice_weight * self.region(inputs, targets)


@torch.no_grad()
def compute_pos_weight(loader, max_batches: int | None = 50, device=None) -> float:
    """Estimate ``(#negative / #positive)`` pixels over the training set.

    That ratio is the canonical ``pos_weight`` for balanced BCE. Sampling 50
    batches is plenty — the estimate is stable to two decimals well before then.
    """
    pos = 0
    total = 0
    for i, (_, labels) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        if device is not None:
            labels = labels.to(device)
        pos += int((labels > 0.5).sum().item())
        total += int(labels.numel())
    if pos == 0:
        raise ValueError(
            "No positive pixels found while estimating pos_weight — check that "
            "images and masks are correctly paired (see F-04)."
        )
    return (total - pos) / pos


def build_loss(cfg, loader=None, device=None) -> nn.Module:
    """Construct the training loss from a :class:`config.TrainConfig`.

    If ``cfg.pos_weight`` is None the weight is estimated from ``loader``.
    """
    pw = cfg.pos_weight
    if pw is None:
        if loader is None:
            raise ValueError("pos_weight is None and no loader was given to estimate it")
        pw = compute_pos_weight(loader, device=device)
        print(f"[loss] estimated pos_weight = {pw:.3f} from the training set")
    loss = ComboLoss(pos_weight=pw, dice_weight=cfg.dice_weight)
    return loss.to(device) if device is not None else loss
