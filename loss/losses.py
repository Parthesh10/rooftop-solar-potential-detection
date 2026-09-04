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

__all__ = ["DiceLoss", "TverskyLoss", "ComboLoss", "compute_pos_weight",
           "build_loss", "IGNORE_INDEX", "split_targets"]


# Targets are {0, 1} normally. A third state is needed when a pixel's label is
# genuinely unknown rather than negative — see scripts/build_osm_labels.py,
# where OpenStreetMap supplies the positives and a hand-drawn envelope marks
# the surrounding area as "roof or alley, we cannot tell". Calling those pixels
# background would teach the model that real rooftops are background, which is
# the exact failure being fixed.
#
# Encoded in the target tensor rather than passed as a fourth argument, so the
# training loop's ``loss_fn(outputs, labels)`` call is unchanged and every
# existing two-state dataset behaves exactly as before.
IGNORE_INDEX = -1.0


def split_targets(targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``targets`` -> ``(clean_targets, valid_mask)``.

    Ignored pixels are zeroed in ``clean_targets`` so they cannot contribute a
    gradient, and marked False in ``valid_mask`` so they are excluded from every
    average.
    """
    valid = targets >= 0
    return targets * valid, valid


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
        targets, valid = split_targets(targets)
        # Masking the *probabilities* is what keeps an ignored pixel out of the
        # denominator: zeroing only the target would still penalise a positive
        # prediction there as a false positive.
        probs = probs * valid
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
        targets, valid = split_targets(targets)
        probs = probs * valid
        tp = (probs * targets).sum(dim=1)
        fp = (probs * (1 - targets) * valid).sum(dim=1)
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
        clean, valid = split_targets(targets)

        if bool(valid.all()):
            bce = F.binary_cross_entropy_with_logits(
                inputs, clean, pos_weight=self.pos_weight)
        else:
            # Average over the valid pixels only. Reducing over everything and
            # dividing by numel would quietly shrink the loss in proportion to
            # how much of the tile was ignored.
            per_px = F.binary_cross_entropy_with_logits(
                inputs, clean, pos_weight=self.pos_weight, reduction="none")
            denom = valid.sum().clamp(min=1)
            bce = (per_px * valid).sum() / denom

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
        total += int((labels >= 0).sum().item())   # ignored pixels are not data
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
