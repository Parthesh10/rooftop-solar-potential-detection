"""Tests for the loss fix (F-02): pos_weight, not weight."""

import pytest
import torch
import torch.nn.functional as F

from loss.losses import ComboLoss, DiceLoss, TverskyLoss, compute_pos_weight


def test_weight_arg_is_a_uniform_rescale_not_a_class_weight():
    """Documents the 2023 bug directly.

    BCEWithLogitsLoss(weight=4) multiplies the whole loss by 4 — identical to
    scaling the learning rate. It does not change the FP/FN balance at all.
    """
    logits = torch.randn(4, 8, 8)
    targets = (torch.rand(4, 8, 8) > 0.9).float()

    plain = F.binary_cross_entropy_with_logits(logits, targets)
    weighted = F.binary_cross_entropy_with_logits(
        logits, targets, weight=torch.tensor([4.0])
    )
    assert weighted == pytest.approx(4.0 * plain.item(), rel=1e-5)


def test_pos_weight_actually_penalises_false_negatives_harder():
    """The behaviour the report described but the code never had."""
    targets = torch.zeros(1, 4, 4)
    targets[0, 0, 0] = 1.0

    # A false negative: confidently predicts background on the one positive.
    fn_logits = torch.full((1, 4, 4), -5.0)
    # A false positive: confidently predicts rooftop on one background pixel.
    fp_logits = torch.full((1, 4, 4), -5.0)
    fp_logits[0, 0, 0] = 5.0
    fp_logits[0, 1, 1] = 5.0

    pw = torch.tensor([9.0])
    fn_loss = F.binary_cross_entropy_with_logits(fn_logits, targets, pos_weight=pw)
    fp_loss = F.binary_cross_entropy_with_logits(fp_logits, targets, pos_weight=pw)
    assert fn_loss > fp_loss

    # With the scalar `weight` argument instead, the ordering is unchanged from
    # the unweighted case — the knob does nothing for imbalance.
    w = torch.tensor([9.0])
    assert F.binary_cross_entropy_with_logits(
        fn_logits, targets, weight=w
    ) == pytest.approx(
        9.0 * F.binary_cross_entropy_with_logits(fn_logits, targets).item(), rel=1e-5
    )


def test_dice_loss_is_zero_for_a_perfect_prediction():
    targets = (torch.rand(2, 16, 16) > 0.5).float()
    logits = (targets * 2 - 1) * 30  # ~ +-30 logits => saturated sigmoid
    assert DiceLoss()(logits, targets).item() == pytest.approx(0.0, abs=1e-3)


def test_dice_loss_is_near_one_for_an_inverted_prediction():
    targets = (torch.rand(2, 16, 16) > 0.5).float()
    logits = (targets * 2 - 1) * -30
    assert DiceLoss()(logits, targets).item() == pytest.approx(1.0, abs=1e-2)


def test_tversky_beta_biases_towards_recall():
    targets = torch.zeros(1, 8, 8)
    targets[0, :4] = 1.0
    misses = torch.full((1, 8, 8), -5.0)          # all false negatives
    extras = torch.full((1, 8, 8), 5.0)           # all false positives

    recall_biased = TverskyLoss(alpha=0.3, beta=0.7)
    assert recall_biased(misses, targets) > recall_biased(extras, targets)


def test_combo_loss_moves_with_the_module_device():
    """The old code hardcoded .cuda(), which crashed on CPU-only machines."""
    loss = ComboLoss(pos_weight=3.0, dice_weight=0.5)
    loss = loss.to("cpu")
    assert loss.pos_weight.device.type == "cpu"
    out = loss(torch.randn(2, 8, 8), (torch.rand(2, 8, 8) > 0.5).float())
    assert torch.isfinite(out)


def test_combo_loss_reduces_to_bce_when_dice_weight_is_zero():
    logits = torch.randn(2, 8, 8)
    targets = (torch.rand(2, 8, 8) > 0.5).float()
    combo = ComboLoss(pos_weight=None, dice_weight=0.0)
    assert combo(logits, targets).item() == pytest.approx(
        F.binary_cross_entropy_with_logits(logits, targets).item(), rel=1e-6
    )


def test_compute_pos_weight_recovers_the_negative_positive_ratio():
    labels = torch.zeros(4, 10, 10)
    labels[:, :1] = 1.0  # 10% positive
    loader = [(torch.zeros(4, 3, 10, 10), labels)]
    assert compute_pos_weight(loader) == pytest.approx(9.0)


def test_compute_pos_weight_rejects_an_all_negative_set():
    loader = [(torch.zeros(2, 3, 8, 8), torch.zeros(2, 8, 8))]
    with pytest.raises(ValueError, match="No positive pixels"):
        compute_pos_weight(loader)
