"""Tri-state labels: positive / negative / ignore.

The ignore state exists because OpenStreetMap is incomplete and the hand labels
merged buildings, so some pixels are genuinely unknown. Getting this wrong is
expensive and silent: treat "unknown" as background and the model learns that
real rooftops are background, which is the exact failure the whole Indian
fine-tuning effort is fixing.
"""

import numpy as np
import pytest
import torch

from loss.losses import IGNORE_INDEX, ComboLoss, DiceLoss, TverskyLoss, split_targets


def test_split_targets_separates_the_three_states():
    t = torch.tensor([[1.0, 0.0, IGNORE_INDEX]])
    clean, valid = split_targets(t)
    assert clean.tolist() == [[1.0, 0.0, 0.0]]
    assert valid.tolist() == [[True, True, False]]


def test_a_two_state_target_is_entirely_valid():
    """Every existing dataset must be unaffected."""
    _, valid = split_targets(torch.tensor([[1.0, 0.0, 1.0]]))
    assert bool(valid.all())


@pytest.mark.parametrize("loss_fn", [
    ComboLoss(pos_weight=2.4, dice_weight=0.6),
    ComboLoss(pos_weight=None, dice_weight=0.0),
    DiceLoss(),
    TverskyLoss(),
])
def test_the_loss_ignores_what_an_ignored_pixel_claims(loss_fn):
    """The one invariant that matters, stated once per objective.

    Note what is *not* asserted: that ignoring leaves the loss unchanged. It
    should not — ignored pixels are dropped from the average, so the number
    legitimately moves. What must hold is that the label underneath an ignored
    pixel has no effect at all. Flip every ignored label and the loss must not
    budge; if it does, those labels are leaking into the gradient.
    """
    torch.manual_seed(0)
    logits = torch.randn(2, 1, 16, 16)
    target = (torch.rand(2, 1, 16, 16) > 0.7).float()
    region = (slice(None), slice(None), slice(0, 6), slice(0, 6))

    as_is = target.clone()
    as_is[region] = IGNORE_INDEX

    flipped = target.clone()
    flipped[region] = 1.0 - flipped[region]
    flipped[region] = IGNORE_INDEX

    a, b = loss_fn(logits, as_is), loss_fn(logits, flipped)
    assert torch.allclose(a, b, atol=1e-6), (
        f"ignored labels leaked: {float(a):.6f} vs {float(b):.6f}")

    # And ignoring must actually do something — a mechanism that changes
    # nothing at all would also pass the assertion above.
    assert not torch.allclose(a, loss_fn(logits, target), atol=1e-6)


def test_ignoring_is_not_the_same_as_labelling_background():
    """A regression guard for the tempting shortcut."""
    torch.manual_seed(1)
    logits = torch.full((1, 1, 8, 8), 3.0)          # confidently positive
    as_background = torch.zeros(1, 1, 8, 8)
    as_ignored = torch.full((1, 1, 8, 8), IGNORE_INDEX)
    as_ignored[0, 0, 0, 0] = 1.0                     # one real positive

    loss = ComboLoss(pos_weight=2.4, dice_weight=0.6)
    assert loss(logits, as_background) > loss(logits, as_ignored)


def test_bce_averages_over_valid_pixels_only():
    """Ignoring must not shrink the loss in proportion to the area ignored."""
    logits = torch.zeros(1, 1, 10, 10)
    small = torch.zeros(1, 1, 10, 10)
    small[0, 0, :2] = 1.0

    big = small.clone()
    big[0, 0, 5:] = IGNORE_INDEX                     # ignore half the tile

    loss = ComboLoss(pos_weight=1.0, dice_weight=0.0)
    a, b = float(loss(logits, small)), float(loss(logits, big))
    # Same per-pixel difficulty in the valid region, so the means must be close;
    # a sum-based reduction would roughly halve the second.
    assert b == pytest.approx(a, rel=0.35)
    assert b > a * 0.6


def test_dataset_decodes_the_ignore_grey_level(tmp_path):
    from PIL import Image

    from process_data.data_loader import DataLoaderSegmentation

    img_dir, lbl_dir = tmp_path / "images", tmp_path / "labels"
    img_dir.mkdir(), lbl_dir.mkdir()
    for i in range(2):
        Image.fromarray(np.full((64, 64, 3), 120, np.uint8)).save(img_dir / f"t{i}.png")
        m = np.zeros((64, 64), np.uint8)
        m[:16] = 255      # positive
        m[16:32] = 128    # ignore
        Image.fromarray(m, mode="L").save(lbl_dir / f"t{i}_label.png")

    tri = DataLoaderSegmentation(img_dir, lbl_dir, augment=False, crop=None,
                                 ignore_value=128)
    _, y = tri[0]
    assert float((y == 1).sum()) == 64 * 16
    assert float((y == IGNORE_INDEX).sum()) == 64 * 16
    assert float((y == 0).sum()) == 64 * 32

    # Default (no ignore_value) keeps the old binary behaviour: 128 <= 127 is
    # False... 128 > 127 is True, so the grey band reads as positive.
    binary = DataLoaderSegmentation(img_dir, lbl_dir, augment=False, crop=None)
    _, yb = binary[0]
    assert set(np.unique(yb.numpy()).tolist()) <= {0.0, 1.0}
    assert float((yb == 1).sum()) == 64 * 32


def test_pos_weight_estimate_excludes_ignored_pixels():
    from loss.losses import compute_pos_weight

    class _Loader:
        def __iter__(self):
            y = torch.zeros(1, 1, 10, 10)
            y[0, 0, :2] = 1.0                 # 20 positives
            y[0, 0, 5:] = IGNORE_INDEX        # 50 ignored
            yield torch.zeros(1, 3, 10, 10), y

    # 50 valid pixels, 20 positive -> (50 - 20) / 20 = 1.5
    assert compute_pos_weight(_Loader(), max_batches=1) == pytest.approx(1.5)
