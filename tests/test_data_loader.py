"""Regression tests for the dataloader fixes (F-04, F-07, F-13, F-14)."""

import numpy as np
import pytest
import torch
from PIL import Image

from process_data.data_loader import D4_OPS, DataLoaderSegmentation, _apply_d4


def test_masks_are_paired_by_filename(paired_dataset):
    """F-04: the old code globbed both folders and zipped them positionally."""
    ds = DataLoaderSegmentation(
        paired_dataset["images"], paired_dataset["labels"], augment=False, crop=None
    )
    for img_path, mask_path in zip(ds.img_files, ds.mask_files):
        assert mask_path.stem == f"{img_path.stem}_label"


def test_missing_mask_raises_instead_of_misaligning(paired_dataset):
    """The failure the old code hid: one deleted mask shifted every later pair."""
    victim = sorted(paired_dataset["labels"].glob("*.png"))[2]
    victim.unlink()
    with pytest.raises(FileNotFoundError, match="have no matching mask"):
        DataLoaderSegmentation(
            paired_dataset["images"], paired_dataset["labels"], augment=False, crop=None
        )


def test_non_strict_drops_unpaired_rather_than_shifting(paired_dataset):
    sorted(paired_dataset["labels"].glob("*.png"))[2].unlink()
    ds = DataLoaderSegmentation(
        paired_dataset["images"], paired_dataset["labels"],
        augment=False, crop=None, strict=False,
    )
    assert len(ds) == 5
    for img_path, mask_path in zip(ds.img_files, ds.mask_files):
        assert mask_path.stem == f"{img_path.stem}_label"


def test_crop_must_be_multiple_of_32(paired_dataset):
    """F-07: the original cropped to 248, which divides by neither 16 nor 32."""
    with pytest.raises(ValueError, match="multiple of 32"):
        DataLoaderSegmentation(
            paired_dataset["images"], paired_dataset["labels"], augment=True, crop=248
        )


def test_eval_keeps_native_geometry(paired_dataset):
    """Eval must not crop; the model boundary pads instead (F-07)."""
    ds = DataLoaderSegmentation(
        paired_dataset["images"], paired_dataset["labels"], augment=False, crop=None
    )
    x, y = ds[0]
    assert x.shape == (3, 64, 64)
    assert y.shape == (64, 64)


def test_train_crop_applies_to_image_and_mask_identically(paired_dataset):
    ds = DataLoaderSegmentation(
        paired_dataset["images"], paired_dataset["labels"], augment=True, crop=32
    )
    x, y = ds[0]
    assert x.shape == (3, 32, 32)
    assert y.shape == (32, 32)


def test_mask_is_binary_after_transform(paired_dataset):
    ds = DataLoaderSegmentation(
        paired_dataset["images"], paired_dataset["labels"], augment=False, crop=None
    )
    _, y = ds[0]
    assert set(torch.unique(y).tolist()) <= {0.0, 1.0}


def test_palette_mode_mask_is_converted(tmp_path):
    """F-13: without .convert('L'), channel 0 of a P-mode PNG is a palette index."""
    img_dir, lbl_dir = tmp_path / "images", tmp_path / "labels"
    img_dir.mkdir()
    lbl_dir.mkdir()
    Image.fromarray(np.zeros((32, 32, 3), np.uint8)).save(img_dir / "a.png")

    arr = np.zeros((32, 32), np.uint8)
    arr[:16] = 255
    Image.fromarray(arr, mode="L").convert("P").save(lbl_dir / "a_label.png")

    ds = DataLoaderSegmentation(img_dir, lbl_dir, augment=False, crop=None)
    _, y = ds[0]
    assert y[:16].mean() == pytest.approx(1.0)
    assert y[16:].mean() == pytest.approx(0.0)


def test_d4_ops_are_lossless_and_distinct():
    """F-14: D4 via PIL transposes is an exact pixel permutation — no interpolation."""
    arr = np.arange(16, dtype=np.uint8).reshape(4, 4)
    img = Image.fromarray(arr, mode="L")
    results = {_apply_d4(img, ops).tobytes() for ops in D4_OPS}
    assert len(D4_OPS) == 8
    assert len(results) == 8  # all eight are genuinely different
    for ops in D4_OPS:
        out = np.array(_apply_d4(img, ops))
        assert sorted(out.ravel().tolist()) == sorted(arr.ravel().tolist())


def test_size_mismatch_between_image_and_mask_raises(tmp_path):
    img_dir, lbl_dir = tmp_path / "images", tmp_path / "labels"
    img_dir.mkdir()
    lbl_dir.mkdir()
    Image.fromarray(np.zeros((32, 32, 3), np.uint8)).save(img_dir / "a.png")
    Image.fromarray(np.zeros((16, 16), np.uint8), mode="L").save(lbl_dir / "a_label.png")
    ds = DataLoaderSegmentation(img_dir, lbl_dir, augment=False, crop=None)
    with pytest.raises(ValueError, match="size mismatch"):
        _ = ds[0]


def test_unlabelled_dataset_returns_zero_masks(paired_dataset):
    ds = DataLoaderSegmentation(paired_dataset["images"], None, augment=False, crop=None)
    _, y = ds[0]
    assert float(y.sum()) == 0.0
