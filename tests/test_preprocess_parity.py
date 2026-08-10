"""The most important test in the repo: inference must match training (F-01).

The 2023 inference helper fed the network a diagonally-transposed, unnormalised
0-255 tensor. These tests make that class of drift impossible to reintroduce
without a red build.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from config import norm_stats
from infer import preprocess
from process_data.data_loader import DataLoaderSegmentation


def test_preprocess_matches_dataloader_transform(paired_dataset):
    """infer.preprocess and DataLoaderSegmentation.transform must agree exactly."""
    ds = DataLoaderSegmentation(
        paired_dataset["images"], paired_dataset["labels"], augment=False, crop=None
    )
    path = ds.img_files[0]

    from_loader, _ = ds[0]
    from_infer = preprocess(path, stats_key=ds.stats_key)[0]

    assert from_infer.shape == from_loader.shape
    torch.testing.assert_close(from_infer, from_loader, rtol=1e-6, atol=1e-6)


def test_preprocess_does_not_transpose_the_image():
    """F-01 bug 1: `np.transpose(hwc)` reverses ALL axes, giving (C, W, H)."""
    h, w = 8, 16  # deliberately non-square, so a transpose changes the shape
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[0, :, :] = 255  # bright top row

    x = preprocess(arr)
    assert x.shape == (1, 3, h, w), "spatial dims must survive in (H, W) order"

    # The bright row must still be row 0, not column 0.
    assert x[0, 0, 0, :].mean() > x[0, 0, 1:, :].mean()


def test_preprocess_scales_to_unit_range_before_normalising():
    """F-01 bug 2: the old path fed raw 0-255 values, ~60x too large."""
    mean, std = norm_stats("all")
    arr = np.full((8, 8, 3), 255, dtype=np.uint8)
    x = preprocess(arr, stats_key="all")[0]
    # A saturated white pixel maps to (1.0 - mean) / std, not (255 - mean) / std.
    for c in range(3):
        assert x[c].mean() == pytest.approx((1.0 - mean[c]) / std[c], rel=1e-5)
    assert x.abs().max() < 10, "values this large mean the /255 scaling was skipped"


def test_preprocess_accepts_bgr_from_cv2():
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[..., 0] = 200  # pure red in RGB
    bgr = rgb[..., ::-1].copy()
    torch.testing.assert_close(preprocess(rgb), preprocess(bgr, bgr=True))


def test_preprocess_accepts_pil_path_and_array_identically(paired_dataset):
    path = sorted(paired_dataset["images"].glob("*.png"))[0]
    pil = Image.open(path).convert("RGB")
    arr = np.asarray(pil)
    torch.testing.assert_close(preprocess(path), preprocess(pil))
    torch.testing.assert_close(preprocess(path), preprocess(arr))


def test_preprocess_rejects_bad_shapes():
    with pytest.raises(ValueError):
        preprocess(np.zeros((4, 4, 1), dtype=np.uint8))
