"""Inria windowed-read dataset (process_data/inria.py).

This path had never run before the general-model pivot; these tests cover the
window contract, the deterministic grid, the empty-window filter, and the
missing-ground-truth guard, using small synthetic .tif tiles.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

tifffile = pytest.importorskip("tifffile")

from process_data.inria import InriaWindowDataset  # noqa: E402

TILE = 512


@pytest.fixture
def inria_tiles(tmp_path: Path):
    """austin1..2 + vienna1..2 as 512x512 tiles with a bright building block."""
    img_dir = tmp_path / "images"
    gt_dir = tmp_path / "gt"
    img_dir.mkdir()
    gt_dir.mkdir()
    rng = np.random.default_rng(0)
    for name in ("austin1", "austin2", "vienna1", "vienna2"):
        img = rng.integers(40, 90, size=(TILE, TILE, 3), dtype=np.uint8)
        gt = np.zeros((TILE, TILE), dtype=np.uint8)
        gt[100:300, 100:300] = 255          # one "building"
        img[100:300, 100:300] = 210
        tifffile.imwrite(img_dir / f"{name}.tif", img)
        tifffile.imwrite(gt_dir / f"{name}.tif", gt)
    return {"images": img_dir, "gt": gt_dir,
            "files": sorted(img_dir.glob("*.tif"))}


def test_window_shapes_and_normalisation(inria_tiles):
    ds = InriaWindowDataset(
        inria_tiles["files"], inria_tiles["gt"], window=256,
        samples_per_tile=4, augment=False, stats_key="imagenet",
    )
    assert len(ds) == 4 * 4
    x, y = ds[0]
    assert x.shape == (3, 256, 256) and x.dtype == torch.float32
    assert y.shape == (256, 256)
    assert set(torch.unique(y).tolist()) <= {0.0, 1.0}
    # imagenet-normalised, not raw [0, 1]
    assert x.min() < 0.0


def test_deterministic_grid_is_reproducible(inria_tiles):
    kw = dict(window=256, augment=False, deterministic=True, stride=256)
    a = InriaWindowDataset(inria_tiles["files"], inria_tiles["gt"], **kw)
    b = InriaWindowDataset(inria_tiles["files"], inria_tiles["gt"], **kw)
    assert len(a) == len(b)
    xa, ya = a[3]
    xb, yb = b[3]
    torch.testing.assert_close(xa, xb)
    torch.testing.assert_close(ya, yb)


def test_min_positive_filter_prefers_building_windows(inria_tiles):
    """With empty_ratio=0 every sampled window must clear min_positive."""
    ds = InriaWindowDataset(
        inria_tiles["files"], inria_tiles["gt"], window=256,
        samples_per_tile=8, min_positive=0.02, empty_ratio=0.0,
        augment=False, seed=1,
    )
    hits = sum(float(ds[i][1].mean()) >= 0.02 for i in range(len(ds)))
    # the retry budget is bounded, so allow a few misses, but the filter should
    # still dominate versus unfiltered (~25% of windows overlap the block)
    assert hits >= 0.8 * len(ds)


def test_missing_ground_truth_is_rejected(inria_tiles):
    (inria_tiles["gt"] / "austin1.tif").unlink()
    with pytest.raises(FileNotFoundError, match="no ground truth"):
        InriaWindowDataset(inria_tiles["files"], inria_tiles["gt"], window=256)


def test_window_must_divide_by_32(inria_tiles):
    with pytest.raises(ValueError, match="multiple of 32"):
        InriaWindowDataset(inria_tiles["files"], inria_tiles["gt"], window=250)
