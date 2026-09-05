"""Tests for the India/OpenStreetMap scoring harness.

The harness decided which model ships, so its arithmetic is worth pinning. The
two things that can silently go wrong are the pooled metric (a per-window mean
would read differently and look plausible) and the rasterisation that puts OSM
rings onto the same pixel grid the model ran on — a sign flip or a transposed
axis there moves every number without raising anything.
"""

import numpy as np
import pytest

from scripts.eval_india_osm import pooled_scores, rasterise_rings
from webapp.tiles import tile_grid_for_bounds


def test_pooled_scores_perfect_overlap():
    a = np.zeros((10, 10), dtype=bool)
    a[2:6, 2:6] = True
    s = pooled_scores(a, a)
    assert s["iou"] == 1.0
    assert s["precision"] == 1.0
    assert s["recall"] == 1.0


def test_pooled_scores_disjoint_is_zero():
    pred = np.zeros((10, 10), dtype=bool)
    pred[0:3, 0:3] = True
    ref = np.zeros((10, 10), dtype=bool)
    ref[7:10, 7:10] = True
    s = pooled_scores(pred, ref)
    assert s["iou"] == 0.0
    assert s["precision"] == 0.0
    assert s["recall"] == 0.0


def test_pooled_scores_half_overlap():
    """A prediction twice the size of the reference: recall 1, precision 0.5."""
    pred = np.zeros((10, 10), dtype=bool)
    pred[0:4, 0:4] = True                      # 16 px
    ref = np.zeros((10, 10), dtype=bool)
    ref[0:4, 0:2] = True                       # 8 px, wholly inside pred
    s = pooled_scores(pred, ref)
    assert s["recall"] == 1.0
    assert s["precision"] == 0.5
    assert s["iou"] == 0.5                     # 8 / 16


def test_pooled_is_global_not_a_per_window_mean():
    """One dense region and one sparse must not be weighted equally.

    This is fact 14 in miniature: a mean of per-window IoUs would report ~0.5
    here (one window perfect, one window zero), while the pooled figure is
    dominated by the region that actually holds most of the pixels.
    """
    pred = np.zeros((10, 20), dtype=bool)
    ref = np.zeros((10, 20), dtype=bool)
    # Left half: 90 px agreeing perfectly.
    pred[0:9, 0:10] = True
    ref[0:9, 0:10] = True
    # Right half: a single reference pixel the model missed.
    ref[0, 15] = True

    s = pooled_scores(pred, ref)
    assert s["iou"] == pytest.approx(90 / 91, abs=1e-4)
    assert s["iou"] > 0.98                     # not the ~0.5 a per-window mean gives


def test_pooled_scores_handles_empty_prediction():
    ref = np.zeros((4, 4), dtype=bool)
    ref[1, 1] = True
    s = pooled_scores(np.zeros((4, 4), dtype=bool), ref)
    assert s["iou"] == 0.0
    assert s["precision"] is None              # undefined, not silently 0
    assert s["recall"] == 0.0


def test_rasterise_rings_lands_inside_the_bbox():
    """A ring drawn from the AOI's own corners must fill roughly the mosaic."""
    west, south, east, north = 77.654, 12.984, 77.658, 12.987
    grid = tile_grid_for_bounds(west, south, east, north, 19)
    shape = (grid.height_px, grid.width_px)

    ring = [(west, south), (east, south), (east, north), (west, north), (west, south)]
    mask = rasterise_rings([ring], grid, shape)

    assert mask.dtype == bool
    assert mask.any(), "the AOI's own bounds rasterised to nothing"
    # The tile grid is the smallest block *covering* the AOI, so the polygon is
    # a strict subset of the mosaic — present, but not the whole thing.
    assert 0.1 < mask.mean() < 1.0


def test_rasterise_rings_round_trips_through_the_grid():
    """A ring built from pixel coords must come back to those pixels.

    Guards the lon/lat <-> pixel conversion in both directions at once: build a
    box from known mosaic pixels, convert it to lon/lat, rasterise it, and check
    the filled area lands where it started.
    """
    grid = tile_grid_for_bounds(77.654, 12.984, 77.658, 12.987, 19)
    shape = (grid.height_px, grid.width_px)

    corners_px = [(100, 80), (300, 80), (300, 240), (100, 240), (100, 80)]
    ring = [grid.pixel_to_lonlat(x, y) for x, y in corners_px]
    mask = rasterise_rings([ring], grid, shape)

    ys, xs = np.nonzero(mask)
    assert xs.min() == pytest.approx(100, abs=2)
    assert xs.max() == pytest.approx(300, abs=2)
    assert ys.min() == pytest.approx(80, abs=2)
    assert ys.max() == pytest.approx(240, abs=2)


def test_rasterise_rings_ignores_rings_entirely_outside():
    grid = tile_grid_for_bounds(77.654, 12.984, 77.658, 12.987, 19)
    shape = (grid.height_px, grid.width_px)
    far = [(10.0, 50.0), (10.001, 50.0), (10.001, 50.001), (10.0, 50.001),
           (10.0, 50.0)]
    assert not rasterise_rings([far], grid, shape).any()


def test_rasterise_rings_empty_input():
    grid = tile_grid_for_bounds(77.654, 12.984, 77.658, 12.987, 19)
    shape = (grid.height_px, grid.width_px)
    assert not rasterise_rings([], grid, shape).any()
