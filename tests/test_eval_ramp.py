"""Tests for the ramp benchmark harness.

The dangerous failure here is silent: a wrong sign or a swapped axis in the
geotransform still produces a plausible-looking roof fraction while every
polygon sits somewhere other than its building, and the benchmark then reports a
confidently wrong IoU. Fact 35 is the same lesson from the label-building side.
"""

import json

import numpy as np
import pytest

from scripts.eval_ramp import geotransform, rasterise


class _Tag:
    def __init__(self, name, value):
        self.name = name
        self.value = value


class _Tags:
    def __init__(self, d):
        self._d = d

    def values(self):
        return [_Tag(k, v) for k, v in self._d.items()]


class _Page:
    """Just enough of a tifffile page for geotransform()."""

    def __init__(self, tie, scale):
        d = {}
        if tie is not None:
            d["ModelTiepointTag"] = tie
        if scale is not None:
            d["ModelPixelScaleTag"] = scale
        self.tags = _Tags(d)


# Values taken from a real ramp Karnataka tile.
TIE = (0.0, 0.0, 0.0, 75.98779134455837, 12.46125992296657, 0.0)
SCALE = (2.8615425050457376e-06, 2.8615425050457376e-06, 0.0)


def test_geotransform_reads_tags():
    lon0, lat0, dlon, dlat = geotransform(_Page(TIE, SCALE))
    assert lon0 == pytest.approx(75.98779134, abs=1e-6)
    assert lat0 == pytest.approx(12.46125992, abs=1e-6)
    assert dlon == pytest.approx(2.8615425e-06, rel=1e-6)
    assert dlat == pytest.approx(2.8615425e-06, rel=1e-6)


def test_geotransform_pixel_scale_is_about_30cm():
    """ramp Karnataka is documented as 30 cm; the tags must agree.

    This is the reason the dataset is usable at all — Inria is 0.30 m/px and the
    app serves Esri z19 at ~0.30, so no rescaling is needed. If a future dataset
    is added at a different scale this test is where it should surface.
    """
    _, lat0, dlon, _ = geotransform(_Page(TIE, SCALE))
    metres_per_deg_lon = 111_320.0 * np.cos(np.radians(lat0))
    assert dlon * metres_per_deg_lon == pytest.approx(0.31, abs=0.03)


def test_geotransform_missing_tags_raises():
    with pytest.raises(ValueError, match="ModelTiepoint"):
        geotransform(_Page(None, SCALE))
    with pytest.raises(ValueError, match="ModelPixelScale"):
        geotransform(_Page(TIE, None))


def _square(lon0, lat0, dlon, dlat, x0, y0, w, h):
    """A polygon covering pixels [x0, x0+w) x [y0, y0+h)."""
    return {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[
        [lon0 + x0 * dlon, lat0 - y0 * dlat],
        [lon0 + (x0 + w) * dlon, lat0 - y0 * dlat],
        [lon0 + (x0 + w) * dlon, lat0 - (y0 + h) * dlat],
        [lon0 + x0 * dlon, lat0 - (y0 + h) * dlat],
        [lon0 + x0 * dlon, lat0 - y0 * dlat],
    ]]}}


def test_rasterise_puts_a_polygon_where_it_belongs():
    """North-up: +lon is +x and +lat is -y. A sign error here is invisible in
    the roof fraction and fatal to the IoU."""
    gt = (75.0, 12.0, 1e-5, 1e-5)
    feats = [_square(*gt, 10, 20, 30, 40)]
    m = rasterise(feats, gt, (256, 256))
    ys, xs = np.nonzero(m)
    assert xs.min() == pytest.approx(10, abs=1)
    assert xs.max() == pytest.approx(40, abs=1)
    assert ys.min() == pytest.approx(20, abs=1)
    assert ys.max() == pytest.approx(60, abs=1)


def test_rasterise_area_matches_polygon():
    gt = (75.0, 12.0, 1e-5, 1e-5)
    m = rasterise([_square(*gt, 50, 50, 20, 20)], gt, (256, 256))
    assert m.sum() == pytest.approx(20 * 20, rel=0.15)


def test_rasterise_handles_multipolygon():
    gt = (75.0, 12.0, 1e-5, 1e-5)
    a = _square(*gt, 10, 10, 10, 10)["geometry"]["coordinates"]
    b = _square(*gt, 100, 100, 10, 10)["geometry"]["coordinates"]
    feats = [{"type": "Feature",
              "geometry": {"type": "MultiPolygon", "coordinates": [a, b]}}]
    m = rasterise(feats, gt, (256, 256))
    assert m[15, 15] and m[105, 105]
    # cv2.fillPoly is inclusive of both boundaries, so a 10x10-pixel polygon
    # covers 11x11 = 121 px. Two of them is 242, not 200.
    assert m.sum() == pytest.approx(2 * 11 * 11, rel=0.1)


def test_rasterise_empty_and_degenerate():
    gt = (75.0, 12.0, 1e-5, 1e-5)
    assert not rasterise([], gt, (64, 64)).any()
    # A ramp label file for an empty tile is a FeatureCollection with no
    # features; two-point "polygons" and null geometries must not raise.
    junk = [{"type": "Feature", "geometry": None},
            {"type": "Feature", "geometry": {"type": "Point",
                                             "coordinates": [75.0, 12.0]}},
            {"type": "Feature", "geometry": {"type": "Polygon",
                                             "coordinates": [[[75.0, 12.0],
                                                              [75.1, 12.1]]]}}]
    assert not rasterise(junk, gt, (64, 64)).any()


def test_rasterise_clips_to_tile():
    """A building crossing the tile edge is clipped, not wrapped."""
    gt = (75.0, 12.0, 1e-5, 1e-5)
    m = rasterise([_square(*gt, -20, -20, 40, 40)], gt, (64, 64))
    assert m[0, 0]
    assert m.sum() < 64 * 64
    assert not m[40:, 40:].any()


def test_real_label_file_shape():
    """The on-disk format this harness assumes: a FeatureCollection of Polygons
    with a 'building' label property."""
    doc = json.loads('{"type":"FeatureCollection","features":['
                     '{"type":"Feature","properties":{"label":"building"},'
                     '"geometry":{"type":"Polygon","coordinates":'
                     '[[[75.9878,12.4611],[75.9879,12.4611],'
                     '[75.9879,12.4610],[75.9878,12.4611]]]}}]}')
    gt = (75.98779134455837, 12.46125992296657,
          2.8615425050457376e-06, 2.8615425050457376e-06)
    m = rasterise(doc["features"], gt, (256, 256))
    assert m.any(), "a real ramp polygon rasterised to nothing"
