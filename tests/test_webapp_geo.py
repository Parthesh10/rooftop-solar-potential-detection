"""Web Mercator maths, geodesic area, and vectorisation.

The area chain is where a silent bug would be most expensive: it produces a
confident number with no visible symptom. These tests pin it against values that
can be checked by hand.
"""

import math

import numpy as np
import pytest

pytest.importorskip("pyproj")

from webapp import geometry  # noqa: E402
from webapp.tiles import (  # noqa: E402
    lonlat_to_pixel,
    lonlat_to_tile,
    metres_per_pixel,
    pixel_to_lonlat,
    tile_grid_for_bounds,
)


# --------------------------------------------------------------------------- #
# Web Mercator
# --------------------------------------------------------------------------- #
def test_metres_per_pixel_matches_the_published_identity():
    """156543.0339 * cos(lat) / 2**z — the number the whole app rests on."""
    assert metres_per_pixel(0.0, 19) == pytest.approx(0.2986, abs=1e-4)
    # Bhopal: the value quoted throughout the docs.
    assert metres_per_pixel(23.26, 19) == pytest.approx(0.2743, abs=1e-3)
    # One zoom level = a factor of two.
    assert metres_per_pixel(0.0, 18) == pytest.approx(2 * metres_per_pixel(0.0, 19))


def test_pixel_roundtrip_is_lossless():
    for lon, lat in [(0, 0), (77.40, 23.21), (-97.74, 30.27), (16.37, 48.21)]:
        x, y = lonlat_to_pixel(lon, lat, 19)
        back_lon, back_lat = pixel_to_lonlat(x, y, 19)
        assert back_lon == pytest.approx(lon, abs=1e-9)
        assert back_lat == pytest.approx(lat, abs=1e-9)


def test_mercator_clamps_at_the_poles():
    """Beyond ~85.05 deg the projection diverges; it must clamp, not blow up."""
    x, y = lonlat_to_pixel(0, 89.9, 10)
    assert math.isfinite(x) and math.isfinite(y)


def test_tile_grid_covers_the_bounds():
    w, s, e, n = -97.7450, 30.2680, -97.7405, 30.2712
    grid = tile_grid_for_bounds(w, s, e, n, 19)
    assert grid.n_tiles == grid.n_cols * grid.n_rows
    # every corner of the box falls inside the tile block
    for lon, lat in [(w, s), (e, s), (e, n), (w, n)]:
        tx, ty = lonlat_to_tile(lon, lat, 19)
        assert grid.x0 <= tx <= grid.x1
        assert grid.y0 <= ty <= grid.y1


def test_grid_pixel_to_lonlat_lands_inside_the_grid():
    grid = tile_grid_for_bounds(77.400, 23.212, 77.406, 23.217, 19)
    lon, lat = grid.pixel_to_lonlat(0, 0)             # top-left
    lon2, lat2 = grid.pixel_to_lonlat(grid.width_px, grid.height_px)
    assert lon < lon2 and lat > lat2                  # x east, y south


# --------------------------------------------------------------------------- #
# Geodesic area
# --------------------------------------------------------------------------- #
def test_geodesic_area_of_a_known_square():
    """A 0.001 deg box at the equator is ~111.32 m x 111.32 m."""
    d = 0.001
    ring = [(0, 0), (d, 0), (d, d), (0, d), (0, 0)]
    area = geometry.polygon_area_m2(ring)
    assert area == pytest.approx(111320 * 0.001 * 111320 * 0.001, rel=0.01)


def test_area_shrinks_with_latitude_as_cos():
    """The reason area is never computed in Web Mercator."""
    d = 0.001
    def box(lat):
        return [(0, lat), (d, lat), (d, lat + d), (0, lat + d), (0, lat)]
    eq = geometry.polygon_area_m2(box(0.0))
    at60 = geometry.polygon_area_m2(box(60.0))
    assert at60 / eq == pytest.approx(math.cos(math.radians(60.0)), rel=0.02)


def test_area_is_orientation_independent():
    ring = [(0, 0), (0.001, 0), (0.001, 0.001), (0, 0.001), (0, 0)]
    assert geometry.polygon_area_m2(ring) == pytest.approx(
        geometry.polygon_area_m2(list(reversed(ring))))


def test_degenerate_rings_are_zero_not_an_exception():
    assert geometry.polygon_area_m2([]) == 0.0
    assert geometry.polygon_area_m2([(0, 0), (1, 1)]) == 0.0


def test_bounds_area_matches_a_manual_product():
    w, s, e, n = 77.400, 23.210, 77.410, 23.220
    area = geometry.bounds_area_m2(w, s, e, n)
    mid = (s + n) / 2
    expect = (geometry.haversine_m(w, mid, e, mid)
              * geometry.haversine_m(w, s, w, n))
    assert area == pytest.approx(expect, rel=0.01)


# --------------------------------------------------------------------------- #
# Vectorisation
# --------------------------------------------------------------------------- #
@pytest.fixture
def grid():
    return tile_grid_for_bounds(77.400, 23.210, 77.410, 23.220, 19)


def test_a_solid_square_becomes_one_building(grid):
    mask = np.zeros((512, 512), dtype=bool)
    mask[100:300, 100:300] = True
    probs = np.where(mask, 0.9, 0.05).astype(np.float32)

    out = geometry.mask_to_buildings(mask, probs, grid)
    assert len(out) == 1
    b = out[0]

    mpp = metres_per_pixel(23.215, 19)
    assert b.area_m2 == pytest.approx((200 * mpp) ** 2, rel=0.05)
    assert b.confidence == pytest.approx(0.9, abs=0.05)


def test_a_courtyard_is_subtracted_not_counted(grid):
    """A hole in a footprint is not roof — it must reduce the area."""
    solid = np.zeros((512, 512), dtype=bool)
    solid[100:300, 100:300] = True
    holed = solid.copy()
    holed[160:240, 160:240] = False

    probs = np.full((512, 512), 0.9, dtype=np.float32)
    a = geometry.mask_to_buildings(solid, probs, grid)[0]
    b = geometry.mask_to_buildings(holed, probs, grid)[0]

    assert b.interiors, "the hole should be emitted as an interior ring"
    assert b.area_m2 < a.area_m2
    # hole is 80x80 of a 200x200 square = 16% of it
    assert b.area_m2 / a.area_m2 == pytest.approx(1 - (80 * 80) / (200 * 200), rel=0.12)


def test_specks_below_the_area_floor_are_dropped(grid):
    mask = np.zeros((512, 512), dtype=bool)
    mask[10:14, 10:14] = True          # ~4x4 px ≈ 1.2 m² at 0.27 m/px
    mask[200:340, 200:340] = True      # a real building
    probs = np.full((512, 512), 0.9, dtype=np.float32)

    out = geometry.mask_to_buildings(mask, probs, grid, min_area_m2=10.0)
    assert len(out) == 1


def test_buildings_are_sorted_largest_first(grid):
    mask = np.zeros((512, 512), dtype=bool)
    mask[20:80, 20:80] = True
    mask[150:350, 150:350] = True
    mask[400:460, 400:440] = True
    probs = np.full((512, 512), 0.8, dtype=np.float32)

    out = geometry.mask_to_buildings(mask, probs, grid)
    assert len(out) == 3
    assert out[0].area_m2 > out[1].area_m2 > out[2].area_m2


def test_empty_mask_yields_no_buildings(grid):
    mask = np.zeros((256, 256), dtype=bool)
    probs = np.zeros((256, 256), dtype=np.float32)
    assert geometry.mask_to_buildings(mask, probs, grid) == []


def test_clean_mask_removes_salt_and_fills_pinholes():
    m = np.zeros((64, 64), dtype=bool)
    m[20:44, 20:44] = True
    m[30, 30] = False        # pinhole -> should close
    m[5, 5] = True           # speck   -> should open away
    out = geometry.clean_mask(m, kernel_px=3)
    assert out[30, 30]
    assert not out[5, 5]


def test_point_in_ring():
    ring = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
    assert geometry.point_in_ring((5, 5), ring)
    assert not geometry.point_in_ring((15, 5), ring)
    assert not geometry.point_in_ring((5, -1), ring)


def test_geojson_carries_usable_area_at_the_packing_factor(grid):
    mask = np.zeros((512, 512), dtype=bool)
    mask[100:300, 100:300] = True
    probs = np.full((512, 512), 0.9, dtype=np.float32)
    out = geometry.mask_to_buildings(mask, probs, grid)

    fc = geometry.buildings_to_geojson(out, packing_factor=0.75)
    assert fc["type"] == "FeatureCollection"
    p = fc["features"][0]["properties"]
    assert p["usable_area_m2"] == pytest.approx(p["roof_area_m2"] * 0.75, rel=1e-3)
    assert fc["features"][0]["geometry"]["type"] == "Polygon"
