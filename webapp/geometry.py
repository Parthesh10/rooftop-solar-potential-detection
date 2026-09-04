"""Mask -> polygons -> real-world area.

Two things here are easy to get wrong and expensive to get wrong:

1. **Never measure area in Web Mercator.** Mercator inflates area by
   ``1/cos^2(phi)`` — +18% at Bhopal's 23 N, +117% at Stockholm's 59 N. Polygons
   are converted to lon/lat and measured with ``pyproj.Geod``, which integrates
   on the WGS84 ellipsoid and is correct everywhere.

2. **Holes matter.** A courtyard inside a building footprint is not roof. The
   contour hierarchy is walked so inner rings are subtracted rather than
   counted, and they are emitted as GeoJSON Polygon interior rings so the map
   draws them as holes too.

Vectorisation is OpenCV ``findContours`` + Douglas-Peucker, not
``rasterio.features.shapes``: cv2 is already a dependency and rasterio would
pull in the whole GDAL stack for one function.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from pyproj import Geod

from webapp.config import (
    MIN_BUILDING_AREA_M2,
    MORPH_KERNEL_PX,
    POLYGON_SIMPLIFY_FRAC,
)
from webapp.tiles import TileGrid

GEOD = Geod(ellps="WGS84")

__all__ = ["Building", "clean_mask", "mask_to_buildings", "polygon_area_m2",
           "point_in_ring", "buildings_to_geojson"]


@dataclass
class Building:
    """One detected roof: its rings in lon/lat, its area, its confidence."""

    exterior: list[tuple[float, float]]          # [(lon, lat), ...]
    interiors: list[list[tuple[float, float]]]
    area_m2: float
    confidence: float
    n_pixels: int

    def to_feature(self, index: int, packing_factor: float) -> dict:
        return {
            "type": "Feature",
            "id": index,
            "geometry": {
                "type": "Polygon",
                "coordinates": [[list(p) for p in self.exterior]]
                + [[list(p) for p in ring] for ring in self.interiors],
            },
            "properties": {
                "id": index,
                "roof_area_m2": round(self.area_m2, 1),
                "usable_area_m2": round(self.area_m2 * packing_factor, 1),
                "confidence": round(self.confidence, 3),
            },
        }


def clean_mask(mask: np.ndarray, kernel_px: int = MORPH_KERNEL_PX) -> np.ndarray:
    """Morphological open then close: drop specks, fill pinholes."""
    if kernel_px < 1:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_px, kernel_px))
    m = (mask.astype(np.uint8) * 255)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return m > 127


def polygon_area_m2(ring: list[tuple[float, float]]) -> float:
    """Geodesic area of a lon/lat ring, in square metres. Always positive."""
    if len(ring) < 3:
        return 0.0
    lons = [p[0] for p in ring]
    lats = [p[1] for p in ring]
    area, _ = GEOD.polygon_area_perimeter(lons, lats)
    return abs(area)


def point_in_ring(pt: tuple[float, float], ring: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon, used to attach holes to their parent."""
    x, y = pt
    inside = False
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xint = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-30) + x1
            if x < xint:
                inside = not inside
    return inside


def _contour_to_lonlat(contour: np.ndarray, grid: TileGrid) -> list[tuple[float, float]]:
    """(N, 1, 2) pixel contour -> closed lon/lat ring."""
    ring = [grid.pixel_to_lonlat(float(p[0][0]), float(p[0][1])) for p in contour]
    if ring and ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


def mask_to_buildings(mask: np.ndarray, probs: np.ndarray, grid: TileGrid,
                      min_area_m2: float = MIN_BUILDING_AREA_M2,
                      simplify_frac: float = POLYGON_SIMPLIFY_FRAC,
                      morph_kernel_px: int = MORPH_KERNEL_PX) -> list[Building]:
    """Vectorise a boolean mask into geodesically-measured buildings.

    ``morph_kernel_px`` is chosen per area by
    ``calibration.choose_morph_kernel``: the shipped 3 px bridges ~0.9 m at the
    serving resolution, which is wider than the alley between two Indian row
    houses and merges them into one polygon.
    """
    mask = clean_mask(mask, morph_kernel_px)
    binary = (mask.astype(np.uint8) * 255)

    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        return []
    hierarchy = hierarchy[0]  # (N, 4): next, prev, first_child, parent

    # RETR_CCOMP gives two levels: parent==-1 is an outer boundary, otherwise a
    # hole. Group holes under their parent.
    holes_by_parent: dict[int, list[int]] = {}
    for i, h in enumerate(hierarchy):
        parent = int(h[3])
        if parent != -1:
            holes_by_parent.setdefault(parent, []).append(i)

    def simplify(c: np.ndarray) -> np.ndarray:
        eps = simplify_frac * cv2.arcLength(c, True)
        return cv2.approxPolyDP(c, eps, True)

    buildings: list[Building] = []
    for i, contour in enumerate(contours):
        if int(hierarchy[i][3]) != -1:
            continue  # a hole; handled with its parent
        outer = simplify(contour)
        if len(outer) < 3:
            continue

        exterior = _contour_to_lonlat(outer, grid)
        area = polygon_area_m2(exterior)

        interiors: list[list[tuple[float, float]]] = []
        for j in holes_by_parent.get(i, []):
            hole = simplify(contours[j])
            if len(hole) < 3:
                continue
            ring = _contour_to_lonlat(hole, grid)
            hole_area = polygon_area_m2(ring)
            if hole_area <= 0:
                continue
            interiors.append(ring)
            area -= hole_area   # a courtyard is not roof

        if area < min_area_m2:
            continue

        # Mean predicted probability inside the footprint — an honest confidence
        # signal for the UI, not a made-up score.
        stencil = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(stencil, [outer], -1, 1, thickness=cv2.FILLED)
        for j in holes_by_parent.get(i, []):
            cv2.drawContours(stencil, [simplify(contours[j])], -1, 0,
                             thickness=cv2.FILLED)
        n_px = int(stencil.sum())
        conf = float(probs[stencil.astype(bool)].mean()) if n_px else 0.0

        buildings.append(Building(exterior=exterior, interiors=interiors,
                                  area_m2=max(area, 0.0), confidence=conf,
                                  n_pixels=n_px))

    buildings.sort(key=lambda b: -b.area_m2)
    return buildings


def buildings_to_geojson(buildings: list[Building], packing_factor: float) -> dict:
    return {
        "type": "FeatureCollection",
        "features": [b.to_feature(i, packing_factor)
                     for i, b in enumerate(buildings)],
    }


def bounds_area_m2(west: float, south: float, east: float, north: float) -> float:
    """Geodesic area of a lon/lat bounding box — used to report AOI size."""
    ring = [(west, south), (east, south), (east, north), (west, north), (west, south)]
    return polygon_area_m2(ring)


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in metres. Used only for sanity checks."""
    r = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))
