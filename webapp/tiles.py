"""Web Mercator maths and tile fetching.

The whole area calculation rests on one identity: at zoom ``z`` and latitude
``phi``, a Web Mercator pixel is

    metres_per_pixel = 156543.0339 * cos(phi) / 2**z

That is why the app serves at z=19 (0.299 m/px at the equator, 0.274 at 23 N) —
it matches Inria's 0.3 m/px training resolution, so the network sees roofs at
the scale it was trained on and pixel counts convert to real area exactly.

Nothing here computes *area*, though. Area in Web Mercator overestimates by
1/cos^2(phi) — about +18% at Bhopal's latitude and far worse further north — so
polygons are converted to lon/lat here and measured geodesically in
``geometry.py``.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image

from webapp.config import (
    TILE_FETCH_CONCURRENCY,
    TILE_PX,
    TILE_TIMEOUT_S,
    USER_AGENT,
)

EARTH_CIRCUMFERENCE_M = 40075016.685578488
ORIGIN_SHIFT = EARTH_CIRCUMFERENCE_M / 2.0

__all__ = [
    "TileGrid",
    "lonlat_to_pixel",
    "pixel_to_lonlat",
    "lonlat_to_tile",
    "metres_per_pixel",
    "tile_grid_for_bounds",
    "fetch_mosaic",
]


def metres_per_pixel(lat_deg: float, zoom: int) -> float:
    """Ground resolution of one pixel at this latitude and zoom."""
    return (EARTH_CIRCUMFERENCE_M * math.cos(math.radians(lat_deg))
            / (TILE_PX * (2 ** zoom)))


def lonlat_to_pixel(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    """Lon/lat (WGS84 degrees) -> global pixel coordinates at ``zoom``."""
    lat = max(min(lat, 85.05112878), -85.05112878)  # Mercator poles
    n = TILE_PX * (2 ** zoom)
    x = (lon + 180.0) / 360.0 * n
    sin_lat = math.sin(math.radians(lat))
    y = (0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)) * n
    return x, y


def pixel_to_lonlat(x: float, y: float, zoom: int) -> tuple[float, float]:
    """Global pixel coordinates -> lon/lat. Inverse of :func:`lonlat_to_pixel`."""
    n = TILE_PX * (2 ** zoom)
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    return lon, math.degrees(lat_rad)


def lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    """Lon/lat -> (tile_x, tile_y) at ``zoom``."""
    px, py = lonlat_to_pixel(lon, lat, zoom)
    return int(px // TILE_PX), int(py // TILE_PX)


@dataclass
class TileGrid:
    """A rectangular block of tiles covering an AOI, plus its pixel origin.

    ``origin_px`` is the global pixel coordinate of the mosaic's top-left corner,
    which is what turns a pixel inside the stitched image back into lon/lat.
    """

    zoom: int
    x0: int
    y0: int
    x1: int          # inclusive
    y1: int          # inclusive
    origin_px: tuple[int, int]

    @property
    def n_cols(self) -> int:
        return self.x1 - self.x0 + 1

    @property
    def n_rows(self) -> int:
        return self.y1 - self.y0 + 1

    @property
    def n_tiles(self) -> int:
        return self.n_cols * self.n_rows

    @property
    def width_px(self) -> int:
        return self.n_cols * TILE_PX

    @property
    def height_px(self) -> int:
        return self.n_rows * TILE_PX

    def coords(self) -> list[tuple[int, int]]:
        return [(x, y)
                for y in range(self.y0, self.y1 + 1)
                for x in range(self.x0, self.x1 + 1)]

    def pixel_to_lonlat(self, px: float, py: float) -> tuple[float, float]:
        """Mosaic-local pixel -> lon/lat."""
        return pixel_to_lonlat(self.origin_px[0] + px,
                               self.origin_px[1] + py, self.zoom)

    def lonlat_to_pixel(self, lon: float, lat: float) -> tuple[float, float]:
        """Lon/lat -> mosaic-local pixel. Inverse of :meth:`pixel_to_lonlat`.

        Needed to rasterise external reference footprints (OpenStreetMap) onto
        the same grid the model ran on — see :mod:`webapp.calibration`.
        """
        gx, gy = lonlat_to_pixel(lon, lat, self.zoom)
        return gx - self.origin_px[0], gy - self.origin_px[1]


def tile_grid_for_bounds(west: float, south: float, east: float, north: float,
                         zoom: int) -> TileGrid:
    """Smallest tile block covering the lon/lat bounding box."""
    if west > east:
        west, east = east, west
    if south > north:
        south, north = north, south

    x0, y0 = lonlat_to_tile(west, north, zoom)   # north = smaller y
    x1, y1 = lonlat_to_tile(east, south, zoom)
    return TileGrid(zoom=zoom, x0=x0, y0=y0, x1=x1, y1=y1,
                    origin_px=(x0 * TILE_PX, y0 * TILE_PX))


def _blank_tile() -> np.ndarray:
    """Neutral grey stand-in for a tile the provider could not serve."""
    return np.full((TILE_PX, TILE_PX, 3), 128, dtype=np.uint8)


async def _fetch_one(client, url_tmpl: str, z: int, x: int, y: int,
                     sem: asyncio.Semaphore) -> tuple[int, int, np.ndarray, bool]:
    url = url_tmpl.format(z=z, x=x, y=y)
    async with sem:
        try:
            r = await client.get(url, timeout=TILE_TIMEOUT_S,
                                 headers={"User-Agent": USER_AGENT})
            r.raise_for_status()
            img = Image.open(BytesIO(r.content)).convert("RGB")
            if img.size != (TILE_PX, TILE_PX):
                img = img.resize((TILE_PX, TILE_PX), Image.Resampling.LANCZOS)
            return x, y, np.asarray(img, dtype=np.uint8), True
        except Exception:
            # One dead tile must not sink the whole analysis — fill it grey and
            # report the count, so the UI can say coverage was incomplete.
            return x, y, _blank_tile(), False


async def fetch_mosaic(grid: TileGrid, url_template: str,
                       progress=None) -> tuple[np.ndarray, int]:
    """Fetch every tile in ``grid`` and stitch them into one RGB array.

    Returns ``(mosaic, n_failed)``. ``progress`` is an optional callable taking
    ``(done, total)`` — the API uses it to drive the job progress bar.
    """
    import httpx

    coords = grid.coords()
    mosaic = np.zeros((grid.height_px, grid.width_px, 3), dtype=np.uint8)
    sem = asyncio.Semaphore(TILE_FETCH_CONCURRENCY)
    failed = 0
    done = 0

    limits = httpx.Limits(max_connections=TILE_FETCH_CONCURRENCY * 2,
                          max_keepalive_connections=TILE_FETCH_CONCURRENCY)
    async with httpx.AsyncClient(limits=limits, follow_redirects=True) as client:
        tasks = [_fetch_one(client, url_template, grid.zoom, x, y, sem)
                 for x, y in coords]
        for coro in asyncio.as_completed(tasks):
            x, y, arr, ok = await coro
            row = (y - grid.y0) * TILE_PX
            col = (x - grid.x0) * TILE_PX
            mosaic[row:row + TILE_PX, col:col + TILE_PX] = arr
            if not ok:
                failed += 1
            done += 1
            if progress is not None:
                progress(done, len(coords))

    return mosaic, failed
