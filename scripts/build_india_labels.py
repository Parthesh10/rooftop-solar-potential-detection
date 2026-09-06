r"""Build India-wide training tiles from Open Buildings + OpenStreetMap.

This is `scripts/build_osm_labels.py` scaled past the two cities a human could
hand-label. That script needed hand-drawn envelopes to know where OpenStreetMap
could not be trusted, which capped it at ~100 tiles in Bangalore and Bhopal.
Google Open Buildings supplies both the footprints *and* a calibrated confidence
per building, so the "where can I not trust this" signal comes from the data
itself and the pipeline runs anywhere.

The three-way label, and why each source is used the way it is
--------------------------------------------------------------

* **positive** — an OpenStreetMap building (human-drawn, so real), OR an Open
  Buildings detection at or above Google's own 80%-precision threshold for that
  region. That threshold is per-S2-cell and genuinely varies: 0.707 in Chennai
  against 0.833 in Mumbai, so one global cutoff would be wrong in both.

* **ignore** — an Open Buildings detection *below* that threshold. Not
  trustworthy enough to train on, far too likely to be real to call background.
  Confidence tracks size here (measured in Chennai: median 53 m2 in the
  0.65-0.70 band against 228 m2 above 0.85), and small roofs are precisely what
  the model is already silent on — so marking these negative would reinforce the
  exact failure this data exists to fix.

* **negative** — everything else.

Written as 255 / 128 / 0, which ``DataLoaderSegmentation(ignore_value=128)``
decodes to 1 / -1 / 0, the same contract `build_osm_labels.py` already uses.

Imagery comes from the same provider and zoom the web app serves, so training
and deployment see the same pixels.

    python scripts/download_open_buildings.py     # first — fetches the footprints
    python scripts/build_india_labels.py --out data/india_wide
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import cv2
import numpy as np
from PIL import Image

from webapp import calibration
from webapp.config import tile_provider_config
from webapp.tiles import fetch_mosaic, tile_grid_for_bounds

POSITIVE, IGNORE, NEGATIVE = 255, 128, 0
WINDOW = 512


def rings_to_mask(rings, grid, shape, offset=(0, 0)) -> np.ndarray:
    """Rasterise lon/lat rings onto a window of the mosaic grid."""
    x_off, y_off = offset
    mask = np.zeros(shape, np.uint8)
    for ring in rings:
        pts = []
        for lon, lat in ring:
            px, py = grid.lonlat_to_pixel(lon, lat)
            pts.append([round(px - x_off), round(py - y_off)])
        arr = np.array(pts, np.int32)
        # cv2 clips for us; skip only what is nowhere near the window.
        if arr[:, 0].max() < -64 or arr[:, 1].max() < -64:
            continue
        if arr[:, 0].min() > shape[1] + 64 or arr[:, 1].min() > shape[0] + 64:
            continue
        cv2.fillPoly(mask, [arr], 1)
    return mask


async def fetch_osm_cached(name, bbox, cache: Path) -> list:
    """OSM buildings for an AOI, cached — Overpass rate-limits bursts."""
    store = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    if name in store:
        return store[name]
    try:
        rings, _ = await calibration.fetch_reference_buildings(*bbox)
    except Exception as exc:
        print(f"    OSM failed ({type(exc).__name__}: {exc}) — "
              f"continuing on Open Buildings alone")
        rings = []
    store[name] = rings
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(store), encoding="utf-8")
    return rings


async def build_aoi(name: str, payload: dict, out_img: Path, out_lbl: Path,
                    osm_cache: Path, args) -> list[dict]:
    bbox = tuple(payload["bounds"])
    thr = float(payload["positive_threshold"])
    buildings = payload["buildings"]

    pos_rings = [b["ring"] for b in buildings if b["confidence"] >= thr]
    unsure_rings = [b["ring"] for b in buildings if b["confidence"] < thr]

    grid = tile_grid_for_bounds(*bbox, args.zoom)
    provider = tile_provider_config()
    print(f"  {name}: {grid.n_tiles} tiles, {len(pos_rings)} positive / "
          f"{len(unsure_rings)} unsure buildings (thr {thr:.3f})", flush=True)

    mosaic, n_failed = await fetch_mosaic(grid, provider["url"])
    if n_failed:
        print(f"    {n_failed}/{grid.n_tiles} imagery tiles failed")

    osm_rings = await fetch_osm_cached(name, bbox, osm_cache)
    print(f"    OSM adds {len(osm_rings)} human-drawn buildings", flush=True)

    h, w = mosaic.shape[:2]
    full_pos = rings_to_mask(pos_rings + osm_rings, grid, (h, w))
    full_unsure = rings_to_mask(unsure_rings, grid, (h, w))

    rows = []
    gray = mosaic.mean(axis=2)
    for y0 in range(0, h - WINDOW + 1, WINDOW):
        for x0 in range(0, w - WINDOW + 1, WINDOW):
            patch = mosaic[y0:y0 + WINDOW, x0:x0 + WINDOW]
            pos = full_pos[y0:y0 + WINDOW, x0:x0 + WINDOW]
            uns = full_unsure[y0:y0 + WINDOW, x0:x0 + WINDOW]

            # A window of water, canopy or bare field teaches background and
            # little else, and there is no shortage of background elsewhere.
            if float(gray[y0:y0 + WINDOW, x0:x0 + WINDOW].std()) < args.min_built_up:
                continue
            pos_frac = float(pos.mean())
            if pos_frac < args.min_positive_frac:
                continue

            label = np.full((WINDOW, WINDOW), NEGATIVE, np.uint8)
            label[(uns > 0) & (pos == 0)] = IGNORE
            label[pos > 0] = POSITIVE

            stem = f"{name}_{y0 // WINDOW:02d}{x0 // WINDOW:02d}"
            Image.fromarray(patch).save(out_img / f"{stem}.png")
            Image.fromarray(label, mode="L").save(out_lbl / f"{stem}_label.png")
            rows.append({"filename": f"{stem}.png", "aoi": name,
                         "positive_frac": round(pos_frac, 4),
                         "ignore_frac": round(float((label == IGNORE).mean()), 4)})
    print(f"    wrote {len(rows)} tiles", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--buildings", default="data/open_buildings",
                    help="output of scripts/download_open_buildings.py")
    ap.add_argument("--out", default="data/india_wide")
    ap.add_argument("--zoom", type=int, default=19)
    ap.add_argument("--min-built-up", type=float, default=8.0,
                    help="grayscale std floor — drops water, canopy, bare field")
    ap.add_argument("--min-positive-frac", type=float, default=0.02,
                    help="skip windows with almost no roof in them")
    ap.add_argument("--skip-existing", action="store_true",
                    help="leave AOIs that already have tiles alone — makes the "
                        "run resumable after an interrupted fetch")
    ap.add_argument("--aoi-timeout", type=float, default=900.0,
                    help="give up on one AOI after this many seconds. A stalled "
                        "tile fetch would otherwise hang the whole run: measured "
                        "2026-09-06, one AOI sat on a dead connection for 20 "
                        "minutes at zero CPU while seven others waited")
    args = ap.parse_args()

    src = Path(args.buildings)
    if not src.is_absolute():
        src = REPO / src
    payloads = sorted(p for p in src.glob("*.json") if not p.name.startswith("_"))
    if not payloads:
        raise SystemExit(f"no AOI files in {src} — run "
                         f"scripts/download_open_buildings.py first")

    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out
    out_img, out_lbl = out / "images", out / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    async def run_all():
        rows = []
        for p in payloads:
            payload = json.loads(p.read_text(encoding="utf-8"))
            name = payload["aoi"]
            if args.skip_existing and any(out_img.glob(f"{name}_*.png")):
                print(f"  {name}: already built, skipping")
                continue
            try:
                rows.extend(await asyncio.wait_for(
                    build_aoi(name, payload, out_img, out_lbl,
                              out / "osm_cache.json", args),
                    timeout=args.aoi_timeout))
            except asyncio.TimeoutError:
                print(f"  {name}: TIMED OUT after {args.aoi_timeout:.0f}s "
                      f"— skipping, re-run with --skip-existing to retry it")
        return rows

    print(f"building tiles from {len(payloads)} AOI(s)\n")
    rows = asyncio.run(run_all())

    # Rebuild the manifest from what is actually on disk rather than from this
    # run's rows. A resumed or interrupted run otherwise describes only the AOIs
    # it happened to touch, while the rest of the tiles sit there unlisted.
    seen = {r["filename"] for r in rows}
    for img in sorted(out_img.glob("*.png")):
        if img.name in seen:
            continue
        lbl = out_lbl / f"{img.stem}_label.png"
        if not lbl.exists():
            continue
        arr = np.array(Image.open(lbl).convert("L"))
        rows.append({"filename": img.name,
                     "aoi": img.stem.rsplit("_", 1)[0],
                     "positive_frac": round(float(np.mean(arr == POSITIVE)), 4),
                     "ignore_frac": round(float(np.mean(arr == IGNORE)), 4)})
    rows.sort(key=lambda r: r["filename"])
    if not rows:
        raise SystemExit("no tiles written — check the AOIs and the imagery fetch")

    manifest_path = out / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wtr.writeheader()
        wtr.writerows(rows)

    pf = np.array([r["positive_frac"] for r in rows])
    ig = np.array([r["ignore_frac"] for r in rows])
    print(f"\n{len(rows)} tiles -> {out}")
    print(f"  positive coverage : mean {pf.mean():.3f}  median {np.median(pf):.3f}")
    print(f"  ignored           : mean {ig.mean():.3f}  median {np.median(ig):.3f}")
    by_aoi: dict[str, int] = {}
    for r in rows:
        by_aoi[r["aoi"]] = by_aoi.get(r["aoi"], 0) + 1
    for k, v in sorted(by_aoi.items()):
        print(f"  {k:24s} {v:4d} tiles")


if __name__ == "__main__":
    main()
