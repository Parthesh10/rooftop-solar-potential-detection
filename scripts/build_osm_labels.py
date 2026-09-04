r"""Build per-building training labels from OpenStreetMap, with an ignore mask.

Why this exists instead of using the hand labels directly
--------------------------------------------------------
The hand labelling merged adjacent buildings and kept the alleys between them,
so it teaches "cluster envelope" rather than "roof" (see finetune/README.md and
results/RESULTS.md). Filtering the good tiles out of it does not work either —
measured 2026-09-05, shape tells you almost nothing (solidity 1.00 and
rectangularity 0.89 for merged *and* single polygons alike), area is a weak
separator, and a strict cut leaves 11-20 usable tiles out of 91.

Neither is OpenStreetMap alone the answer: across the same 91 tiles it covers
only 26.7% of the area the labeller marked built-up, so training on it as-is
would call a lot of real roof "background" — the precise failure being fixed.

So use each source for what it is actually reliable at:

* **positive**  — inside an OSM building. Human-drawn, one polygon per
  building, correct semantics.
* **ignore**    — inside a hand-drawn envelope but not inside an OSM building.
  Genuinely unknown: an alley, or a roof nobody mapped. Excluded from the loss
  entirely rather than guessed at.
* **negative**  — everything else.

Written as an 8-bit PNG using 255 / 128 / 0, which
``DataLoaderSegmentation(ignore_value=128)`` decodes into 1 / -1 / 0.

    python scripts/build_osm_labels.py --out data/finetune_osm

The OSM fetch is one Overpass query per AOI (not per tile) and is cached, so
re-running is free.
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

from scripts.select_finetune_tiles import DEFAULT_AOIS
from webapp import calibration
from webapp.tiles import tile_grid_for_bounds

POSITIVE, IGNORE, NEGATIVE = 255, 128, 0
TILE_PX = 512


async def fetch_osm_for_aois(aois, cache_path: Path) -> dict:
    """One Overpass query per AOI, cached to disk."""
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    store: dict[str, list] = {}
    for name, bbox in aois:
        try:
            rings, _ = await calibration.fetch_reference_buildings(*bbox)
            store[name] = rings
            print(f"  {name}: {len(rings)} OSM buildings")
        except Exception as exc:
            print(f"  {name}: FAILED ({type(exc).__name__}: {exc})")
            store[name] = []
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(store), encoding="utf-8")
    return store


def hand_envelope_mask(json_path: Path) -> np.ndarray:
    """The labeller's polygons, if this tile was labelled at all."""
    mask = np.zeros((TILE_PX, TILE_PX), np.uint8)
    if not json_path.exists():
        return mask
    data = json.loads(json_path.read_text(encoding="utf-8"))
    for shape in data.get("shapes", []):
        if str(shape.get("label", "")).strip().lower() in {
                "not-building", "not_building", "ignore", "exclude"}:
            continue
        pts = shape.get("points") or []
        if len(pts) < 3:
            continue
        cv2.fillPoly(mask, [np.array([[round(x), round(y)] for x, y in pts],
                                     np.int32)], 1)
    return mask


def osm_mask_for_tile(rings, bounds) -> np.ndarray:
    grid = tile_grid_for_bounds(*bounds, 19)
    mask = np.zeros((TILE_PX, TILE_PX), np.uint8)
    for ring in rings:
        px = [grid.lonlat_to_pixel(lon, lat) for lon, lat in ring]
        # Keep polygons that merely clip the tile edge — fillPoly clips for us —
        # but skip anything wildly outside, which is just wasted work.
        if all(-2000 < x < 2000 and -2000 < y < 2000 for x, y in px):
            cv2.fillPoly(mask, [np.array([[round(x), round(y)] for x, y in px],
                                         np.int32)], 1)
    return mask


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", default="data/finetune_candidates",
                    help="output of scripts/select_finetune_tiles.py")
    ap.add_argument("--out", default="data/finetune_osm")
    ap.add_argument("--min-positive-frac", type=float, default=0.02,
                    help="skip tiles whose OSM buildings cover less than this "
                        "fraction of the tile — with almost no positives, the "
                        "tile teaches background and little else")
    ap.add_argument("--no-ignore", action="store_true",
                    help="write plain binary masks (no ignore band). Use only "
                        "to measure how much the ignore mask is worth")
    args = ap.parse_args()

    cand = Path(args.candidates)
    img_dir = cand / "images"
    manifest = cand / "manifest.csv"
    if not manifest.exists():
        raise SystemExit(f"no manifest at {manifest} — run "
                         f"scripts/select_finetune_tiles.py first")

    bounds: dict[str, tuple[float, float, float, float]] = {}
    with manifest.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            bounds[row["filename"][:-4]] = (
                float(row["west"]), float(row["south"]),
                float(row["east"]), float(row["north"]))

    print("fetching OpenStreetMap footprints (cached after the first run)...")
    osm = asyncio.run(fetch_osm_for_aois(
        DEFAULT_AOIS, cand / "osm_buildings.json"))
    if not any(osm.values()):
        raise SystemExit(
            "no OSM buildings for any AOI — Overpass may be rate-limiting. "
            "Wait a minute and re-run; the result is cached.")

    out = Path(args.out)
    out_img, out_lbl = out / "images", out / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    written = skipped = 0
    stats = []
    for stem, bb in sorted(bounds.items()):
        aoi = stem.rsplit("_", 1)[0]
        rings = osm.get(aoi) or []
        src = img_dir / f"{stem}.png"
        if not rings or not src.exists():
            skipped += 1
            continue

        pos = osm_mask_for_tile(rings, bb)
        pos_frac = float(pos.mean())
        if pos_frac < args.min_positive_frac:
            skipped += 1
            continue

        label = np.full((TILE_PX, TILE_PX), NEGATIVE, np.uint8)
        if not args.no_ignore:
            hand = hand_envelope_mask(img_dir / f"{stem}.json")
            label[(hand > 0) & (pos == 0)] = IGNORE
        label[pos > 0] = POSITIVE

        Image.open(src).convert("RGB").save(out_img / f"{stem}.png")
        Image.fromarray(label, mode="L").save(out_lbl / f"{stem}_label.png")
        written += 1
        stats.append((pos_frac, float((label == IGNORE).mean())))

    if not written:
        raise SystemExit("no tiles written — check the AOIs and the OSM fetch")

    pf = np.array([s[0] for s in stats])
    ig = np.array([s[1] for s in stats])
    print(f"\nwrote {written} tiles to {out}  ({skipped} skipped)")
    print(f"  positive coverage : mean {pf.mean():.3f}  median {np.median(pf):.3f}")
    print(f"  ignored           : mean {ig.mean():.3f}  median {np.median(ig):.3f}")
    print(f"\nTrain with:\n  python scripts/finetune_indian.py "
          f"--data-dir {out} --ignore-value 128 "
          f"--label-semantics building-footprint --run-name finetune_osm")


if __name__ == "__main__":
    main()
