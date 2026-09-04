r"""Select candidate Indian tiles for hand-labelling and fine-tuning.

Answers a narrower question than "grab some Bangalore imagery": *which* tiles
are worth a human's time. Measured on a Bangalore block (results/RESULTS.md,
2026-09-04), the model is not merely under-confident on Indian rooftops — 56%
of mapped buildings score below probability 0.10, a confident negative no
threshold reaches. Labelling tiles at random would spend most of the budget on
easy tiles the model already gets right. Instead this script:

1. Fetches real imagery for a handful of named AOIs (the tile-fetch path is
   identical to the web app's — same provider, same zoom).
2. Runs the shipped model over each AOI to get its probability map (same
   sliding-window inference the app uses).
3. Slices into non-overlapping 512x512 windows (the model's native window).
4. Scores each window two ways: **built-up-ness** (grayscale std — filters out
   blank fields, water, uniform tree canopy, which are not worth a human's
   time) and **model uncertainty** (how far the window's probabilities sit from
   confident 0/1 — the windows most likely to contain the buildings the model
   is silently missing).
5. Keeps the built-up windows, ranked by uncertainty, and writes them as PNGs
   plus a manifest ready for a labelling tool.

Usage:

    python scripts/select_finetune_tiles.py --out data/finetune_candidates

Then hand-label (see finetune/README.md) and fine-tune with
scripts/finetune_indian.py.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
from PIL import Image

from webapp.config import tile_provider_config
from webapp.inference import load_model, predict_mask
from webapp.tiles import fetch_mosaic, metres_per_pixel, tile_grid_for_bounds

WINDOW = 512

# Named AOIs to draw candidates from. Centred on locations already confirmed
# dense-residential this session (CV Raman Nagar, Bangalore — see
# results/RESULTS.md) plus Bhopal, the project's other long-standing reference
# point (coverage.py's TRAINING_CITIES distance check, CLAUDE.md fact 13).
# Each box is sized to stay well under the tile provider's fetch cap.
# Each box is 14x12 tiles at z19 (168 tiles, 7x6 = 42 non-overlapping 512px
# windows) — sized with webapp.tiles' own tile math, not eyeballed degrees, so
# every AOI actually lands under MAX_TILES_PER_AOI.
DEFAULT_AOIS: list[tuple[str, tuple[float, float, float, float]]] = [
    ("bangalore_cvraman_a", (77.650681, 12.98181, 77.660294, 12.989839)),
    ("bangalore_cvraman_b", (77.643127, 12.987162, 77.65274, 12.995191)),
    ("bhopal_central_a", (77.400055, 23.249548, 77.409668, 23.257118)),
    ("bhopal_central_b", (77.419968, 23.259642, 77.429581, 23.267212)),
]

MIN_TILES_PER_AOI = 4     # sanity floor — an AOI this small is a bug, not data
MAX_TILES_PER_AOI = 240   # keep each AOI under the provider's MAX_TILES guard


@dataclass
class Candidate:
    aoi: str
    idx: int
    x0: int
    y0: int
    built_up: float
    uncertainty: float
    mean_prob: float
    bounds: tuple[float, float, float, float]  # west, south, east, north


def _window_bounds(grid, x0: int, y0: int, w: int, h: int
                   ) -> tuple[float, float, float, float]:
    lon0, lat0 = grid.pixel_to_lonlat(x0, y0 + h)   # bottom-left
    lon1, lat1 = grid.pixel_to_lonlat(x0 + w, y0)   # top-right
    return (min(lon0, lon1), min(lat0, lat1), max(lon0, lon1), max(lat0, lat1))


def score_windows(mosaic: np.ndarray, probs: np.ndarray, grid,
                  aoi_name: str, window: int = WINDOW) -> list[Candidate]:
    h, w = mosaic.shape[:2]
    gray = mosaic.mean(axis=2)
    out: list[Candidate] = []
    idx = 0
    for y0 in range(0, h - window + 1, window):
        for x0 in range(0, w - window + 1, window):
            patch_gray = gray[y0:y0 + window, x0:x0 + window]
            patch_prob = probs[y0:y0 + window, x0:x0 + window]
            built_up = float(patch_gray.std())
            # Uncertainty: mean distance of each pixel's probability from a
            # confident 0 or 1. Maximal at p=0.5, zero at p=0 or p=1 — exactly
            # the windows where the model has not made up its mind, which is
            # where a label teaches it the most.
            uncertainty = float((0.5 - np.abs(patch_prob - 0.5)).mean())
            out.append(Candidate(aoi=aoi_name, idx=idx, x0=x0, y0=y0,
                                 built_up=built_up, uncertainty=uncertainty,
                                 mean_prob=float(patch_prob.mean()),
                                 bounds=_window_bounds(grid, x0, y0, window, window)))
            idx += 1
    return out


async def process_aoi(name: str, bounds: tuple[float, float, float, float],
                      bundle, zoom: int) -> tuple[np.ndarray, list[Candidate]]:
    west, south, east, north = bounds
    provider = tile_provider_config()
    grid = tile_grid_for_bounds(west, south, east, north, zoom)
    if grid.n_tiles < MIN_TILES_PER_AOI:
        raise ValueError(f"{name}: only {grid.n_tiles} tiles — bounds too small")
    if grid.n_tiles > MAX_TILES_PER_AOI:
        raise ValueError(f"{name}: {grid.n_tiles} tiles exceeds the "
                         f"{MAX_TILES_PER_AOI} cap — shrink the bounding box")

    mosaic, n_failed = await fetch_mosaic(grid, provider["url"])
    if n_failed:
        print(f"  [{name}] {n_failed}/{grid.n_tiles} tiles failed to load")

    _, probs = predict_mask(mosaic, bundle, threshold=0.5)
    candidates = score_windows(mosaic, probs, grid, name)
    return mosaic, candidates


def select_top(all_candidates: list[Candidate], n: int,
               min_built_up: float, uncertain_fraction: float
               ) -> list[Candidate]:
    """Keep the built-up windows; mostly the uncertain ones, some confident ones.

    A purely uncertainty-ranked set would only ever show the model its own
    blind spots and nothing it already handles, which risks a fine-tune that
    forgets what it knew. ``uncertain_fraction`` reserves the rest for the
    windows the model is *most* confident about, so the labelled set still
    spans the full range.
    """
    pool = [c for c in all_candidates if c.built_up >= min_built_up]
    if not pool:
        return []

    n_uncertain = int(round(n * uncertain_fraction))
    by_uncertainty = sorted(pool, key=lambda c: -c.uncertainty)
    uncertain_pick = by_uncertainty[:n_uncertain]

    remaining = [c for c in pool if c not in uncertain_pick]
    by_confidence = sorted(remaining, key=lambda c: c.uncertainty)
    confident_pick = by_confidence[:max(n - len(uncertain_pick), 0)]

    return uncertain_pick + confident_pick


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/finetune_candidates")
    ap.add_argument("--n", type=int, default=120,
                    help="candidate tiles to keep (label about 100 of these; "
                        "the rest cover ones that turn out empty or cloudy)")
    ap.add_argument("--min-built-up", type=float, default=8.0,
                    help="grayscale std floor — drops blank fields/water/canopy")
    ap.add_argument("--uncertain-fraction", type=float, default=0.75,
                    help="fraction of picks biased toward model uncertainty; "
                        "the rest sample the model's confident windows too")
    ap.add_argument("--zoom", type=int, default=19)
    args = ap.parse_args()

    out_dir = Path(args.out)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    print("loading the shipped model...")
    bundle = load_model()
    print(f"  {bundle.name} ({bundle.runtime})")

    all_candidates: list[Candidate] = []
    mosaics: dict[str, np.ndarray] = {}

    async def run_all():
        for name, bounds in DEFAULT_AOIS:
            print(f"fetching {name} {bounds} ...")
            mosaic, cands = await process_aoi(name, bounds, bundle, args.zoom)
            mosaics[name] = mosaic
            all_candidates.extend(cands)
            print(f"  {len(cands)} windows, "
                 f"{sum(c.built_up >= args.min_built_up for c in cands)} built-up")

    asyncio.run(run_all())

    chosen = select_top(all_candidates, args.n, args.min_built_up,
                        args.uncertain_fraction)
    if not chosen:
        raise SystemExit(
            "no windows passed the built-up filter — check the AOIs actually "
            "cover urban imagery, or lower --min-built-up")

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "aoi", "west", "south", "east", "north",
                        "built_up", "uncertainty", "mean_model_prob"])
        for c in sorted(chosen, key=lambda c: -c.uncertainty):
            stem = f"{c.aoi}_{c.idx:04d}"
            patch = mosaics[c.aoi][c.y0:c.y0 + WINDOW, c.x0:c.x0 + WINDOW]
            Image.fromarray(patch).save(img_dir / f"{stem}.png")
            writer.writerow([f"{stem}.png", c.aoi, *[round(b, 6) for b in c.bounds],
                            round(c.built_up, 1), round(c.uncertainty, 4),
                            round(c.mean_prob, 4)])

    print(f"\n{len(chosen)} candidate tiles written to {img_dir}")
    print(f"manifest: {manifest_path}")
    print(f"  mean uncertainty of the picked set: "
         f"{np.mean([c.uncertainty for c in chosen]):.4f} "
         f"(pool mean: {np.mean([c.uncertainty for c in all_candidates]):.4f})")
    print("\nNext: hand-label these — see finetune/README.md — then run "
         "scripts/finetune_indian.py.")


if __name__ == "__main__":
    main()
