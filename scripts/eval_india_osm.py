r"""Score checkpoints against OpenStreetMap footprints on a held-out Indian block.

This is the other half of the model decision. ``scripts/eval_inria.py`` says
whether a model still works on the five Western cities it was trained on; this
says whether it works where the project actually cares about deploying, and the
two must be read together — every fine-tune so far has bought one at the cost of
the other.

**What the numbers here mean, and do not mean.**

* **Recall is the trustworthy column.** OpenStreetMap is badly incomplete in
  India, so a building it *has* was drawn by a human and a miss is real.
* **Precision is a property of OpenStreetMap, not of the model.** Three models
  with recall 0.35 / 0.69 / 0.80 all scored precision ~0.49 on this block,
  because roughly half of what *any* model predicts is simply not mapped. Read
  it as a constant; never quote it as a model quality figure.
* **The IoU here is not comparable with an Inria IoU** — different labels,
  different country, and an incomplete reference. Never put 0.7712 and a number
  from this script in the same table without saying so.

Every model is scored against **one** mosaic fetched once, so a difference
between two rows is the model and not the imagery or a re-fetch.

    python scripts/eval_india_osm.py --ckpt results/unetpp_effb0_inria_20260903.pt
    python scripts/eval_india_osm.py --ckpt a.pt --ckpt b.pt --out results/cmp.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import cv2
import numpy as np

from infer import load_manifest
from webapp import tiles as tiles_mod
from webapp.calibration import _recall_by_size, _ring_scores, fetch_reference_buildings
from webapp.config import TILE_PROVIDERS
from webapp.inference import ModelBundle, predict_mask

# The CV Raman Nagar benchmark block in Bangalore. Dense low-rise residential —
# the exact built form the Inria-only model is silent on — and held out of the
# fine-tuning set in full, so it is a fair test for every model here.
DEFAULT_BOUNDS = (77.654, 12.984, 77.658, 12.987)

# A footprint whose median probability is below this is one the model is
# *silent* on rather than merely unsure about. Tracking it separately matters
# because silence cannot be fixed by moving the threshold, and under-confidence
# can — that distinction is what redirected this project from threshold tuning
# to fine-tuning.
SILENT_PROB = 0.10


def build_torch_bundle(ckpt: Path, arch: str, encoder: str,
                       stats_key: str, window: int, threshold: float) -> ModelBundle:
    """A ModelBundle wrapping a raw .pt, so predict_mask can be reused verbatim.

    Deliberately does NOT go through webapp.inference.load_model: that reads
    whatever is in webapp/models/, and exporting each candidate there to compare
    them is exactly the "newest .onnx silently becomes everyone's model" trap.
    Scoring must never mutate what the app serves.
    """
    import threading

    import torch

    from config import norm_stats
    from model.registry import build_model

    mean, std = norm_stats(stats_key)
    model = build_model(arch, encoder, encoder_weights=None)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()
    return ModelBundle(
        name=ckpt.stem,
        manifest={"mean": list(mean), "std": list(std), "window": window,
                  "stride": window // 2, "threshold": threshold,
                  "arch": arch, "encoder": encoder},
        runtime="torch", _torch_model=model, _lock=threading.Lock(),
    )


def rasterise_rings(rings, grid, shape) -> np.ndarray:
    """OSM lon/lat rings -> a boolean mask on the same pixel grid the model ran on.

    Rasterising the reference (rather than vectorising the prediction) keeps the
    comparison in pixel space, where a pooled IoU is well defined and no polygon
    simplification tolerance can quietly move the answer.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    h, w = shape
    for ring in rings:
        pts = np.array([[int(round(px)), int(round(py))]
                        for px, py in (grid.lonlat_to_pixel(lon, lat)
                                       for lon, lat in ring)], dtype=np.int32)
        # Keep buildings that are only partly inside; clipping is what cv2 does
        # anyway, and dropping them would bias the reference toward block
        # interiors.
        if pts.size and pts[:, 0].max() >= 0 and pts[:, 1].max() >= 0 \
                and pts[:, 0].min() < w and pts[:, 1].min() < h:
            cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def pooled_scores(pred: np.ndarray, ref: np.ndarray) -> dict:
    """Global intersection/union over the whole mosaic — not a per-window mean."""
    inter = float(np.logical_and(pred, ref).sum())
    union = float(np.logical_or(pred, ref).sum())
    p_sum, r_sum = float(pred.sum()), float(ref.sum())
    return {
        "iou": round(inter / union, 4) if union else None,
        "precision": round(inter / p_sum, 4) if p_sum else None,
        "recall": round(inter / r_sum, 4) if r_sum else None,
        "pred_px": int(p_sum),
        "ref_px": int(r_sum),
    }


async def fetch_inputs(bounds, zoom, provider_key):
    west, south, east, north = bounds
    provider = TILE_PROVIDERS[provider_key]
    zoom = min(zoom, provider.get("max_zoom", zoom))
    grid = tiles_mod.tile_grid_for_bounds(west, south, east, north, zoom)
    print(f"[tiles]  {grid.n_tiles} tiles at z{zoom} "
          f"({grid.width_px}x{grid.height_px} px)")
    mosaic, n_failed = await tiles_mod.fetch_mosaic(grid, provider["url"])
    if n_failed:
        print(f"[tiles]  WARNING: {n_failed} tile(s) failed to fetch")
    rings, source = await fetch_reference_buildings(west, south, east, north)
    print(f"[osm]    {len(rings)} building ways ({source})")
    return grid, mosaic, rings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True,
                    help="checkpoint to score; repeat to compare several")
    ap.add_argument("--arch", default=None, help="override the manifest arch")
    ap.add_argument("--encoder", default=None)
    ap.add_argument("--stats-key", default=None)
    # NOTE: a western-hemisphere box starts with a minus sign, and argparse
    # reads "-97.745,..." as a flag rather than a value. Passing it as
    # --bounds=-97.745,... works, but so does accepting a leading space, and
    # only one of those is discoverable from --help.
    ap.add_argument("--bounds", default=None,
                    help="west,south,east,north (default: CV Raman Nagar "
                         "block). Negative longitudes are fine: either use "
                         "--bounds=-97.7,30.2,-97.6,30.3 or quote it with a "
                         "leading space, ' -97.7,30.2,...'")
    ap.add_argument("--zoom", type=int, default=19)
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--provider", default="esri")
    ap.add_argument("--out", default=None, help="write results as JSON here")
    args = ap.parse_args()

    # .strip() so a leading space — the usual way to smuggle a negative
    # longitude past argparse — does not become a float() error.
    bounds = (tuple(float(v) for v in args.bounds.strip().split(","))
              if args.bounds else DEFAULT_BOUNDS)
    if len(bounds) != 4:
        raise SystemExit("--bounds needs west,south,east,north")
    if not (-180 <= bounds[0] < bounds[2] <= 180
            and -90 <= bounds[1] < bounds[3] <= 90):
        raise SystemExit(
            f"--bounds {bounds} is not west,south,east,north with west<east "
            f"and south<north. A western-hemisphere box needs "
            f"--bounds=-97.7,30.2,-97.6,30.3 (the '=' keeps argparse from "
            f"reading the minus sign as a flag).")

    grid, mosaic, rings = asyncio.run(
        fetch_inputs(bounds, args.zoom, args.provider))

    mpp = tiles_mod.metres_per_pixel((bounds[1] + bounds[3]) / 2, grid.zoom)
    px_m2 = mpp * mpp
    ref = rasterise_rings(rings, grid, mosaic.shape[:2])
    ref_area = float(ref.sum()) * px_m2
    print(f"[osm]    reference area {ref_area:,.0f} m²  "
          f"({mpp:.3f} m/px)\n")

    manifest = load_manifest().get("models", {})
    results = []
    for raw in args.ckpt:
        ckpt = Path(raw)
        if not ckpt.is_absolute():
            ckpt = REPO / ckpt
        if not ckpt.exists():
            raise SystemExit(f"checkpoint not found: {ckpt}")
        entry = manifest.get(ckpt.name, {})
        arch = args.arch or entry.get("arch", "unet++")
        encoder = args.encoder or entry.get("encoder", "efficientnet-b0")
        stats_key = args.stats_key or entry.get("stats_key", "imagenet")

        print(f"[model]  {ckpt.name}  ({arch} / {encoder}, norm={stats_key})",
              flush=True)
        bundle = build_torch_bundle(ckpt, arch, encoder, stats_key,
                                    args.window, args.threshold)
        pred, probs = predict_mask(mosaic, bundle, threshold=args.threshold)

        scores = pooled_scores(pred, ref)
        ring_scores = _ring_scores(rings, probs, grid)
        meds = np.array([s["median"] for s in ring_scores], dtype=np.float64)
        pred_area = scores["pred_px"] * px_m2
        row = {
            "checkpoint": ckpt.name,
            "arch": arch, "encoder": encoder,
            "threshold": args.threshold,
            **{k: v for k, v in scores.items() if k not in ("pred_px", "ref_px")},
            "predicted_area_m2": round(pred_area),
            "area_ratio_vs_osm": round(pred_area / ref_area, 3) if ref_area else None,
            "footprints_scored": len(ring_scores),
            # Recall over whole footprints, which is what a user perceives —
            # distinct from the pixel recall above.
            "footprint_recall": round(float((meds > args.threshold).mean()), 3)
            if len(meds) else None,
            "silent_fraction": round(float((meds < SILENT_PROB).mean()), 3)
            if len(meds) else None,
            "by_size": _recall_by_size(ring_scores, args.threshold, mpp),
        }
        results.append(row)
        print(f"         IoU {row['iou']}  precision {row['precision']}  "
              f"pixel-recall {row['recall']}")
        print(f"         footprint recall {row['footprint_recall']}  "
              f"silent {row['silent_fraction']}  "
              f"area {row['predicted_area_m2']:,} m² "
              f"({row['area_ratio_vs_osm']}x OSM)\n", flush=True)

    print("=" * 78)
    print("model                              IoU   prec  pxRec  fpRec  silent  areaX")
    print("-" * 78)
    for r in results:
        print(f"{r['checkpoint'][:34].ljust(34)}"
              f"{r['iou']:>6.3f}{r['precision']:>7.3f}{r['recall']:>7.3f}"
              f"{r['footprint_recall']:>7.3f}{r['silent_fraction']:>8.3f}"
              f"{r['area_ratio_vs_osm']:>7.2f}")
    print("")
    print("precision ~0.49 for every model measures OpenStreetMap's "
          "incompleteness, not the model.")

    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = REPO / out
        out.write_text(json.dumps({
            "bounds": list(bounds), "zoom": grid.zoom,
            "provider": args.provider,
            "metres_per_px": round(mpp, 4),
            "reference": "OpenStreetMap building ways (incomplete — recall only)",
            "reference_buildings": len(rings),
            "reference_area_m2": round(ref_area),
            "threshold": args.threshold,
            "models": results,
        }, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
