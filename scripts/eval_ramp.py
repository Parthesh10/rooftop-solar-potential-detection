r"""Score checkpoints against ramp's human-drawn rooftop labels.

**This is the project's first evaluation outside Inria with complete, human-drawn
ground truth.** Everything else non-Western has been OSM-anchored *recall*,
because OpenStreetMap is incomplete wherever the model is interesting — and an
incomplete reference makes precision and IoU meaningless (fact 28). ramp's tiles
are exhaustively labelled and human-reviewed, so here **precision and IoU mean
what they say**.

Three things make the comparison fair without any rescaling:

* ramp Karnataka is **30 cm/px**; Inria is 0.30 and the app serves Esri z19 at
  ~0.30. No resampling, so no scale shift of the kind that silently changes what
  the network sees.
* Labels are **rooftops**, which is what this project actually needs — it turns
  roof area into kWh.
* Tiles are exhaustively labelled, so a false positive is genuinely a false
  positive rather than an unmapped building.

Two honest caveats, both reported alongside the numbers:

* **Imagery is Maxar, not Esri.** Same resolution, different sensor and
  processing. A gap here is domain shift between providers, not necessarily
  model quality.
* **Tiles are 256x256 and the model's window is 512.** ``predict_mask`` reflect-
  pads up to one window, so the model sees a mirrored border it would never see
  when serving. ``--erode-border`` drops a margin from scoring to check whether
  that matters; it is reported both ways.

    python scripts/eval_ramp.py --ckpt results/joint_v3_world25_20260908.pt
    python scripts/eval_ramp.py --ckpt a.pt --ckpt b.pt --out results/ramp.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import cv2
import numpy as np
import tifffile

from infer import load_manifest
from scripts.eval_india_osm import build_torch_bundle, pooled_scores
from webapp.inference import predict_mask


def epsg_of(page) -> int:
    """The tile's CRS, from the GeoTIFF GeoKeyDirectory.

    **This is not a formality — ramp mixes CRSs across its datasets.** Karnataka
    is EPSG:4326 (degrees) while Dhaka, Accra and Nairobi are UTM zones 46N,
    30N and 37S (metres). The labels are GeoJSON, which is *always* WGS84
    lon/lat, so a harness that assumes the raster is also lon/lat produces a
    silently empty mask on three datasets out of four — roof fraction 0.000
    rather than an exception. Read the CRS; never infer it.
    """
    gk = {t.name: t.value for t in page.tags.values()}.get("GeoKeyDirectoryTag")
    if not gk:
        return 4326
    # Header is 4 shorts, then 4-short entries: (key, location, count, value).
    for i in range(4, len(gk) - 3, 4):
        key, _loc, _cnt, val = gk[i:i + 4]
        if key == 3072:            # ProjectedCSTypeGeoKey
            return int(val)
        if key == 2048:            # GeographicTypeGeoKey
            return int(val)
    return 4326


def geotransform(page) -> tuple[float, float, float, float]:
    """(x0, y0, dx, dy) for a north-up GeoTIFF tile, in the tile's own CRS.

    ramp tiles carry ModelTiepointTag + ModelPixelScaleTag rather than a full
    transform, which is the simple north-up case: pixel (0,0) sits at the tie
    point and each pixel steps by the scale. Read straight from the tags so the
    project keeps its no-GDAL, no-rasterio dependency footing.

    Units follow the CRS — degrees for EPSG:4326, metres for a UTM zone — so
    pair this with :func:`epsg_of` rather than assuming.
    """
    tags = {t.name: t.value for t in page.tags.values()}
    tie = tags.get("ModelTiepointTag")
    scale = tags.get("ModelPixelScaleTag")
    if not tie or not scale:
        raise ValueError("tile has no ModelTiepointTag/ModelPixelScaleTag")
    return float(tie[3]), float(tie[4]), float(scale[0]), float(scale[1])


_TRANSFORMERS: dict[int, object] = {}


def _to_raster_crs(epsg: int):
    """WGS84 lon/lat -> the raster's CRS. Identity for 4326. Cached, because
    building a pyproj Transformer per polygon is far slower than the rasterising."""
    if epsg == 4326:
        return None
    if epsg not in _TRANSFORMERS:
        from pyproj import Transformer
        _TRANSFORMERS[epsg] = Transformer.from_crs(4326, epsg, always_xy=True)
    return _TRANSFORMERS[epsg]


def rasterise(features, gt, shape, epsg: int = 4326) -> np.ndarray:
    """ramp polygons -> boolean mask on the tile's own pixel grid.

    GeoJSON coordinates are lon/lat by specification; the raster may not be
    (see :func:`epsg_of`), so project the ring into the raster's CRS before
    touching the geotransform.
    """
    lon0, lat0, dlon, dlat = gt
    tf = _to_raster_crs(epsg)
    mask = np.zeros(shape, np.uint8)
    for feat in features:
        geom = feat.get("geometry") or {}
        kind = geom.get("type")
        if kind == "Polygon":
            rings = [geom.get("coordinates", [None])[0]]
        elif kind == "MultiPolygon":
            rings = [poly[0] for poly in geom.get("coordinates", []) if poly]
        else:
            continue
        for ring in rings:
            if not ring or len(ring) < 3:
                continue
            xy = ring if tf is None else [tf.transform(lon, lat)
                                          for lon, lat in ring]
            pts = np.array([[int(round((x - lon0) / dlon)),
                             int(round((lat0 - y) / dlat))]
                            for x, y in xy], np.int32)
            cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def load_tile(stem: str, data: Path):
    with tifffile.TiffFile(data / "source" / f"{stem}.tif") as tfl:
        page = tfl.pages[0]
        img = page.asarray()
        gt = geotransform(page)
        epsg = epsg_of(page)
    if img.ndim == 2:
        img = np.stack([img] * 3, -1)
    img = img[..., :3]
    feats = json.loads((data / "labels" / f"{stem}.geojson").read_text(
        encoding="utf-8")).get("features", [])
    return img, rasterise(feats, gt, img.shape[:2], epsg), len(feats)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/ramp/ramp_karnataka_india")
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--arch", default=None)
    ap.add_argument("--encoder", default=None)
    ap.add_argument("--stats-key", default=None)
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="0 = every tile")
    ap.add_argument("--erode-border", type=int, default=32,
                    help="pixels dropped from each edge in the second score, to "
                        "test how much the reflect-padded border distorts things")
    ap.add_argument("--sweep", action="store_true",
                    help="score every threshold from 0.30 to 0.65. Probabilities "
                        "are computed ONCE per tile and every cut point scored "
                        "against the same cached array, so the sweep costs one "
                        "forward pass rather than eight")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = Path(args.data)
    if not data.is_absolute():
        data = REPO / data
    # Require BOTH halves. The downloader writes the .tif before the .geojson,
    # so a run interrupted mid-pair leaves an image with no label — scoring
    # against a missing label would either crash or, worse, silently count a
    # fully-labelled tile as empty background.
    all_tifs = sorted(p.stem for p in (data / "source").glob("*.tif"))
    stems = [s for s in all_tifs if (data / "labels" / f"{s}.geojson").exists()]
    orphans = len(all_tifs) - len(stems)
    if args.limit:
        stems = stems[:args.limit]
    if not stems:
        raise SystemExit(f"no complete tile/label pairs in {data} — run "
                         f"scripts/download_ramp.py first")
    if orphans:
        print(f"[ramp] skipping {orphans} image(s) with no label "
              f"(download still running?)")

    print(f"[ramp] {data.name}: {len(stems)} tiles", flush=True)
    tiles = []
    n_feat = 0
    for s in stems:
        try:
            img, ref, nf = load_tile(s, data)
        except Exception as exc:
            print(f"  skip {s}: {type(exc).__name__}: {exc}")
            continue
        tiles.append((img, ref))
        n_feat += nf
    ref_frac = float(np.mean([r.mean() for _, r in tiles]))
    print(f"[ramp] {n_feat:,} labelled buildings, "
          f"mean {ref_frac:.1%} of each tile is roof\n", flush=True)

    manifest = load_manifest().get("models", {})
    results = []
    for raw in args.ckpt:
        ckpt = Path(raw)
        if not ckpt.is_absolute():
            ckpt = REPO / ckpt
        if not ckpt.exists():
            raise SystemExit(f"checkpoint not found: {ckpt}")
        e = manifest.get(ckpt.name, {})
        arch = args.arch or e.get("arch", "unet++")
        enc = args.encoder or e.get("encoder", "efficientnet-b0")
        sk = args.stats_key or e.get("stats_key", "imagenet")
        print(f"[model] {ckpt.name}", flush=True)
        bundle = build_torch_bundle(ckpt, arch, enc, sk, args.window,
                                    args.threshold)

        # Pooled over the whole set, not a mean of per-tile IoUs: a tile with
        # three roofs must not weigh the same as a tile with three hundred
        # (fact 14).
        inter = union = p_sum = r_sum = 0.0
        b = args.erode_border
        i2 = u2 = 0.0
        thresholds = (np.round(np.arange(0.30, 0.66, 0.05), 2).tolist()
                      if args.sweep else [])
        sweep = {t: [0.0, 0.0, 0.0] for t in thresholds}   # inter, union, pred
        for img, ref in tiles:
            pred, probs = predict_mask(img, bundle, threshold=args.threshold)
            pred = pred[:ref.shape[0], :ref.shape[1]]
            inter += np.logical_and(pred, ref).sum()
            union += np.logical_or(pred, ref).sum()
            p_sum += pred.sum()
            r_sum += ref.sum()
            if b and ref.shape[0] > 2 * b and ref.shape[1] > 2 * b:
                pc, rc = pred[b:-b, b:-b], ref[b:-b, b:-b]
                i2 += np.logical_and(pc, rc).sum()
                u2 += np.logical_or(pc, rc).sum()
            # One forward pass already happened; every extra threshold is just a
            # comparison against the cached probability array.
            if thresholds:
                pr = probs[:ref.shape[0], :ref.shape[1]]
                for t in thresholds:
                    m = pr > t
                    acc = sweep[t]
                    acc[0] += np.logical_and(m, ref).sum()
                    acc[1] += np.logical_or(m, ref).sum()
                    acc[2] += m.sum()

        row = {
            "checkpoint": ckpt.name, "arch": arch, "encoder": enc,
            "threshold": args.threshold,
            "iou": round(inter / union, 4) if union else None,
            "precision": round(inter / p_sum, 4) if p_sum else None,
            "recall": round(inter / r_sum, 4) if r_sum else None,
            "predicted_over_reference": round(p_sum / r_sum, 3) if r_sum else None,
            "iou_border_eroded": round(i2 / u2, 4) if u2 else None,
        }
        if thresholds:
            rows = []
            for t in thresholds:
                i, u, p = sweep[t]
                rows.append({"threshold": t,
                             "iou": round(i / u, 4) if u else None,
                             "recall": round(i / r_sum, 4) if r_sum else None,
                             "area_ratio": round(p / r_sum, 3) if r_sum else None})
            best = max(rows, key=lambda r: r["iou"] or 0)
            row["sweep"] = rows
            row["best_threshold"] = best["threshold"]
            row["best_iou"] = best["iou"]

        results.append(row)
        print(f"        IoU {row['iou']}  precision {row['precision']}  "
              f"recall {row['recall']}  area {row['predicted_over_reference']}x"
              f"  (IoU excl. {b}px border {row['iou_border_eroded']})",
              flush=True)
        if thresholds:
            print("        thr    IoU   recall  area")
            for r in row["sweep"]:
                mark = "  <- best" if r["threshold"] == row["best_threshold"] else ""
                print(f"        {r['threshold']:.2f} {r['iou']:>7.4f} "
                      f"{r['recall']:>7.3f} {r['area_ratio']:>6.2f}{mark}")
        print("", flush=True)

    print("=" * 74)
    print("model                                IoU   prec  recall  areaX  IoU-noedge")
    print("-" * 74)
    for r in results:
        print(f"{r['checkpoint'][:34].ljust(35)}"
              f"{r['iou']:>6.3f}{r['precision']:>7.3f}{r['recall']:>8.3f}"
              f"{r['predicted_over_reference']:>7.2f}{r['iou_border_eroded']:>11.3f}")
    print("")
    print("Human-drawn, exhaustively labelled ROOFTOPS at 30 cm. Unlike every")
    print("OSM-anchored number in this project, precision and IoU are real here.")

    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = REPO / out
        out.write_text(json.dumps({
            "dataset": data.name,
            "source": "ramp (source.coop), CC-BY-NC-4.0",
            "tiles": len(tiles), "labelled_buildings": n_feat,
            "reference_roof_fraction": round(ref_frac, 4),
            "label_semantics": "rooftop (not ground footprint)",
            "imagery": "Maxar ODP at 30 cm — NOT the Esri imagery the app serves",
            "threshold": args.threshold,
            "models": results,
        }, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
