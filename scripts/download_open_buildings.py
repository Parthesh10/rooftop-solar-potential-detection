r"""Pull Google Open Buildings v3 footprints for a set of Indian AOIs.

Why this dataset
----------------
Measured 2026-09-06, the shipped model recovers 24% of mapped buildings on a
Bangalore block and goes silent on 56% of them, and **more Inria data makes that
worse, not better** (fact 30: doubling the windows per tile bought +0.013 Inria
IoU and cost a fifth of the Indian recall). The fix has to be new geography, and
hand-labelling it does not scale past the ~100 tiles already done.

Open Buildings is ML-derived from satellite imagery, covers all of India, and is
CC-BY-4.0 / ODbL. It carries a per-building ``confidence`` in [0.65, 1.0], which
is what makes it usable as a *training* label despite being machine-generated:
the uncertain band can be routed to an ignore mask instead of being guessed at,
exactly as OpenStreetMap's incompleteness was handled in
``scripts/build_osm_labels.py``.

How the download works
----------------------
Shards live in a public GCS bucket with no auth. They are partitioned by S2
cell. Level-4 shards are country-sized — covering the cities below would mean
**45 GB** — so this uses level-6 (~100-500 MB each) and computes the covering
tokens with ``s2sphere``, which is pure Python and installs on Windows, unlike
the ``s2geometry`` bindings Google's own notebook uses.

Nothing is written to disk except the extracted buildings. The CSV columns are

    latitude, longitude, area_in_meters, confidence, geometry(WKT), plus_code

and the two centroid columns come *before* the geometry, so each row can be
bbox-tested on a cheap ``split(",", 4)`` and only the survivors pay for WKT
parsing. A 486 MB shard streams through in one pass and yields a few thousand
buildings.

    python scripts/download_open_buildings.py
    python scripts/download_open_buildings.py --aoi delhi_lajpat --min-confidence 0.7
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BUCKET = ("https://storage.googleapis.com/open-buildings-data/v3/"
          "polygons_s2_level_6_gzip_no_header/{token}_buildings.csv.gz")
THRESHOLDS_URL = ("https://storage.googleapis.com/open-buildings-data/v3/"
                  "score_thresholds_s2_level_4.csv")
USER_AGENT = "rooftop-solar-potential-detection/1.0 (research)"

# Google publishes, per S2 level-4 cell, the confidence needed to reach a given
# precision. Using it beats picking one global number: measured 2026-09-06 the
# 80%-precision threshold across the Indian cells here runs from 0.707 (Chennai)
# to 0.833 (Mumbai), so any single cutoff is simultaneously too strict in one
# city and too loose in another.
PRECISION_COLUMN = "confidence_threshold_80%_precision"

# Below the cell's threshold a detection is not trustworthy enough to train on
# as a positive — but it is far too likely to be a real building to teach as
# background. Confidence correlates strongly with size here (measured in Chennai:
# median 53 m2 in the 0.65-0.70 band against 228 m2 above 0.85), and small roofs
# are exactly what the model is already silent on. Calling them negative would
# reinforce the failure being fixed, so they become the ignore band instead.
ABSOLUTE_FLOOR = 0.65          # the dataset's own minimum confidence

# Dense residential AOIs across India, each the same size as the existing
# Bangalore/Bhopal boxes in scripts/select_finetune_tiles.py (~1.0 x 0.9 km,
# 168 tiles at z19 -> 42 non-overlapping 512px windows).
#
# Deliberately NOT Bangalore or Bhopal: those two cities already supply the 103
# OSM-relabelled tiles, and the CV Raman Nagar block is the held-out benchmark.
# Reusing either would make the eval numbers meaningless.
#
# A box that turns out to sit on water, forest or an airport shows up as a very
# low building count in this script's own summary — that is the intended check,
# rather than trusting coordinates typed from memory.
INDIA_AOIS: dict[str, tuple[float, float, float, float]] = {
    "delhi_lajpat":      (77.238, 28.564, 77.248, 28.572),
    "mumbai_andheri":    (72.840, 19.061, 72.850, 19.069),
    "chennai_adyar":     (80.250, 13.031, 80.260, 13.039),
    "kolkata_saltlake":  (88.401, 22.575, 88.411, 22.583),
    "hyderabad_kphb":    (78.440, 17.436, 78.450, 17.444),
    "jaipur_malviya":    (75.800, 26.851, 75.810, 26.859),
    "ahmedabad_maninagar": (72.600, 22.996, 72.610, 23.004),
    "pune_kothrud":      (73.810, 18.501, 73.820, 18.509),
}


def precision_thresholds(cache: Path | None = None) -> list[dict]:
    """Google's per-cell confidence thresholds, with the cell's lon/lat bbox."""
    if cache and cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    import csv
    import io

    req = urllib.request.Request(THRESHOLDS_URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as r:
        text = r.read().decode("utf-8")

    out = []
    for row in csv.DictReader(io.StringIO(text)):
        wkt = row.get("geometry", "")
        rings = parse_wkt_rings(wkt)
        if not rings:
            continue
        lons = [p[0] for p in rings[0]]
        lats = [p[1] for p in rings[0]]
        try:
            thr = float(row[PRECISION_COLUMN])
        except (KeyError, ValueError):
            continue
        out.append({"token": row.get("s2_token"), "threshold": thr,
                    "bbox": [min(lons), min(lats), max(lons), max(lats)]})
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(out), encoding="utf-8")
    return out


def threshold_for(lon: float, lat: float, table: list[dict],
                  fallback: float = 0.75) -> float:
    """The 80%-precision confidence threshold covering this point."""
    for cell in table:
        w, s, e, n = cell["bbox"]
        if w <= lon <= e and s <= lat <= n:
            return cell["threshold"]
    return fallback


def tokens_for_bbox(west: float, south: float, east: float, north: float,
                    level: int = 6) -> set[str]:
    """S2 cell tokens at ``level`` covering the bbox."""
    import s2sphere as s2

    coverer = s2.RegionCoverer()
    coverer.min_level = level
    coverer.max_level = level
    coverer.max_cells = 128
    rect = s2.LatLngRect(s2.LatLng.from_degrees(south, west),
                         s2.LatLng.from_degrees(north, east))
    return {c.to_token() for c in coverer.get_covering(rect)}


def parse_wkt_rings(wkt: str) -> list[list[tuple[float, float]]]:
    """WKT POLYGON / MULTIPOLYGON -> exterior rings as [(lon, lat), ...].

    Interior rings are dropped. A building footprint with a hole in it is not a
    distinction this model can represent — the mask is binary — and keeping them
    would punch holes in the positives.
    """
    body = wkt.strip()
    if body.upper().startswith("MULTIPOLYGON"):
        body = body[body.index("(") + 1:]
        chunks = [c for c in body.split("))") if "(" in c]
    elif body.upper().startswith("POLYGON"):
        chunks = [body[body.index("(") + 1:]]
    else:
        return []

    rings = []
    for chunk in chunks:
        # The exterior ring is whatever sits inside the first "((" or "(".
        start = chunk.rfind("((")
        piece = chunk[start + 2:] if start >= 0 else chunk.lstrip("(")
        piece = piece.split(")")[0]
        pts = []
        for pair in piece.split(","):
            bits = pair.split()
            if len(bits) >= 2:
                try:
                    pts.append((float(bits[0]), float(bits[1])))
                except ValueError:
                    pass
        if len(pts) >= 4:
            rings.append(pts)
    return rings


def stream_shard(token: str, wanted: list[tuple[str, tuple]],
                 min_confidence: float, progress_every: int = 2_000_000):
    """Stream one shard, yielding (aoi_name, confidence, ring) for hits.

    The shard is never stored. Rows are rejected on the centroid, which is the
    first two columns, so the expensive WKT parse only runs for buildings that
    are actually inside one of the AOIs.
    """
    url = BUCKET.format(token=token)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    n_rows = n_hit = 0
    with urllib.request.urlopen(req, timeout=180) as resp:
        with gzip.GzipFile(fileobj=resp) as gz:
            for raw in gz:
                n_rows += 1
                if n_rows % progress_every == 0:
                    print(f"    ...{n_rows:,} rows, {n_hit:,} kept", flush=True)
                try:
                    line = raw.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                parts = line.split(",", 4)
                if len(parts) < 5:
                    continue
                try:
                    lat = float(parts[0])
                    lon = float(parts[1])
                    conf = float(parts[3])
                except ValueError:
                    continue
                if conf < min_confidence:
                    continue
                for name, (w, s, e, n) in wanted:
                    if w <= lon <= e and s <= lat <= n:
                        wkt = parts[4]
                        q0 = wkt.find('"')
                        q1 = wkt.rfind('"')
                        if q0 >= 0 and q1 > q0:
                            wkt = wkt[q0 + 1:q1]
                        for ring in parse_wkt_rings(wkt):
                            yield name, conf, ring
                            n_hit += 1
                        break
    print(f"    {n_rows:,} rows scanned, {n_hit:,} buildings kept", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/open_buildings")
    ap.add_argument("--aoi", action="append",
                    help="restrict to these AOI names (repeatable)")
    ap.add_argument("--min-confidence", type=float, default=0.65,
                    help="drop rows below this before storing. The dataset's own "
                        "floor is 0.65; the positive/ignore split happens later "
                        "in build_india_labels.py, so keep everything here")
    args = ap.parse_args()

    aois = dict(INDIA_AOIS)
    if args.aoi:
        missing = [a for a in args.aoi if a not in aois]
        if missing:
            raise SystemExit(f"unknown AOI(s): {missing}. "
                             f"Known: {sorted(aois)}")
        aois = {k: v for k, v in aois.items() if k in args.aoi}

    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out
    out.mkdir(parents=True, exist_ok=True)

    # Group AOIs by shard so each shard is streamed exactly once even when
    # several cities share one S2 cell.
    by_token: dict[str, list[tuple[str, tuple]]] = {}
    for name, bbox in aois.items():
        for tok in tokens_for_bbox(*bbox):
            by_token.setdefault(tok, []).append((name, bbox))

    print(f"{len(aois)} AOI(s) -> {len(by_token)} shard(s): {sorted(by_token)}\n")

    print("resolving Google's per-region precision thresholds...")
    table = precision_thresholds(out / "_score_thresholds.json")
    thresholds = {name: threshold_for((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2, table)
                  for name, bb in aois.items()}
    for name, thr in sorted(thresholds.items()):
        print(f"  {name:22s} positive at confidence >= {thr:.3f}")
    print()

    collected: dict[str, list] = {name: [] for name in aois}
    for tok, wanted in sorted(by_token.items()):
        print(f"shard {tok}  ({', '.join(n for n, _ in wanted)})", flush=True)
        try:
            for name, conf, ring in stream_shard(tok, wanted, args.min_confidence):
                collected[name].append({"confidence": round(conf, 4), "ring": ring})
        except Exception as exc:
            print(f"    FAILED: {type(exc).__name__}: {exc}", flush=True)

    print("\nresults:")
    for name, items in sorted(collected.items()):
        path = out / f"{name}.json"
        thr = thresholds[name]
        path.write_text(json.dumps({
            "aoi": name,
            "bounds": list(aois[name]),
            "source": "Google Open Buildings v3 (CC-BY-4.0 / ODbL)",
            "positive_threshold": thr,
            "positive_threshold_basis": (
                f"Google's {PRECISION_COLUMN} for the S2 level-4 cell covering "
                f"this AOI. Below it, treat as ignore rather than negative."),
            "buildings": items,
        }), encoding="utf-8")
        confs = [b["confidence"] for b in items]
        hi = sum(c >= thr for c in confs)
        flag = "" if len(items) >= 200 else "   <-- suspiciously few, check the bbox"
        print(f"  {name:22s} {len(items):6,} buildings  "
              f"({hi:,} positive at >={thr:.3f}, "
              f"{len(items) - hi:,} ignore){flag}")

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
