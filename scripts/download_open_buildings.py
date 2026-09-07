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
import collections
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

# Round 2, added 2026-09-07 after the ten-city model was promoted.
#
# Fact 36 is why this exists: going from two Indian cities to ten moved held-out
# recall in six of six new cities (Chennai 0.302 -> 0.872, Pune 0.202 -> 0.839)
# and cost 0.0022 Inria IoU. The trade everyone feared is barely measurable, so
# the thing to do is push breadth much further rather than tune the weighting.
#
# Two groups, for two different reasons:
#
# * **Indian tier-2 cities.** The eight metros above share a built form. The
#   weakest held-out scores were Ahmedabad (0.414) and Jaipur (0.353), which are
#   also the least metro-like of the set — a hint that the model has learned
#   "Indian metro" rather than "Indian". Smaller cities test that directly.
# * **Three new continents.** Open Buildings covers Africa, South and Southeast
#   Asia and Latin America, and the project's stated scope is a model that
#   detects a roof *anywhere*. None of this imagery has ever been in training.
#
# NOTE: unlike every AOI above, these include southern latitudes and western
# longitudes. Anything that assumed positive coordinates will surface here —
# which is how the --bounds argparse bug in scripts/eval_india_osm.py was found.
WORLD_AOIS: dict[str, tuple[float, float, float, float]] = {
    # --- India, tier-2: different density, roof material and plot size ---
    "lucknow_gomtinagar":   (80.995, 26.855, 81.005, 26.863),
    "nagpur_dharampeth":    (79.060, 21.135, 79.070, 21.143),
    "surat_adajan":         (72.780, 21.185, 72.790, 21.193),
    "indore_vijaynagar":    (75.890, 22.750, 75.900, 22.758),
    "patna_kankarbagh":     (85.150, 25.590, 85.160, 25.598),
    "coimbatore_rspuram":   (76.960, 11.010, 76.970, 11.018),
    "vizag_mvpcolony":      (83.320, 17.740, 83.330, 17.748),
    "bhubaneswar_saheed":   (85.820, 20.290, 85.830, 20.298),
    # --- Africa ---
    "lagos_surulere":       (3.350, 6.495, 3.360, 6.503),
    "nairobi_umoja":        (36.885, -1.283, 36.895, -1.275),
    "accra_osu":            (-0.190, 5.560, -0.180, 5.568),
    "addis_bole":           (38.780, 8.990, 38.790, 8.998),
    # --- Southeast and South Asia beyond India ---
    "dhaka_mirpur":         (90.365, 23.800, 90.375, 23.808),
    "jakarta_tebet":        (106.850, -6.235, 106.860, -6.227),
    "manila_sampaloc":      (120.995, 14.605, 121.005, 14.613),
    # --- Latin America ---
    "saopaulo_tatuape":     (-46.575, -23.545, -46.565, -23.537),
    "lima_sanjuan":         (-76.970, -12.160, -76.960, -12.152),
}

ALL_AOIS: dict[str, tuple[float, float, float, float]] = {
    **INDIA_AOIS, **WORLD_AOIS}


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

    aois = dict(ALL_AOIS)
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

    def write_aoi(name: str, items: list) -> None:
        thr = thresholds[name]
        (out / f"{name}.json").write_text(json.dumps({
            "aoi": name,
            "bounds": list(aois[name]),
            "source": "Google Open Buildings v3 (CC-BY-4.0 / ODbL)",
            "positive_threshold": thr,
            "positive_threshold_basis": (
                f"Google's {PRECISION_COLUMN} for the S2 level-4 cell covering "
                f"this AOI. Below it, treat as ignore rather than negative."),
            "buildings": items,
        }), encoding="utf-8")
        hi = sum(b["confidence"] >= thr for b in items)
        flag = "" if len(items) >= 200 else "   <-- suspiciously few, check the bbox"
        print(f"  wrote {name:22s} {len(items):6,} buildings  "
              f"({hi:,} positive at >={thr:.3f}, "
              f"{len(items) - hi:,} ignore){flag}", flush=True)

    # How many shards each AOI is still waiting on, so an AOI can be written the
    # moment its last shard lands rather than at the end of the whole run.
    # Worth the bookkeeping: these runs take hours, the shards are hundreds of
    # MB each, and a run that writes only at the end loses everything if the
    # process dies — which is exactly what happened on 2026-09-07, throwing away
    # 13 minutes of streaming across four continents.
    pending = collections.Counter()
    for tok, wanted in by_token.items():
        for name, _ in wanted:
            pending[name] += 1

    collected: dict[str, list] = {name: [] for name in aois}
    for tok, wanted in sorted(by_token.items()):
        print(f"shard {tok}  ({', '.join(n for n, _ in wanted)})", flush=True)
        try:
            for name, conf, ring in stream_shard(tok, wanted, args.min_confidence):
                collected[name].append({"confidence": round(conf, 4), "ring": ring})
        except Exception as exc:
            print(f"    FAILED: {type(exc).__name__}: {exc}", flush=True)
        for name, _ in wanted:
            pending[name] -= 1
            if pending[name] == 0:
                write_aoi(name, collected.pop(name))

    # Anything still held (an AOI whose every shard failed) still gets a file,
    # so the empty result is visible rather than silently absent.
    for name, items in sorted(collected.items()):
        write_aoi(name, items)

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
