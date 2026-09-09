r"""Fetch a ramp building-footprint dataset from source.coop.

Why this dataset is worth the trouble
-------------------------------------
Every label this project has trained or evaluated on so far is one of:
Inria (footprints, five Western cities), OpenStreetMap (footprints, incomplete
outside the West) or Google Open Buildings (footprints, machine-generated). ramp
is different on all three counts at once:

* **The labels are rooftops.** ramp's own Shanghai README says the SpaceNet
  labels were revised "to be consistent with the ramp datasets notion of
  rooftop as the building footprint". This project multiplies *roof* area into
  kWh, so that is the semantics it actually wants — closer than a ground
  footprint, which for a tall building is displaced from the visible roof.
* **It is human-reviewed.** "Tier 1" means thoroughly reviewed and improved.
  As of 2026-09-08 not one of the project's 25 training regions had any
  human-drawn ground truth, so every non-Inria number was OSM-anchored recall
  that could not yield a trustworthy IoU.
* **It is 30 cm.** Inria is 0.30 m/px and the app serves Esri z19 (~0.30 m/px),
  so no rescaling is needed. Most alternatives fail here — Open Cities is 2-20 cm
  drone imagery. (ramp's own Accra set *is* Open Cities, resampled to 30 cm.)

Licence: **CC-BY-NC-4.0**, i.e. non-commercial. That is more restrictive than
Inria, Open Buildings (CC-BY) or OSM (ODbL). Fine for research and for this
project; it would need a deliberate decision before any commercial use, which is
why the licence is written into every manifest entry this data touches.

Data layout at source.coop: paired ``source/<uuid>.tif`` (256x256 RGB GeoTIFF)
and ``labels/<uuid>.geojson`` (WGS84 polygons). No authentication.

    python scripts/download_ramp.py --dataset ramp_karnataka_india --limit 400
    python scripts/download_ramp.py --list
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import random
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BASE = "https://data.source.coop/ramp/ramp"
USER_AGENT = "rooftop-solar-potential-detection/1.0 (research)"
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def _get(url: str, timeout: float = 60.0, attempts: int = 4) -> bytes:
    """GET with backoff.

    source.coop returns a transient 503 under load, and without a retry that
    lands on the *listing* call and kills an entire dataset download before a
    single tile is fetched — which is exactly how a 400-tile Dhaka run failed on
    2026-09-09. Retrying here rather than in a shell loop means a hiccup costs
    seconds instead of restarting the whole listing.
    """
    last: Exception | None = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as exc:                      # 503, timeout, reset
            last = exc
            if i < attempts - 1:
                time.sleep(2 ** i)                    # 1s, 2s, 4s
    raise last if last else RuntimeError(f"failed: {url}")


def list_datasets() -> list[str]:
    """The 22 regional datasets in the bucket."""
    xml = _get(f"{BASE}/?list-type=2&delimiter=/").decode("utf-8")
    root = ET.fromstring(xml)
    out = []
    for cp in root.findall(f"{S3_NS}CommonPrefixes"):
        p = cp.findtext(f"{S3_NS}Prefix", "")
        name = p.rstrip("/").split("/")[-1]
        if name.startswith("ramp_"):
            out.append(name)
    return sorted(out)


def list_keys(dataset: str, sub: str) -> list[str]:
    """Every key under ``<dataset>/<sub>/``, following continuation tokens.

    The endpoint silently prepends ``ramp/`` to whatever prefix is asked for,
    so the prefix here is dataset-relative and the returned keys are not.
    """
    keys: list[str] = []
    token = None
    while True:
        url = f"{BASE}/?list-type=2&prefix={dataset}/{sub}/&max-keys=1000"
        if token:
            token = urllib.parse.quote(token, safe="")
            url += f"&continuation-token={token}"
        root = ET.fromstring(_get(url).decode("utf-8"))
        for c in root.findall(f"{S3_NS}Contents"):
            k = c.findtext(f"{S3_NS}Key", "")
            if k:
                keys.append(k)
        if root.findtext(f"{S3_NS}IsTruncated", "false") != "true":
            break
        token = root.findtext(f"{S3_NS}NextContinuationToken")
        if not token:
            break
    return keys


def readme(dataset: str) -> str:
    try:
        return _get(f"{BASE}/{dataset}/README.md").decode("utf-8")
    except Exception:
        return ""


def fetch_pair(uuid: str, dataset: str, out: Path) -> tuple[str, bool]:
    img = out / "source" / f"{uuid}.tif"
    lbl = out / "labels" / f"{uuid}.geojson"
    if img.exists() and lbl.exists():
        return uuid, True
    try:
        img.write_bytes(_get(f"{BASE}/{dataset}/source/{uuid}.tif"))
        lbl.write_bytes(_get(f"{BASE}/{dataset}/labels/{uuid}.geojson"))
        return uuid, True
    except Exception as exc:
        print(f"    {uuid}: {type(exc).__name__}: {exc}", flush=True)
        for p in (img, lbl):
            p.unlink(missing_ok=True)
        return uuid, False


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="ramp_karnataka_india")
    ap.add_argument("--out", default=None,
                    help="default: data/ramp/<dataset>")
    ap.add_argument("--limit", type=int, default=400,
                    help="tiles to fetch. A few hundred is already a bigger "
                        "reference than any OSM block this project has used")
    ap.add_argument("--seed", type=int, default=0,
                    help="which random subset — fixed so a benchmark is "
                        "reproducible rather than 'whatever downloaded'")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--list", action="store_true", help="list datasets and exit")
    args = ap.parse_args()

    if args.list:
        print("ramp datasets at source.coop (CC-BY-NC-4.0):\n")
        for d in list_datasets():
            r = readme(d)
            import re
            tiles = re.search(r"contains ([\d,]+) tiles", r)
            blds = re.search(r"([\d,]+) (?:individual )?buildings", r)
            res = re.search(r"resolution is (\d+) ?cm", r)
            tier = re.search(r"Tier (\d)", r)
            kw = re.search(r"keywords: ([^.\n]*)", r)
            print(f"  {d:32s} {tiles.group(1) if tiles else '?':>7s} tiles  "
                  f"{blds.group(1) if blds else '?':>8s} bldgs  "
                  f"{(res.group(1) + 'cm') if res else '?':>5s}  "
                  f"Tier {tier.group(1) if tier else '?'}  "
                  f"{(kw.group(1).strip()[:38]) if kw else ''}")
        return

    out = Path(args.out) if args.out else REPO / "data" / "ramp" / args.dataset
    (out / "source").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    info = readme(args.dataset)
    if info:
        (out / "README.md").write_text(info, encoding="utf-8")
        first = next((ln for ln in info.splitlines() if ln.startswith("This ")), "")
        print(first[:300] + "\n")

    print(f"listing {args.dataset} ...", flush=True)
    keys = list_keys(args.dataset, "labels")
    uuids = [k.rsplit("/", 1)[-1].removesuffix(".geojson") for k in keys]
    if not uuids:
        raise SystemExit(f"no labels found for {args.dataset} — check --list")
    print(f"  {len(uuids):,} tile/label pairs available")

    # A fixed random subset, not the first N. The keys are UUID-sorted, which is
    # not spatially meaningful, but taking a seeded sample makes the choice
    # explicit and repeatable rather than incidental.
    random.seed(args.seed)
    picked = uuids if args.limit <= 0 else random.sample(
        uuids, min(args.limit, len(uuids)))
    print(f"  fetching {len(picked):,} (seed {args.seed})", flush=True)

    ok = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(fetch_pair, u, args.dataset, out) for u in picked]
        for i, f in enumerate(cf.as_completed(futs), 1):
            _, good = f.result()
            ok += good
            if i % 50 == 0 or i == len(picked):
                print(f"    {i}/{len(picked)}  ({ok} ok)", flush=True)

    (out / "_subset.json").write_text(json.dumps(
        {"dataset": args.dataset, "seed": args.seed, "requested": len(picked),
         "downloaded": ok, "available": len(uuids),
         "licence": "CC-BY-NC-4.0 (non-commercial)",
         "source": f"{BASE}/{args.dataset}"}, indent=2), encoding="utf-8")
    print(f"\n{ok}/{len(picked)} pairs in {out}")
    print(f"\nScore a checkpoint on it:\n"
          f"  python scripts/eval_ramp.py --data {out.relative_to(REPO)} "
          f"--ckpt results/joint_v3_world25_20260908.pt")


if __name__ == "__main__":
    main()
