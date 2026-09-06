r"""Merge every Indian tile set into one directory and stage it for Kaggle.

``scripts/train_inria.py`` takes a single ``--extra-data-dir``, so the Bangalore/
Bhopal tiles from ``build_osm_labels.py`` and the eight-city set from
``build_india_labels.py`` have to arrive as one directory rather than two mounts.
Merging here keeps the training script unchanged.

Both sources already use the same contract — RGB 512x512 PNG plus a
``<stem>_label.png`` of 255 / 128 / 0 — because ``build_india_labels.py`` was
written to match. Filenames are prefixed per source so two AOIs can never
collide.

    python scripts/build_india_labels.py
    python scripts/pack_india_dataset.py
    python -m kaggle datasets version -p data/kaggle_india -m "8 more cities" --dir-mode zip
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SOURCES = [
    ("obw", "data/india_wide"),      # Open Buildings + OSM, eight cities
    ("osm", "data/finetune_osm"),    # the original Bangalore/Bhopal relabelling
]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/kaggle_india")
    ap.add_argument("--slug", default="partheshgupta/rooftop-solar-india-wide-tiles")
    ap.add_argument("--title", default="Rooftop Solar India Wide Tiles")
    args = ap.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out
    img_dir, lbl_dir = out / "images", out / "labels"
    for d in (img_dir, lbl_dir):
        d.mkdir(parents=True, exist_ok=True)

    total = 0
    per_source = {}
    for prefix, rel in SOURCES:
        src = REPO / rel
        if not (src / "images").is_dir():
            print(f"  {rel}: absent, skipping")
            continue
        n = 0
        for img in sorted((src / "images").glob("*.png")):
            lbl = src / "labels" / f"{img.stem}_label.png"
            if not lbl.exists():
                continue
            shutil.copy2(img, img_dir / f"{prefix}_{img.name}")
            shutil.copy2(lbl, lbl_dir / f"{prefix}_{img.stem}_label.png")
            n += 1
        per_source[rel] = n
        total += n
        print(f"  {rel}: {n} tiles -> {prefix}_*")

    if not total:
        raise SystemExit("nothing to pack — run build_india_labels.py first")

    (out / "dataset-metadata.json").write_text(json.dumps({
        "title": args.title,
        "id": args.slug,
        "licenses": [{"name": "CC-BY-SA-4.0"}],
    }, indent=2), encoding="utf-8")

    (out / "README.md").write_text(
        "# Indian rooftop tiles\n\n"
        "512x512 RGB tiles at Web Mercator z19 (Esri World Imagery) with\n"
        "three-way masks: 255 positive, 128 ignore, 0 negative.\n\n"
        f"{total} tiles from {len(per_source)} source(s):\n\n"
        + "".join(f"- `{k}` — {v} tiles\n" for k, v in per_source.items())
        + "\nLabels combine Google Open Buildings v3 (CC-BY-4.0 / ODbL) above\n"
          "Google's own per-region 80%-precision confidence threshold, plus\n"
          "OpenStreetMap footprints, as positives. Open Buildings detections\n"
          "*below* that threshold are the ignore band — they skew small, and\n"
          "small roofs are what the model is already silent on, so calling them\n"
          "background would reinforce the failure this data exists to fix.\n",
        encoding="utf-8")

    print(f"\n{total} tiles staged in {out}")
    print("\nUpload with:")
    print(f"  python -m kaggle datasets version -p {out.relative_to(REPO)} "
          f'-m "india-wide tiles" --dir-mode zip')
    print(f"  (first time: python -m kaggle datasets create -p "
          f"{out.relative_to(REPO)} --dir-mode zip)")


if __name__ == "__main__":
    main()
