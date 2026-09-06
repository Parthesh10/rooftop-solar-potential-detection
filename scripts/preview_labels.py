r"""Contact sheet of imagery with its labels overlaid, for eyeballing a label set.

Statistics do not catch a georeferencing slip or an inverted mask — the positive
fraction looks perfectly reasonable while every polygon sits ten metres north of
its roof. This renders tiles with the label painted over them so that class of
bug is visible in one glance.

    python scripts/preview_labels.py --data data/india_wide --out preview.png

green = positive (train on it as roof)
amber = ignore   (excluded from the loss entirely)
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
from PIL import Image

POSITIVE, IGNORE = 255, 128


def overlay(img: np.ndarray, label: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    out = img.astype(np.float32).copy()
    pos = label == POSITIVE
    ign = label == IGNORE
    for mask, colour in ((pos, (0, 255, 90)), (ign, (255, 176, 0))):
        if mask.any():
            for c in range(3):
                out[..., c][mask] = (1 - alpha) * out[..., c][mask] + alpha * colour[c]
    return out.clip(0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/india_wide")
    ap.add_argument("--out", default="label_preview.png")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--cell", type=int, default=384, help="px per tile in the sheet")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-aoi", action="store_true",
                    help="one tile from each AOI instead of a random sample")
    args = ap.parse_args()

    root = Path(args.data)
    if not root.is_absolute():
        root = REPO / root
    imgs = sorted((root / "images").glob("*.png"))
    if not imgs:
        raise SystemExit(f"no tiles in {root / 'images'}")

    if args.per_aoi:
        seen: dict[str, Path] = {}
        for p in imgs:
            aoi = p.stem.rsplit("_", 1)[0]
            seen.setdefault(aoi, p)
        picks = list(seen.values())[:args.cols * args.rows]
    else:
        random.seed(args.seed)
        picks = random.sample(imgs, min(len(imgs), args.cols * args.rows))

    cell = args.cell
    sheet = Image.new("RGB", (args.cols * cell, args.rows * cell), (16, 16, 18))
    for i, p in enumerate(picks):
        lp = root / "labels" / f"{p.stem}_label.png"
        if not lp.exists():
            continue
        img = np.array(Image.open(p).convert("RGB"))
        lab = np.array(Image.open(lp).convert("L"))
        tile = Image.fromarray(overlay(img, lab)).resize((cell, cell))
        sheet.paste(tile, ((i % args.cols) * cell, (i // args.cols) * cell))

    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out
    sheet.save(out)
    print(f"wrote {out}  ({len(picks)} tiles)")
    for p in picks:
        lab = np.array(Image.open(root / "labels" / f"{p.stem}_label.png").convert("L"))
        print(f"  {p.stem:28s} positive {np.mean(lab == POSITIVE):.3f}  "
              f"ignore {np.mean(lab == IGNORE):.3f}")


if __name__ == "__main__":
    main()
