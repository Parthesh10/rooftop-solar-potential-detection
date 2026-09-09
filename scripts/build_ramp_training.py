r"""Turn ramp tiles into the images/ + labels/ pairs train_inria.py consumes.

**Read the licence note before using the output.** ramp is CC-BY-NC-4.0 —
non-commercial, and stricter than Inria, Open Buildings (CC-BY) or OSM (ODbL).
A model trained on it inherits that constraint, which is why:

* the resulting checkpoint is registered ``"default": false`` and is **not**
  published to a GitHub release, and
* the shipped default stays free of ramp data.

**And read the semantics note, which is the more subtle trap.** ramp labels
**rooftops**; Inria labels **ground footprints**. Training one model on both
teaches two different answers to "where does a roof end" — precisely the
incoherence that made the hand-drawn envelope fine-tune unusable (fact 25).
So this script exists to build a *separate* ramp-semantics model, not to blend
ramp into the existing joint set. `--mix-inria` is deliberately not an option
here.

The upside is real: ramp is human-drawn, exhaustive, 30 cm — the same resolution
Inria and the app's Esri z19 use — and rooftop semantics are what a solar tool
actually wants, since it multiplies roof area into kWh.

Output matches the existing convention: ``images/<stem>.png`` beside
``labels/<stem>_label.png``, 255 = building, 0 = background. No ignore band —
ramp tiles are exhaustively labelled, so a negative really is a negative, unlike
the Open Buildings sets where low-confidence detections must be ignored
(fact 34).

    python scripts/build_ramp_training.py --dataset ramp_dhaka_bangladesh
    python scripts/build_ramp_training.py --all --out data/ramp_train
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
from PIL import Image

from scripts.eval_ramp import load_tile

POSITIVE, NEGATIVE = 255, 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="data/ramp",
                    help="where download_ramp.py put things")
    ap.add_argument("--dataset", action="append",
                    help="repeatable; default is every dataset under --src")
    ap.add_argument("--out", default="data/ramp_train")
    ap.add_argument("--min-positive-frac", type=float, default=0.005,
                    help="skip near-empty tiles. Lower than the Open Buildings "
                        "pipeline's 0.02 because ramp deliberately includes "
                        "rural and agricultural tiles, and those sparse tiles "
                        "are exactly the built form the model is worst at")
    ap.add_argument("--holdout", action="append", default=["ramp_karnataka_india"],
                    help="datasets to EXCLUDE from training. Karnataka is held "
                        "out by default because it is the project's only "
                        "human-drawn benchmark — training on it would destroy "
                        "the one honest non-Inria number there is")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.is_absolute():
        src = REPO / src
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    names = args.dataset or sorted(
        p.name for p in src.iterdir() if p.is_dir() and (p / "source").is_dir())
    held = [n for n in names if n in args.holdout]
    names = [n for n in names if n not in args.holdout]
    if held:
        print(f"holding out (benchmark, never train on it): {', '.join(held)}\n")
    if not names:
        raise SystemExit(f"nothing to convert under {src}")

    written = skipped = 0
    per_ds = {}
    for name in names:
        d = src / name
        stems = [p.stem for p in sorted((d / "source").glob("*.tif"))
                 if (d / "labels" / f"{p.stem}.geojson").exists()]
        n_ok = 0
        for stem in stems:
            try:
                img, ref, _ = load_tile(stem, d)
            except Exception as exc:
                print(f"  {name}/{stem[:8]}: {type(exc).__name__}: {exc}")
                skipped += 1
                continue
            if float(ref.mean()) < args.min_positive_frac:
                skipped += 1
                continue
            # Prefix the stem so tiles from different regions cannot collide and
            # so a later split can hold out a whole region by name.
            key = f"{name.removeprefix('ramp_')}_{stem[:12]}"
            Image.fromarray(img).save(out / "images" / f"{key}.png")
            Image.fromarray(
                np.where(ref, POSITIVE, NEGATIVE).astype(np.uint8), mode="L"
            ).save(out / "labels" / f"{key}_label.png")
            written += 1
            n_ok += 1
        per_ds[name] = n_ok
        print(f"  {name:32s} {n_ok:5d} tiles")

    if not written:
        raise SystemExit("no tiles written — check --src and --min-positive-frac")

    fracs = []
    for p in sorted((out / "labels").glob("*_label.png"))[:400]:
        fracs.append(float((np.array(Image.open(p)) == POSITIVE).mean()))
    print(f"\n{written} tiles -> {out}   ({skipped} skipped as near-empty)")
    print(f"  positive coverage: mean {np.mean(fracs):.3f}  "
          f"median {np.median(fracs):.3f}")
    print("\n  LICENCE: CC-BY-NC-4.0. Any checkpoint trained on this is "
          "non-commercial;\n  keep it non-default and do not publish it.")
    print("  SEMANTICS: rooftops, NOT Inria ground footprints — do not mix "
          "the two\n  in one training run (fact 25).")


if __name__ == "__main__":
    main()
