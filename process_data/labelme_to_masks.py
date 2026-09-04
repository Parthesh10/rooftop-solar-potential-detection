r"""Convert labelme polygon annotations into the binary masks the training
loaders expect.

``labelme`` (``pip install labelme``) is the recommended tool for the
hand-labelling step — see ``finetune/README.md``. It writes one JSON file per
image, each holding a list of polygons. Training needs a PNG mask instead:
``process_data.data_loader.DataLoaderSegmentation`` expects
``images/<stem>.png`` next to ``labels/<stem>_label.png``, mirroring the Swiss
DOP25 layout exactly (see ``conftest.py``'s ``_make_tile`` fixture for the same
convention). This script bridges the two.

Every polygon whose label is not explicitly "not-building" / "ignore" counts as
roof — labelme's default single-class workflow only ever produces one label
anyway ("building" is what ``finetune/README.md`` tells a labeller to name it),
but a second label lets a labeller flag something to exclude (e.g. a shadow
misread as a building) without deleting the polygon.

    python -m process_data.labelme_to_masks \
        --images data/finetune_candidates/images \
        --annotations data/finetune_candidates/images \
        --out data/finetune_indian
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
from PIL import Image

__all__ = ["polygons_to_mask", "convert_one", "convert_dir"]

EXCLUDE_LABELS = {"not-building", "not_building", "ignore", "exclude"}


def polygons_to_mask(shapes: list[dict], height: int, width: int) -> np.ndarray:
    """labelme ``shapes`` -> a ``{0, 255}`` uint8 mask, ``(height, width)``.

    Only polygon and closed-path shapes are rasterised; a point or line shape
    (someone marking "check this later") is silently skipped rather than
    raising, since a labeller should be free to leave notes.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for shape in shapes:
        label = str(shape.get("label", "")).strip().lower()
        if label in EXCLUDE_LABELS:
            continue
        if shape.get("shape_type", "polygon") not in ("polygon", "linestrip"):
            continue
        pts = shape.get("points") or []
        if len(pts) < 3:
            continue
        poly = np.array([[round(x), round(y)] for x, y in pts], dtype=np.int32)
        cv2.fillPoly(mask, [poly], 255)
    return mask


def convert_one(json_path: Path, images_dir: Path,
                out_images: Path, out_labels: Path) -> bool:
    """Convert one labelme JSON. Returns False (and skips) if there is nothing
    to rasterise, so an unlabelled candidate does not silently become an
    all-background training example."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    shapes = data.get("shapes") or []
    if not shapes:
        return False

    stem = json_path.stem
    src_image = images_dir / f"{stem}.png"
    if not src_image.exists():
        # labelme also supports embedding the image in the JSON; fall back to
        # imageHeight/imageWidth for the mask size and re-derive the source
        # image from the embedded data if present, else skip.
        if "imageData" not in data or not data["imageData"]:
            raise FileNotFoundError(
                f"{json_path.name} has no matching image at {src_image} and no "
                f"embedded imageData — run labelme without --nodata, or keep "
                f"the source PNGs next to the JSON files")
        import base64
        from io import BytesIO
        img = Image.open(BytesIO(base64.b64decode(data["imageData"]))).convert("RGB")
    else:
        img = Image.open(src_image).convert("RGB")

    w, h = img.size
    mask = polygons_to_mask(shapes, h, w)
    if not mask.any():
        return False

    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    img.save(out_images / f"{stem}.png")
    Image.fromarray(mask, mode="L").save(out_labels / f"{stem}_label.png")
    return True


def convert_dir(images_dir: Path, annotations_dir: Path, out_dir: Path) -> dict:
    """Convert every ``*.json`` in ``annotations_dir``. Returns a summary dict."""
    out_images = out_dir / "images"
    out_labels = out_dir / "labels"
    json_files = sorted(annotations_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(
            f"no .json annotations in {annotations_dir}. Label the candidates "
            f"first — see finetune/README.md.")

    converted, skipped = [], []
    for jp in json_files:
        try:
            ok = convert_one(jp, images_dir, out_images, out_labels)
        except Exception as exc:
            print(f"  ! {jp.name}: {exc}")
            skipped.append(jp.name)
            continue
        (converted if ok else skipped).append(jp.name)

    return {"converted": converted, "skipped": skipped,
            "out_images": out_images, "out_labels": out_labels}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="directory of source PNGs")
    ap.add_argument("--annotations", required=True,
                    help="directory of labelme .json files (often the same "
                        "directory as --images, since labelme writes there "
                        "by default)")
    ap.add_argument("--out", required=True,
                    help="output root; writes out/images and out/labels")
    args = ap.parse_args()

    summary = convert_dir(Path(args.images), Path(args.annotations), Path(args.out))
    print(f"converted: {len(summary['converted'])}")
    print(f"skipped (no polygons or an error): {len(summary['skipped'])}")
    for name in summary["skipped"]:
        print(f"  - {name}")
    print(f"\nimages -> {summary['out_images']}")
    print(f"labels -> {summary['out_labels']}")
    if len(summary["converted"]) < 10:
        print("\nWarning: fewer than 10 labelled tiles. finetune_indian.py will "
             "refuse to run below its floor — label more before fine-tuning.")


if __name__ == "__main__":
    main()
