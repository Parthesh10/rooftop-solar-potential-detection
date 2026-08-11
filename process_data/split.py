"""Leakage-free dataset splitting.

Replaces ``process_data/spliter.ipynb``, which did:

    np.random.shuffle(allFileNames)
    train, val, test = np.split(np.array(allFileNames), [int(n*0.8), int(n*0.9)])

The Swiss DOP25 filenames encode LV03 coordinates —
``DOP25_LV03_1301_11_2015_1_15_497812.5_120937.5.png`` — and the crops are
250x250 windows spaced **62.5 m apart**: directly adjacent views of one
continuous aerial survey. A random shuffle therefore puts a tile's immediate
neighbour in train and the tile itself in test. They share buildings, lighting,
sun angle, season, sensor and roof style, so the measured train->test gap
(0.752 -> 0.556) is an *optimistic* estimate of true generalisation (F-05).

This module splits by geography instead: whole spatial blocks are assigned to
one split, so no tile is ever adjacent to a tile in a different split.

The old notebook also called ``shutil.rmtree`` on the target folders
unconditionally and copied rather than linked, tripling disk usage. This writes
manifest files by default and only copies when asked.

Usage
-----
    # inspect the split without touching the filesystem
    python -m process_data.split --root data --dry-run

    # write train.txt / val.txt / test.txt manifests
    python -m process_data.split --root data --out data/splits

    # Inria official protocol (tiles 1-5 of each city -> val)
    python -m process_data.split --inria --root ../dataset/AerialImageDataset/train
"""

from __future__ import annotations

import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

__all__ = ["parse_lv03", "block_split", "inria_official_split", "write_manifests"]

# ...<easting>_<northing>.png, both as decimal metres.
_LV03_RE = re.compile(r"_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)(?:_label)?\.png$", re.IGNORECASE)

# austin1.tif, tyrol-w12.tif, ...
_INRIA_RE = re.compile(r"^([a-z\-]+)(\d+)\.(?:tif|tiff|png)$", re.IGNORECASE)


def parse_lv03(path: Path | str) -> tuple[float, float] | None:
    """Extract ``(easting, northing)`` in metres from a DOP25 filename."""
    m = _LV03_RE.search(Path(path).name)
    return (float(m.group(1)), float(m.group(2))) if m else None


def block_split(
    files: list[Path],
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    block_size: float = 1000.0,
    seed: int = 0,
    buffer: float = 0.0,
) -> dict[str, list[Path]]:
    """Assign whole ``block_size``-metre blocks to train / val / test.

    Blocks, not files, are shuffled — so a tile and its neighbours always land
    in the same split. Ratios are approximate because blocks are indivisible.

    ``buffer`` (metres) additionally **drops training tiles within that distance
    of any val/test tile.** Blocking alone is not sufficient: a tile at the edge
    of a training block can still be one 62.5 m step from a tile at the edge of a
    test block, which is precisely the adjacency F-05 is about. A buffer of two
    tile-steps (125 m) removes it entirely. Val and test are never trimmed, so
    the evaluation sets stay at full size and only training data is sacrificed.

    Raises if any filename lacks parseable coordinates; a silent fallback to
    random splitting would reintroduce exactly the leak this exists to prevent.
    """
    blocks: dict[tuple[int, int], list[Path]] = defaultdict(list)
    unparsed: list[Path] = []
    for f in files:
        coords = parse_lv03(f)
        if coords is None:
            unparsed.append(f)
            continue
        e, n = coords
        blocks[(int(e // block_size), int(n // block_size))].append(f)

    if unparsed:
        raise ValueError(
            f"{len(unparsed)} filenames carry no LV03 coordinates, e.g. "
            f"{[p.name for p in unparsed[:3]]}. Refusing to fall back to a random "
            f"split — that is the leak described in F-05."
        )

    keys = sorted(blocks)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)

    total = len(files)
    want_val = val_ratio * total
    want_test = test_ratio * total

    out: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    n_val = n_test = 0
    for key in keys:
        group = sorted(blocks[key])
        if n_test < want_test:
            out["test"].extend(group)
            n_test += len(group)
        elif n_val < want_val:
            out["val"].extend(group)
            n_val += len(group)
        else:
            out["train"].extend(group)

    print(
        f"[split] {len(files)} tiles in {len(keys)} blocks of {block_size:.0f} m -> "
        f"train {len(out['train'])} / val {len(out['val'])} / test {len(out['test'])}"
    )

    if buffer > 0:
        held = np.array([parse_lv03(f) for f in out["val"] + out["test"]], dtype=float)
        if len(held):
            train_xy = np.array([parse_lv03(f) for f in out["train"]], dtype=float)
            # Chebyshev distance: tiles sit on a regular grid, so max-axis
            # distance is the natural "how many tile-steps away" measure.
            d = np.abs(train_xy[:, None, :] - held[None, :, :]).max(axis=2).min(axis=1)
            keep = d > buffer
            dropped = int((~keep).sum())
            out["train"] = [f for f, k in zip(out["train"], keep) if k]
            print(
                f"[split] buffer {buffer:.0f} m dropped {dropped} training tiles too "
                f"close to val/test -> train {len(out['train'])}"
            )
    return out


def inria_official_split(files: list[Path], n_val_per_city: int = 5) -> dict[str, list[Path]]:
    """Inria's published protocol: tiles 1-5 of each city -> validation.

    Using this makes your numbers directly comparable to published Inria
    building-segmentation results. Note that Inria's own *test* cities
    (bellingham, bloomington, innsbruck, sfo, tyrol-e) ship without ground
    truth, so they cannot be scored locally.
    """
    out: dict[str, list[Path]] = {"train": [], "val": []}
    by_city: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for f in files:
        m = _INRIA_RE.match(f.name)
        if not m:
            raise ValueError(f"unrecognised Inria filename: {f.name}")
        by_city[m.group(1).lower()].append((int(m.group(2)), f))

    for city, items in sorted(by_city.items()):
        items.sort()
        out["val"].extend(p for idx, p in items if idx <= n_val_per_city)
        out["train"].extend(p for idx, p in items if idx > n_val_per_city)

    print(
        f"[split] Inria official: {len(by_city)} cities -> "
        f"train {len(out['train'])} / val {len(out['val'])}"
    )
    return out


def city_holdout_split(files: list[Path], holdout: str) -> dict[str, list[Path]]:
    """Hold out an entire city. The strictest generalisation test available."""
    out: dict[str, list[Path]] = {"train": [], "val": []}
    for f in files:
        m = _INRIA_RE.match(f.name)
        if not m:
            raise ValueError(f"unrecognised Inria filename: {f.name}")
        key = "val" if m.group(1).lower() == holdout.lower() else "train"
        out[key].append(f)
    if not out["val"]:
        raise ValueError(f"no tiles found for holdout city {holdout!r}")
    print(f"[split] holdout '{holdout}': train {len(out['train'])} / val {len(out['val'])}")
    return out


def write_manifests(split: dict[str, list[Path]], out_dir: Path, root: Path) -> None:
    """Write ``<split>.txt`` files of root-relative paths. No data is copied."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, files in split.items():
        lines = "\n".join(str(Path(f).relative_to(root)).replace("\\", "/") for f in files)
        (out_dir / f"{name}.txt").write_text(lines + "\n", encoding="utf-8")
        print(f"[split] wrote {out_dir / f'{name}.txt'} ({len(files)} entries)")


def materialize(
    split: dict[str, list[Path]],
    dest_root: Path,
    image_dir: Path,
    label_dir: Path | None,
    mask_suffix: str = "_label",
) -> None:
    """Copy files into ``dest_root/<split>/{images,labels}``.

    Only use this if a tool genuinely needs the on-disk layout — manifests are
    cheaper and reversible. Refuses to overwrite a non-empty destination.
    """
    dest_root = Path(dest_root)
    for name, files in split.items():
        for sub in ("images", "labels") if label_dir else ("images",):
            d = dest_root / name / sub
            if d.exists() and any(d.iterdir()):
                raise FileExistsError(f"{d} is not empty; refusing to overwrite")
            d.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(f, dest_root / name / "images" / Path(f).name)
            if label_dir:
                lbl = Path(label_dir) / f"{Path(f).stem}{mask_suffix}.png"
                if not lbl.exists():
                    lbl = Path(label_dir) / f"{Path(f).stem}.png"
                if not lbl.exists():
                    raise FileNotFoundError(f"no label for {f.name}")
                shutil.copy2(lbl, dest_root / name / "labels" / lbl.name)
        print(f"[split] materialized {name}: {len(files)} pairs")


def main() -> None:
    ap = argparse.ArgumentParser(description="Leakage-free dataset splitting (F-05)")
    ap.add_argument("--root", required=True, help="directory containing the images")
    ap.add_argument("--images", default="images", help="image subdirectory under --root")
    ap.add_argument("--labels", default="labels", help="label subdirectory under --root")
    ap.add_argument("--out", default=None, help="where to write manifests")
    ap.add_argument("--inria", action="store_true", help="use Inria's official 1-5 protocol")
    ap.add_argument("--holdout-city", default=None, help="hold out one Inria city entirely")
    ap.add_argument("--block-size", type=float, default=1000.0, help="metres, Swiss data")
    ap.add_argument("--buffer", type=float, default=125.0,
                    help="metres; drop training tiles this close to val/test (0 disables)")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--materialize", default=None, help="copy files into this directory")
    args = ap.parse_args()

    root = Path(args.root)
    exts = {".png", ".tif", ".tiff"}

    if (root / args.images).is_dir():
        image_dir = root / args.images
        candidates = list(image_dir.iterdir())
    elif list(root.glob(f"*/{args.images}")):
        # Pooled mode: data/{train,val,test}/images/*.png. A geographic re-split
        # has to see every tile at once, not one pre-existing split at a time.
        image_dir = root
        candidates = list(root.glob(f"*/{args.images}/*"))
        print(f"[split] pooling tiles from {root}/*/{args.images}")
    else:
        image_dir = root
        candidates = list(root.iterdir())

    files = sorted(
        p for p in candidates
        if p.is_file() and p.suffix.lower() in exts and "_label" not in p.stem
    )
    if not files:
        raise SystemExit(f"no images found under {root}")

    # Deduplicate by filename. data/residencial/ is a *subset* of the same
    # imagery already present in data/{train,val,test}/, so pooling naively puts
    # byte-identical tiles into different splits — the very leak this module
    # exists to prevent, in its most literal form.
    seen: dict[str, Path] = {}
    dupes = 0
    for p in files:
        if p.name in seen:
            dupes += 1
            continue
        seen[p.name] = p
    if dupes:
        print(f"[split] dropped {dupes} duplicate filenames (kept {len(seen)} unique tiles)")
    files = sorted(seen.values())

    if args.holdout_city:
        split = city_holdout_split(files, args.holdout_city)
    elif args.inria:
        split = inria_official_split(files)
    else:
        split = block_split(
            files, val_ratio=args.val_ratio, test_ratio=args.test_ratio,
            block_size=args.block_size, seed=args.seed, buffer=args.buffer,
        )

    if args.dry_run:
        for name, fs in split.items():
            print(f"  {name}: {len(fs)} — e.g. {[p.name for p in fs[:2]]}")
        return

    if args.out:
        write_manifests(split, Path(args.out), root)
    if args.materialize:
        label_dir = root / args.labels if (root / args.labels).is_dir() else None
        materialize(split, Path(args.materialize), image_dir, label_dir)


if __name__ == "__main__":
    main()
