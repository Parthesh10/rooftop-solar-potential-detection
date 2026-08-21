"""Trustworthy evaluation harness.

Replaces ``hyperparameters.select_param.test_model``, which had three defects
that between them invalidate every number this project has ever reported:

* **F-03** — it never called ``model.eval()``. BatchNorm therefore normalised
  with the *current batch's* statistics at batch size 2, so each image's
  prediction depended on whichever image happened to share its batch. The
  evaluation was neither deterministic nor representative of deployment.
* **F-06** — no ``torch.no_grad()``, so a full autograd graph was built for
  every validation forward pass.
* **F-16** — the inner loop reused the outer loop's index variable ``i``.

Plus the metric fixes in ``loss/loss.py`` (F-08 empty-tile inversion, F-11
divide-by-zero in ``recall``).

Usage
-----
    python evaluate.py --ckpt "model/path raise 130.pt" --split test
    python evaluate.py --all            # re-score every checkpoint, all splits
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import DATA_ROOT, REPO_ROOT
from loss.loss import accuracy, f1, iou, precision, recall, summarize
from process_data.data_loader import DataLoaderSegmentation
from utils import get_device

__all__ = ["evaluate", "evaluate_arrays", "make_eval_loader"]

_METRICS = {
    "iou": iou,
    "f1": f1,
    "accuracy": accuracy,
    "recall": recall,
    "precision": precision,
}


@torch.no_grad()
def evaluate(
    loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device | str | None = None,
    threshold: float = 0.5,
    tta: bool = False,
    per_tile: bool = False,
    loss_fn=None,
    progress: bool = False,
) -> dict:
    """Score a model over a loader.

    Returns a dict of nan-aware means plus ``<metric>_undefined`` counts, so
    empty tiles are reported rather than silently scored as perfect (F-08).

    Pass ``loss_fn`` to also accumulate the validation loss in the *same* pass —
    the training loop needs both, and on a small GPU a second sweep over the
    validation set is pure waste.
    """
    from infer import predict_probs  # local import: avoids a cycle at module load

    device = torch.device(device) if device is not None else get_device()
    model = model.to(device)
    was_training = model.training
    model.eval()  # ← F-03, the line that was missing

    scores: dict[str, list[float]] = {name: [] for name in _METRICS}
    n_images = 0
    n_positive_pixels = 0
    n_pixels = 0
    loss_total, loss_batches = 0.0, 0

    iterator = loader
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(loader, desc="val", leave=False, unit="b", dynamic_ncols=True)
        except ImportError:
            pass

    try:
        for images, labels in iterator:  # F-16: no index shadowing
            images = images.to(device, non_blocking=True)

            probs = predict_probs(model, images, device=device, tta=tta)

            if loss_fn is not None:
                # logit = log(p / (1-p)); recovering it avoids a second forward
                # pass just to feed a *WithLogits loss.
                p = probs.clamp(1e-6, 1 - 1e-6)
                logits = torch.log(p / (1 - p)).squeeze(1)
                loss_total += float(loss_fn(logits, labels.to(device)).item())
                loss_batches += 1

            preds = (probs.squeeze(1) > threshold).cpu().numpy()
            gts = labels.numpy() > 0.5

            n_positive_pixels += int(gts.sum())
            n_pixels += int(gts.size)

            for b in range(preds.shape[0]):
                n_images += 1
                for name, fn in _METRICS.items():
                    scores[name].append(fn(preds[b], gts[b]))
    finally:
        if was_training:
            model.train()

    out = summarize(scores)
    out["n_images"] = n_images
    out["positive_pixel_rate"] = (n_positive_pixels / n_pixels) if n_pixels else float("nan")
    out["threshold"] = threshold
    out["tta"] = tta
    if loss_fn is not None:
        out["loss"] = loss_total / max(loss_batches, 1)
    if per_tile:
        out["_per_tile"] = scores
    return out


def evaluate_arrays(preds, targets) -> dict:
    """Score already-computed boolean arrays. Handy in tests and notebooks."""
    scores = {name: [fn(p, t) for p, t in zip(preds, targets)] for name, fn in _METRICS.items()}
    return summarize(scores)


def make_eval_loader(
    split: str,
    data_root: Path | None = None,
    stats_key: str = "all",
    batch_size: int = 4,
    num_workers: int = 0,
) -> DataLoader:
    """Build a non-augmenting loader for a split directory under ``DATA_ROOT``."""
    root = Path(data_root) if data_root is not None else DATA_ROOT
    ds = DataLoaderSegmentation(
        root / split / "images",
        root / split / "labels",
        augment=False,   # no crop, no flips — evaluate on the full native tile
        crop=None,
        stats_key=stats_key,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def _fmt(d: dict) -> str:
    keys = ["iou", "f1", "accuracy", "recall", "precision"]
    parts = [f"{k}={d[k]:.4f}" for k in keys if k in d and np.isfinite(d[k])]
    undef = d.get("iou_undefined", 0)
    parts.append(f"n={d.get('n_images', 0)}")
    if undef:
        parts.append(f"undefined_tiles={undef}")
    parts.append(f"pos_rate={d.get('positive_pixel_rate', float('nan')):.4f}")
    return "  ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-score checkpoints with a correct harness")
    ap.add_argument("--ckpt", type=str, help="path to a .pt checkpoint")
    ap.add_argument("--all", action="store_true", help="score every checkpoint in model/")
    ap.add_argument("--split", type=str, default="test",
                    choices=["train", "val", "test", "all"])
    ap.add_argument("--data-root", type=str, default=None)
    ap.add_argument("--stats-key", type=str, default="all")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--out", type=str, default=None, help="write results as JSON here")
    args = ap.parse_args()

    from infer import load_manifest, load_model

    device = get_device()
    print(f"device: {device}")

    ckpts: list[Path]
    if args.all:
        ckpts = sorted((REPO_ROOT / "model").glob("*.pt"))
    elif args.ckpt:
        ckpts = [Path(args.ckpt)]
    else:
        ap.error("pass --ckpt or --all")

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    manifest = load_manifest()
    results: dict[str, dict] = {}

    for ckpt in ckpts:
        print(f"\n=== {ckpt.name} ===")
        model, entry = load_model(ckpt, device=device, manifest=manifest)
        stats_key = entry.get("stats_key", args.stats_key)
        threshold = args.threshold if args.threshold != 0.5 else entry.get("threshold", 0.5)

        results[ckpt.name] = {}
        for split in splits:
            loader = make_eval_loader(
                split, data_root=args.data_root, stats_key=stats_key,
                batch_size=args.batch_size,
            )
            res = evaluate(loader, model, device=device, threshold=threshold, tta=args.tta)
            results[ckpt.name][split] = res
            print(f"  {split:<6} {_fmt(res)}")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
