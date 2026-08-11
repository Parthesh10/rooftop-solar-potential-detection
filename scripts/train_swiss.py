"""Train the U-Net on the Swiss DOP25 set using the leakage-free geographic split.

Prerequisite — generate the split manifests once:

    python -m process_data.split --root data --out data/splits --block-size 1000 --buffer 125

Then:

    python scripts/train_swiss.py                 # defaults tuned for a 4 GB GPU
    python scripts/train_swiss.py --batch-size 4 --epochs 40
    python scripts/train_swiss.py --wandb

Notes for small GPUs (this was written against a GTX 1650, 4 GB):

* AMP uses **float16**, not bfloat16. bf16 needs compute capability 8.0
  (Ampere); Turing is 7.5, so bf16 silently falls back or errors. ``torch.amp``
  defaults to fp16 on CUDA, which is what we want, and ``GradScaler`` handles
  the loss scaling.
* The GTX 16-series has no tensor cores, so AMP buys memory headroom more than
  speed. Expect maybe 1.2-1.4x, not 2-3x.
* ``--batch-size`` auto-halves on CUDA OOM rather than dying.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch
from torch.utils.data import DataLoader

from config import TrainConfig
from evaluate import evaluate
from loss.losses import build_loss
from model.unet import UNet
from process_data.data_loader import DataLoaderSegmentation
from train.train import build_optimizer, build_scheduler, training_model
from utils import count_parameters, get_device, seed_torch


def build_loaders(cfg: TrainConfig, splits_dir: Path, data_root: Path, batch_size: int):
    ds = {}
    for name, augment, crop in (
        ("train", True, cfg.crop),
        ("val", False, None),
        ("test", False, None),
    ):
        manifest = splits_dir / f"{name}.txt"
        if not manifest.exists():
            raise SystemExit(
                f"missing {manifest}. Generate it first:\n"
                f"  python -m process_data.split --root data --out data/splits "
                f"--block-size 1000 --buffer 125"
            )
        ds[name] = DataLoaderSegmentation.from_manifest(
            manifest, data_root, augment=augment, crop=crop, stats_key=cfg.stats_key
        )
        print(f"  {name:<6} {ds[name]}")

    loaders = {
        "train": DataLoader(ds["train"], batch_size=batch_size, shuffle=True,
                            num_workers=cfg.num_workers, drop_last=False,
                            pin_memory=torch.cuda.is_available()),
        # batch 1 for eval: tiles are full-size 250x250 and this keeps peak
        # memory low on a 4 GB card. Evaluation is not the bottleneck.
        "val": DataLoader(ds["val"], batch_size=1, shuffle=False,
                          num_workers=cfg.num_workers),
        "test": DataLoader(ds["test"], batch_size=1, shuffle=False,
                           num_workers=cfg.num_workers),
    }
    return ds, loaders


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--crop", type=int, default=224)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=15)
    args = ap.parse_args()

    cfg = TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, crop=args.crop,
        num_workers=args.workers, amp=not args.no_amp, seed=args.seed,
        early_stop_patience=args.patience, wandb=args.wandb,
        wandb_project="rooftop-solar", stats_key="all",
    )

    device = get_device()
    seed_torch(cfg.seed, deterministic=False)  # benchmark mode: ~20-30% faster

    print(f"device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"gpu:    {props.name}  {props.total_memory / 1024**3:.1f} GB  "
              f"sm_{props.major}{props.minor}")
        if props.major < 8 and cfg.amp:
            print("note:   compute < 8.0 -> AMP uses fp16 (no bf16, no tensor cores)")

    print("datasets:")
    batch_size = args.batch_size
    ds, loaders = build_loaders(cfg, Path(args.splits), Path(args.data_root), batch_size)

    model = UNet(3, 1, False).to(device)
    print(f"model:  UNet, {count_parameters(model):,} trainable parameters")

    loss_fn = build_loss(cfg, loader=loaders["train"], device=device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(loaders["train"]))

    while True:
        try:
            history = training_model(
                loaders["train"], loss_fn, optimizer, model,
                num_epochs=cfg.epochs, scheduler=scheduler, val_loader=loaders["val"],
                cfg=cfg, device=device, threshold=cfg.threshold,
            )
            break
        except torch.cuda.OutOfMemoryError:
            batch_size //= 2
            if batch_size < 1:
                raise
            print(f"\n[oom] retrying with batch_size={batch_size}\n")
            torch.cuda.empty_cache()
            cfg.batch_size = batch_size
            ds, loaders = build_loaders(cfg, Path(args.splits), Path(args.data_root),
                                        batch_size)
            model = UNet(3, 1, False).to(device)
            loss_fn = build_loss(cfg, loader=loaders["train"], device=device)
            optimizer = build_optimizer(model, cfg)
            scheduler = build_scheduler(optimizer, cfg,
                                        steps_per_epoch=len(loaders["train"]))

    # Final scoring uses the BEST checkpoint, not whatever the last epoch left
    # behind — the 2023 run had no best-checkpoint selection at all (F-10).
    best = Path(history.run_dir) / "best.pt"
    if best.exists():
        model.load_state_dict(torch.load(best, map_location=device, weights_only=True))
        print(f"\nloaded best checkpoint (epoch {history.best_epoch})")

    print("\n=== final metrics (geographic split, correct eval harness) ===")
    for name in ("train", "val", "test"):
        res = evaluate(loaders[name], model, device=device, threshold=cfg.threshold)
        print(f"  {name:<6} IoU {res['iou']:.4f}  F1 {res['f1']:.4f}  "
              f"acc {res['accuracy']:.4f}  P {res['precision']:.4f}  "
              f"R {res['recall']:.4f}  (n={res['n_images']}, "
              f"undefined={res['iou_undefined']})")
    print(f"\nartefacts: {history.run_dir}")


if __name__ == "__main__":
    main()
