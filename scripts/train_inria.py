r"""Train a segmentation model on the Inria Aerial Image Labeling dataset.

Inria is the training backbone for a *general* rooftop-detection model: 180
tiles of 5000x5000 @ 0.3 m/px over five cities (Austin, Chicago, Kitsap WA,
Vienna, Tyrol), with **building-footprint** labels. That label semantics is
deliberate — "where is the roof" is unambiguous everywhere; the usable-for-PV
fraction is a downstream packing factor (see plan.md §5.2).

Data comes from ``process_data.inria.InriaWindowDataset``: random 512x512
windows read straight out of the GeoTIFFs (no pre-cutting to disk), filtered so
the model does not spend all its compute on Kitsap forest and water. The split
is Inria's published protocol — tiles 1-5 of each city are validation, 6-36 are
training (``process_data.split.inria_official_split``) — so val IoU is directly
comparable to published Inria building-segmentation work. Target: >= 0.72.

Usage
-----
    # local copy
    python scripts/train_inria.py --inria-root ../dataset/AerialImageDataset \
        --arch unet --encoder resnet34 --encoder-weights imagenet

    # Kaggle (mounts the public dataset — see kaggle_inria/)
    python scripts/train_inria.py \
        --inria-root /kaggle/input/inria-aerial-image-labeling-dataset/AerialImageDataset

Pause / stop / resume works exactly as in train_swiss.py (runs/<ts>/CONTROL.md).
Inria is ~10-20 GPU-h, over Kaggle's 12 h session cap, so ``--resume`` across
two sessions is the expected path — model, optimizer, scheduler, AMP scaler,
epoch and RNG state are checkpointed every epoch and mid-epoch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch
from torch.utils.data import DataLoader

from config import INRIA_WINDOW, RUNS_ROOT, TrainConfig
from evaluate import evaluate
from loss.losses import build_loss
from model.registry import build_model, model_spec, recommended_stats_key
from process_data.inria import InriaWindowDataset
from process_data.split import inria_official_split
from train.train import build_optimizer, build_scheduler, training_model
from utils import count_parameters, get_device, seed_torch


def _resolve_inria_root(root: Path) -> Path:
    """Accept .../AerialImageDataset or .../AerialImageDataset/train."""
    if (root / "images").is_dir() and (root / "gt").is_dir():
        return root
    if (root / "train" / "images").is_dir():
        return root / "train"
    raise SystemExit(
        f"could not find images/ and gt/ under {root}. Point --inria-root at the "
        f"AerialImageDataset directory (or its train/ subdirectory)."
    )


def build_inria_loaders(cfg: TrainConfig, inria_root: Path, batch_size: int,
                        val_stride: int):
    root = _resolve_inria_root(Path(inria_root))
    img_files = sorted((root / "images").glob("*.tif"))
    if not img_files:
        raise SystemExit(f"no .tif tiles in {root / 'images'}")
    gt_dir = root / "gt"

    split = inria_official_split(img_files)

    train_ds = InriaWindowDataset(
        split["train"], gt_dir, window=cfg.crop,
        samples_per_tile=cfg.samples_per_tile,
        min_positive=cfg.min_positive, empty_ratio=cfg.empty_ratio,
        augment=True, stats_key=cfg.stats_key, seed=cfg.seed,
    )
    val_ds = InriaWindowDataset(
        split["val"], gt_dir, window=cfg.crop,
        augment=False, stats_key=cfg.stats_key,
        deterministic=True, stride=val_stride,
    )
    print(f"  train  {train_ds}")
    print(f"  val    {val_ds}")

    loaders = {
        "train": DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=cfg.num_workers, drop_last=True,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=cfg.num_workers > 0,
        ),
        "val": DataLoader(
            val_ds, batch_size=max(batch_size // 2, 1), shuffle=False,
            num_workers=cfg.num_workers,
        ),
    }
    return {"train": train_ds, "val": val_ds}, loaders


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inria-root", default="../dataset/AerialImageDataset",
                    help="AerialImageDataset directory (local) or the Kaggle mount")
    ap.add_argument("--arch", default="unet",
                    help="'unet', 'unet++', 'deeplabv3+', ... (model/registry.SMP_ARCHS)")
    ap.add_argument("--encoder", default="resnet34",
                    help="smp encoder, or 'scratch' for the verbatim U-Net")
    ap.add_argument("--encoder-weights", default="imagenet",
                    help="'imagenet' (default) or 'none'")
    ap.add_argument("--stats-key", default=None,
                    help="normalisation key; default: imagenet for a pretrained encoder")
    ap.add_argument("--window", type=int, default=INRIA_WINDOW,
                    help="crop size, multiple of 32 (default 512)")
    ap.add_argument("--samples-per-tile", type=int, default=32,
                    help="windows sampled per tile per epoch (155 train tiles)")
    ap.add_argument("--val-stride", type=int, default=512,
                    help="stride of the deterministic validation grid")
    ap.add_argument("--min-positive", type=float, default=0.005)
    ap.add_argument("--empty-ratio", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--amp", default="auto", choices=["auto", "fp16", "bf16", "off"])
    ap.add_argument("--no-amp", action="store_true", help="alias for --amp off")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--pos-weight", type=float, default=None,
                    help="BCE pos_weight (F-02). Default: estimate from the train set")
    ap.add_argument("--dice-weight", type=float, default=0.5)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--resume", nargs="?", const="auto", default=None, metavar="RUN",
                    help="resume the newest run with a state.pt, or a named run dir")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--gpu-mem-fraction", type=float, default=0.9)
    ap.add_argument("--gpu-util-target", type=float, default=80.0)
    ap.add_argument("--gpu-temp-limit", type=float, default=78.0)
    ap.add_argument("--checkpoint-every", type=float, default=120.0)
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    if args.window % 32:
        ap.error(f"--window must be a multiple of 32 (got {args.window})")

    ew = None if str(args.encoder_weights).lower() in ("none", "", "null") else args.encoder_weights
    stats_key = args.stats_key or recommended_stats_key(args.encoder, ew)

    cfg = TrainConfig(
        arch=args.arch, encoder=args.encoder, encoder_weights=ew,
        crop=args.window, samples_per_tile=args.samples_per_tile,
        min_positive=args.min_positive, empty_ratio=args.empty_ratio,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        num_workers=args.workers, amp=("off" if args.no_amp else args.amp),
        seed=args.seed, early_stop_patience=args.patience, wandb=args.wandb,
        pos_weight=args.pos_weight, dice_weight=args.dice_weight,
        wandb_project="rooftop-solar-inria", stats_key=stats_key,
        gpu_mem_fraction=args.gpu_mem_fraction or None,
        gpu_util_target=(args.gpu_util_target if args.gpu_util_target < 100 else None),
        gpu_temp_limit=args.gpu_temp_limit or None,
        checkpoint_every_seconds=args.checkpoint_every,
    )

    device = get_device()
    seed_torch(cfg.seed, deterministic=False)

    print(f"device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"gpu:    {props.name}  {props.total_memory / 1024**3:.1f} GB  "
              f"sm_{props.major}{props.minor}")

    print("datasets:")
    batch_size = args.batch_size
    ds, loaders = build_inria_loaders(cfg, args.inria_root, batch_size, args.val_stride)

    def make_model():
        return build_model(cfg.arch, cfg.encoder, cfg.encoder_weights).to(device)

    model = make_model()
    enc = "scratch" if cfg.encoder in (None, "scratch") else (
        f"{cfg.encoder} ({cfg.encoder_weights or 'random init'})")
    print(f"model:  {cfg.arch} / {enc}, norm='{cfg.stats_key}', "
          f"{count_parameters(model):,} trainable parameters")

    loss_fn = build_loss(cfg, loader=loaders["train"], device=device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(loaders["train"]))

    while True:
        try:
            history = training_model(
                loaders["train"], loss_fn, optimizer, model,
                num_epochs=cfg.epochs, scheduler=scheduler, val_loader=loaders["val"],
                cfg=cfg, device=device, threshold=cfg.threshold,
                resume=args.resume, progress=not args.no_progress,
                run_dir=(RUNS_ROOT / args.run_name) if args.run_name else None,
            )
            break
        except torch.cuda.OutOfMemoryError:
            batch_size //= 2
            if batch_size < 1:
                raise
            print(f"\n[oom] retrying with batch_size={batch_size}\n")
            torch.cuda.empty_cache()
            cfg.batch_size = batch_size
            ds, loaders = build_inria_loaders(cfg, args.inria_root, batch_size,
                                              args.val_stride)
            model = make_model()
            loss_fn = build_loss(cfg, loader=loaders["train"], device=device)
            optimizer = build_optimizer(model, cfg)
            scheduler = build_scheduler(optimizer, cfg,
                                        steps_per_epoch=len(loaders["train"]))

    best = Path(history.run_dir) / "best.pt"
    if best.exists():
        model.load_state_dict(torch.load(best, map_location=device, weights_only=True))
        print(f"\nloaded best checkpoint (epoch {history.best_epoch})")

    print("\n=== final metrics (Inria official val, correct eval harness) ===")
    final = {}
    for name in ("val",):
        res = evaluate(loaders[name], model, device=device, threshold=cfg.threshold)
        final[name] = res
        print(f"  {name:<6} IoU {res['iou']:.4f}  F1 {res['f1']:.4f}  "
              f"acc {res['accuracy']:.4f}  P {res['precision']:.4f}  "
              f"R {res['recall']:.4f}  (n={res['n_images']}, "
              f"undefined={res['iou_undefined']})")

    summary = {
        "run": Path(history.run_dir).name,
        "dataset": "inria-official",
        "best_val_iou": history.best_val_iou,
        "best_epoch": history.best_epoch,
        "epochs_run": len(history.epochs),
        "model": model_spec(cfg.arch, cfg.encoder, cfg.encoder_weights),
        "config": {"arch": cfg.arch, "encoder": cfg.encoder,
                   "encoder_weights": cfg.encoder_weights, "stats_key": cfg.stats_key,
                   "window": cfg.crop, "samples_per_tile": cfg.samples_per_tile,
                   "pos_weight": cfg.pos_weight, "dice_weight": cfg.dice_weight,
                   "lr": cfg.lr, "batch_size": cfg.batch_size,
                   "epochs": cfg.epochs, "patience": cfg.early_stop_patience},
        "metrics": {k: {m: float(v[m]) for m in
                        ("iou", "f1", "accuracy", "precision", "recall")}
                    for k, v in final.items()},
    }
    (Path(history.run_dir) / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nartefacts: {history.run_dir}")


if __name__ == "__main__":
    main()
