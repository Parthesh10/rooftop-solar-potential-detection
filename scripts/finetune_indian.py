r"""Fine-tune the shipped Inria checkpoint on hand-labelled Indian rooftops.

Prerequisite — a labelled directory (see finetune/README.md for the full
labelling workflow, from scripts/select_finetune_tiles.py through
process_data/labelme_to_masks.py):

    data/finetune_indian/images/<stem>.png
    data/finetune_indian/labels/<stem>_label.png

Then:

    python scripts/finetune_indian.py

This is CLAUDE.md's "cheapest first experiment" (next-steps #2): fine-tune at a
low learning rate with the encoder **frozen for the first few epochs**, so the
decoder adapts to what the encoder already extracts before the encoder itself
is allowed to drift — a ~100-tile dataset is nowhere near enough to retrain a
5.3 M-parameter EfficientNet-B0 encoder from a cold start without overfitting
or wrecking the Inria-trained features being fine-tuned *from*.

What this cannot do: turn ~100 tiles into a generalising model on its own.
Its job is to test the hypothesis from results/RESULTS.md — that the Bangalore
failure is a data problem, not a capacity one — and produce a checkpoint worth
comparing against the shipped one. Compare with:

    python scripts/eval_inria.py --limit-tiles 5     # catastrophic forgetting check
    # then re-run POST /api/calibrate on the same Bangalore block used for the
    # baseline measurement and compare recall by roof size.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch
from torch.utils.data import DataLoader

from config import TrainConfig
from loss.losses import build_loss
from model.registry import build_model
from process_data.data_loader import DataLoaderSegmentation
from train.train import build_optimizer, save_checkpoint, training_model, unwrap
from utils import count_parameters, get_device, seed_torch

MIN_TILES = 20   # below this a train/val split and a loss estimate are noise


def split_files(images_dir: Path, val_fraction: float, seed: int
                ) -> tuple[list[Path], list[Path]]:
    files = sorted(images_dir.glob("*.png"))
    if len(files) < MIN_TILES:
        raise SystemExit(
            f"only {len(files)} labelled tiles in {images_dir} (need at least "
            f"{MIN_TILES}). Label more with labelme first — see "
            f"finetune/README.md.")
    rng = random.Random(seed)
    shuffled = files[:]
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_fraction))
    return shuffled[n_val:], shuffled[:n_val]


def set_encoder_trainable(model: torch.nn.Module, trainable: bool) -> int:
    """Freeze/unfreeze ``model.encoder`` (every smp architecture has one).

    Returns the number of parameters whose ``requires_grad`` changed, so the
    caller can print something a human can sanity-check against "that sounds
    like the whole encoder".
    """
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        raise AttributeError(
            f"{type(model).__name__} has no .encoder — only smp architectures "
            f"(unet++, deeplabv3+, ...) support freeze/unfreeze here")
    n = 0
    for p in encoder.parameters():
        p.requires_grad = trainable
        n += p.numel()
    return n


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/finetune_indian")
    ap.add_argument("--checkpoint", default="results/unetpp_effb0_inria_20260903.pt",
                    help="the shipped checkpoint to fine-tune from")
    ap.add_argument("--arch", default="unet++")
    ap.add_argument("--encoder", default="efficientnet-b0")
    ap.add_argument("--stats-key", default="imagenet",
                    help="must match the checkpoint's manifest stats_key")
    ap.add_argument("--freeze-epochs", type=int, default=5,
                    help="epochs with the encoder frozen before unfreezing")
    ap.add_argument("--epochs", type=int, default=25,
                    help="total epochs, including the frozen ones")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--pos-weight", type=float, default=2.4,
                    help="BCE pos_weight. Defaults to the value already proven "
                        "on this project (hard-won fact #3) rather than "
                        "auto-estimating from a ~100-tile set, which is too "
                        "small for a stable estimate and historically "
                        "over-predicted at ~5.9")
    ap.add_argument("--dice-weight", type=float, default=0.6)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--amp", default="auto", choices=["auto", "fp16", "bf16", "off"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", default="finetune_indian")
    ap.add_argument("--out", default=None,
                    help="output checkpoint path (default: results/<run-name>.pt)")
    args = ap.parse_args()

    seed_torch(args.seed)
    device = get_device()
    print(f"device: {device}")

    data_dir = Path(args.data_dir)
    train_files, val_files = split_files(data_dir / "images", args.val_fraction,
                                         args.seed)
    print(f"labelled tiles: {len(train_files)} train / {len(val_files)} val")

    train_ds = DataLoaderSegmentation.from_files(
        train_files, augment=True, crop=None, stats_key=args.stats_key,
        mask_suffix="_label")
    val_ds = DataLoaderSegmentation.from_files(
        val_files, augment=False, crop=None, stats_key=args.stats_key,
        mask_suffix="_label")
    print(f"  train: {train_ds}")
    print(f"  val:   {val_ds}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers,
                              pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=args.workers)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    print(f"building {args.arch}/{args.encoder} and loading {ckpt_path.name} ...")
    model = build_model(args.arch, args.encoder, encoder_weights=None)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    print(f"  {count_parameters(model):,} parameters")

    cfg = TrainConfig(
        arch=args.arch, encoder=args.encoder, encoder_weights=None,
        stats_key=args.stats_key, lr=args.lr, batch_size=args.batch_size,
        pos_weight=args.pos_weight, dice_weight=args.dice_weight,
        epochs=args.epochs, early_stop_patience=args.patience,
        scheduler="none",   # a hand-labelled ~100-tile set is not where a
                            # cosine schedule earns its keep; a flat lr 1e-5
                            # keeps this experiment easy to reason about
        amp=args.amp, seed=args.seed,
    )
    loss_fn = build_loss(cfg, loader=train_loader, device=device)
    optimizer = build_optimizer(model, cfg)

    run_dir = REPO / "runs" / args.run_name

    n_frozen = set_encoder_trainable(model, trainable=False)
    n_total = count_parameters(model, trainable_only=False)
    print(f"\nphase 1: encoder frozen ({n_frozen:,}/{n_total:,} params not "
         f"trainable), epochs 0-{args.freeze_epochs}")
    training_model(
        train_loader, loss_fn, optimizer, model,
        num_epochs=args.freeze_epochs, val_loader=val_loader, cfg=cfg,
        device=device, run_dir=run_dir, resume=None,
    )

    set_encoder_trainable(model, trainable=True)
    print(f"\nphase 2: encoder unfrozen, epochs {args.freeze_epochs}-{args.epochs}")
    history = training_model(
        train_loader, loss_fn, optimizer, model,
        num_epochs=args.epochs, val_loader=val_loader, cfg=cfg,
        device=device, run_dir=run_dir, resume=run_dir,
    )

    out_path = Path(args.out) if args.out else REPO / "results" / f"{args.run_name}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(unwrap(model).state_dict(), out_path)
    (out_path.with_suffix(".metadata.json")).write_text(json.dumps({
        "base_checkpoint": str(ckpt_path.name),
        "arch": args.arch, "encoder": args.encoder, "stats_key": args.stats_key,
        "freeze_epochs": args.freeze_epochs, "total_epochs": args.epochs,
        "lr": args.lr, "pos_weight": args.pos_weight, "dice_weight": args.dice_weight,
        "n_train_tiles": len(train_files), "n_val_tiles": len(val_files),
        "best_val_iou": history.best_val_iou, "best_epoch": history.best_epoch,
    }, indent=2), encoding="utf-8")

    print(f"\nsaved: {out_path}")
    print(f"best val IoU on the held-out Indian tiles: "
         f"{history.best_val_iou:.4f} @ epoch {history.best_epoch}")
    print("\nNext steps:")
    print(f"  1. Check it didn't forget Inria:")
    print(f"       python scripts/eval_inria.py --limit-tiles 5")
    print(f"     (compare against the shipped 0.7712 pooled IoU in "
         f"results/RESULTS.md)")
    print(f"  2. Export and compare against the Bangalore baseline:")
    print(f"       python scripts/export_onnx.py --ckpt {out_path}")
    print(f"     then re-run POST /api/calibrate on the same block used in "
         f"results/RESULTS.md's 'Out-of-distribution recall, measured' "
         f"section and compare recall by roof size.")


if __name__ == "__main__":
    main()
