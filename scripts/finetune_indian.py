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
import shutil
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
    ap.add_argument("--batch-size", type=int, default=4,
                    help="auto-halves on CUDA OOM rather than dying")
    ap.add_argument("--crop", type=int, default=256,
                    help="random crop side, a multiple of 32 (F-07). The tiles "
                        "are 512x512, but U-Net++'s dense skip connections at "
                        "that resolution overflow a 4 GB card — 256 fits, adds "
                        "crop augmentation, and costs nothing at serving time "
                        "because the network is fully convolutional. Pass 512 "
                        "on a bigger GPU to train at the native window size")
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
    ap.add_argument("--wandb", action="store_true",
                    help="log to Weights & Biases (needs `wandb login`); "
                        "no-ops if wandb is missing")
    ap.add_argument("--label-semantics", default="building-cluster-envelope",
                    choices=["building-footprint", "building-cluster-envelope"],
                    help="what the hand-drawn labels actually outline. "
                        "'building-cluster-envelope' means adjacent buildings "
                        "and the alleys between them were merged into one "
                        "polygon, so predicted area includes non-roof gaps and "
                        "needs a lower packing factor downstream. Recorded in "
                        "the checkpoint metadata so export_onnx.py and the app "
                        "cannot silently treat it as an Inria footprint model")
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

    if args.crop % 32:
        raise SystemExit(f"--crop must be a multiple of 32, got {args.crop} "
                         f"(the U-Net downsamples by 32; see F-07)")

    def build_loaders(batch_size: int):
        train_ds = DataLoaderSegmentation.from_files(
            train_files, augment=True, crop=args.crop, stats_key=args.stats_key,
            mask_suffix="_label")
        # Validation keeps the native 512 tile: the model is fully convolutional
        # and boundary-pads, so scoring at the serving size is both free and
        # more representative than scoring on crops.
        val_ds = DataLoaderSegmentation.from_files(
            val_files, augment=False, crop=None, stats_key=args.stats_key,
            mask_suffix="_label")
        return train_ds, val_ds, (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                       num_workers=args.workers,
                       pin_memory=torch.cuda.is_available()),
            DataLoader(val_ds, batch_size=1, shuffle=False,
                       num_workers=args.workers))

    batch_size = args.batch_size
    train_ds, val_ds, (train_loader, val_loader) = build_loaders(batch_size)
    print(f"  train: {train_ds}  (crop {args.crop})")
    print(f"  val:   {val_ds}  (native size)")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    run_dir = REPO / "runs" / args.run_name

    def make_model():
        model = build_model(args.arch, args.encoder, encoder_weights=None)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return model

    def run_two_phase(batch_size: int, train_loader, val_loader):
        """Freeze the encoder, train, unfreeze, continue. Returns the History.

        Both phases share one run directory: phase 2 resumes from the
        ``state.pt`` phase 1 left behind, which is what carries the optimizer
        state, the epoch counter and the best-checkpoint bookkeeping across the
        unfreeze. ``requires_grad`` is not part of a state_dict, so flipping it
        on the live model between calls is exactly what makes the two phases
        differ.
        """
        print(f"building {args.arch}/{args.encoder} and loading "
             f"{ckpt_path.name} ...")
        model = make_model()
        print(f"  {count_parameters(model):,} parameters")

        cfg = TrainConfig(
            arch=args.arch, encoder=args.encoder, encoder_weights=None,
            stats_key=args.stats_key, lr=args.lr, batch_size=batch_size,
            crop=args.crop,
            pos_weight=args.pos_weight, dice_weight=args.dice_weight,
            epochs=args.epochs, early_stop_patience=args.patience,
            scheduler="none",   # a hand-labelled ~100-tile set is not where a
                                # cosine schedule earns its keep; a flat lr 1e-5
                                # keeps this experiment easy to reason about
            amp=args.amp, seed=args.seed,
            wandb=args.wandb, wandb_project="rooftop-solar-finetune",
            wandb_run_name=args.run_name,
        )
        loss_fn = build_loss(cfg, loader=train_loader, device=device)
        optimizer = build_optimizer(model, cfg)

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
        print(f"\nphase 2: encoder unfrozen, epochs "
             f"{args.freeze_epochs}-{args.epochs}")
        history = training_model(
            train_loader, loss_fn, optimizer, model,
            num_epochs=args.epochs, val_loader=val_loader, cfg=cfg,
            device=device, run_dir=run_dir, resume=run_dir,
        )
        return model, history

    # Same guard train_swiss.py uses: a 4 GB card shared with the desktop can
    # lose to a background process at any moment, and dying after N epochs of
    # real work is not an acceptable failure mode.
    while True:
        try:
            model, history = run_two_phase(batch_size, train_loader, val_loader)
            break
        except torch.cuda.OutOfMemoryError:
            batch_size //= 2
            if batch_size < 1:
                raise
            print(f"\n[oom] retrying with batch_size={batch_size}\n")
            torch.cuda.empty_cache()
            # Start the run clean: a state.pt written at the larger batch size
            # would be resumed into a differently-shaped run.
            if run_dir.exists():
                shutil.rmtree(run_dir, ignore_errors=True)
            train_ds, val_ds, (train_loader, val_loader) = build_loaders(batch_size)

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
        # Carried into the ONNX sidecar by export_onnx.py. The app reads it to
        # decide what the predicted area *is* — a cluster envelope includes the
        # alleys between merged buildings, so it is not comparable to an Inria
        # footprint IoU and it needs a lower packing factor downstream.
        "label_semantics": args.label_semantics,
        "trained_on": "hand-labelled-indian",
        "not_comparable_to": (
            "Inria pooled IoU 0.7712 — different label semantics, different "
            "cities, and a val set of a few dozen hand-drawn tiles"),
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
