r"""Train the U-Net on the Swiss DOP25 set using the leakage-free geographic split.

Prerequisite — generate the split manifests once:

    python -m process_data.split --root data --out data/splits --block-size 1000 --buffer 125

Then:

    python scripts/train_swiss.py                 # defaults tuned for a 4 GB GPU
    python scripts/train_swiss.py --batch-size 4 --epochs 40
    python scripts/train_swiss.py --wandb

Pause / stop / resume (see runs/<ts>/CONTROL.md):

    New-Item runs\<ts>\PAUSE -ItemType File     # pause at the next step
    Remove-Item runs\<ts>\PAUSE                 # resume
    New-Item runs\<ts>\STOP  -ItemType File     # graceful stop (or Ctrl+C once)
    python scripts/train_swiss.py --resume      # continue the newest run

Notes for small GPUs — **measured** on a GTX 1650 4 GB (sm_75, cuDNN 9.10.2),
batch 8 @ 224, not assumed:

    fp32   497 ms/step   3.32 GB   correct
    bf16  1299 ms/step   1.48 GB   correct, but 2.6x SLOWER (emulated)
    fp16  1851 ms/step   3.15 GB   NaN on every step

fp16 comes back NaN because of a bad cuDNN kernel for 64->64 convolutions on
this architecture — the whole forward pass is NaN from the first block, and
neither cudnn.benchmark nor cudnn.deterministic changes it. So AMP is *not* a
free win on pre-Ampere cards: ``--amp auto`` (the default) turns it off there
and NaN-probes whatever dtype it does pick. ``--batch-size`` auto-halves on
CUDA OOM rather than dying.
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
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--crop", type=int, default=224)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--amp", default="auto", choices=["auto", "fp16", "bf16", "off"],
                    help="mixed precision. 'auto' disables it on pre-Ampere GPUs "
                         "and NaN-probes whatever it picks (default: auto)")
    ap.add_argument("--no-amp", action="store_true", help="alias for --amp off")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--resume", nargs="?", const="auto", default=None,
                    metavar="RUN",
                    help="resume: bare flag = newest run with a state.pt, "
                         "or pass a run directory name")
    ap.add_argument("--gpu-mem-fraction", type=float, default=0.9,
                    help="cap this process at a fraction of total VRAM (0 = off)")
    ap.add_argument("--gpu-util-target", type=float, default=80.0,
                    help="duty-cycle to roughly this average GPU utilisation "
                         "(100 = no throttling)")
    ap.add_argument("--gpu-temp-limit", type=float, default=78.0,
                    help="pause training above this GPU temperature in C (0 = off)")
    ap.add_argument("--checkpoint-every", type=float, default=60.0,
                    help="also save a resumable state this often mid-epoch, seconds")
    ap.add_argument("--no-progress", action="store_true",
                    help="disable the live progress bar (for piping to a log file)")
    args = ap.parse_args()

    cfg = TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, crop=args.crop,
        num_workers=args.workers, amp=("off" if args.no_amp else args.amp),
        seed=args.seed,
        early_stop_patience=args.patience, wandb=args.wandb,
        wandb_project="rooftop-solar", stats_key="all",
        gpu_mem_fraction=args.gpu_mem_fraction or None,
        gpu_util_target=(args.gpu_util_target if args.gpu_util_target < 100 else None),
        gpu_temp_limit=args.gpu_temp_limit or None,
        checkpoint_every_seconds=args.checkpoint_every,
    )

    if args.resume is None:
        from runstate import find_latest_run
        from config import RUNS_ROOT
        latest = find_latest_run(RUNS_ROOT)
        if latest is not None:
            print(f"note:   '{latest.name}' is resumable. Pass --resume to continue "
                  f"it instead of starting a new run.")

    device = get_device()
    seed_torch(cfg.seed, deterministic=False)  # benchmark mode: ~20-30% faster

    print(f"device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"gpu:    {props.name}  {props.total_memory / 1024**3:.1f} GB  "
              f"sm_{props.major}{props.minor}")
        if props.major < 8:
            print("note:   compute < 8.0 — no tensor cores. Measured on a GTX 1650: "
                  "fp32 497 ms/step, bf16 1299 ms (correct but emulated), "
                  "fp16 1851 ms and NaN. AMP='auto' therefore picks fp32 here.")

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
                resume=args.resume, progress=not args.no_progress,
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
