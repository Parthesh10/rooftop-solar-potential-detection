r"""Score a checkpoint on the Inria official val split, and sweep the cheap knobs.

Two post-hoc levers cost no retraining at all:

* **Threshold.** Everything so far reports 0.5. The shipped model has recall
  0.857 against precision 0.823, i.e. it over-predicts slightly — so a higher
  threshold may buy IoU for free.
* **Test-time augmentation.** Averaging the 8 dihedral transforms is typically
  worth +1-2 IoU for 8x the inference cost. Worth it offline; a toggle online.

Probabilities are computed **once** per window and every threshold is scored
against the same cached logits, so sweeping 17 thresholds costs one forward pass,
not 17.

Metrics are global (pooled intersection / union over the whole split), not a
mean of per-tile IoUs. Per-tile averaging is what F-08 distorted, and it weights
a tile with three roofs the same as a tile with three hundred.

    python scripts/eval_inria.py                          # threshold sweep
    python scripts/eval_inria.py --tta                    # + 8x dihedral TTA
    python scripts/eval_inria.py --ckpt results/other.pt --batch-size 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import norm_stats
from infer import load_manifest
from model.registry import build_model
from process_data.inria import InriaWindowDataset
from process_data.split import inria_official_split
from utils import get_device

DEFAULT_CKPT = "results/unetpp_effb0_inria_20260903.pt"

_TTA_D4 = ((0, False), (1, False), (2, False), (3, False),
           (0, True), (1, True), (2, True), (3, True))


@torch.inference_mode()
def _forward(model, x: torch.Tensor, tta: bool) -> torch.Tensor:
    if not tta:
        return torch.sigmoid(model(x))
    acc = None
    for k, flip in _TTA_D4:
        v = torch.rot90(x, k, dims=(-2, -1))
        if flip:
            v = torch.flip(v, dims=(-1,))
        out = model(v)
        if flip:
            out = torch.flip(out, dims=(-1,))
        out = torch.rot90(out, -k, dims=(-2, -1))
        acc = out if acc is None else acc + out
    return torch.sigmoid(acc / len(_TTA_D4))


def evaluate_thresholds(model, loader, device, thresholds, tta=False,
                        progress_every=25):
    """One forward pass; pooled TP/FP/FN accumulated per threshold."""
    n = len(thresholds)
    tp = np.zeros(n, dtype=np.int64)
    fp = np.zeros(n, dtype=np.int64)
    fn = np.zeros(n, dtype=np.int64)
    n_pos = n_px = 0

    model.eval()
    t0 = time.time()
    for i, (images, labels) in enumerate(loader):
        probs = _forward(model, images.to(device, non_blocking=True), tta)
        probs = probs[:, 0]
        gt = labels.to(device) > 0.5
        n_pos += int(gt.sum().item())
        n_px += int(gt.numel())

        for j, t in enumerate(thresholds):
            pred = probs > t
            tp[j] += int((pred & gt).sum().item())
            fp[j] += int((pred & ~gt).sum().item())
            fn[j] += int((~pred & gt).sum().item())

        if progress_every and (i + 1) % progress_every == 0:
            done = (i + 1) / len(loader)
            el = time.time() - t0
            print(f"    {done*100:5.1f}%  {el:6.1f}s elapsed, "
                  f"~{el/done - el:5.1f}s left", flush=True)

    rows = []
    for j, t in enumerate(thresholds):
        inter, union = tp[j], tp[j] + fp[j] + fn[j]
        iou = inter / union if union else float("nan")
        prec = tp[j] / (tp[j] + fp[j]) if (tp[j] + fp[j]) else float("nan")
        rec = tp[j] / (tp[j] + fn[j]) if (tp[j] + fn[j]) else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else float("nan")
        rows.append({"threshold": round(float(t), 3), "iou": float(iou),
                     "f1": float(f1), "precision": float(prec),
                     "recall": float(rec)})
    return rows, {"positive_pixel_rate": n_pos / n_px if n_px else float("nan"),
                  "seconds": time.time() - t0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--inria-root", default="../dataset/AerialImageDataset")
    ap.add_argument("--arch", default=None)
    ap.add_argument("--encoder", default=None)
    ap.add_argument("--stats-key", default=None)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tta", action="store_true", help="8x dihedral TTA")
    ap.add_argument("--limit-tiles", type=int, default=None,
                    help="score only the first N val tiles (quick check)")
    ap.add_argument("--out", default=None, help="write results as JSON here")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    if not ckpt.is_absolute():
        ckpt = REPO / ckpt
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")

    entry = load_manifest().get("models", {}).get(ckpt.name, {})
    arch = args.arch or entry.get("arch", "unet")
    encoder = args.encoder or entry.get("encoder", "scratch")
    stats_key = args.stats_key or entry.get("stats_key", "imagenet")

    root = Path(args.inria_root)
    if not root.is_absolute():
        root = (REPO / root).resolve()
    if (root / "train" / "images").is_dir():
        root = root / "train"

    files = sorted((root / "images").glob("*.tif"))
    if not files:
        raise SystemExit(f"no Inria tiles under {root / 'images'}")
    val_files = inria_official_split(files)["val"]
    if args.limit_tiles:
        val_files = val_files[:args.limit_tiles]

    ds = InriaWindowDataset(val_files, root / "gt", window=args.window,
                            augment=False, stats_key=stats_key,
                            deterministic=True, stride=args.stride)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers,
                        pin_memory=torch.cuda.is_available())

    device = get_device()
    model = build_model(arch, encoder, encoder_weights=None).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))

    print(f"checkpoint: {ckpt.name}")
    print(f"model:      {arch} / {encoder}   norm='{stats_key}'")
    print(f"device:     {device}")
    print(f"val:        {len(val_files)} tiles -> {len(ds)} windows "
          f"@ {args.window} stride {args.stride}")
    print(f"tta:        {'8x dihedral' if args.tta else 'off'}")
    print()

    thresholds = np.round(np.arange(0.20, 0.86, 0.05), 2)
    rows, meta = evaluate_thresholds(model, loader, device, thresholds, tta=args.tta)

    print()
    print("  thr     IoU      F1       P        R")
    print("  " + "-" * 42)
    best = max(rows, key=lambda r: r["iou"])
    for r in rows:
        star = "  <- best IoU" if r is best else ""
        print(f"  {r['threshold']:.2f}   {r['iou']:.4f}  {r['f1']:.4f}  "
              f"{r['precision']:.4f}  {r['recall']:.4f}{star}")

    at50 = next(r for r in rows if abs(r["threshold"] - 0.50) < 1e-6)
    print()
    print(f"  at 0.50 (shipped):  IoU {at50['iou']:.4f}  F1 {at50['f1']:.4f}  "
          f"P {at50['precision']:.4f}  R {at50['recall']:.4f}")
    print(f"  best  {best['threshold']:.2f}       :  IoU {best['iou']:.4f}  "
          f"F1 {best['f1']:.4f}  P {best['precision']:.4f}  R {best['recall']:.4f}")
    print(f"  delta               :  IoU {best['iou'] - at50['iou']:+.4f}  "
          f"F1 {best['f1'] - at50['f1']:+.4f}")
    print()
    print(f"  positive pixel rate {meta['positive_pixel_rate']:.4f} | "
          f"{meta['seconds']:.0f}s")

    payload = {"checkpoint": ckpt.name, "arch": arch, "encoder": encoder,
               "tta": args.tta, "window": args.window, "stride": args.stride,
               "n_tiles": len(val_files), "n_windows": len(ds),
               "rows": rows, "best": best, "at_0.5": at50, **meta}
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
