r"""Export a trained checkpoint to ONNX for the serving path.

Why ONNX: the web API then needs neither torch nor
``segmentation_models_pytorch`` (together ~2.5 GB installed), starts in
milliseconds instead of seconds, and runs 2-3x faster on CPU. ``webapp`` prefers
the ONNX file and falls back to torch only if it is missing.

    python scripts/export_onnx.py                       # the shipped model
    python scripts/export_onnx.py --ckpt results/foo.pt --arch unet --encoder resnet34

Writes ``<out>.onnx`` plus ``<out>.json`` — the sidecar manifest carrying the
normalisation constants, window size, threshold and metrics. **The API reads
preprocessing from that sidecar, never from a literal in its own source**; that
is the F-01 class of bug and it must not be able to come back.
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

from config import norm_stats
from infer import load_manifest
from model.registry import build_model

DEFAULT_CKPT = "results/unetpp_effb0_inria_20260903.pt"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--arch", default=None, help="override the manifest entry")
    ap.add_argument("--encoder", default=None, help="override the manifest entry")
    ap.add_argument("--stats-key", default=None)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--out", default=None, help="output .onnx path")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--check", action="store_true", default=True,
                    help="verify ONNX output matches torch (default: on)")
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
    window = args.window or entry.get("window", 512)
    threshold = args.threshold if args.threshold is not None else entry.get("threshold", 0.5)

    out = Path(args.out) if args.out else (REPO / "webapp" / "models" / f"{ckpt.stem}.onnx")
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"checkpoint: {ckpt.name}")
    print(f"model:      {arch} / {encoder}   norm='{stats_key}'  window={window}")

    model = build_model(arch, encoder, encoder_weights=None)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 3, window, window)
    torch.onnx.export(
        model, dummy, str(out),
        input_names=["input"], output_names=["logits"],
        opset_version=args.opset,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        do_constant_folding=True,
    )
    print(f"wrote {out}  ({out.stat().st_size / 1024**2:.1f} MB)")

    mean, std = norm_stats(stats_key)
    sidecar = {
        "name": ckpt.stem,
        "onnx": out.name,
        "source_checkpoint": ckpt.name,
        "arch": arch,
        "encoder": encoder,
        "in_channels": 3,
        "out_channels": 1,
        "window": window,
        "stride": window // 2,
        "threshold": threshold,
        "tta": bool(entry.get("tta", False)),
        "stats_key": stats_key,
        "mean": mean,
        "std": std,
        "output": "logits",
        "gsd_m_per_px": entry.get("gsd_m_per_px", 0.3),
        "serving_zoom": entry.get("serving_zoom", 19),
        "label_semantics": entry.get("label_semantics", "building-footprint"),
        # A cluster-envelope model predicts roof *plus* the alleys between
        # merged buildings, so it needs a smaller packing factor than a
        # footprint model or the energy estimate inherits the gap area.
        "recommended_packing_factor": entry.get("recommended_packing_factor"),
        "default": bool(entry.get("default", False)),
        "trained_on": entry.get("trained_on"),
        "metrics": entry.get("metrics_verified", {}),
        "limitations": entry.get("limitations", []),
        "opset": args.opset,
    }
    side = out.with_suffix(".json")
    side.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(f"wrote {side}")

    if args.check:
        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError:
            print("[check] onnxruntime not installed — skipping parity check")
            return
        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])

        # What this checks, and why it is not the obvious thing.
        #
        # The original check compared raw LOGITS on a fresh torch.randn draw
        # against a hard 1e-3 tolerance. All three of those were wrong:
        #
        #   * Gaussian noise is nothing like normalised imagery, and a model
        #     trained on more varied data produces more extreme logits on
        #     garbage input, which amplifies harmless float differences.
        #   * Logits are unbounded, so an absolute tolerance on them has no
        #     fixed meaning. What is served is the *probability*.
        #   * A fresh random draw each run makes the verdict flaky.
        #
        # Measured 2026-09-08, that combination inverted the answer: it failed
        # joint_v3 (4 of 5 draws) and passed joint_v2, while on real imagery
        # joint_v3's probability error was 8.7e-05 against joint_v2's 4.2e-04
        # and joint_v3 flipped FEWER pixels (2 vs 3 per million). It blocked the
        # better export and shipped the worse one.
        #
        # So: fixed seed, an input shaped like normalised imagery, and gate on
        # the probability error plus the fraction of thresholded decisions that
        # actually change — which is the only thing a user can observe.
        torch.manual_seed(0)
        x = torch.randn(2, 3, window, window) * 0.9   # ~ImageNet-normalised range
        with torch.no_grad():
            ref = model(x).numpy()
        got = sess.run(None, {"input": x.numpy()})[0]

        sig = lambda z: 1.0 / (1.0 + np.exp(-z))
        logit_diff = float(np.abs(ref - got).max())
        prob_diff = float(np.abs(sig(ref) - sig(got)).max())
        flips = int(((sig(ref) > 0.5) != (sig(got) > 0.5)).sum())
        flip_frac = flips / ref.size

        ok = prob_diff < 1e-3 and flip_frac < 1e-4
        print(f"[check] max |torch - onnx|: logits {logit_diff:.2e}, "
              f"probabilities {prob_diff:.2e}")
        print(f"[check] thresholded decisions changed: {flips} / {ref.size:,} "
              f"({flip_frac:.1e})  {'OK' if ok else 'TOO LARGE'}")
        if not ok:
            raise SystemExit(
                "ONNX export does not match torch — do not ship this file. "
                f"probability error {prob_diff:.2e} (limit 1e-3), "
                f"decision flips {flip_frac:.1e} (limit 1e-4).")


if __name__ == "__main__":
    main()
