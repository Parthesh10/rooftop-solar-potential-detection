"""The serving inference path: ONNX first, torch as a fallback.

Preprocessing constants are read from the model's sidecar manifest
(``<model>.json``, written by ``scripts/export_onnx.py``), **never** from a
literal here. That is the whole point: F-01 was a normalisation mismatch between
training and inference, and the only durable fix is one file that both sides
read.

Large images are handled by a sliding window with Hann-weighted blending, so
tile seams do not appear as a grid in the output mask. This mirrors
``infer.predict_large`` in the training repo; the two are kept behaviourally
identical and ``tests/test_webapp_inference.py`` asserts it.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from webapp.config import MODELS_DIR, SERVED_MODEL

__all__ = ["ModelBundle", "load_model", "predict_mask"]


@dataclass
class ModelBundle:
    """A loaded model plus everything needed to preprocess for it."""

    name: str
    manifest: dict
    runtime: str                     # "onnxruntime" | "torch"
    _session: object = None
    _torch_model: object = None
    _lock: threading.Lock = None

    @property
    def window(self) -> int:
        return int(self.manifest.get("window", 512))

    @property
    def stride(self) -> int:
        return int(self.manifest.get("stride", self.window // 2))

    @property
    def threshold(self) -> float:
        return float(self.manifest.get("threshold", 0.5))

    @property
    def mean(self) -> np.ndarray:
        return np.asarray(self.manifest["mean"], dtype=np.float32).reshape(3, 1, 1)

    @property
    def std(self) -> np.ndarray:
        return np.asarray(self.manifest["std"], dtype=np.float32).reshape(3, 1, 1)

    def card(self) -> dict:
        """The public model card — what /api/model returns."""
        m = self.manifest
        return {
            "name": m.get("name", self.name),
            "architecture": m.get("arch"),
            "encoder": m.get("encoder"),
            "runtime": self.runtime,
            "window": self.window,
            "threshold": self.threshold,
            "tta": bool(m.get("tta", False)),
            "trained_on": m.get("trained_on"),
            "gsd_m_per_px": m.get("gsd_m_per_px"),
            "serving_zoom": m.get("serving_zoom"),
            "label_semantics": m.get("label_semantics"),
            "recommended_packing_factor": m.get("recommended_packing_factor"),
            "metrics": m.get("metrics", {}),
            "limitations": m.get("limitations", []),
        }


def _sidecar_says_default(onnx: Path) -> bool:
    try:
        return bool(json.loads(
            onnx.with_suffix(".json").read_text(encoding="utf-8")).get("default"))
    except Exception:
        return False


def _find_model(models_dir: Path) -> tuple[Path | None, Path]:
    """Locate ``(onnx_path_or_None, sidecar_json)``.

    Priority: the ``RSOLAR_MODEL`` stem, then whichever sidecar declares
    ``"default": true``, then the newest file. Only the last of those is a
    guess, and it is the one that bites when a second model is exported — see
    ``config.SERVED_MODEL``.
    """
    onnx_files = sorted(models_dir.glob("*.onnx"),
                        key=lambda p: p.stat().st_mtime, reverse=True)

    if SERVED_MODEL:
        pinned = models_dir / f"{SERVED_MODEL}.onnx"
        if not pinned.exists():
            available = ", ".join(sorted(p.stem for p in onnx_files)) or "none"
            raise FileNotFoundError(
                f"RSOLAR_MODEL={SERVED_MODEL!r} but {pinned.name} is not in "
                f"{models_dir}. Exported models: {available}")
        return pinned, pinned.with_suffix(".json")

    for onnx in onnx_files:
        if _sidecar_says_default(onnx):
            return onnx, onnx.with_suffix(".json")

    if onnx_files:
        onnx = onnx_files[0]
        return onnx, onnx.with_suffix(".json")
    sidecars = sorted(models_dir.glob("*.json"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
    if sidecars:
        return None, sidecars[0]
    raise FileNotFoundError(
        f"no model in {models_dir}. Export one first:\n"
        f"  python scripts/export_onnx.py")


def load_model(models_dir: Path | None = None) -> ModelBundle:
    """Load the newest exported model. ONNX if present, else torch."""
    models_dir = Path(models_dir) if models_dir else MODELS_DIR
    onnx_path, sidecar = _find_model(models_dir)

    if not sidecar.exists():
        raise FileNotFoundError(
            f"model sidecar manifest missing: {sidecar}. It carries the "
            f"normalisation constants — refusing to guess them (see F-01).")
    manifest = json.loads(sidecar.read_text(encoding="utf-8"))

    for key in ("mean", "std", "window"):
        if key not in manifest:
            raise ValueError(f"sidecar {sidecar.name} has no '{key}'")

    if onnx_path is not None:
        try:
            import onnxruntime as ort

            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                         if p in ort.get_available_providers()]
            sess = ort.InferenceSession(str(onnx_path), so, providers=providers)
            return ModelBundle(name=onnx_path.stem, manifest=manifest,
                               runtime="onnxruntime", _session=sess,
                               _lock=threading.Lock())
        except ImportError:
            pass  # fall through to torch

    # Torch fallback: needs the training repo importable.
    import sys

    import torch

    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from model.registry import build_model

    ckpt_name = manifest.get("source_checkpoint")
    ckpt = repo / "results" / ckpt_name if ckpt_name else None
    if ckpt is None or not ckpt.exists():
        raise FileNotFoundError(
            f"neither an .onnx file nor the source checkpoint ({ckpt}) is "
            f"available. Run: python scripts/export_onnx.py")

    model = build_model(manifest.get("arch", "unet"),
                        manifest.get("encoder", "scratch"), encoder_weights=None)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()
    return ModelBundle(name=ckpt.stem, manifest=manifest, runtime="torch",
                       _torch_model=model, _lock=threading.Lock())


def _preprocess(patch: np.ndarray, bundle: ModelBundle) -> np.ndarray:
    """(H, W, 3) uint8 -> (1, 3, H, W) float32, normalised exactly as in training."""
    x = patch.astype(np.float32).transpose(2, 0, 1) / 255.0
    x = (x - bundle.mean) / bundle.std
    return x[None, ...]


def _raw_logits(batch: np.ndarray, bundle: ModelBundle) -> np.ndarray:
    """(N, 3, H, W) float32 -> (N, H, W) logits."""
    if bundle.runtime == "onnxruntime":
        with bundle._lock:
            out = bundle._session.run(None, {"input": batch})[0]
    else:
        import torch

        with bundle._lock, torch.no_grad():
            out = bundle._torch_model(torch.from_numpy(batch)).numpy()
    return out[:, 0] if out.ndim == 4 else out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable — a raw 1/(1+exp(-x)) overflows on saturated logits."""
    z = np.clip(x, -60, 60)
    pos = 1.0 / (1.0 + np.exp(-z))
    ez = np.exp(z)
    return np.where(x >= 0, pos, ez / (1.0 + ez))


# The 8 elements of the dihedral group D4: 4 rotations x optional mirror.
_TTA_D4 = ((0, False), (1, False), (2, False), (3, False),
           (0, True), (1, True), (2, True), (3, True))


def _forward(batch: np.ndarray, bundle: ModelBundle,
             tta: bool = False) -> np.ndarray:
    """(N, 3, H, W) float32 -> (N, H, W) probabilities.

    ``tta`` averages the model over all 8 dihedral transforms of the input,
    undoing each transform on the way out. Measured on the Inria val split:
    **IoU 0.7712 -> 0.7809, precision 0.845 -> 0.865** for 8x the compute. The
    averaging happens in logit space, before the sigmoid, which is the correct
    place — averaging probabilities pulls confident predictions toward 0.5.
    """
    if not tta:
        return _sigmoid(_raw_logits(batch, bundle))

    acc = None
    for k, flip in _TTA_D4:
        v = np.rot90(batch, k, axes=(-2, -1))
        if flip:
            v = v[..., ::-1]
        out = _raw_logits(np.ascontiguousarray(v), bundle)
        if flip:
            out = out[..., ::-1]
        out = np.rot90(out, -k, axes=(-2, -1))
        acc = out if acc is None else acc + out
    return _sigmoid(acc / len(_TTA_D4))


def _hann2d(n: int) -> np.ndarray:
    """2-D Hann window, floored so no pixel gets zero weight."""
    w = np.hanning(n + 2)[1:-1].astype(np.float32)
    return np.maximum(np.outer(w, w), 1e-3)


def predict_mask(image: np.ndarray, bundle: ModelBundle,
                 threshold: float | None = None,
                 tta: bool = False,
                 progress=None) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over a full mosaic. Returns ``(mask_bool, probs_float32)``.

    Overlapping windows are blended with a Hann weight rather than averaged
    uniformly or last-write-wins, either of which leaves visible seams.

    ``tta`` is worth ~+1 IoU and ~+2 precision for 8x the time. See ``_forward``.
    """
    thr = bundle.threshold if threshold is None else threshold
    win, stride = bundle.window, bundle.stride
    h, w = image.shape[:2]

    # Pad up to at least one window so small AOIs work unchanged.
    pad_h = max(win - h, 0)
    pad_w = max(win - w, 0)
    if pad_h or pad_w:
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    H, W = image.shape[:2]

    ys = list(range(0, max(H - win, 0) + 1, stride)) or [0]
    xs = list(range(0, max(W - win, 0) + 1, stride)) or [0]
    if ys[-1] + win < H:
        ys.append(H - win)
    if xs[-1] + win < W:
        xs.append(W - win)

    weight = _hann2d(win)
    acc = np.zeros((H, W), dtype=np.float32)
    wsum = np.zeros((H, W), dtype=np.float32)

    total = len(ys) * len(xs)
    done = 0
    for y in ys:
        for x in xs:
            patch = image[y:y + win, x:x + win]
            probs = _forward(_preprocess(patch, bundle), bundle, tta=tta)[0]
            ph, pw = patch.shape[:2]
            acc[y:y + ph, x:x + pw] += probs[:ph, :pw] * weight[:ph, :pw]
            wsum[y:y + ph, x:x + pw] += weight[:ph, :pw]
            done += 1
            if progress is not None:
                progress(done, total)

    probs = acc / np.maximum(wsum, 1e-6)
    probs = probs[:h, :w]
    return probs > thr, probs
