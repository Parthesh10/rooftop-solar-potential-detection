"""Single source of truth for paths, normalisation constants and defaults.

Every module that touches pixels reads its normalisation stats from here (or,
at inference time, from the model manifest written alongside a checkpoint).
Hardcoding these in more than one place is what caused F-01: the training
pipeline normalised to zero-mean/unit-std while the inference helper fed the
network raw 0-255 values.

See potential-fixes.md, F-01 / F-07 / F-09 / F-22.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Point DATA_ROOT at a canonical dataset location to avoid the duplicated copies
# described in F-23. Defaults to the in-repo `data/` for backwards compatibility.
DATA_ROOT = Path(os.environ.get("DATA_ROOT", REPO_ROOT / "data"))

RUNS_ROOT = Path(os.environ.get("RUNS_ROOT", REPO_ROOT / "runs"))


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
# Channel mean/std computed over each training split. Previously these lived as
# three hardcoded lines in data_loader.py with two commented out (F-01) — a
# silent accuracy killer if the wrong one shipped.
NORM_STATS: dict[str, dict[str, list[float]]] = {
    "all": {
        "mean": [0.4066, 0.4768, 0.4383],
        "std": [0.2121, 0.1899, 0.1618],
    },
    "residencial": {
        "mean": [0.3268, 0.5080, 0.3735],
        "std": [0.2853, 0.2395, 0.2063],
    },
    "industrial": {
        "mean": [0.3877, 0.5270, 0.4675],
        "std": [0.2972, 0.2621, 0.2189],
    },
    # ImageNet stats — use these when training with a pretrained encoder.
    "imagenet": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
}

DEFAULT_STATS_KEY = "all"


def norm_stats(key: str = DEFAULT_STATS_KEY) -> tuple[list[float], list[float]]:
    """Return ``(mean, std)`` for a named dataset. Raises on an unknown key."""
    if key not in NORM_STATS:
        raise KeyError(f"Unknown normalisation key {key!r}. Known: {sorted(NORM_STATS)}")
    s = NORM_STATS[key]
    return s["mean"], s["std"]


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
# The U-Net has 4 max-pools, so any input must divide by 16; we require 32 to
# leave headroom for deeper encoders. The legacy Swiss crops are 250x250, which
# divides by neither — so:
#
#   * training uses a random 224x224 crop (224 = 7 x 32), preserving native GSD;
#   * evaluation keeps the full 250x250 and pads to 256 at the model boundary,
#     then crops the prediction back to 250 (see utils.pad_to_multiple).
#
# Both paths therefore see identical ground-sample-distance, and nothing is
# discarded at eval time. See F-07.
TRAIN_CROP = 224
SIZE_DIVISOR = 32

# Inria tiles are 5000x5000 @ 0.3 m/px; 512 windows with 256 stride.
INRIA_WINDOW = 512
INRIA_STRIDE = 256

# Web Mercator zoom whose ground resolution best matches the training GSD.
# metres_per_pixel = 156543.0339 * cos(lat) / 2**z  ->  0.299 m/px at z=19, lat 0.
SERVING_ZOOM = 19


@dataclass
class TrainConfig:
    """Defaults for a training run. Overridable from the notebook or a CLI."""

    # model — F-04.2 (plan.md §4.2): a pretrained encoder is the change expected
    # to actually move test IoU past 0.5442. "scratch" keeps the verbatim 2023
    # U-Net as the control. See model/registry.py.
    arch: str = "unet"                     # "unet" | "unet++" | "deeplabv3+" | ...
    encoder: str = "scratch"               # "scratch" | "resnet34" | "efficientnet-b0" | ...
    encoder_weights: str | None = None     # "imagenet" to pull pretrained weights

    # data
    stats_key: str = DEFAULT_STATS_KEY
    train_dir: str = "train"
    val_dir: str = "val"
    test_dir: str = "test"
    crop: int = TRAIN_CROP
    batch_size: int = 4
    num_workers: int = 0  # >0 needs `if __name__ == "__main__"` guarding on Windows

    # Inria windowed sampling (scripts/train_inria.py). Ignored by train_swiss.
    samples_per_tile: int = 32   # windows drawn per 5000² tile per epoch
    min_positive: float = 0.005  # reject windows below this building-pixel fraction
    empty_ratio: float = 0.1     # ...but let this fraction of empty windows through

    # optimisation — F-09: was Adam @ lr=0.01 with a gamma=1 (no-op) scheduler.
    epochs: int = 80
    lr: float = 3e-4
    weight_decay: float = 1e-4
    scheduler: str = "cosine"  # "cosine" | "onecycle" | "step" | "none"
    warmup_epochs: int = 5

    # loss — F-02: pos_weight, not weight. None => computed from the training set.
    pos_weight: float | None = None
    dice_weight: float = 0.5

    # loop — F-10: validate every epoch, keep the best checkpoint, early-stop.
    val_every: int = 1
    early_stop_patience: int = 15
    threshold: float = 0.5

    seed: int = 0
    # "auto" | "fp16" | "bf16" | False. "auto" disables AMP on pre-Ampere GPUs
    # and probes the chosen dtype for NaN against the real model before
    # training — see utils.select_amp.
    amp: bool | str = "auto"

    # GPU safety limits — added after a hard machine crash during training on
    # 2026-08-11. See sysmon.GpuGovernor for what each one actually controls.
    gpu_mem_fraction: float | None = 0.9   # cap this process's VRAM share
    gpu_util_target: float | None = 80.0   # duty-cycle to this average utilisation
    gpu_temp_limit: float | None = 78.0    # pause above this, resume 6C lower

    # Crash resilience: also write a resumable state mid-epoch, this often.
    # Per-epoch alone loses everything if the machine dies inside epoch 0 —
    # which is what happened. 0 disables mid-epoch saves.
    checkpoint_every_seconds: float = 60.0

    # tracking — F-24. Requires `pip install wandb` and `wandb login`.
    # Degrades to a no-op if either is missing; runs/<ts>/history.json is
    # written either way, so nothing depends on the network.
    wandb: bool = False
    wandb_project: str = "rooftop-solar"
    wandb_run_name: str | None = None

    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


__all__ = [
    "REPO_ROOT",
    "DATA_ROOT",
    "RUNS_ROOT",
    "NORM_STATS",
    "DEFAULT_STATS_KEY",
    "norm_stats",
    "TRAIN_CROP",
    "SIZE_DIVISOR",
    "INRIA_WINDOW",
    "INRIA_STRIDE",
    "SERVING_ZOOM",
    "TrainConfig",
]
