"""Training loop.

Fixes, per potential-fixes.md:

* **F-06** — validation ran without ``torch.no_grad()``, building a full
  autograd graph for every val forward pass.
* **F-09** — the scheduler was constructed as ``StepLR(..., gamma=1)``, a no-op,
  so the "decreasing learning rate" the report describes never happened. Real
  cosine / one-cycle / step schedules with warmup are available here.
* **F-10** — the run that produced the shipped weights was launched *without* a
  ``val_loader``, so ``history_val_*`` came back empty and there was no early
  stopping and no best-checkpoint selection: you got whatever epoch 130 left
  behind. Validation now runs every epoch by default and the best-by-val-IoU
  checkpoint is kept.
* **F-15** — ``torch.squeeze()`` with no dim removes *every* size-1 axis, so a
  trailing batch of 1 collapsed the batch dimension. Now ``squeeze(1)``.
* **F-17** — ``torch.autograd.Variable`` has been a no-op alias since PyTorch
  0.4, and the ``if torch.cuda.is_available()`` guards meant tensors were never
  moved at all on CPU-only machines. One resolved ``device`` is threaded through.
* **F-18** — checkpoints were named via ``input()`` at a notebook prompt,
  producing ``path raise 130.pt`` with no record of the config, data or metrics.
  Runs now write ``runs/<timestamp>/{best.pt,last.pt,metadata.json,history.json}``.
* **F-19** — this module no longer imports ``hyperparameters.select_param``,
  breaking the circular import; shared helpers live in ``utils`` and ``evaluate``.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from config import RUNS_ROOT, TrainConfig
from evaluate import evaluate
from utils import get_device, pad_to_multiple, unpad

__all__ = ["History", "training_model", "build_optimizer", "build_scheduler", "save_checkpoint"]


@dataclass
class History:
    """Training history.

    Supports tuple unpacking in the legacy order
    ``(train_loss, val_loss, train_iou, val_iou)`` so existing notebook cells
    keep working, while carrying the richer per-epoch record for new code.
    """

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_iou: list[float] = field(default_factory=list)
    val_iou: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)
    epochs: list[int] = field(default_factory=list)
    best_epoch: int | None = None
    best_val_iou: float = float("-inf")
    run_dir: str | None = None

    def __iter__(self):
        return iter((self.train_loss, self.val_loss, self.train_iou, self.val_iou))

    def to_dict(self) -> dict:
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_iou": self.train_iou,
            "val_iou": self.val_iou,
            "lr": self.lr,
            "epochs": self.epochs,
            "best_epoch": self.best_epoch,
            "best_val_iou": None if self.best_val_iou == float("-inf") else self.best_val_iou,
            "run_dir": self.run_dir,
        }


# --------------------------------------------------------------------------- #
# Optimiser / scheduler
# --------------------------------------------------------------------------- #
def build_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    """AdamW at ``cfg.lr`` (default 3e-4).

    The 2023 run used ``Adam(lr=0.01)`` — 10-100x the usual U-Net setting — and
    F-02's accidental x4 loss scale pushed the effective rate to ~0.04. The
    training curve (loss still falling and IoU still climbing at epoch 125)
    is the signature of a run that never converged.
    """
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: TrainConfig, steps_per_epoch: int = 1
):
    """Per-epoch LR schedule. Returns None for ``cfg.scheduler == 'none'``."""
    kind = (cfg.scheduler or "none").lower()
    if kind == "none":
        return None
    if kind == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
    if kind == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=cfg.lr, epochs=cfg.epochs,
            steps_per_epoch=max(steps_per_epoch, 1),
        )
    if kind == "cosine":
        warmup = max(int(cfg.warmup_epochs), 0)

        def lr_lambda(epoch: int) -> float:
            if warmup and epoch < warmup:
                return (epoch + 1) / warmup
            span = max(cfg.epochs - warmup, 1)
            progress = min((epoch - warmup) / span, 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    raise ValueError(f"unknown scheduler {cfg.scheduler!r}")


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #
def save_checkpoint(
    model: torch.nn.Module,
    run_dir: Path,
    name: str,
    metadata: dict | None = None,
) -> Path:
    """Save weights plus a self-describing ``metadata.json`` (F-18)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / name
    torch.save(model.state_dict(), path)
    if metadata is not None:
        (run_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str), encoding="utf-8"
        )
    return path


def _git_sha() -> str | None:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(Path(__file__).resolve().parents[1]),
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def training_model(
    train_loader,
    loss_function,
    optimizer,
    model,
    num_epochs: int | None = None,
    scheduler=None,
    val_loader=None,
    *,
    cfg: TrainConfig | None = None,
    device=None,
    run_dir: str | Path | None = None,
    threshold: float = 0.5,
    amp: bool | None = None,
    early_stop_patience: int | None = None,
    verbose: bool = True,
) -> History:
    """Train ``model`` and return a :class:`History`.

    Backwards compatible with the old positional signature
    ``training_model(train_loader, loss_function, optimizer, model, num_epochs)``
    and with ``a, b, c, d = training_model(...)`` unpacking.

    Unlike the original, this validates **every epoch** (not every 25), keeps the
    best-by-val-IoU checkpoint, and early-stops (F-10).
    """
    cfg = cfg or TrainConfig()
    if num_epochs is None:
        num_epochs = cfg.epochs
    if early_stop_patience is None:
        early_stop_patience = cfg.early_stop_patience

    device = torch.device(device) if device is not None else get_device()
    model = model.to(device)
    if hasattr(loss_function, "to"):
        loss_function = loss_function.to(device)

    use_amp = cfg.amp if amp is None else amp
    use_amp = bool(use_amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if run_dir is None:
        run_dir = RUNS_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    history = History(run_dir=str(run_dir))
    if val_loader is None and verbose:
        print(
            "[train] WARNING: no val_loader — there will be no early stopping and "
            "no best-checkpoint selection. This is exactly the mistake that produced "
            "the 2023 weights (F-10)."
        )

    epochs_since_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0
        train_inter = 0
        train_union = 0
        t0 = time.time()

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)   # F-17
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                padded, pad = pad_to_multiple(images)
                logits = unpad(model(padded), pad).squeeze(1)  # F-15: named dim
                loss = loss_function(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            n_batches += 1

            # Running IoU accumulated as global intersection/union — cheap, and
            # it avoids the per-tile averaging that F-08 distorted.
            with torch.no_grad():
                pred = logits.detach().float() > 0.0  # logit>0  <=>  sigmoid>0.5
                gt = labels > 0.5
                train_inter += int((pred & gt).sum().item())
                train_union += int((pred | gt).sum().item())

            if scheduler is not None and isinstance(
                scheduler, torch.optim.lr_scheduler.OneCycleLR
            ):
                scheduler.step()

        if scheduler is not None and not isinstance(
            scheduler, torch.optim.lr_scheduler.OneCycleLR
        ):
            scheduler.step()

        epoch_loss = running_loss / max(n_batches, 1)
        epoch_iou = (train_inter / train_union) if train_union else float("nan")

        history.epochs.append(epoch)
        history.train_loss.append(epoch_loss)
        history.train_iou.append(epoch_iou)
        history.lr.append(float(optimizer.param_groups[0]["lr"]))

        msg = (
            f"epoch {epoch:>4}/{num_epochs}  loss {epoch_loss:.4f}  "
            f"train_iou {epoch_iou:.4f}  lr {history.lr[-1]:.2e}  "
            f"{time.time() - t0:.1f}s"
        )

        improved = False
        if val_loader is not None and (epoch % max(cfg.val_every, 1) == 0
                                       or epoch == num_epochs - 1):
            val_loss = _validate_loss(model, val_loader, loss_function, device, use_amp)
            val = evaluate(val_loader, model, device=device, threshold=threshold)
            history.val_loss.append(val_loss)
            history.val_iou.append(val["iou"])
            msg += f"  |  val_loss {val_loss:.4f}  val_iou {val['iou']:.4f}"

            if val["iou"] > history.best_val_iou:
                history.best_val_iou = float(val["iou"])
                history.best_epoch = epoch
                improved = True
                save_checkpoint(
                    model, run_dir, "best.pt",
                    metadata={
                        "created": datetime.now().isoformat(timespec="seconds"),
                        "git_sha": _git_sha(),
                        "config": cfg.to_dict(),
                        "epoch": epoch,
                        "val_metrics": {k: v for k, v in val.items() if not k.startswith("_")},
                        "device": str(device),
                        "amp": use_amp,
                    },
                )
                msg += "  *best*"

        if verbose:
            print(msg)

        epochs_since_improvement = 0 if improved else epochs_since_improvement + 1
        if (
            val_loader is not None
            and early_stop_patience
            and epochs_since_improvement >= early_stop_patience
        ):
            if verbose:
                print(
                    f"[train] early stop at epoch {epoch}: no val IoU improvement in "
                    f"{early_stop_patience} epochs (best {history.best_val_iou:.4f} "
                    f"@ epoch {history.best_epoch})"
                )
            break

    save_checkpoint(model, run_dir, "last.pt")
    (run_dir / "history.json").write_text(
        json.dumps(history.to_dict(), indent=2), encoding="utf-8"
    )
    if verbose:
        print(f"[train] artefacts written to {run_dir}")

    return history


@torch.no_grad()  # F-06
def _validate_loss(model, loader, loss_function, device, use_amp: bool) -> float:
    model.eval()  # F-03
    total, n = 0.0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            padded, pad = pad_to_multiple(images)
            logits = unpad(model(padded), pad).squeeze(1)
            total += float(loss_function(logits, labels).item())
        n += 1
    model.train()
    return total / max(n, 1)
