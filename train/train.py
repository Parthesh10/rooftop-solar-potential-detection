"""Training loop.

Fixes, per potential-fixes.md:

* **F-06** — validation ran without ``torch.no_grad()``.
* **F-09** — the scheduler was ``StepLR(..., gamma=1)``, a no-op, so the
  "decreasing learning rate" the report describes never happened.
* **F-10** — the run that produced the shipped weights was launched *without* a
  ``val_loader``: no early stopping, no best-checkpoint selection.
* **F-15** — bare ``torch.squeeze()`` collapsed the batch dimension on a
  trailing batch of 1.
* **F-17** — ``Variable`` and ``.cuda()`` replaced by one resolved ``device``.
* **F-18** — checkpoints named via ``input()`` at a notebook prompt.
* **F-19** — no longer imports ``hyperparameters.select_param``.

Operationally it also provides:

* a live progress bar with loss, IoU, throughput and GPU/CPU/RAM metrics;
* **pause / resume** via a ``PAUSE`` file in the run directory;
* **graceful stop** via a ``STOP`` file or a single Ctrl+C;
* **crash-safe resume** — model, optimizer, scheduler, AMP scaler, epoch,
  history and RNG state are written atomically every epoch, so a reboot costs
  at most one epoch rather than the whole run.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch

from config import RUNS_ROOT, TrainConfig
from evaluate import evaluate
from runstate import (
    RunControl,
    find_latest_run,
    load_state,
    restore_rng,
    save_state,
    write_control_help,
)
from sysmon import GpuGovernor, SysMon, fmt_bytes
from tracking import Tracker
from utils import get_device, pad_to_multiple, select_amp, unpad

__all__ = [
    "History",
    "training_model",
    "build_optimizer",
    "build_scheduler",
    "save_checkpoint",
]


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

    @classmethod
    def from_dict(cls, d: dict) -> "History":
        h = cls()
        for k in ("train_loss", "val_loss", "train_iou", "val_iou", "lr", "epochs"):
            setattr(h, k, list(d.get(k) or []))
        h.best_epoch = d.get("best_epoch")
        bvi = d.get("best_val_iou")
        h.best_val_iou = float("-inf") if bvi is None else float(bvi)
        h.run_dir = d.get("run_dir")
        return h


# --------------------------------------------------------------------------- #
# Optimiser / scheduler
# --------------------------------------------------------------------------- #
def build_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    """AdamW at ``cfg.lr`` (default 3e-4).

    The 2023 run used ``Adam(lr=0.01)`` — 10-100x the usual U-Net setting — and
    F-02's accidental x4 loss scale pushed the effective rate to ~0.04.
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


def _tqdm(iterable=None, **kwargs):
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, **kwargs)
    except ImportError:
        return iterable if iterable is not None else _NullBar()


class _NullBar:
    """Stand-in so the loop works without tqdm installed."""

    def update(self, *a, **k): pass
    def set_postfix_str(self, *a, **k): pass
    def set_description(self, *a, **k): pass
    def close(self): pass
    def write(self, msg): print(msg)
    def __iter__(self): return iter(())


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
    resume: str | Path | bool | None = None,
    progress: bool = True,
) -> History:
    """Train ``model`` and return a :class:`History`.

    Backwards compatible with the old positional signature and with
    ``a, b, c, d = training_model(...)`` unpacking.

    Args:
        resume: ``True`` / ``"auto"`` picks the newest run directory containing a
            ``state.pt``; a path resumes that specific run. ``None`` starts fresh.
        progress: live progress bar with loss, IoU, throughput and GPU metrics.
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

    # Probe the requested autocast dtype against the real model before training
    # — a bad cuDNN fp16 kernel silently NaNs the whole forward pass. See
    # utils.select_amp for the measurements behind this.
    use_amp, amp_dtype = select_amp(
        model, device, cfg.amp if amp is None else amp,
        probe_shape=(2, 3, cfg.crop, cfg.crop),
    )
    # GradScaler is only needed for fp16; bf16 has fp32 range.
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype is torch.float16)

    # ---------------- resume ---------------- #
    history = History()
    start_epoch = 0
    resumed_from = None

    if resume:
        target = (
            find_latest_run(RUNS_ROOT)
            if resume is True or str(resume) in ("auto", "True")
            else Path(resume)
        )
        if target is not None and not (target / "state.pt").exists():
            alt = RUNS_ROOT / str(resume)
            target = alt if (alt / "state.pt").exists() else target
        state = load_state(Path(target) / "state.pt", map_location=device) if target else None
        if state is None:
            print(f"[resume] nothing resumable found ({target}) — starting fresh")
        else:
            run_dir = target
            model.load_state_dict(state["model"])
            if state.get("optimizer"):
                optimizer.load_state_dict(state["optimizer"])
            if state.get("scheduler") and scheduler is not None:
                scheduler.load_state_dict(state["scheduler"])
            if state.get("scaler"):
                scaler.load_state_dict(state["scaler"])
            if state.get("history"):
                history = History.from_dict(state["history"])
            start_epoch = int(state.get("epoch", -1)) + 1
            restore_rng(state)
            resumed_from = Path(target)
            print(
                f"[resume] {resumed_from.name}: continuing at epoch {start_epoch}"
                f"/{num_epochs} (best val IoU {history.best_val_iou:.4f} "
                f"@ epoch {history.best_epoch})"
            )

    if run_dir is None:
        run_dir = RUNS_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    history.run_dir = str(run_dir)
    write_control_help(run_dir)

    if start_epoch >= num_epochs:
        print(f"[train] already at epoch {start_epoch}/{num_epochs} — nothing to do")
        return history

    tracker = Tracker(
        enabled=cfg.wandb,
        project=cfg.wandb_project,
        run_name=cfg.wandb_run_name or run_dir.name,
        config={**cfg.to_dict(), "git_sha": _git_sha(), "device": str(device)},
    )
    mon = SysMon(enabled=True)
    gov = GpuGovernor(
        mon,
        util_target=cfg.gpu_util_target,
        temp_limit=cfg.gpu_temp_limit,
        mem_fraction=cfg.gpu_mem_fraction,
    ) if device.type == "cuda" else None

    if val_loader is None and verbose:
        print(
            "[train] WARNING: no val_loader — there will be no early stopping and "
            "no best-checkpoint selection. This is exactly the mistake that produced "
            "the 2023 weights (F-10)."
        )

    print(f"[train] run dir: {run_dir}")
    print(f"[train] pause: create {run_dir / 'PAUSE'}   stop: create {run_dir / 'STOP'} "
          f"(or Ctrl+C once)")

    epochs_since_improvement = 0
    stopped_early = False

    def _persist(epoch: int) -> None:
        save_state(
            run_dir, model=model, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler, epoch=epoch, history=history.to_dict(),
            best_val_iou=history.best_val_iou, best_epoch=history.best_epoch,
            config=cfg.to_dict(),
            extra={"git_sha": _git_sha(), "device": str(device), "amp": use_amp},
        )

    status_path = run_dir / "status.json"
    log_path = run_dir / "train.log"

    def _log(line: str) -> None:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line.rstrip() + "\n")

    def _status(**kw) -> None:
        """Heartbeat the dashboard reads. Written atomically; never fatal."""
        payload = {
            "run": run_dir.name, "updated": time.time(),
            "num_epochs": num_epochs, "device": str(device),
            "amp": bool(use_amp), "amp_dtype": str(amp_dtype),
            "batch_size": getattr(train_loader, "batch_size", None),
            "train_batches": len(train_loader),
            "best_val_iou": None if history.best_val_iou == float("-inf")
                            else history.best_val_iou,
            "best_epoch": history.best_epoch,
            "history": history.to_dict(),
            "sys": mon.sample().to_dict(),
            **kw,
        }
        try:
            tmp = status_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, default=str), encoding="utf-8")
            import os as _os
            _os.replace(tmp, status_path)
        except Exception:
            pass

    _log(f"[train] run {run_dir.name} starting at epoch {start_epoch}/{num_epochs}")
    control = RunControl(run_dir)
    outer = _tqdm(
        range(start_epoch, num_epochs), desc="epochs", unit="ep",
        dynamic_ncols=True, initial=0, disable=not progress,
    )

    try:
        for epoch in outer:
            model.train()
            mon.reset_peak()
            running_loss = 0.0
            n_batches = 0
            train_inter = 0
            train_union = 0
            n_seen = 0
            t0 = time.time()

            inner = _tqdm(
                train_loader, desc=f"ep {epoch:>3}", unit="b", leave=False,
                dynamic_ncols=True, disable=not progress,
            )

            last_status = 0.0
            last_ckpt = time.time()
            for step, (images, labels) in enumerate(inner):
                step_t0 = time.time()
                if control.check(
                    on_pause=lambda: _status(state="paused", epoch=epoch),
                    on_resume=lambda: _status(state="running", epoch=epoch),
                ):
                    break

                images = images.to(device, non_blocking=True)   # F-17
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    padded, pad = pad_to_multiple(images)
                    logits = unpad(model(padded), pad).squeeze(1)  # F-15: named dim
                    loss = loss_function(logits, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                n_batches += 1
                n_seen += images.shape[0]

                # Running IoU as global intersection/union — cheap, and it avoids
                # the per-tile averaging that F-08 distorted.
                with torch.no_grad():
                    pred = logits.detach().float() > 0.0  # logit>0 <=> sigmoid>0.5
                    gt = labels > 0.5
                    train_inter += int((pred & gt).sum().item())
                    train_union += int((pred | gt).sum().item())

                if scheduler is not None and isinstance(
                    scheduler, torch.optim.lr_scheduler.OneCycleLR
                ):
                    scheduler.step()

                now = time.time()
                ips = n_seen / max(now - t0, 1e-9)
                iou_so_far = (train_inter / train_union) if train_union else 0.0

                if progress and (step % 5 == 0 or step == len(train_loader) - 1):
                    inner.set_postfix_str(
                        f"loss {running_loss / n_batches:.4f} "
                        f"iou {iou_so_far:.4f} "
                        f"{ips:.1f} img/s | {mon.sample().compact()}"
                    )

                # Heartbeat ~1 Hz so the dashboard shows within-epoch progress,
                # not just a jump once per epoch.
                if now - last_status > 1.0:
                    last_status = now
                    _status(state="running", epoch=epoch, step=step + 1,
                            running_loss=running_loss / n_batches,
                            running_iou=iou_so_far, images_per_sec=ips,
                            throttled_seconds=(gov.throttled_seconds if gov else 0.0))

                # Mid-epoch crash insurance. Per-epoch saving alone meant a
                # machine crash inside epoch 0 left nothing resumable at all —
                # which is exactly what happened on 2026-08-11.
                if cfg.checkpoint_every_seconds and (
                    now - last_ckpt > cfg.checkpoint_every_seconds
                ):
                    last_ckpt = now
                    _persist(epoch - 1)  # resume re-runs this epoch from its start
                    _log(f"[ckpt] mid-epoch save at epoch {epoch} step {step + 1}")

                # Keep the card inside its thermal / utilisation envelope.
                if gov is not None:
                    gov.after_step(
                        time.time() - step_t0,
                        on_wait=lambda: _status(state="cooling", epoch=epoch,
                                                step=step + 1),
                    )

            inner.close()

            if control.should_stop:
                print(f"\n[control] stopping after epoch {epoch} — saving state")
                _log(f"[control] stopped during epoch {epoch}")
                _status(state="stopped", epoch=epoch)
                _persist(epoch - 1 if n_batches == 0 else epoch)
                break

            if scheduler is not None and not isinstance(
                scheduler, torch.optim.lr_scheduler.OneCycleLR
            ):
                scheduler.step()

            epoch_loss = running_loss / max(n_batches, 1)
            epoch_iou = (train_inter / train_union) if train_union else float("nan")
            secs = time.time() - t0

            history.epochs.append(epoch)
            history.train_loss.append(epoch_loss)
            history.train_iou.append(epoch_iou)
            history.lr.append(float(optimizer.param_groups[0]["lr"]))

            msg = (
                f"ep {epoch:>3}/{num_epochs - 1}  loss {epoch_loss:.4f}  "
                f"iou {epoch_iou:.4f}  lr {history.lr[-1]:.2e}  {secs:.0f}s"
            )

            improved = False
            val = None
            if val_loader is not None and (
                epoch % max(cfg.val_every, 1) == 0 or epoch == num_epochs - 1
            ):
                # One pass for both loss and metrics (F-06: under no_grad, F-03:
                # in eval mode).
                val = evaluate(
                    val_loader, model, device=device, threshold=threshold,
                    loss_fn=loss_function, progress=progress,
                )
                history.val_loss.append(val.get("loss", float("nan")))
                history.val_iou.append(val["iou"])
                msg += f"  |  val_loss {history.val_loss[-1]:.4f}  val_iou {val['iou']:.4f}"

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
                            "val_metrics": {
                                k: v for k, v in val.items() if not k.startswith("_")
                            },
                            "device": str(device),
                            "amp": use_amp, "amp_dtype": str(amp_dtype),
                        },
                    )
                    msg += "  *best*"

            peak = mon.peak_vram()
            if peak:
                msg += f"  peak {fmt_bytes(peak)}G"
            if gov is not None and gov.throttled_seconds > 0:
                msg += f"  throttle {gov.throttled_seconds:.0f}s"

            outer.write(msg) if hasattr(outer, "write") else print(msg)
            _log(msg)
            _status(state="running", epoch=epoch, step=len(train_loader))

            payload = {
                "train/loss": epoch_loss,
                "train/iou": epoch_iou,
                "lr": history.lr[-1],
                "epoch_seconds": secs,
                "gpu/peak_vram_gb": (peak / 1024**3) if peak else None,
            }
            if val is not None:
                payload.update({
                    "val/loss": history.val_loss[-1],
                    "val/iou": val["iou"],
                    "val/f1": val.get("f1"),
                    "val/precision": val.get("precision"),
                    "val/recall": val.get("recall"),
                    "val/undefined_tiles": val.get("iou_undefined"),
                })
            tracker.log(payload, step=epoch)

            # Resumable state every epoch: a reboot costs one epoch, not the run.
            _persist(epoch)

            epochs_since_improvement = 0 if improved else epochs_since_improvement + 1
            if (
                val_loader is not None
                and early_stop_patience
                and epochs_since_improvement >= early_stop_patience
            ):
                print(
                    f"[train] early stop at epoch {epoch}: no val IoU improvement in "
                    f"{early_stop_patience} epochs (best {history.best_val_iou:.4f} "
                    f"@ epoch {history.best_epoch})"
                )
                stopped_early = True
                break
    except KeyboardInterrupt:
        print("\n[train] aborted by user — last saved state is still resumable")
    finally:
        outer.close() if hasattr(outer, "close") else None
        save_checkpoint(model, run_dir, "last.pt")
        (run_dir / "history.json").write_text(
            json.dumps(history.to_dict(), indent=2), encoding="utf-8"
        )
        if history.best_epoch is not None:
            tracker.summary({
                "best/val_iou": history.best_val_iou,
                "best/epoch": history.best_epoch,
            })
            best_path = run_dir / "best.pt"
            if best_path.exists():
                tracker.save_artifact(best_path, name=f"unet-{run_dir.name}")
        tracker.finish()
        if gov is not None:
            _log(f"[gpu] {gov.summary()}")
        mon.close()
        control.close()

    _status(state="finished", epoch=history.epochs[-1] if history.epochs else start_epoch)
    _log("[train] finished")
    if verbose:
        print(f"[train] artefacts written to {run_dir}"
              + ("  (early stop)" if stopped_early else ""))
    return history
