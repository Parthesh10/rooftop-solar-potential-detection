"""Crash-safe checkpointing, pause/stop control, and graceful interrupt.

Three separate needs, all served by files inside the run directory:

* **Resume after a crash or reboot.** ``save_state`` writes model + optimizer +
  scheduler + AMP scaler + epoch + history + RNG states, so a resumed run
  continues from exactly where it stopped rather than restarting the epoch with
  a fresh optimizer. Writes are **atomic** — a temp file plus ``os.replace`` —
  because the failure mode that matters here is losing power *during* a save,
  which a naive ``torch.save`` turns into a truncated, unloadable checkpoint.
  A ``.bak`` of the previous state is kept for the same reason.

* **Pause / resume on demand.** Create a ``PAUSE`` file in the run directory and
  the loop blocks at the next step boundary, holding VRAM but doing no work.
  Delete it and training continues. Nothing needs to be restarted.

* **Graceful stop.** Create a ``STOP`` file, or press Ctrl+C once, and the loop
  finishes the current step, saves a resumable checkpoint, and exits cleanly.
  A second Ctrl+C aborts immediately.
"""

from __future__ import annotations

import json
import os
import random
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

__all__ = ["RunControl", "save_state", "load_state", "find_latest_run", "STATE_NAME"]

STATE_NAME = "state.pt"
PAUSE_FILE = "PAUSE"
STOP_FILE = "STOP"


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #
def save_state(
    run_dir: str | Path,
    *,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    epoch: int = 0,
    history: dict | None = None,
    best_val_iou: float | None = None,
    best_epoch: int | None = None,
    config: dict | None = None,
    extra: dict | None = None,
    name: str = STATE_NAME,
) -> Path:
    """Write a fully resumable training state, atomically."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / name

    state: dict[str, Any] = {
        "format": 1,
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "history": history,
        "best_val_iou": best_val_iou,
        "best_epoch": best_epoch,
        "config": config,
        "extra": extra or {},
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "saved_at": time.time(),
    }

    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        torch.save(state, fh)
        fh.flush()
        os.fsync(fh.fileno())  # ← see below
    if path.exists():
        bak = path.with_suffix(path.suffix + ".bak")
        try:
            os.replace(path, bak)
        except OSError:
            pass
    os.replace(tmp, path)  # atomic on Windows and POSIX
    return path


# Why the fsync above is not optional
# -----------------------------------
# os.replace makes the *rename* atomic; it says nothing about whether the
# temp file's bytes ever reached the platter. On 2026-08-11 this machine hard
# crashed mid-run and left a 910-byte status.json consisting entirely of NUL
# bytes: the metadata operation had been committed while the data was still in
# the page cache. A NUL-filled status.json is merely confusing — a NUL-filled
# state.pt would be an unrecoverable checkpoint, which is the exact failure
# this module exists to prevent. flush + fsync before the rename closes it.


def load_state(path: str | Path, map_location=None) -> dict | None:
    """Load a state written by :func:`save_state`, falling back to the ``.bak``.

    Returns None if neither file is loadable, so a corrupt checkpoint downgrades
    to "start fresh" instead of crashing.
    """
    path = Path(path)
    for candidate in (path, path.with_suffix(path.suffix + ".bak")):
        if not candidate.exists():
            continue
        try:
            # weights_only=False: this payload includes RNG state and the config
            # dict, which the safe unpickler rejects. Only ever load your own
            # checkpoints with this.
            state = torch.load(candidate, map_location=map_location, weights_only=False)
            if candidate != path:
                print(f"[resume] primary checkpoint unreadable, used {candidate.name}")
            return state
        except Exception as exc:
            print(f"[resume] could not load {candidate.name}: {exc}")
    return None


def _as_cpu_byte(t):
    """RNG states must be CPU uint8 tensors.

    ``load_state`` passes ``map_location=device``, which moves *every* tensor in
    the payload to the GPU — including these — and ``set_rng_state`` then rejects
    them. Casting back is the fix; without it resume silently loses reproducible
    shuffling with only a warning.
    """
    if t is None:
        return None
    if isinstance(t, torch.Tensor):
        return t.detach().to(device="cpu", dtype=torch.uint8)
    return torch.tensor(t, dtype=torch.uint8, device="cpu")


def restore_rng(state: dict) -> None:
    rng = state.get("rng") or {}
    try:
        if rng.get("python"):
            random.setstate(rng["python"])
        if rng.get("numpy") is not None:
            np.random.set_state(rng["numpy"])
        if rng.get("torch") is not None:
            torch.set_rng_state(_as_cpu_byte(rng["torch"]))
        if rng.get("cuda") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([_as_cpu_byte(t) for t in rng["cuda"]])
    except Exception as exc:
        print(f"[resume] RNG restore skipped: {exc}")


def find_latest_run(runs_root: str | Path) -> Path | None:
    """Most recent run directory containing a resumable state file."""
    runs_root = Path(runs_root)
    if not runs_root.is_dir():
        return None
    candidates = [
        d for d in runs_root.iterdir()
        if d.is_dir() and (d / STATE_NAME).exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda d: (d / STATE_NAME).stat().st_mtime)


# --------------------------------------------------------------------------- #
# Pause / stop control
# --------------------------------------------------------------------------- #
class RunControl:
    """Watches for PAUSE / STOP files and traps Ctrl+C.

    Call :meth:`check` at a safe boundary — between steps or epochs. It blocks
    while paused and returns True when the caller should stop.
    """

    def __init__(self, run_dir: str | Path, poll_seconds: float = 1.0,
                 install_signal_handler: bool = True):
        self.run_dir = Path(run_dir)
        self.poll_seconds = poll_seconds
        self.pause_path = self.run_dir / PAUSE_FILE
        self.stop_path = self.run_dir / STOP_FILE
        self._interrupted = False
        self._force = False
        self._prev_handler = None

        if install_signal_handler:
            try:
                self._prev_handler = signal.signal(signal.SIGINT, self._on_sigint)
            except (ValueError, OSError):
                pass  # not on the main thread; skip

    def _on_sigint(self, signum, frame):
        if self._interrupted:
            print("\n[control] second interrupt — aborting now")
            if callable(self._prev_handler):
                self._prev_handler(signum, frame)
            raise KeyboardInterrupt
        self._interrupted = True
        print(
            "\n[control] interrupt received — finishing this step, saving a "
            "resumable checkpoint, then exiting. Press Ctrl+C again to abort."
        )

    @property
    def should_stop(self) -> bool:
        return self._interrupted or self.stop_path.exists()

    def check(self, on_pause=None, on_resume=None) -> bool:
        """Block while paused. Returns True if the caller should stop."""
        paused = False
        while self.pause_path.exists() and not self.should_stop:
            if not paused:
                paused = True
                print(
                    f"\n[control] paused — delete {self.pause_path} to resume "
                    f"(VRAM stays reserved)"
                )
                if on_pause:
                    on_pause()
            time.sleep(self.poll_seconds)
        if paused:
            print("[control] resumed")
            if on_resume:
                on_resume()
        return self.should_stop

    def close(self) -> None:
        if self._prev_handler is not None:
            try:
                signal.signal(signal.SIGINT, self._prev_handler)
            except (ValueError, OSError):
                pass
        for p in (self.pause_path, self.stop_path):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass

    def __enter__(self) -> "RunControl":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def write_control_help(run_dir: str | Path) -> None:
    """Drop a README in the run directory explaining the control files."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "CONTROL.md").write_text(
        f"""# Controlling this run

From any terminal, in this directory:

    # pause (blocks at the next step; VRAM stays reserved)
    New-Item PAUSE -ItemType File          # PowerShell
    touch PAUSE                            # bash

    # resume
    Remove-Item PAUSE                      # PowerShell
    rm PAUSE                               # bash

    # graceful stop: finishes the step, saves {STATE_NAME}, exits
    New-Item STOP -ItemType File

Ctrl+C once does the same as STOP. Twice aborts immediately.

## Resuming

    python scripts/train_swiss.py --resume            # newest run with a state file
    python scripts/train_swiss.py --resume {run_dir.name}

`{STATE_NAME}` holds model, optimizer, scheduler, AMP scaler, epoch, history and
RNG state, so a resumed run continues rather than restarting the epoch. It is
written atomically every epoch, with a `.bak` of the previous one — losing power
mid-save cannot leave you without a loadable checkpoint.
""",
        encoding="utf-8",
    )
