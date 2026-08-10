"""Optional experiment tracking (F-24).

The 2023 runs were tracked by hand-naming ``.npy`` files —
``history_train_ioubatch5loss4_1000.npy`` — and the "loss4" part referred to
the broken ``weight=`` argument from F-02, so 40+ files in ``plots/`` are named
after a quantity that was never being swept.

This wraps Weights & Biases so a run records its own config, metrics, curves,
environment and git SHA. It is entirely optional: if ``wandb`` is not installed
or the user is not logged in, :class:`Tracker` degrades to a no-op and training
proceeds unchanged. Local ``runs/<ts>/history.json`` is written either way, so
nothing depends on the network.

Setup
-----
    uv pip install wandb
    wandb login          # paste the key from https://wandb.ai/authorize

Then set ``TrainConfig(wandb=True)`` (or ``RSOLAR_WANDB=1`` in the environment).
"""

from __future__ import annotations

import os
from typing import Any

__all__ = ["Tracker"]


class Tracker:
    """Thin, failure-tolerant wrapper around ``wandb``.

    Every method is a no-op when tracking is disabled or unavailable. Any
    exception from the tracker is swallowed with a warning — a logging backend
    should never be able to kill a training run.
    """

    def __init__(
        self,
        enabled: bool = False,
        project: str = "rooftop-solar",
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ):
        self.run = None
        self.enabled = bool(enabled) or os.environ.get("RSOLAR_WANDB") == "1"
        if not self.enabled:
            return

        try:
            import wandb
        except ImportError:
            print("[tracking] wandb not installed — skipping. `uv pip install wandb`")
            self.enabled = False
            return

        try:
            self.run = wandb.init(
                project=project, name=run_name, config=config or {}, tags=tags or [],
            )
            print(f"[tracking] wandb run: {self.run.url}")
        except Exception as exc:  # not logged in, offline, quota, ...
            print(f"[tracking] wandb init failed ({exc}) — continuing without it. "
                  "Run `wandb login` if you want tracking.")
            self.enabled = False
            self.run = None

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if not self.run:
            return
        try:
            self.run.log({k: v for k, v in metrics.items() if v is not None}, step=step)
        except Exception as exc:
            print(f"[tracking] log failed: {exc}")

    def summary(self, values: dict[str, Any]) -> None:
        if not self.run:
            return
        try:
            self.run.summary.update(values)
        except Exception as exc:
            print(f"[tracking] summary update failed: {exc}")

    def save_artifact(self, path: "str | os.PathLike[str]", name: str,
                      kind: str = "model") -> None:
        """Upload a checkpoint. Skipped silently when tracking is off."""
        if not self.run:
            return
        try:
            import wandb

            art = wandb.Artifact(name, type=kind)
            art.add_file(str(path))
            self.run.log_artifact(art)
        except Exception as exc:
            print(f"[tracking] artifact upload failed: {exc}")

    def finish(self) -> None:
        if not self.run:
            return
        try:
            self.run.finish()
        except Exception as exc:
            print(f"[tracking] finish failed: {exc}")
        finally:
            self.run = None

    def __enter__(self) -> "Tracker":
        return self

    def __exit__(self, *exc_info) -> None:
        self.finish()
