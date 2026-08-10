"""Training-curve plots."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

__all__ = ["plot_train_val", "plot_history"]


def plot_train_val(m_train, m_val, period=1, al_param=False, metric="IoU", save=None):
    """Plot a metric's evolution on train and validation.

    Args:
        m_train: history of the metric on the training set.
        m_val: history of the metric on the validation set. May be empty — in
            which case a warning is printed rather than silently drawing a
            single-point line. In 2023 this was *always* empty, because the run
            was launched without a val_loader, and the empty series went into
            report Figures 15-20 unnoticed (F-10).
        period: epochs between two evaluations, for the x axis.
        al_param: epochs per learning rate, drawn as vertical rules.
        metric: axis label.
        save: path to write the figure to. ``None`` skips saving — the original
            unconditionally wrote ``evol_<metric>.png`` into the repo root.
    """
    m_train = list(m_train or [])
    m_val = list(m_val or [])

    if not m_val:
        print(
            f"[plots] WARNING: the validation history for '{metric}' is empty. "
            "The model was trained without a val_loader, so there was no early "
            "stopping and no best-checkpoint selection (F-10)."
        )

    plt.figure(figsize=(8, 5))
    plt.title(f"Evolution of the {metric} with respect to the number of epochs", fontsize=13)

    if al_param:
        steps = np.arange(1, int(len(m_train) * period / al_param) + 1) * al_param
        for step in steps:
            plt.axvline(step, color="black", lw=0.7)

    if m_train:
        plt.plot(np.arange(len(m_train)) * period, m_train,
                 color="tab:blue", marker="o", ms=3, ls=":", label=f"{metric} train")
    if m_val:
        plt.plot(np.arange(len(m_val)) * period, m_val,
                 color="tab:red", marker="o", ms=3, ls=":", label=f"{metric} val")

    plt.xlabel("Number of epochs")
    plt.ylabel(metric)
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.show()


def plot_history(history, save_prefix: str | None = None):
    """Plot loss, IoU and learning rate from a :class:`train.train.History`.

    Also accepts a plain dict with the same keys (e.g. a loaded history.json).
    """
    h = history.to_dict() if hasattr(history, "to_dict") else dict(history)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(h["train_loss"], color="tab:blue", label="train")
    if h.get("val_loss"):
        axes[0].plot(h["val_loss"], color="tab:red", label="val")
    axes[0].set_title("Loss")

    axes[1].plot(h["train_iou"], color="tab:blue", label="train")
    if h.get("val_iou"):
        axes[1].plot(h["val_iou"], color="tab:red", label="val")
    best = h.get("best_epoch")
    if best is not None:
        axes[1].axvline(best, color="black", ls="--", lw=0.8,
                        label=f"best @ {best} ({h.get('best_val_iou', float('nan')):.4f})")
    axes[1].set_title("IoU")

    axes[2].plot(h.get("lr", []), color="tab:green")
    axes[2].set_title("Learning rate")
    axes[2].set_yscale("log")

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
    for ax in axes[:2]:
        ax.legend(loc="best")

    fig.tight_layout()
    if save_prefix:
        fig.savefig(f"{save_prefix}_history.png", dpi=150, bbox_inches="tight")
    plt.show()
