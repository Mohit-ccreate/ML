"""Plotting helpers. All functions save to disk and never call plt.show(),
so they work in headless environments (CI, servers, this sandbox)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")  # headless backend — must come before pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

from .data import CLASSES, MEAN, STD  # noqa: E402


def _denormalize(img: torch.Tensor) -> np.ndarray:
    """Undo Normalize(MEAN, STD) and return an HxW float array in [0, 1]."""
    img = img.clone().squeeze(0) * STD[0] + MEAN[0]
    return img.clamp(0, 1).numpy()


def plot_history(history: List[Dict[str, float]], out_path: str | Path) -> Path:
    """Loss and accuracy curves for train vs. validation, side by side."""
    epochs = [h["epoch"] for h in history]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))

    ax_loss.plot(epochs, [h["train_loss"] for h in history], marker="o", label="train")
    ax_loss.plot(epochs, [h["val_loss"] for h in history], marker="o", label="val")
    ax_loss.set(title="Cross-entropy loss", xlabel="epoch", ylabel="loss")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend()

    ax_acc.plot(epochs, [100 * h["train_acc"] for h in history], marker="o", label="train")
    ax_acc.plot(epochs, [100 * h["val_acc"] for h in history], marker="o", label="val")
    ax_acc.set(title="Accuracy", xlabel="epoch", ylabel="accuracy (%)")
    ax_acc.grid(alpha=0.3)
    ax_acc.legend()

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str | Path,
    class_names: Sequence[str] = CLASSES,
    normalize: bool = True,
) -> Path:
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ticks = np.arange(len(class_names))
    ax.set(
        xticks=ticks,
        yticks=ticks,
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion matrix" + (" (row-normalised)" if normalize else ""),
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f"{cm[i, j]:.2f}" if normalize else f"{cm[i, j]:d}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_sample_grid(
    images: torch.Tensor,
    y_true: Sequence[int],
    y_pred: Sequence[int] | None,
    out_path: str | Path,
    class_names: Sequence[str] = CLASSES,
    ncols: int = 8,
    title: str | None = None,
) -> Path:
    """Grid of images with true (and optionally predicted) labels.
    Correct predictions are titled in green, mistakes in red."""
    n = len(images)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.6 * ncols, 1.9 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for k, ax in enumerate(axes):
        ax.axis("off")
        if k >= n:
            continue
        ax.imshow(_denormalize(images[k]), cmap="gray")
        t = class_names[int(y_true[k])]
        if y_pred is None:
            ax.set_title(t, fontsize=8)
        else:
            p = class_names[int(y_pred[k])]
            ok = int(y_true[k]) == int(y_pred[k])
            ax.set_title(f"{p}\n({t})" if not ok else p, fontsize=7, color="green" if ok else "red")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
