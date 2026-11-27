"""Helpers for summarising and plotting training runs (no TensorFlow needed)."""

from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

History = Dict[str, List[float]]


def summarize_history(history: History, test_loss: float, test_acc: float,
                      augmented: bool, aug_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Condense a Keras history dict + test results into one metrics dict.

    "Best epoch" is the epoch with the lowest validation loss, which is the
    epoch EarlyStopping restores.
    """
    best = int(np.argmin(history["val_loss"]))
    train_acc = float(history["accuracy"][best])
    val_acc = float(history["val_accuracy"][best])
    return {
        "augmented": bool(augmented),
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "best_epoch": best + 1,
        "epochs_run": len(history["val_loss"]),
        "train_accuracy_at_best": train_acc,
        "best_val_accuracy": val_acc,
        "best_val_loss": float(history["val_loss"][best]),
        "generalization_gap": train_acc - val_acc,   # train acc minus val acc: lower = less overfitting
        "augmentation": aug_config if augmented else None,
    }


def plot_history(history: History, path: str, title: str = "") -> None:
    """Save loss and accuracy curves (train vs validation) to `path`."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (tk, vk, name) in zip(
        axes, [("loss", "val_loss", "Loss"), ("accuracy", "val_accuracy", "Accuracy")]
    ):
        ax.plot(history[tk], label="train")
        ax.plot(history[vk], label="validation")
        ax.set_title(f"{name} {title}".strip())
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_augmentation_grid(originals: np.ndarray, augmented_versions: Sequence[np.ndarray],
                           labels: Sequence[int], path: str, class_names: Sequence[str]) -> None:
    """Row 0 = originals, rows 1..k = independent augmentations of the same images."""
    n = len(originals)
    rows = 1 + len(augmented_versions)
    fig, axes = plt.subplots(rows, n, figsize=(1.6 * n, 1.7 * rows))
    axes = np.atleast_2d(axes).reshape(rows, n)
    for j in range(n):
        axes[0, j].set_title(class_names[int(labels[j])], fontsize=8)
        for i, imgs in enumerate([originals] + list(augmented_versions)):
            axes[i, j].imshow(imgs[j])
            axes[i, j].axis("off")
    axes[0, 0].text(-0.15, 0.5, "original", transform=axes[0, 0].transAxes,
                    rotation=90, va="center", ha="right", fontsize=8)
    for i in range(1, rows):
        axes[i, 0].text(-0.15, 0.5, f"aug {i}", transform=axes[i, 0].transAxes,
                        rotation=90, va="center", ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
