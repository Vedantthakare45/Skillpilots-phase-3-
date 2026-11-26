"""Step 4: evaluate accuracy on the test set and plot a confusion matrix.

Usage:
    python evaluate.py
    python evaluate.py --model outputs/best_model.keras
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from data import CLASS_NAMES


def plot_confusion_matrix(cm, class_names, path, normalize=True):
    """Save a labelled confusion-matrix heatmap. Rows = actual, columns = predicted."""
    cm = np.asarray(cm, dtype=float)
    shown = cm / cm.sum(axis=1, keepdims=True) if normalize else cm

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix" + (" (row-normalised)" if normalize else ""))
    threshold = shown.max() / 2
    for i in range(shown.shape[0]):
        for j in range(shown.shape[1]):
            text = f"{shown[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if shown[i, j] > threshold else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Evaluate the trained CNN on the CIFAR-10 test set.")
    p.add_argument("--model", default="outputs/best_model.keras")
    p.add_argument("--output-dir", default="outputs")
    args = p.parse_args()

    from tensorflow import keras
    from data import load_cifar10

    if not os.path.exists(args.model):
        raise SystemExit(f"error: model not found: {args.model}. Run train.py first.")
    os.makedirs(args.output_dir, exist_ok=True)

    _, (x_test, y_test) = load_cifar10()
    model = keras.models.load_model(args.model)

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss:     {loss:.4f}")
    print(f"Test accuracy: {acc:.4f}")

    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=3)
    print(report)
    with open(os.path.join(args.output_dir, "classification_report.txt"), "w") as f:
        f.write(f"Test accuracy: {acc:.4f}\n\n{report}")

    cm = confusion_matrix(y_test, y_pred)
    path = os.path.join(args.output_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, CLASS_NAMES, path)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
