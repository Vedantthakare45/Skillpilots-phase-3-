"""Step 3: compile and train with early stopping.

Usage:
    python train.py
    python train.py --epochs 60 --batch-size 128 --patience 8
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras

from data import load_cifar10
from model import build_cnn


def parse_args():
    p = argparse.ArgumentParser(description="Train a CNN on CIFAR-10.")
    p.add_argument("--epochs", type=int, default=50, help="max epochs (early stopping may end sooner)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--patience", type=int, default=8, help="early-stopping patience (epochs)")
    p.add_argument("--val-split", type=float, default=0.1, help="fraction of train data used for validation")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def plot_history(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (train_key, val_key, title) in zip(
        axes, [("loss", "val_loss", "Loss"), ("accuracy", "val_accuracy", "Accuracy")]
    ):
        ax.plot(history[train_key], label="train")
        ax.plot(history[val_key], label="validation")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    keras.utils.set_random_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    (x_train, y_train), _ = load_cifar10()

    model = build_cnn()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",  # labels are integers 0-9
        metrics=["accuracy"],
    )
    model.summary()

    best_path = os.path.join(args.output_dir, "best_model.keras")
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.patience, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(best_path, monitor="val_loss", save_best_only=True),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_split=args.val_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)
    plot_history(history.history, os.path.join(args.output_dir, "training_curves.png"))
    print(f"Best model saved to {best_path}")
    print("Next: python evaluate.py && python export.py")


if __name__ == "__main__":
    main()
