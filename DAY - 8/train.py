"""Train the CNN with or without imgaug augmentation.

Usage:
    python train.py                 # augmented  -> outputs/augmented/
    python train.py --no-augment    # baseline   -> outputs/baseline/
    python train.py --epochs 60 --batch-size 128 --patience 10
    python train.py --deterministic # exactly repeatable runs (slower on GPU)
"""

import argparse
import json
import os
from typing import Any, Dict, Optional

from augmentation import AugConfig, resolve_config
from reporting import plot_history, summarize_history
from seeding import set_global_seed


def train(augment: bool = True, epochs: int = 50, batch_size: int = 64, patience: int = 8,
          val_split: float = 0.1, output_dir: Optional[str] = None, seed: int = 42,
          aug_config: Optional[AugConfig] = None, deterministic: bool = False) -> Dict[str, Any]:
    """Train one model, save everything into `output_dir`, and return its metrics dict.

    Saved files: best_model.keras, history.json, metrics.json, training_curves.png.
    """
    from tensorflow import keras

    from data import load_cifar10_raw
    from data_pipeline import make_datasets
    from model import build_cnn

    output_dir = output_dir or os.path.join("outputs", "augmented" if augment else "baseline")
    os.makedirs(output_dir, exist_ok=True)
    cfg = resolve_config(aug_config)

    set_global_seed(seed, deterministic)
    (x_train, y_train), (x_test, y_test) = load_cifar10_raw()
    train_ds, val_ds, test_ds = make_datasets(
        x_train, y_train, x_test, y_test, batch_size=batch_size, augment=augment,
        val_split=val_split, seed=seed, aug_config=cfg, deterministic=deterministic,
    )

    model = build_cnn()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",   # integer labels 0-9
        metrics=["accuracy"],
    )

    best_path = os.path.join(output_dir, "best_model.keras")
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3,
                                          min_lr=1e-5, verbose=1),
        keras.callbacks.ModelCheckpoint(best_path, monitor="val_loss", save_best_only=True),
    ]

    print(f"\n=== Training ({'WITH' if augment else 'WITHOUT'} augmentation) -> {output_dir} ===")
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                        callbacks=callbacks, verbose=2)
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    metrics = summarize_history(hist, test_loss, test_acc, augment, cfg)
    metrics.update({"seed": seed, "deterministic": deterministic})

    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump(hist, f, indent=2)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    plot_history(hist, os.path.join(output_dir, "training_curves.png"),
                 title="(augmented)" if augment else "(baseline)")

    print(f"Test accuracy: {test_acc:.4f}   Best model: {best_path}")
    return metrics


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a CNN on CIFAR-10 with imgaug augmentation.")
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True,
                   help="use the imgaug pipeline (default on); --no-augment for the baseline")
    p.add_argument("--epochs", type=int, default=50, help="max epochs (early stopping may end sooner)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--patience", type=int, default=8, help="early-stopping patience (epochs)")
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--output-dir", default=None, help="default: outputs/augmented or outputs/baseline")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true",
                   help="exactly repeatable results (deterministic TF ops, single-thread augmentation)")
    return p.parse_args(argv)


def main() -> None:
    a = parse_args()
    train(a.augment, a.epochs, a.batch_size, a.patience, a.val_split, a.output_dir,
          a.seed, deterministic=a.deterministic)
    print("Next: python compare.py   (or: python evaluate.py && python export.py)")


if __name__ == "__main__":
    main()
