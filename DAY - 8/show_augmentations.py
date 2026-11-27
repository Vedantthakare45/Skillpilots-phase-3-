"""Preview what the imgaug pipeline does to CIFAR-10 images.

Usage:
    python show_augmentations.py
    python show_augmentations.py --n 10 --versions 3
Saves outputs/augmentation_examples.png (top row = originals, other rows = augmented).
"""

import argparse
import os

from augmentation import AUG_CONFIG, augment_batch, build_augmenter
from data import CLASS_NAMES
from reporting import plot_augmentation_grid


def main() -> None:
    p = argparse.ArgumentParser(description="Visualise the augmentation pipeline.")
    p.add_argument("--n", type=int, default=8, help="images to show")
    p.add_argument("--versions", type=int, default=3, help="augmented versions per image")
    p.add_argument("--output", default="outputs/augmentation_examples.png")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    from data import load_cifar10_raw

    (x_train, y_train), _ = load_cifar10_raw()
    images, labels = x_train[: a.n], y_train[: a.n]

    augmenter = build_augmenter(AUG_CONFIG, seed=a.seed)
    versions = [augment_batch(images, augmenter) for _ in range(a.versions)]

    os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)
    plot_augmentation_grid(images, versions, labels, a.output, CLASS_NAMES)
    print(f"Saved {a.output}")


if __name__ == "__main__":
    main()
