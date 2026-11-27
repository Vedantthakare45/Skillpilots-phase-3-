"""Load and preprocess CIFAR-10.

load_cifar10_raw() returns uint8 images (needed by imgaug).
load_cifar10()     returns float32 images in [0, 1] (used by evaluate.py).
"""

from typing import Tuple

import numpy as np

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def preprocess(images: np.ndarray) -> np.ndarray:
    """Scale pixel values from [0, 255] to [0, 1] as float32."""
    return images.astype("float32") / 255.0


def load_cifar10_raw() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Download CIFAR-10 (cached after the first run).

    Returns (x_train, y_train), (x_test, y_test):
      images: uint8, shape (N, 32, 32, 3), values 0-255
      labels: int64, shape (N,), values 0-9
    """
    from tensorflow import keras  # imported here so other helpers work without TF

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    return (
        (x_train, y_train.ravel().astype("int64")),
        (x_test, y_test.ravel().astype("int64")),
    )


def load_cifar10() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Like load_cifar10_raw() but with images scaled to float32 [0, 1]."""
    (x_train, y_train), (x_test, y_test) = load_cifar10_raw()
    return (preprocess(x_train), y_train), (preprocess(x_test), y_test)


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_cifar10_raw()
    print("train:", x_train.shape, x_train.dtype, y_train.shape)
    print("test: ", x_test.shape, y_test.shape)
    print("class counts (train):", np.bincount(y_train))
