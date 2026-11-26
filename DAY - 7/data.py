"""Step 1: download and preprocess CIFAR-10."""

import numpy as np

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def preprocess(images: np.ndarray) -> np.ndarray:
    """Scale pixel values from [0, 255] to [0, 1] as float32."""
    return images.astype("float32") / 255.0


def load_cifar10():
    """Download CIFAR-10 (cached after the first run) and preprocess it.

    Returns (x_train, y_train), (x_test, y_test) with:
      images: float32, shape (N, 32, 32, 3), values in [0, 1]
      labels: int64,   shape (N,), values 0-9 (use sparse_categorical_crossentropy)
    """
    from tensorflow import keras  # imported here so other helpers work without TF

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    return (
        (preprocess(x_train), y_train.ravel().astype("int64")),
        (preprocess(x_test), y_test.ravel().astype("int64")),
    )


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    print("train:", x_train.shape, y_train.shape)
    print("test: ", x_test.shape, y_test.shape)
    print("pixel range:", x_train.min(), "-", x_train.max())
    print("class counts (train):", np.bincount(y_train))
