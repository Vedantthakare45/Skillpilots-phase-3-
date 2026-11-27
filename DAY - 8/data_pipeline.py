"""tf.data pipeline that applies the imgaug augmenter to each batch on the fly.

Memory use stays low: only the compact uint8 arrays are held, and each batch is
converted to float32 and augmented as it is requested. The same
`apply_augmentation` step works on ANY batched (uint8 images, labels) dataset,
for example one read lazily from disk (see README, "Large datasets").
"""

import threading
from typing import Callable, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from augmentation import AugConfig, augment_batch, build_augmenter


def make_numpy_augment_fn(augmenter) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap the augmenter in a lock.

    tf.data may call the function from several threads at once, while imgaug's
    random state is not thread-safe. The lock keeps results valid.
    """
    lock = threading.Lock()

    def _augment(batch: np.ndarray) -> np.ndarray:
        with lock:
            return augment_batch(batch, augmenter)

    return _augment


def apply_augmentation(ds, aug_config: Optional[AugConfig] = None,
                       seed: Optional[int] = None, deterministic: bool = False):
    """Map imgaug over a batched dataset of (uint8 images, labels).

    Returns a dataset of (float32 images in [0, 1], labels). With
    deterministic=True the map runs on one thread so the augmentation sequence is
    identical on every run (needs a fixed seed); otherwise it runs in parallel.
    """
    import tensorflow as tf

    np_augment = make_numpy_augment_fn(build_augmenter(aug_config, seed=seed))

    def augment_fn(images, labels):
        augmented = tf.numpy_function(np_augment, [images], tf.uint8)
        augmented.set_shape(images.shape)  # numpy_function loses the static shape
        return tf.cast(augmented, tf.float32) / 255.0, labels

    return ds.map(augment_fn, num_parallel_calls=1 if deterministic else tf.data.AUTOTUNE)


def make_datasets(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 64,
    augment: bool = True,
    val_split: float = 0.1,
    seed: int = 42,
    aug_config: Optional[AugConfig] = None,
    deterministic: bool = False,
) -> Tuple["tf.data.Dataset", "tf.data.Dataset", "tf.data.Dataset"]:  # noqa: F821
    """Build (train_ds, val_ds, test_ds) tf.data.Datasets from RAW uint8 images.

    - Stratified train/validation split (identical for augmented and baseline runs
      when the seed is the same, so the comparison is fair).
    - Train: shuffle -> batch -> [imgaug augment] -> scale to [0,1] -> prefetch
    - Val / test: batch -> scale to [0,1] -> prefetch (never augmented)
    """
    import tensorflow as tf

    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train, y_train, test_size=val_split, stratify=y_train, random_state=seed
    )

    def to_float(images, labels):
        return tf.cast(images, tf.float32) / 255.0, labels

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_tr, y_tr))
        .shuffle(len(x_tr), seed=seed, reshuffle_each_iteration=True)
        .batch(batch_size)
    )
    if augment:
        train_ds = apply_augmentation(train_ds, aug_config, seed=seed, deterministic=deterministic)
    else:
        train_ds = train_ds.map(to_float, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = (tf.data.Dataset.from_tensor_slices((x_val, y_val))
              .batch(batch_size).map(to_float).prefetch(tf.data.AUTOTUNE))
    test_ds = (tf.data.Dataset.from_tensor_slices((x_test, y_test))
               .batch(batch_size).map(to_float).prefetch(tf.data.AUTOTUNE))
    return train_ds, val_ds, test_ds
