"""Tests for seeding and the thread-safe augmentation wrapper (no TensorFlow needed)."""

import os
import random
import threading

import numpy as np

from data_pipeline import make_numpy_augment_fn
from seeding import set_global_seed


def test_set_global_seed_makes_python_and_numpy_repeatable():
    set_global_seed(123)
    first = (random.random(), float(np.random.rand()))
    set_global_seed(123)
    second = (random.random(), float(np.random.rand()))
    assert first == second
    set_global_seed(124)
    assert (random.random(), float(np.random.rand())) != first


def test_set_global_seed_sets_hash_seed_env():
    set_global_seed(7)
    assert os.environ["PYTHONHASHSEED"] == "7"


def test_augment_wrapper_serialises_concurrent_calls():
    """imgaug's RNG is not thread-safe, so the wrapper must never run two calls at once."""

    class Fragile:
        active = 0
        overlaps = 0

        def __call__(self, images):
            Fragile.active += 1
            if Fragile.active > 1:
                Fragile.overlaps += 1
            sum(range(20000))  # give other threads a chance to collide
            Fragile.active -= 1
            return images

    fn = make_numpy_augment_fn(Fragile())
    batch = np.zeros((2, 32, 32, 3), dtype=np.uint8)
    threads = [threading.Thread(target=lambda: [fn(batch) for _ in range(20)]) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert Fragile.overlaps == 0


def test_augment_wrapper_returns_augmenter_output():
    fn = make_numpy_augment_fn(lambda images: images[:, :, ::-1, :])
    batch = np.random.default_rng(0).integers(0, 256, (3, 32, 32, 3), dtype=np.uint8)
    np.testing.assert_array_equal(fn(batch), batch[:, :, ::-1, :])
