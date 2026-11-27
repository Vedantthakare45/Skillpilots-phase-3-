"""Reproducibility: seed every random number generator in one call."""

import os
import random

import numpy as np


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and (if installed) TensorFlow.

    deterministic=True additionally forces TensorFlow to use deterministic GPU
    ops. Results then repeat exactly, at the cost of slower training.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
    except ImportError:
        return
    tf.keras.utils.set_random_seed(seed)  # seeds python, numpy and tensorflow together
    if deterministic:
        tf.config.experimental.enable_op_determinism()
