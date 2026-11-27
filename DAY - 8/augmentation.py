"""imgaug augmentation pipeline: flips, rotations, crops and colour jitter.

All settings live in AUG_CONFIG so they are easy to change and are saved with
every training run's metrics.
"""

import copy
import warnings
from typing import Any, Dict, Optional

import numpy as np

# imgaug 0.4.0 still uses aliases that NumPy >= 1.24 removed (np.bool, np.int ...).
# Restore them before imgaug is imported. (NumPy 2 is NOT supported; see requirements.txt.)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for _name, _alias in (("bool", np.bool_), ("int", int), ("float", float),
                          ("complex", complex), ("object", object)):
        if not hasattr(np, _name):
            setattr(np, _name, _alias)

AugConfig = Dict[str, Any]

AUG_CONFIG: AugConfig = {
    "fliplr_prob": 0.5,            # horizontal flip applied to 50% of images
    "rotate_deg": 15,              # random rotation in [-15, +15] degrees
    "rotate_mode": "reflect",      # fill rotated corners by mirroring (no black borders)
    "crop_percent": (0.0, 0.10),   # crop 0-10% off each side, then resize back to 32x32
    "color_jitter": {              # set to None to disable colour jitter
        "prob": 0.5,               # fraction of images that get colour jitter
        "brightness": (0.8, 1.2),  # multiply pixel values
        "contrast": (0.8, 1.2),    # linear contrast factor
        "saturation": (0.8, 1.2),  # multiply colour saturation
    },
}


def resolve_config(config: Optional[AugConfig] = None) -> AugConfig:
    """Return AUG_CONFIG with `config` applied on top (a fresh copy, safe to mutate).

    `color_jitter` is merged key by key, so {"color_jitter": {"prob": 0.8}} only
    changes the probability. Pass {"color_jitter": None} to turn colour jitter off.
    """
    cfg = copy.deepcopy(AUG_CONFIG)
    for key, value in (config or {}).items():
        if key == "color_jitter" and isinstance(value, dict):
            cfg["color_jitter"] = {**(cfg["color_jitter"] or {}), **value}
        else:
            cfg[key] = copy.deepcopy(value)
    return cfg


def build_augmenter(config: Optional[AugConfig] = None, seed: Optional[int] = None):
    """Return an imgaug ``Sequential`` augmenter: flip -> rotate -> crop -> colour jitter.

    The augmenter expects uint8 images of shape (N, H, W, 3) with values 0-255.
    Pass ``seed`` for a reproducible sequence of random augmentations.
    """
    import imgaug.augmenters as iaa

    cfg = resolve_config(config)
    steps = [
        iaa.Fliplr(cfg["fliplr_prob"]),
        iaa.Affine(rotate=(-cfg["rotate_deg"], cfg["rotate_deg"]), mode=cfg["rotate_mode"]),
        iaa.Crop(percent=tuple(cfg["crop_percent"]), keep_size=True),
    ]
    cj = cfg["color_jitter"]
    if cj:
        steps.append(
            iaa.Sometimes(
                cj["prob"],
                iaa.Sequential([
                    iaa.Multiply(tuple(cj["brightness"])),
                    iaa.LinearContrast(tuple(cj["contrast"])),
                    iaa.MultiplySaturation(tuple(cj["saturation"])),
                ]),
            )
        )
    return iaa.Sequential(steps, random_order=False, seed=seed)


def augment_batch(images: np.ndarray, augmenter) -> np.ndarray:
    """Augment a uint8 batch (N, H, W, 3). Output has the same shape and dtype."""
    if images.dtype != np.uint8:
        raise ValueError(f"augment_batch expects uint8 images in [0, 255], got {images.dtype}")
    if images.ndim != 4:
        raise ValueError(f"expected a 4-D batch (N, H, W, C), got shape {images.shape}")
    out = augmenter(images=images)
    return np.ascontiguousarray(out)
