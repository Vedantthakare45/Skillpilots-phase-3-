"""Tests for the imgaug augmentation module.

Structural tests use a recording stand-in for imgaug, so they run anywhere.
Behavioural tests use the real imgaug and are skipped if it is not installed.
"""

import sys
import types

import numpy as np
import pytest

import augmentation
from augmentation import AUG_CONFIG, augment_batch, build_augmenter, resolve_config


# ---------- a recording stand-in for imgaug.augmenters ----------
class _Rec:
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs


@pytest.fixture
def fake_imgaug(monkeypatch):
    iaa = types.ModuleType("imgaug.augmenters")
    for name in ("Fliplr", "Affine", "Crop", "Multiply", "LinearContrast",
                 "MultiplySaturation", "Sometimes", "Sequential"):
        setattr(iaa, name, type(name, (_Rec,), {}))
    parent = types.ModuleType("imgaug")
    parent.augmenters = iaa
    monkeypatch.setitem(sys.modules, "imgaug", parent)
    monkeypatch.setitem(sys.modules, "imgaug.augmenters", iaa)
    return iaa


def _names(seq):
    return [type(s).__name__ for s in seq.args[0]]


def test_pipeline_order_and_arguments(fake_imgaug):
    seq = build_augmenter(seed=7)
    assert _names(seq) == ["Fliplr", "Affine", "Crop", "Sometimes"]   # flip, rotate, crop, colour
    flip, affine, crop, color = seq.args[0]
    assert flip.args == (0.5,)
    assert affine.kwargs == {"rotate": (-15, 15), "mode": "reflect"}
    assert crop.kwargs == {"percent": (0.0, 0.10), "keep_size": True}
    assert color.args[0] == 0.5
    assert _names(color.args[1]) == ["Multiply", "LinearContrast", "MultiplySaturation"]
    assert seq.kwargs == {"random_order": False, "seed": 7}


def test_color_jitter_can_be_disabled(fake_imgaug):
    seq = build_augmenter({"color_jitter": None})
    assert _names(seq) == ["Fliplr", "Affine", "Crop"]


def test_custom_settings_reach_imgaug(fake_imgaug):
    seq = build_augmenter({"fliplr_prob": 0.9, "rotate_deg": 30, "color_jitter": {"prob": 0.2}})
    flip, affine, _, color = seq.args[0]
    assert flip.args == (0.9,)
    assert affine.kwargs["rotate"] == (-30, 30)
    assert color.args[0] == 0.2
    assert color.args[1].args[0][0].args == ((0.8, 1.2),)   # brightness range kept from defaults


# ---------- config handling ----------
def test_resolve_config_returns_independent_copy():
    cfg = resolve_config()
    cfg["rotate_deg"] = 99
    cfg["color_jitter"]["prob"] = 1.0
    assert AUG_CONFIG["rotate_deg"] == 15 and AUG_CONFIG["color_jitter"]["prob"] == 0.5


def test_resolve_config_merges_color_jitter_keys():
    cfg = resolve_config({"color_jitter": {"prob": 0.9}})
    assert cfg["color_jitter"]["prob"] == 0.9
    assert cfg["color_jitter"]["brightness"] == (0.8, 1.2)


def test_resolve_config_none_disables_color_jitter():
    assert resolve_config({"color_jitter": None})["color_jitter"] is None


# ---------- augment_batch input validation ----------
class _Identity:
    def __call__(self, images):
        return images


def test_augment_batch_rejects_float_images():
    with pytest.raises(ValueError, match="uint8"):
        augment_batch(np.zeros((2, 32, 32, 3), dtype="float32"), _Identity())


def test_augment_batch_rejects_wrong_dimensions():
    with pytest.raises(ValueError, match="4-D"):
        augment_batch(np.zeros((32, 32, 3), dtype=np.uint8), _Identity())


def test_augment_batch_keeps_shape_and_dtype():
    batch = np.random.default_rng(0).integers(0, 256, (4, 32, 32, 3), dtype=np.uint8)
    out = augment_batch(batch, _Identity())
    assert out.shape == batch.shape and out.dtype == np.uint8


# ---------- behaviour with the REAL imgaug (skipped if not installed) ----------
@pytest.fixture
def batch():
    return np.random.default_rng(0).integers(0, 256, (16, 32, 32, 3), dtype=np.uint8)


def test_real_output_shape_and_dtype(batch):
    pytest.importorskip("imgaug")
    out = augment_batch(batch, build_augmenter(seed=1))
    assert out.shape == batch.shape and out.dtype == np.uint8


def test_real_same_seed_gives_same_output(batch):
    pytest.importorskip("imgaug")
    a = augment_batch(batch, build_augmenter(seed=123))
    b = augment_batch(batch, build_augmenter(seed=123))
    c = augment_batch(batch, build_augmenter(seed=456))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_real_augmentation_actually_changes_images(batch):
    pytest.importorskip("imgaug")
    out = augment_batch(batch, build_augmenter(seed=0))
    assert not np.array_equal(out, batch)


def test_real_flip_only_matches_numpy_flip(batch):
    pytest.importorskip("imgaug")
    cfg = {"fliplr_prob": 1.0, "rotate_deg": 0, "crop_percent": (0.0, 0.0), "color_jitter": None}
    out = augment_batch(batch, build_augmenter(cfg, seed=0))
    np.testing.assert_array_equal(out, batch[:, :, ::-1, :])


def test_real_color_jitter_keeps_geometry(batch):
    pytest.importorskip("imgaug")
    cfg = {"fliplr_prob": 0.0, "rotate_deg": 0, "crop_percent": (0.0, 0.0),
           "color_jitter": {"prob": 1.0, "brightness": (1.3, 1.3), "contrast": (1.0, 1.0),
                            "saturation": (1.0, 1.0)}}
    out = augment_batch(batch, build_augmenter(cfg, seed=0))
    assert out.shape == batch.shape
    assert out.mean() > batch.mean()          # brightness x1.3 makes the batch brighter
