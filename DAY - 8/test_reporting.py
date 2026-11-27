"""Tests for reporting, the before/after comparison and the confusion-matrix plot."""

import json
import os

import numpy as np
import pytest
from sklearn.metrics import confusion_matrix

import compare
import reporting
from augmentation import resolve_config
from data import CLASS_NAMES
from evaluate import plot_confusion_matrix

HIST = {"loss": [1.5, 1.0, 0.8, 0.7], "val_loss": [1.4, 1.0, 0.9, 0.95],
        "accuracy": [0.45, 0.65, 0.72, 0.76], "val_accuracy": [0.50, 0.66, 0.70, 0.69]}


def test_summarize_history_uses_epoch_with_lowest_val_loss():
    m = reporting.summarize_history(HIST, 0.91, 0.705, True, resolve_config())
    assert m["best_epoch"] == 3 and m["epochs_run"] == 4
    assert m["best_val_accuracy"] == pytest.approx(0.70)
    assert m["generalization_gap"] == pytest.approx(0.72 - 0.70)
    assert m["augmentation"] is not None


def test_summarize_history_baseline_has_no_augmentation_settings():
    assert reporting.summarize_history(HIST, 1.0, 0.5, False, resolve_config())["augmentation"] is None


def _metrics(val_acc, gap, augmented):
    return {"augmented": augmented, "test_accuracy": val_acc, "test_loss": 0.7,
            "best_val_accuracy": val_acc, "best_val_loss": 0.7, "train_accuracy_at_best": val_acc + gap,
            "generalization_gap": gap, "best_epoch": 5, "epochs_run": 10,
            "augmentation": resolve_config() if augmented else None, "seed": 42, "deterministic": False}


def test_verdict_improved_and_gap_shrank():
    text = compare.verdict(_metrics(0.78, 0.12, False), _metrics(0.84, 0.02, True))
    assert "IMPROVED" in text and "+6.00 pts" in text and "shrank" in text


def test_verdict_decreased():
    assert "DECREASED" in compare.verdict(_metrics(0.80, 0.05, False), _metrics(0.75, 0.05, True))


def test_verdict_no_meaningful_change():
    text = compare.verdict(_metrics(0.800, 0.05, False), _metrics(0.8004, 0.05, True))
    assert "NO meaningful change" in text


def test_markdown_lists_settings_and_color_jitter():
    md = compare.build_markdown(_metrics(0.78, 0.12, False), _metrics(0.84, 0.02, True))
    for text in ("Test accuracy", "Horizontal flip", "Rotation", "Crop", "Colour jitter", "Seed: 42"):
        assert text in md


def test_write_report_creates_all_outputs(tmp_path):
    for name, augmented, acc in [("baseline", False, 0.78), ("augmented", True, 0.84)]:
        d = tmp_path / name
        d.mkdir()
        n = 12 if augmented else 8
        hist = {"loss": list(np.linspace(1.6, 0.5, n)), "val_loss": list(np.linspace(1.5, 0.7, n)),
                "accuracy": list(np.linspace(0.4, 0.85, n)), "val_accuracy": list(np.linspace(0.45, acc, n))}
        (d / "history.json").write_text(json.dumps(hist))
        metrics = reporting.summarize_history(hist, 0.7, acc, augmented, resolve_config())
        (d / "metrics.json").write_text(json.dumps(metrics))
    md = compare.write_report(str(tmp_path))
    assert "IMPROVED" in md
    for f in ("comparison.md", "comparison.json", "comparison.png"):
        assert os.path.getsize(tmp_path / f) > 100


def test_augmentation_grid_plot(tmp_path):
    rng = np.random.default_rng(0)
    imgs = rng.integers(0, 256, (6, 32, 32, 3), dtype=np.uint8)
    path = str(tmp_path / "grid.png")
    reporting.plot_augmentation_grid(imgs, [imgs[:, :, ::-1], imgs[::-1]], rng.integers(0, 10, 6),
                                     path, CLASS_NAMES)
    assert os.path.getsize(path) > 5000


@pytest.mark.parametrize("normalize", [True, False])
def test_confusion_matrix_plot(tmp_path, normalize):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 10, 400)
    pred = np.where(rng.random(400) < 0.8, y, rng.integers(0, 10, 400))
    path = str(tmp_path / "cm.png")
    plot_confusion_matrix(confusion_matrix(y, pred), CLASS_NAMES, path, normalize=normalize)
    assert os.path.getsize(path) > 5000
