"""Retrain the CNN with and without augmentation and compare the metrics.

Both runs use identical hyper-parameters, seed and train/validation split;
the ONLY difference is the imgaug pipeline.

Usage:
    python compare.py                       # trains both, then writes the comparison
    python compare.py --epochs 40
    python compare.py --skip-training       # just rebuild the report from saved runs
"""

import argparse
import json
import os
from typing import Any, Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

Metrics = Dict[str, Any]
History = Dict[str, list]

# Differences smaller than this (0.1 percentage point) are reported as "no change".
MIN_MEANINGFUL_DELTA = 0.001


def load_run(run_dir: str) -> Tuple[Metrics, History]:
    """Read metrics.json and history.json written by train.py."""
    with open(os.path.join(run_dir, "metrics.json")) as f:
        metrics = json.load(f)
    with open(os.path.join(run_dir, "history.json")) as f:
        history = json.load(f)
    return metrics, history


def _pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def _pts(delta: float) -> str:
    return f"{100 * delta:+.2f} pts"


def verdict(base: Metrics, aug: Metrics) -> str:
    """One honest sentence on whether augmentation helped validation accuracy."""
    delta = aug["best_val_accuracy"] - base["best_val_accuracy"]
    gap_delta = aug["generalization_gap"] - base["generalization_gap"]
    if delta > MIN_MEANINGFUL_DELTA:
        text = f"Validation accuracy IMPROVED by {_pts(delta)} with augmentation."
    elif delta < -MIN_MEANINGFUL_DELTA:
        text = f"Validation accuracy DECREASED by {_pts(abs(delta))} with augmentation."
    else:
        text = "Validation accuracy showed NO meaningful change with augmentation."
    if gap_delta < -MIN_MEANINGFUL_DELTA:
        text += f" The train-validation gap shrank by {_pts(abs(gap_delta))} (less overfitting)."
    elif gap_delta > MIN_MEANINGFUL_DELTA:
        text += f" The train-validation gap grew by {_pts(gap_delta)}."
    return text


def build_markdown(base: Metrics, aug: Metrics) -> str:
    """Return the before/after comparison as a Markdown string."""
    cfg = aug.get("augmentation") or {}
    cj = cfg.get("color_jitter")
    rows = [
        ("Test accuracy", _pct(base["test_accuracy"]), _pct(aug["test_accuracy"]),
         _pts(aug["test_accuracy"] - base["test_accuracy"])),
        ("Test loss", f"{base['test_loss']:.4f}", f"{aug['test_loss']:.4f}",
         f"{aug['test_loss'] - base['test_loss']:+.4f}"),
        ("Best validation accuracy", _pct(base["best_val_accuracy"]), _pct(aug["best_val_accuracy"]),
         _pts(aug["best_val_accuracy"] - base["best_val_accuracy"])),
        ("Train acc at best epoch", _pct(base["train_accuracy_at_best"]), _pct(aug["train_accuracy_at_best"]),
         _pts(aug["train_accuracy_at_best"] - base["train_accuracy_at_best"])),
        ("Train - val gap (overfitting)", _pts(base["generalization_gap"]), _pts(aug["generalization_gap"]),
         _pts(aug["generalization_gap"] - base["generalization_gap"])),
        ("Best epoch", str(base["best_epoch"]), str(aug["best_epoch"]), ""),
        ("Epochs run", str(base["epochs_run"]), str(aug["epochs_run"]), ""),
    ]
    lines = [
        "# Augmentation results: before vs after",
        "",
        f"**{verdict(base, aug)}**",
        "",
        "| Metric | Baseline (no augmentation) | With imgaug augmentation | Change |",
        "|---|---|---|---|",
    ]
    lines += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
    lines += [
        "",
        "## Augmentation settings",
        "",
        f"- Horizontal flip probability: {cfg.get('fliplr_prob')}",
        f"- Rotation: uniform in [-{cfg.get('rotate_deg')}, +{cfg.get('rotate_deg')}] degrees, "
        f"border mode `{cfg.get('rotate_mode')}`",
        f"- Crop: {cfg.get('crop_percent')} of each side removed, then resized back to 32x32",
    ]
    if cj:
        lines.append(
            f"- Colour jitter on {cj.get('prob')} of images: brightness x{cj.get('brightness')}, "
            f"contrast x{cj.get('contrast')}, saturation x{cj.get('saturation')}"
        )
    else:
        lines.append("- Colour jitter: off")
    lines += [
        "- Applied to training batches only (validation and test data are never augmented)",
        f"- Seed: {aug.get('seed')}, deterministic mode: {aug.get('deterministic')}",
        "",
        "Test accuracy is measured once, on the 10,000-image CIFAR-10 test set. "
        "Models were selected on validation loss, not on the test set.",
    ]
    return "\n".join(lines) + "\n"


def plot_comparison(base_hist: History, aug_hist: History, base: Metrics, aug: Metrics,
                    path: str) -> None:
    """Save a 3-panel figure: validation curves, train/val curves, test-accuracy bars."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

    axes[0].plot(base_hist["val_accuracy"], label="baseline")
    axes[0].plot(aug_hist["val_accuracy"], label="augmented")
    axes[0].set_title("Validation accuracy")
    axes[0].set_xlabel("Epoch")

    axes[1].plot(base_hist["accuracy"], "--", label="baseline train")
    axes[1].plot(base_hist["val_accuracy"], label="baseline val")
    axes[1].plot(aug_hist["accuracy"], "--", label="augmented train")
    axes[1].plot(aug_hist["val_accuracy"], label="augmented val")
    axes[1].set_title("Train vs validation (gap = overfitting)")
    axes[1].set_xlabel("Epoch")

    bars = axes[2].bar(["baseline", "augmented"],
                       [100 * base["test_accuracy"], 100 * aug["test_accuracy"]],
                       color=["#9aa0a6", "#6c5ce7"])
    axes[2].bar_label(bars, fmt="%.2f%%")
    lo = min(base["test_accuracy"], aug["test_accuracy"]) * 100
    axes[2].set_ylim(max(0, lo - 10), 100)
    axes[2].set_title("Test accuracy")

    for ax in axes[:2]:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(root: str) -> str:
    """Build comparison.md/.json/.png in `root` from its baseline/ and augmented/ runs."""
    base, base_hist = load_run(os.path.join(root, "baseline"))
    aug, aug_hist = load_run(os.path.join(root, "augmented"))

    md = build_markdown(base, aug)
    with open(os.path.join(root, "comparison.md"), "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(root, "comparison.json"), "w") as f:
        json.dump({"baseline": base, "augmented": aug}, f, indent=2)
    plot_comparison(base_hist, aug_hist, base, aug, os.path.join(root, "comparison.png"))
    return md


def main() -> None:
    p = argparse.ArgumentParser(description="Compare CNN training with vs without augmentation.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true", help="exactly repeatable runs (slower)")
    p.add_argument("--output-root", default="outputs")
    p.add_argument("--skip-training", action="store_true",
                   help="reuse existing outputs/baseline and outputs/augmented")
    a = p.parse_args()

    if not a.skip_training:
        from train import train  # imported here so the report code works without TensorFlow

        for augment, name in [(False, "baseline"), (True, "augmented")]:
            train(augment=augment, epochs=a.epochs, batch_size=a.batch_size, patience=a.patience,
                  val_split=a.val_split, output_dir=os.path.join(a.output_root, name),
                  seed=a.seed, deterministic=a.deterministic)

    for name in ("baseline", "augmented"):
        if not os.path.exists(os.path.join(a.output_root, name, "metrics.json")):
            raise SystemExit(f"error: no results in {a.output_root}/{name}. Run without --skip-training.")

    print(write_report(a.output_root))
    print(f"Saved {a.output_root}/comparison.md, comparison.json and comparison.png")


if __name__ == "__main__":
    main()
