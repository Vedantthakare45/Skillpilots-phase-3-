# CIFAR-10 CNN with imgaug Data Augmentation

A Keras CNN for CIFAR-10 with an **imgaug augmentation pipeline** (flips, rotations, crops, colour jitter) wrapped in `tf.data`, plus a script that trains the model **with and without** augmentation and reports whether validation accuracy improved.

Put all files in **one folder** (they import each other).

## Files

| File | Purpose |
|------|---------|
| `augmentation.py` | imgaug `Sequential` augmenter (flip, rotate, crop, colour jitter) and all its settings |
| `data_pipeline.py` | `tf.data` pipeline: lazily augments each batch with imgaug; reusable `apply_augmentation` |
| `data.py` | Downloads CIFAR-10 (raw uint8 images for imgaug, or scaled float images) |
| `model.py` | CNN: Conv2D, BatchNorm, MaxPooling2D, Dropout (`tf.keras`) |
| `seeding.py` | One-call seeding of Python, NumPy and TensorFlow, with optional deterministic mode |
| `train.py` | Trains one model, with (default) or without (`--no-augment`) augmentation |
| `compare.py` | Trains both and writes the before/after comparison and verdict |
| `reporting.py` | Metric summary and plotting helpers |
| `show_augmentations.py` | Saves a picture of original vs augmented images |
| `evaluate.py` | Test accuracy, classification report, confusion matrix |
| `export.py`, `predict.py` | Export the model and classify your own images |
| `test_*.py` | pytest suite (augmentation, reporting/comparison, seeding, thread-safety) |

## Setup

Use **Python 3.9 to 3.12**.

```bash
pip install -r requirements.txt
pytest -q                      # optional: run the tests
```

`imgaug` 0.4.0 is unmaintained and does not support NumPy 2, so `requirements.txt` pins `numpy<2`. If you hit `np.sctypes` or `np.bool` errors, check your NumPy version first.

## Quick start

```bash
python show_augmentations.py   # look at what the augmentation does first
python compare.py              # trains baseline + augmented, writes the comparison
python evaluate.py             # confusion matrix for the augmented model
python export.py
python predict.py my_photo.jpg
```

Or train one model at a time:

```bash
python train.py                # with augmentation -> outputs/augmented/
python train.py --no-augment   # baseline          -> outputs/baseline/
python train.py --deterministic  # exactly repeatable run
```

A GPU makes this much faster. Augmentation runs on the CPU, so a weak CPU can slow training a little.

## Augmentation settings

Defined in `AUG_CONFIG` in `augmentation.py`:

| Augmentation | imgaug call | Setting |
|---|---|---|
| Horizontal flip | `iaa.Fliplr` | applied to 50% of images |
| Rotation | `iaa.Affine(rotate=...)` | uniform in [-15, +15] degrees; corners filled by mirroring (`mode="reflect"`) |
| Crop | `iaa.Crop(..., keep_size=True)` | removes 0-10% of each side, then resizes back to 32x32 |
| Colour jitter | `iaa.Sometimes(0.5, ...)` | on 50% of images: brightness x0.8-1.2 (`Multiply`), contrast x0.8-1.2 (`LinearContrast`), saturation x0.8-1.2 (`MultiplySaturation`) |

They run in that order inside one `iaa.Sequential`. Change any value by editing `AUG_CONFIG` or passing a dict, e.g. `build_augmenter({"rotate_deg": 10, "color_jitter": {"prob": 0.8}})`. Use `{"color_jitter": None}` to switch colour jitter off. The settings used are saved in each run's `metrics.json`.

Why these choices:
- **No vertical flips:** upside-down airplanes, horses and ships don't appear in the test data.
- **Small rotation, crop and colour changes:** CIFAR images are only 32x32, and strong transforms destroy the object.
- **Training data only:** validation and test batches are never augmented, so the numbers measure real generalisation.

## How the pipeline works

```
raw uint8 images
  -> stratified train / validation split (same seed in both runs)
  -> train:  shuffle -> batch -> imgaug (tf.numpy_function) -> scale to [0,1] -> prefetch
  -> val/test:          batch -> scale to [0,1] -> prefetch
```

- **Augmentation happens on the fly,** one batch at a time, so no augmented copy of the dataset is ever stored.
- A lock serialises the augmenter, because tf.data can call it from several threads and imgaug's random state is not thread-safe.
- The old in-model Keras flip/translate layers are **off** by default (`builtin_augmentation=False` in `model.py`), so the baseline has no augmentation and the augmented run is not augmented twice.

### Memory and large datasets

CIFAR-10 is small: the training images are about 150 MB as uint8, and the pipeline keeps them as uint8 and only converts each batch to float32 when it is used. For datasets that do not fit in memory, read the images lazily from disk and reuse the same augmentation step on that dataset:

```python
import tensorflow as tf
from data_pipeline import apply_augmentation

ds = tf.keras.utils.image_dataset_from_directory(
    "data/train", image_size=(64, 64), batch_size=64, label_mode="int")   # reads files lazily
ds = ds.map(lambda x, y: (tf.cast(x, tf.uint8), y))                       # imgaug needs uint8
ds = apply_augmentation(ds, seed=42).prefetch(tf.data.AUTOTUNE)
```

(This snippet is a usage sketch; the project itself trains on CIFAR-10.)

## Reproducibility

- `seeding.set_global_seed(seed)` seeds Python, NumPy and TensorFlow; it is called at the start of every run. The imgaug augmenter is seeded too.
- `--deterministic` additionally enables deterministic TensorFlow ops and runs the augmentation on one thread, so a run repeats exactly. It is slower, especially on GPU.
- Without `--deterministic`, results with the same seed are close but not bit-identical, because parallel augmentation and GPU kernels are not ordered.

## Comparison method

`compare.py` runs two trainings that differ **only** in the augmentation: same CNN, optimizer, batch size, early-stopping settings, seed and validation split. It writes:

- `outputs/comparison.md`: a one-line verdict (validation accuracy improved / decreased / no meaningful change, and what happened to the train-validation gap), plus a table of test accuracy, test loss, validation accuracy and epochs
- `outputs/comparison.png`: validation curves, train/validation curves, and test-accuracy bars
- `outputs/comparison.json`, and per-run `metrics.json`, `history.json`, `training_curves.png`

## Results

Run `python compare.py` to generate the real numbers; they are written to `outputs/comparison.md`. Paste the table here:

| Metric | Baseline | Augmented | Change |
|---|---|---|---|
| Best validation accuracy | _fill in_ | _fill in_ | |
| Test accuracy | _fill in_ | _fill in_ | |
| Train - val gap | _fill in_ | _fill in_ | |

What to expect (typical for CIFAR-10, not measured for this code): augmentation usually raises validation and test accuracy by a few points and narrows the gap between training and validation accuracy, because the model can no longer memorise exact images. Results vary with the seed, so for a firm conclusion run `compare.py` with a few different `--seed` values.

## Tests

```bash
pytest -q
```

Covers the augmentation pipeline's order and settings, config merging, input validation, the thread-safe wrapper, seeding, the metric summary, the comparison verdict and report, and the plots. Tests that need the real `imgaug` (seeded reproducibility, flip correctness, colour jitter) are skipped automatically if it is not installed.

## Notes

- Use `.keras` files for models; `evaluate.py` and `export.py` default to `outputs/augmented/best_model.keras`.
- `predict.py` scales images to [0, 1] exactly as the training pipeline does, and applies no augmentation.
