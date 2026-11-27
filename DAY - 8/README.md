# CIFAR-10 CNN with imgaug Data Augmentation

A Keras CNN for CIFAR-10 with an **imgaug augmentation pipeline** wrapped in `tf.data`, plus a script that trains the model **with and without** augmentation and compares the results.

Put all files in **one folder** (they import each other).

## Files

| File | Purpose |
|------|---------|
| `augmentation.py` | **Step 1:** imgaug `Sequential` augmenter (flip, rotate, crop) and all its settings |
| `data_pipeline.py` | **Steps 2 and 3:** `tf.data` pipeline that augments each batch; replaces the old in-memory loader |
| `data.py` | Downloads CIFAR-10 (raw uint8 images for imgaug, or scaled float images) |
| `model.py` | CNN: Conv2D, BatchNorm, MaxPooling2D, Dropout |
| `train.py` | Trains one model, with (default) or without (`--no-augment`) augmentation |
| `compare.py` | **Step 4:** trains both and writes the before/after comparison |
| `reporting.py` | Metric summary and plotting helpers |
| `show_augmentations.py` | Saves a picture of original vs augmented images |
| `evaluate.py` | Test accuracy, classification report, confusion matrix |
| `export.py`, `predict.py` | Export the model and classify your own images |

## Setup

Use **Python 3.9 to 3.12**.

```bash
pip install -r requirements.txt
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
```

A GPU makes this much faster. Augmentation runs on the CPU, so a weak CPU can slow training a little.

## Augmentation settings

Defined in `AUG_CONFIG` in `augmentation.py`:

| Augmentation | imgaug call | Setting |
|---|---|---|
| Horizontal flip | `iaa.Fliplr` | applied to 50% of images |
| Rotation | `iaa.Affine(rotate=...)` | uniform in [-15, +15] degrees; corners filled by mirroring (`mode="reflect"`) |
| Crop | `iaa.Crop(..., keep_size=True)` | removes 0-10% of each side, then resizes back to 32x32 |

They run in that order inside one `iaa.Sequential`. Edit `AUG_CONFIG` to change any value, or pass a dict as `aug_config`. Settings are saved in each run's `metrics.json`.

Why these choices:
- **No vertical flips:** upside-down airplanes, horses and ships don't appear in the test data.
- **Small rotation and crop:** CIFAR images are only 32x32, and strong transforms remove the object.
- **Training data only:** validation and test batches are never augmented, so the numbers measure real generalisation.

## How the pipeline works

```
raw uint8 images
  -> stratified train / validation split (same seed in both runs)
  -> train:  shuffle -> batch -> imgaug (tf.numpy_function) -> scale to [0,1] -> prefetch
  -> val/test:          batch -> scale to [0,1] -> prefetch
```

- The augmenter works on whole batches, so imgaug's vectorised code is used.
- A lock serialises the augmenter, because tf.data can call it from several threads and imgaug's random state is not thread-safe.
- The old in-model Keras flip/translate layers are **off** by default (`builtin_augmentation=False` in `model.py`), so the baseline has no augmentation and the augmented run is not augmented twice.

## Comparison method

`compare.py` runs two trainings that differ **only** in the augmentation: same CNN, optimizer, batch size, early-stopping settings, seed and validation split. It writes:

- `outputs/comparison.md`: table of test accuracy, test loss, validation accuracy, train-vs-validation gap, epochs
- `outputs/comparison.png`: validation curves, train/validation curves, and test-accuracy bars
- `outputs/comparison.json`, and per-run `metrics.json`, `history.json`, `training_curves.png`

## Results

Run `python compare.py` to generate the real numbers; they are written to `outputs/comparison.md`. Paste the table here:

| Metric | Baseline | Augmented | Change |
|---|---|---|---|
| Test accuracy | _fill in_ | _fill in_ | |
| Train - val gap | _fill in_ | _fill in_ | |

What to expect (typical for CIFAR-10, not measured for this code): augmentation usually raises test accuracy by a few points and clearly narrows the gap between training and validation accuracy, because the model can no longer memorise exact images. Augmented runs usually train for more epochs before early stopping triggers. Results vary with the seed, so for a firm conclusion run `compare.py` with a few different `--seed` values.

## Notes

- Use `.keras` files for models; `evaluate.py` and `export.py` default to `outputs/augmented/best_model.keras`.
- `predict.py` is unchanged: it scales images to [0, 1] exactly as the training pipeline does, and applies no augmentation.
