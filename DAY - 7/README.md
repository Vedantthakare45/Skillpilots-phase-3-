# CIFAR-10 Image Classification with a CNN

A complete Keras project: data loading, CNN model, training with early stopping, evaluation with a confusion matrix, and export plus an inference script.

Put all files in **one folder** (they import each other).

## Files

| File | Step | Purpose |
|------|------|---------|
| `data.py` | 1 | Downloads CIFAR-10 and scales pixels to [0, 1] |
| `model.py` | 2 | CNN: Conv2D, BatchNorm, MaxPooling2D, Dropout |
| `train.py` | 3 | Compile and train with EarlyStopping, ReduceLROnPlateau and ModelCheckpoint |
| `evaluate.py` | 4 | Test accuracy, classification report, confusion-matrix plot |
| `export.py` | 5 | Export the model (`.keras`, optional SavedModel) |
| `predict.py` | 5 | Classify your own images |
| `requirements.txt` | - | Dependencies |

## Quick start

```bash
pip install -r requirements.txt

python train.py          # downloads CIFAR-10 (~170 MB) on first run, then trains
python evaluate.py       # test accuracy + outputs/confusion_matrix.png
python export.py         # writes exported_model/cifar10_cnn.keras
python predict.py my_photo.jpg
```

Training options:

```bash
python train.py --epochs 60 --batch-size 128 --patience 10
```

A GPU speeds training up a lot. On CPU, expect several minutes per epoch.

## Outputs

- `outputs/best_model.keras`: best checkpoint (lowest validation loss)
- `outputs/training_curves.png` and `outputs/history.json`: loss and accuracy per epoch
- `outputs/classification_report.txt`: per-class precision, recall, F1
- `outputs/confusion_matrix.png`: row-normalised confusion matrix
- `exported_model/cifar10_cnn.keras`: model for inference

## Architecture

```
Input 32x32x3 -> RandomFlip -> RandomTranslation
3 x [Conv2D -> BN -> Conv2D -> BN -> MaxPool -> Dropout]   (32, 64, 128 filters)
Flatten -> Dense(256) -> Dropout(0.5) -> Dense(10, softmax)
```

Loss is `sparse_categorical_crossentropy` because the labels are integers 0-9. It is equivalent to `categorical_crossentropy` on one-hot labels.

## Notes

- **Expected accuracy:** roughly 80-88% on the test set, depending on epochs and seed.
- **Early stopping:** training stops when validation loss hasn't improved for `--patience` epochs and restores the best weights.
- **Inference images:** `predict.py` resizes any image to 32x32. CIFAR-10 images are tiny and low-resolution, so the model works best on a single clear subject of one of the 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). It will always pick one of these, even for unrelated pictures.
- **Why a validation split and a test set:** the validation split is used during training for early stopping. The test set is used only once at the end, in `evaluate.py`.
