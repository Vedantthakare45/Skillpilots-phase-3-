# Keras Notebook: Train, Validate and Save a Model on a Custom Dataset

A Jupyter notebook that trains a Keras `Sequential` classifier on a custom CSV dataset, validates it, and saves it to an `.h5` file.

## Project structure

```
keras_notebook_project/
├── segment_classifier_keras.ipynb   # the notebook
├── data/
│   └── customer_segments.csv        # sample dataset (900 rows, 3 classes)
├── generate_dataset.py              # regenerates the sample dataset
├── requirements.txt
└── README.md
```

Running the notebook creates `models/` (the `.h5` model and preprocessing file) and `logs/` (per-epoch CSV log).

## Setup

```bash
pip install -r requirements.txt
jupyter notebook segment_classifier_keras.ipynb
```

Then choose **Cell > Run All**.

## What the notebook does

| Step | Content |
|------|---------|
| 1 | Load the CSV with pandas; `info`, `describe`, missing values, class balance, plots. Then clean, one-hot encode, split (stratified 80/20) and scale. |
| 2 | `Sequential`: Dense(64) -> BatchNorm -> Dropout -> Dense(32) -> Dropout -> Dense(3, softmax) |
| 3 | Compile with `categorical_crossentropy` and the Adam optimizer |
| 4 | `fit` with `validation_split=0.2` and the `EarlyStopping`, `ReduceLROnPlateau` and `CSVLogger` callbacks. Then loss and accuracy curves, test-set evaluation, classification report and confusion matrix. |
| 5 | Save to `models/segment_classifier.h5`, reload it, check the predictions match, and predict a new customer |

## Using your own dataset

In the first cell of Step 1, change:

```python
DATA_PATH = "data/your_file.csv"
TARGET = "your_target_column"
```

Requirements: one row per sample, a categorical target column (any number of classes), and numeric or text feature columns. The notebook handles missing numeric values and one-hot encodes text columns. The output layer and the one-hot targets adapt to the number of classes automatically.

## Notes

- **`.h5` format:** Keras 3 treats `.h5` as the legacy format and may print a warning when saving. It still saves and loads correctly. For new projects, Keras recommends `.keras`. To switch, change `MODEL_PATH` in Step 5.
- **Inference needs more than the model.** The notebook also saves `models/preprocessing.joblib` (scaler, feature order, class names). Without it you can't prepare new data the same way as the training data.
- **Why `categorical_crossentropy`:** it expects one-hot targets, which is what `to_categorical` produces. If your labels are plain integers, use `sparse_categorical_crossentropy` and skip `to_categorical`.
- **No data leakage:** the scaler is fitted on the training split only, and the test set is never used during training.
