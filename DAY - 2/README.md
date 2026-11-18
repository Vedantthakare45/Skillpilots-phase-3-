# Reusable CSV Preprocessing Pipeline

A command-line script that preprocesses any CSV dataset with scikit-learn. It scales numeric columns, one-hot encodes categorical columns, and saves the fitted pipeline so you can reuse it on new data later.

## What it does

1. Loads a CSV file with pandas
2. Identifies numeric and categorical columns
3. Builds a `ColumnTransformer` with `StandardScaler` (numeric) and `OneHotEncoder` (categorical)
4. Fits the transformer and transforms the data
5. Saves the fitted pipeline with `joblib` for later reuse

Missing values are filled automatically (median for numeric, most frequent for categorical), so it works on messy datasets too.

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python preprocess_pipeline.py data.csv
python preprocess_pipeline.py data.csv --target price --output-dir results
```

- `--target`: column to exclude from preprocessing (your label/target)
- `--output-dir`: where to save outputs (default: `output`)

Outputs:
- `preprocessor.joblib`: the fitted pipeline
- `processed_data.csv`: the transformed dataset

## Reuse the saved pipeline

```python
import joblib

pipeline = joblib.load("output/preprocessor.joblib")
X_new_processed = pipeline.transform(new_dataframe)
```

## Files

- `preprocess_pipeline.py`: the pipeline script
- `requirements.txt`: Python dependencies

## Tech

Python, pandas, scikit-learn, joblib

*Learning project from Skillplot (ML track, Data Preprocessing).*
