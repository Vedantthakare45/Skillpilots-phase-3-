"""
Reusable preprocessing pipeline for any CSV dataset.

Steps:
1. Load a CSV file with pandas
2. Identify numeric and categorical columns
3. Build a ColumnTransformer with StandardScaler and OneHotEncoder
4. Fit the transformer and transform the data
5. Save the fitted pipeline with joblib for later reuse

Usage:
    python preprocess_pipeline.py data.csv
    python preprocess_pipeline.py data.csv --target price
    python preprocess_pipeline.py data.csv --target price --output-dir results

Reuse the saved pipeline later:
    import joblib
    pipeline = joblib.load("results/preprocessor.joblib")
    X_new_processed = pipeline.transform(new_dataframe)
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(csv_path):
    """Step 1: load the CSV file."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {csv_path}: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def identify_columns(df):
    """Step 2: split columns into numeric and categorical."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
    print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols, categorical_cols):
    """Step 3: ColumnTransformer with StandardScaler and OneHotEncoder."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])


def main():
    parser = argparse.ArgumentParser(description="Reusable CSV preprocessing pipeline")
    parser.add_argument("csv_path", help="Path to the input CSV file")
    parser.add_argument("--target", default=None,
                        help="Target column to leave out of preprocessing (optional)")
    parser.add_argument("--output-dir", default="output",
                        help="Folder for the saved pipeline and processed data")
    args = parser.parse_args()

    df = load_data(args.csv_path)

    if args.target:
        if args.target not in df.columns:
            raise SystemExit(f"Target column '{args.target}' not found in the CSV.")
        features = df.drop(columns=[args.target])
    else:
        features = df

    numeric_cols, categorical_cols = identify_columns(features)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    # Step 4: fit and transform
    transformed = preprocessor.fit_transform(features)
    feature_names = preprocessor.get_feature_names_out()
    processed_df = pd.DataFrame(transformed, columns=feature_names, index=features.index)
    print(f"Processed data shape: {processed_df.shape}")

    # Step 5: save the fitted pipeline and the processed data
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, out_dir / "preprocessor.joblib")
    processed_df.to_csv(out_dir / "processed_data.csv", index=False)
    print(f"Saved pipeline to {out_dir / 'preprocessor.joblib'}")
    print(f"Saved processed data to {out_dir / 'processed_data.csv'}")


if __name__ == "__main__":
    main()
