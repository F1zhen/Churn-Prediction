import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd

RAW_DEFAULT = "data/raw/Churn_Modelling.csv"
OUT_FEATURES_DEFAULT = "data/processed/training.csv"
OUT_LABELS_DEFAULT = "data/processed/test.csv"


def read_raw(raw_path: str) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    return df


def clean_and_engineer(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # Drop obvious identifier columns
    df = df.copy()
    for col in ["RowNumber", "CustomerId", "Surname"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Target
    if "Exited" not in df.columns:
        raise ValueError("Expected target column 'Exited' in raw data")
    y = df["Exited"].astype(int)

    # Handle missing values in numeric columns: to numeric then impute median
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Convert coercible non-numeric columns (like 'Age' with decimals etc.)
    for col in df.columns:
        if col not in numeric_cols and col != "Exited":
            # keep for encoding step
            pass

    # Basic cleaning for 'Age' or other numeric-like columns that may be object
    for col in ["Age"]:
        if col in df.columns and not np.issubdtype(df[col].dtype, np.number):
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if col not in numeric_cols:
                numeric_cols.append(col)

    # Impute median for numeric columns
    for col in numeric_cols:
        if col == "Exited":
            continue
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)

    # One-hot encode Geography; binary encode Gender
    geography_dummies = pd.get_dummies(df["Geography"], prefix="Geography") if "Geography" in df.columns else pd.DataFrame()
    if not geography_dummies.empty:
        df = pd.concat([df.drop(columns=["Geography"]), geography_dummies], axis=1)

    if "Gender" in df.columns:
        df["Gender_encoder"] = (df["Gender"].str.lower() == "male").astype(int)
        df = df.drop(columns=["Gender"])  # drop original

    # Remove the target from features
    X = df.drop(columns=["Exited"])

    # Ensure all are numeric and fill any remaining NaNs
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(X.median(numeric_only=True))

    return X, y


def ensure_dirs(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def preprocess(raw_path: str, out_features_path: str, out_labels_path: str) -> None:
    df = read_raw(raw_path)
    X, y = clean_and_engineer(df)
    ensure_dirs(out_features_path)
    ensure_dirs(out_labels_path)
    X.to_csv(out_features_path, index=True)
    y.to_csv(out_labels_path, index=True, header=True, name="Exited")


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess raw churn data into model-ready CSVs")
    parser.add_argument("--raw_path", type=str, default=RAW_DEFAULT)
    parser.add_argument("--out_features_path", type=str, default=OUT_FEATURES_DEFAULT)
    parser.add_argument("--out_labels_path", type=str, default=OUT_LABELS_DEFAULT)
    return parser.parse_args()


def main():
    args = parse_args()
    preprocess(
        raw_path=args.raw_path,
        out_features_path=args.out_features_path,
        out_labels_path=args.out_labels_path,
    )
    print("Preprocessing completed.")


if __name__ == "__main__":
    main()
