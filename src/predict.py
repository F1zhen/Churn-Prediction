import argparse
import json
import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd


def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def load_features(features_path: str) -> pd.DataFrame:
    df = pd.read_csv(features_path, index_col=0)
    # Ensure numeric types
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.fillna(df.median(numeric_only=True))


def align_columns(df: pd.DataFrame, feature_columns_path: Optional[str]) -> pd.DataFrame:
    if feature_columns_path is None:
        return df
    if not os.path.exists(feature_columns_path):
        raise FileNotFoundError(f"Feature columns file not found: {feature_columns_path}")
    with open(feature_columns_path, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    # Reindex to training-time columns, fill missing with 0
    aligned = df.reindex(columns=feature_columns, fill_value=0)
    return aligned


def predict(model_path: str, features_path: str, feature_columns_path: Optional[str], output_path: Optional[str]) -> pd.DataFrame:
    model = load_model(model_path)
    features = load_features(features_path)
    features = align_columns(features, feature_columns_path)

    # Predictions
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[:, 1]
    else:
        # Fall back to decision_function if available and scale to 0-1
        if hasattr(model, "decision_function"):
            scores = model.decision_function(features)
            ranks = pd.Series(scores).rank(pct=True).to_numpy()
            probabilities = ranks
        else:
            probabilities = None

    predictions = model.predict(features)

    result = pd.DataFrame(
        {
            "prediction": predictions.astype(int),
            **({"probability": probabilities} if probabilities is not None else {}),
        },
        index=features.index,
    )

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        result.to_csv(output_path, index=True)

    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Predict churn using a trained model")
    parser.add_argument("--model_path", type=str, default="artifacts/model.joblib")
    parser.add_argument("--features_path", type=str, default="data/processed/training.csv")
    parser.add_argument(
        "--feature_columns_path", type=str, default="artifacts/feature_columns.json"
    )
    parser.add_argument("--output_path", type=str, default="artifacts/predictions.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    df = predict(
        model_path=args.model_path,
        features_path=args.features_path,
        feature_columns_path=args.feature_columns_path,
        output_path=args.output_path,
    )
    # Print a small sample for quick view
    print(df.head().to_string())


if __name__ == "__main__":
    main()
