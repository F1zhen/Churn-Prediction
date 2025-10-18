import argparse
import json
import os
from dataclasses import dataclass
from typing import Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class TrainConfig:
    features_path: str = "data/processed/training.csv"
    labels_path: str = "data/processed/test.csv"
    output_dir: str = "artifacts"
    model_type: str = "rf"  # choices: "rf", "logreg"
    test_size: float = 0.2
    random_state: int = 42
    max_iter: int = 1000  # for logistic regression
    n_estimators: int = 300  # for random forest
    n_jobs: int = -1


def load_features_and_labels(features_path: str, labels_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    # Read with index_col=0 to drop the first unnamed index column if present
    features_df = pd.read_csv(features_path, index_col=0)
    labels_df = pd.read_csv(labels_path, index_col=0)

    # Expect labels to be under column 'Exited'
    if "Exited" not in labels_df.columns:
        raise ValueError("Labels file must contain an 'Exited' column")

    labels = labels_df["Exited"].astype(int)

    # Align by index to ensure consistent row ordering and length
    missing_in_features = labels.index.difference(features_df.index)
    if len(missing_in_features) > 0:
        raise ValueError(
            f"Labels contain indices not present in features: {len(missing_in_features)} missing"
        )

    # Select and align features rows by labels index
    aligned_features_df = features_df.loc[labels.index]

    # Basic sanity checks
    if aligned_features_df.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Features and labels mismatch: X rows={aligned_features_df.shape[0]} vs y rows={labels.shape[0]}"
        )

    # Ensure all feature dtypes are numeric
    for column_name in aligned_features_df.columns:
        if not np.issubdtype(aligned_features_df[column_name].dtype, np.number):
            aligned_features_df[column_name] = pd.to_numeric(
                aligned_features_df[column_name], errors="coerce"
            )
    # Fill any remaining NaNs with column medians
    aligned_features_df = aligned_features_df.fillna(aligned_features_df.median(numeric_only=True))

    return aligned_features_df, labels


def build_model_pipeline(config: TrainConfig) -> Pipeline:
    if config.model_type == "logreg":
        model = LogisticRegression(
            max_iter=config.max_iter,
            class_weight="balanced",
            random_state=config.random_state,
            n_jobs=config.n_jobs if hasattr(LogisticRegression(), "n_jobs") else None,
        )
        steps = [("scaler", StandardScaler()), ("model", model)]
        return Pipeline(steps=steps)
    elif config.model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=config.n_estimators,
            random_state=config.random_state,
            n_jobs=config.n_jobs,
            class_weight="balanced_subsample",
        )
        return Pipeline(steps=[("model", model)])
    else:
        raise ValueError("Unsupported model_type. Choose from: 'rf', 'logreg'")


def evaluate_model(y_true: pd.Series, y_pred: np.ndarray, y_prob: Optional[np.ndarray]) -> dict:
    metrics: dict = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["precision"] = float(precision_score(y_true, y_pred))
    metrics["recall"] = float(recall_score(y_true, y_pred))
    metrics["f1"] = float(f1_score(y_true, y_pred))

    # ROC AUC if probabilities available
    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    # Confusion matrix and classification report (as strings / lists)
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["classification_report"] = classification_report(y_true, y_pred, digits=4)

    return metrics


def save_artifacts(
    pipeline: Pipeline,
    metrics: dict,
    feature_columns: list,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Save model pipeline
    model_path = os.path.join(output_dir, "model.joblib")
    joblib.dump(pipeline, model_path)

    # Save metrics
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Save feature columns for future reference
    columns_path = os.path.join(output_dir, "feature_columns.json")
    with open(columns_path, "w", encoding="utf-8") as f:
        json.dump(feature_columns, f, indent=2, ensure_ascii=False)


def train_and_evaluate(config: TrainConfig) -> dict:
    features_df, labels = load_features_and_labels(config.features_path, config.labels_path)

    X_train, X_valid, y_train, y_valid = train_test_split(
        features_df,
        labels,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=labels,
    )

    pipeline = build_model_pipeline(config)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_valid)
    y_prob: Optional[np.ndarray] = None
    if hasattr(pipeline, "predict_proba"):
        try:
            y_prob = pipeline.predict_proba(X_valid)[:, 1]
        except Exception:
            y_prob = None
    elif hasattr(pipeline, "decision_function"):
        try:
            decision_scores = pipeline.decision_function(X_valid)
            # Map decision scores to [0,1] by rank-normalization if needed
            ranks = pd.Series(decision_scores).rank(pct=True)
            y_prob = ranks.to_numpy()
        except Exception:
            y_prob = None

    metrics = evaluate_model(y_valid, y_pred, y_prob)

    save_artifacts(pipeline, metrics, feature_columns=list(features_df.columns), output_dir=config.output_dir)

    return metrics


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a churn prediction model")
    parser.add_argument("--features_path", type=str, default=TrainConfig.features_path)
    parser.add_argument("--labels_path", type=str, default=TrainConfig.labels_path)
    parser.add_argument("--output_dir", type=str, default=TrainConfig.output_dir)
    parser.add_argument("--model", dest="model_type", type=str, choices=["rf", "logreg"], default=TrainConfig.model_type)
    parser.add_argument("--test_size", type=float, default=TrainConfig.test_size)
    parser.add_argument("--random_state", type=int, default=TrainConfig.random_state)
    parser.add_argument("--max_iter", type=int, default=TrainConfig.max_iter)
    parser.add_argument("--n_estimators", type=int, default=TrainConfig.n_estimators)
    parser.add_argument("--n_jobs", type=int, default=TrainConfig.n_jobs)
    args = parser.parse_args()
    return TrainConfig(
        features_path=args.features_path,
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        model_type=args.model_type,
        test_size=args.test_size,
        random_state=args.random_state,
        max_iter=args.max_iter,
        n_estimators=args.n_estimators,
        n_jobs=args.n_jobs,
    )


def main() -> None:
    config = parse_args()
    metrics = train_and_evaluate(config)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
